"""High-level strategy code"""

import traceback
from collections import deque
from time import time
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.const import State as GameStates
from bridge.router.actions import Action, Actions
from bridge.strategy import debug_draw, guards, params
from bridge.strategy.context import TickContext
from bridge.strategy.debug_draw import DebugDraw
from bridge.strategy.kick_model import KickModel
from bridge.strategy.memory import ROLE_LABELS, Memory, update_memory
from bridge.strategy.plays import (
    ball_placement,
    debug_scenarios,
    free_kick,
    kickoff,
    open_play,
    penalty,
    sparring,
    stop,
)
from bridge.strategy.roles import goalkeeper
from bridge.strategy.roles.assignment import assign_roles
from bridge.strategy.world import WorldModel

# Состояния, в которых может действовать исполнитель стандарта
SET_PIECE_STATES = (
    GameStates.PREPARE_KICKOFF,
    GameStates.KICKOFF,
    GameStates.FREE_KICK,
    GameStates.PREPARE_PENALTY,
    GameStates.PENALTY,
    GameStates.BALL_PLACEMENT,
)
TICK_HISTORY = 250


class Strategy:
    """Main class of strategy"""

    def __init__(
        self,
    ) -> None:
        self.we_active = False
        self.memory = Memory()
        self.kick_model = KickModel()
        self.debug = debug_scenarios.DebugScenarios()
        self.tick_times: deque[float] = deque(maxlen=TICK_HISTORY)

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

        start = time()
        if params.SPARRING_SECOND_TEAM and field.ally_color != const.COLOR:
            return sparring.play(field)

        try:
            actions = self.tick(field, start)
        except Exception as exc:  # pylint: disable=broad-except
            if not params.SAFE_MODE:
                raise
            actions = self.fallback(field, exc, start)

        self.tick_times.append(time() - start)
        return actions

    def tick(self, field: fld.Field, now: float) -> list[Optional[Action]]:
        """Один такт стратегии (ТЗ 5.3)"""
        memory = self.memory
        memory.we_active = self.we_active
        memory.update_state(field.game_state, now)
        if field.game_state not in SET_PIECE_STATES and not memory.set_piece_armed:
            memory.set_piece_kicker_id = None

        world = WorldModel.build(field, memory, now)
        update_memory(memory, field, world, now, self.kick_model)
        draw = DebugDraw(field.strategy_image)
        ctx = TickContext(field, world, memory, now, self.we_active, self.kick_model, draw)
        assign_roles(ctx)

        raw = self.select_play(ctx)
        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
        for r_id, action in raw.items():
            actions[r_id] = action
        actions = guards.apply(ctx, actions)

        self.visualize(ctx)
        return actions

    def select_play(self, ctx: TickContext) -> dict[int, Action]:
        """Розыгрыш по состоянию игры"""
        state = ctx.field.game_state
        world = ctx.world
        match state:
            case GameStates.HALT | GameStates.TIMEOUT:
                return {robot.r_id: Actions.Stop() for robot in world.allies}
            case GameStates.STOP:
                return stop.play(ctx)
            case GameStates.PREPARE_KICKOFF:
                return kickoff.prepare(ctx)
            case GameStates.KICKOFF:
                return kickoff.kickoff(ctx)
            case GameStates.FREE_KICK:
                return free_kick.play(ctx)
            case GameStates.PREPARE_PENALTY:
                return penalty.prepare(ctx)
            case GameStates.PENALTY:
                return penalty.penalty(ctx)
            case GameStates.BALL_PLACEMENT:
                return ball_placement.play(ctx)
            case GameStates.DEBUG:
                return self.debug.play(ctx)
        if world.ball_out_of_field:
            # R-6.9: мяч за полем, игра ещё не остановлена - расстановка как в STOP
            return stop.play(ctx)
        return open_play.play(ctx)

    def fallback(self, field: fld.Field, exc: BaseException, now: float) -> list[Optional[Action]]:
        """R-5.6: исключение в такте - всем Stop, сообщение не чаще раза в секунду"""
        if now - self.memory.last_exception_print_time > params.EXCEPTION_PRINT_PERIOD:
            self.memory.last_exception_print_time = now
            print(f"[STRATEGY] tick failed: {exc!r}\n{traceback.format_exc()}")
        field.strategy_image.send_telemetry("STRATEGY", f"TICK FAILED: {exc!r}")
        self.memory.metric_add("exceptions")
        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
        for robot in field.active_allies(True):
            actions[robot.r_id] = Actions.Stop()
        return actions

    def visualize(self, ctx: TickContext) -> None:
        """Отрисовка и телеметрия (ТЗ, раздел 14)"""
        draw = ctx.draw
        world = ctx.world
        memory = ctx.memory
        for robot in world.allies:
            role = ctx.roles.get(robot.r_id)
            label = ROLE_LABELS[role] if role is not None else "?"
            if ctx.is_banned(robot):
                label += " X2"
            draw.text(robot.get_pos() + aux.Point(0, 160), label, debug_draw.YELLOW, 70)
            target = ctx.targets.get(robot.r_id)
            if target is not None:
                draw.line(robot.get_pos(), target, debug_draw.GRAY, 1)
        for r_id, inter in world.our_intercepts.items():
            if inter.t < params.INTERCEPT_HORIZON and r_id == world.our_best_id:
                draw.circle(inter.point, debug_draw.CYAN, 40)
        if world.their_best_id is not None:
            their = world.their_intercepts[world.their_best_id]
            if their.t < params.INTERCEPT_HORIZON:
                draw.circle(their.point, debug_draw.RED, 40)
        gk = world.gk
        if gk is not None and not world.ball_lost_vision:
            # Сектор створа, который закрывает наш вратарь
            left, right = goalkeeper.blocked_interval(ctx, world.ball, gk.get_pos())
            draw.line(world.ball, left, debug_draw.BLUE, 1)
            draw.line(world.ball, right, debug_draw.BLUE, 1)
        for track in memory.dribble.values():
            draw.circle(track.start_ball_pos, debug_draw.ORANGE, params.DRIBBLE_SOFT_LIMIT)
            draw.circle(track.start_ball_pos, debug_draw.RED, params.RULE_MAX_DRIBBLE_DIST)

        times = list(self.tick_times)
        avg_ms = 1000 * sum(times) / len(times) if times else 0.0
        max_ms = 1000 * max(times) if times else 0.0
        memory.metrics["tick_avg_ms"] = avg_ms
        memory.metric_max("tick_max_ms", max_ms)
        area_time = ctx.now - memory.ball_in_our_area_since if memory.ball_in_our_area_since is not None else 0.0
        decision = memory.attacker_decision
        roles = ", ".join(f"{r_id}={ROLE_LABELS[r]}" for r_id, r in sorted(ctx.roles.items()))
        draw.tele(f"state={ctx.field.game_state.name} we_active={ctx.we_active} possession={world.possession.name}")
        draw.tele(f"roles: {roles}")
        if decision is not None:
            draw.tele(f"attacker: {decision.kind} util={decision.utility: .2f}")
        for r_id in memory.dribble:
            draw.tele(f"dribble {r_id}: {memory.dribble_disp(r_id, world.ball): .0f} mm")
        draw.tele(f"ball in our area: {area_time: .1f} s, double touch ban: {memory.double_touch_banned_id}")
        draw.tele(f"last kick max speed: {memory.last_kick_max_speed: .0f} mm/s, ball race: {world.ball_race: .2f} s")
        draw.tele(f"tick avg {avg_ms: .1f} ms, max {max_ms: .1f} ms")
        metrics = ", ".join(f"{k}={v: .0f}" for k, v in sorted(memory.metrics.items()) if not k.startswith("tick"))
        draw.tele(f"metrics: {metrics}")
        warning = self.kick_model.warning()
        if warning is not None:
            draw.tele(warning)
        for line in ctx.warnings:
            draw.tele(f"WARN {line}")
        draw.flush("STRATEGY")
