"""
Финальная проверка и коррекция действий по Регламенту (ТЗ, раздел 10)

G-1 HALT/TIMEOUT -> Stop; G-2 действие каждому; G-3 белый список; G-4 легальность целей;
G-5 двойное касание; G-6 чрезмерный дриблинг; G-7 скорость мяча; G-11 скопление;
G-12 нет «чипов»; G-13 цели ударов в поле.
"""

import math
from typing import Iterator, Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.router import legality
from bridge.router.actions import (
    Action,
    Actions,
    DribbleActions,
    KickActions,
    StrategyActions,
)
from bridge.router.actions.action import ComposedAction
from bridge.router.actions.dumb_actions import DumbActions
from bridge.strategy import params
from bridge.strategy.context import TickContext, angle_to, carry_limits, unit
from bridge.strategy.memory import Memory, Role

State = const.State

BALL_ACTIONS: tuple[type, ...] = (
    Actions.BallGrab,
    KickActions.Kick,
    KickActions.SetPieceKick,
    DribbleActions.HoldBall,
    DribbleActions.DribbleTo,
    DribbleActions.DribbleKick,
    DribbleActions.TapForward,
    StrategyActions.Receive,
    StrategyActions.CatchBall,
)
VOLTAGE_ACTIONS: tuple[type, ...] = (
    KickActions.Kick,
    KickActions.SetPieceKick,
    DribbleActions.DribbleKick,
    DribbleActions.TapForward,
    DumbActions.ControlVoltageAction,
)
KICK_ACTIONS: tuple[type, ...] = VOLTAGE_ACTIONS + (DumbActions.ShootAction,)
TARGET_ACTIONS = (Actions.GoToPoint, Actions.GoToPointIgnore)


# --- G-4: легальность целей ---


def keep_ball_dist(field: fld.Field, memory: Memory, we_active: bool, robot_id: int) -> float:
    """Минимальное расстояние центра робота до мяча по состоянию игры"""
    state = field.game_state
    keep = 0.0
    is_kicker = memory.set_piece_kicker_id == robot_id and we_active
    if state == State.STOP:
        keep = params.KEEP_BALL_DIST
    elif state in (State.KICKOFF, State.FREE_KICK) and not we_active:
        keep = params.KEEP_BALL_DIST
    elif state == State.PREPARE_KICKOFF and not is_kicker:
        keep = params.KEEP_BALL_DIST
    elif state == State.BALL_PLACEMENT and not (params.AUTO_BALL_PLACEMENT and we_active and is_kicker):
        keep = params.KEEP_BALL_DIST
    elif state in (State.PREPARE_PENALTY, State.PENALTY) and robot_id != field.gk_id and not is_kicker:
        keep = params.PENALTY_OTHERS_DIST
    if memory.double_touch_banned_id == robot_id:
        keep = max(keep, params.DOUBLE_TOUCH_KEEP_DIST)
    return keep


def _push_out_of_stadium(point: aux.Point, a: aux.Point, b: aux.Point, radius: float) -> aux.Point:
    closest = aux.closest_point_on_line(a, b, point, "S")
    vec = point - closest
    if vec.mag() >= radius:
        return point
    if vec.mag() < 1:
        vec = aux.rotate(b - a, math.pi / 2) if (b - a).mag() > 1 else aux.Point(0, 1)
    return closest + vec.unity() * (radius + legality.PUSH_EPS)


def legal_target(field: fld.Field, memory: Memory, we_active: bool, robot_id: int, point: aux.Point) -> aux.Point:
    """
    G-4: спроецировать цель в допустимую область: поле (R-0.8), штрафные, отступ от
    чужой штрафной, дистанция до мяча, пенальти, своя половина в начальном ударе, «стадион».
    """
    state = field.game_state
    ball = field.ball.get_pos()
    keep = keep_ball_dist(field, memory, we_active, robot_id)
    is_kicker = memory.set_piece_kicker_id == robot_id and we_active
    is_gk = robot_id == field.gk_id
    attack = -field.polarity

    p = point
    for _ in range(legality.LEGAL_ITERATIONS):
        p = legality.legal_target(field, robot_id, p, keep_ball_dist=keep)
        changed = False
        if state in (State.PREPARE_KICKOFF, State.KICKOFF) and not is_kicker:
            # 5.3.2: своя половина, вне центрального круга
            max_x_att = -params.KICKOFF_HALF_GAP
            if p.x * attack > max_x_att:
                p = aux.Point(max_x_att * attack, p.y)
                changed = True
            circle_r = params.KICKOFF_CIRCLE_R + const.ROBOT_R + params.SAFETY_MARGIN
            if aux.dist(p, aux.Point(0, 0)) < circle_r:
                p = legality.push_out_of_circle(p, aux.Point(0, 0), circle_r, aux.Point(-attack, 0))
                changed = True
        if state in (State.PREPARE_PENALTY, State.PENALTY) and not is_gk and not is_kicker:
            # Не между мячом и воротами: за линией мяча относительно атакуемых ворот
            penalty_dir = attack if we_active else -attack
            limit = ball.x * penalty_dir - const.ROBOT_R
            if p.x * penalty_dir > limit:
                p = aux.Point(limit * penalty_dir, p.y)
                changed = True
        if state == State.BALL_PLACEMENT and field.ball_placement_pos is not None and not is_kicker:
            new_p = _push_out_of_stadium(p, ball, field.ball_placement_pos, params.KEEP_BALL_DIST)
            changed = changed or new_p != p
            p = new_p
        if not changed:
            break
    p = legality.legal_target(field, robot_id, p, keep_ball_dist=keep)
    if _extra_ok(field, memory, we_active, robot_id, p):
        return p

    def ok(q: aux.Point) -> bool:
        base = legality.is_legal_target(field, robot_id, q) and aux.dist(q, ball) >= keep + legality.PUSH_EPS
        return base and _extra_ok(field, memory, we_active, robot_id, q)

    found = legality.nearest_valid(point, ok)
    return found if found is not None else p


def _extra_ok(field: fld.Field, memory: Memory, we_active: bool, robot_id: int, p: aux.Point) -> bool:
    """Ограничения стратегии сверх роутерных: своя половина в начальном ударе, сторона в пенальти, «стадион»"""
    state = field.game_state
    ball = field.ball.get_pos()
    is_kicker = memory.set_piece_kicker_id == robot_id and we_active
    attack = -field.polarity
    if state in (State.PREPARE_KICKOFF, State.KICKOFF) and not is_kicker:
        circle_r = params.KICKOFF_CIRCLE_R + const.ROBOT_R + params.SAFETY_MARGIN
        if p.x * attack > -params.KICKOFF_HALF_GAP + 1 or aux.dist(p, aux.Point(0, 0)) < circle_r - 1:
            return False
    if state in (State.PREPARE_PENALTY, State.PENALTY) and robot_id != field.gk_id and not is_kicker:
        penalty_dir = attack if we_active else -attack
        if p.x * penalty_dir > ball.x * penalty_dir - const.ROBOT_R + 1:
            return False
    if state == State.BALL_PLACEMENT and field.ball_placement_pos is not None and not is_kicker:
        closest = aux.closest_point_on_line(ball, field.ball_placement_pos, p, "S")
        if aux.dist(closest, p) < params.KEEP_BALL_DIST - 1:
            return False
    return True


def legal(ctx: TickContext, robot: rbt.Robot, point: aux.Point) -> aux.Point:
    """legal_target в контексте такта"""
    return legal_target(ctx.field, ctx.memory, ctx.we_active, robot.r_id, point)


def go_to(
    ctx: TickContext,
    robot: rbt.Robot,
    point: aux.Point,
    angle: float,
    *,
    ignore_ball: bool = False,
    avoid_ball: Optional[bool] = None,
    target_vel: Optional[aux.Point] = None,
) -> Action:
    """GoToPoint в легальную цель (с запоминанием цели для отрисовки)"""
    target = legal(ctx, robot, point)
    ctx.targets[robot.r_id] = target
    return Actions.GoToPoint(
        target,
        angle,
        ignore_ball=ignore_ball,
        ignore_robots={},
        avoid_ball=avoid_ball,
        target_vel=target_vel if target_vel is not None else aux.Point(0, 0),
    )


def hold_position(ctx: TickContext, robot: rbt.Robot) -> Action:
    """Безопасное действие: остаться на месте (легально) лицом к мячу"""
    pos = robot.get_pos()
    return go_to(ctx, robot, pos, angle_to(pos, ctx.world.ball))


# --- Обход дерева действий ---


def walk(action: Action) -> Iterator[Action]:
    """Все действия в статической структуре compose"""
    yield action
    if isinstance(action, ComposedAction):
        yield from walk(action.first)
        yield from walk(action.second)


def has_type(action: Action, types: tuple[type, ...]) -> bool:
    """В дереве есть действие одного из типов"""
    return any(isinstance(a, types) for a in walk(action))


def only_receive(action: Action) -> bool:
    """Все действия с мячом в дереве - приём паса"""
    return all(isinstance(a, StrategyActions.Receive) for a in walk(action) if isinstance(a, BALL_ACTIONS))


# --- G-3: белый список ---


def ball_actions_allowed(ctx: TickContext, robot: rbt.Robot, action: Action) -> bool:
    """Можно ли этому роботу действие с мячом в текущем состоянии"""
    state = ctx.field.game_state
    memory = ctx.memory
    is_kicker = memory.set_piece_kicker_id == robot.r_id
    if state in (State.STOP, State.PREPARE_KICKOFF, State.PREPARE_PENALTY, State.HALT, State.TIMEOUT):
        return False
    if state in (State.KICKOFF, State.FREE_KICK):
        if not ctx.we_active:
            return False
        return is_kicker or only_receive(action)
    if state == State.BALL_PLACEMENT:
        return params.AUTO_BALL_PLACEMENT and ctx.we_active and is_kicker
    if state == State.PENALTY:
        if ctx.we_active:
            return is_kicker
        return ctx.is_gk(robot) and memory.penalty_ball_in_play
    return True


def _check_whitelist(ctx: TickContext, robot: rbt.Robot, action: Action) -> Action:
    if has_type(action, BALL_ACTIONS) and not ball_actions_allowed(ctx, robot, action):
        ctx.warnings.append(f"G-3: robot {robot.r_id} ball action blocked in {ctx.field.game_state.name}")
        return hold_position(ctx, robot)
    if ctx.field.game_state == State.STOP:
        for sub in walk(action):
            if isinstance(sub, Actions.GoToPoint) and sub.avoid_ball is False:
                sub.avoid_ball = None
    return action


# --- G-4 ---


def _check_targets(ctx: TickContext, robot: rbt.Robot, action: Action) -> None:
    for sub in walk(action):
        if isinstance(sub, (Actions.GoToPoint, Actions.GoToPointIgnore)):
            sub.target_pos = legal(ctx, robot, sub.target_pos)


# --- G-5 ---


def _check_double_touch(ctx: TickContext, robot: rbt.Robot, action: Action) -> Action:
    if not ctx.is_banned(robot):
        return action
    if has_type(action, BALL_ACTIONS):
        ctx.warnings.append(f"G-5: robot {robot.r_id} must not touch the ball")
        ball = ctx.world.ball
        away = robot.get_pos() - ball
        if away.mag() < 1:
            away = ctx.field.ally_goal.center - ball
        point = ball + away.unity() * (params.DOUBLE_TOUCH_KEEP_DIST + const.ROBOT_R)
        return go_to(ctx, robot, point, angle_to(point, ball))
    return action


# --- G-6 ---


def _check_dribble(ctx: TickContext, robot: rbt.Robot, action: Action) -> Action:
    if not ctx.has_ball(robot):
        return action
    disp = ctx.memory.dribble_disp(robot.r_id, ctx.world.ball)
    if disp < params.DRIBBLE_HARD_LIMIT or has_type(action, KICK_ACTIONS):
        return action
    ctx.warnings.append(f"G-6: robot {robot.r_id} dribble {disp: .0f} mm, forced release")
    heading = robot.get_angle()
    attack = unit(0.0 if ctx.world.attack_sign > 0 else math.pi)
    if aux.scal_mult(unit(heading), attack) > -0.3:
        voltage = ctx.kick_model.tap_voltage(params.TAP_DIST_MAX)
        ctx.memory.kick_voltage_issued[robot.r_id] = voltage
        return DribbleActions.TapForward(heading, voltage, params.TAP_ALIGN_TOL, carry_limits())
    return DribbleActions.HoldBall(
        angle_to(ctx.world.ball, ctx.field.enemy_goal.center),
        carry_limits(),
        dribbler_speed=params.DRIBBLER_HOLD,
        hold_dist=params.BALL_HOLD_DIST,
    )


# --- G-7, G-12, G-13 ---


def _is_shot_target(ctx: TickContext, target: aux.Point) -> bool:
    goal = ctx.field.enemy_goal
    return abs(target.x - goal.center.x) < const.BALL_R * 3 and abs(target.y - goal.center.y) <= const.GOAL_DY / 2


def _check_kicks(ctx: TickContext, robot: rbt.Robot, action: Action) -> None:
    max_voltage = ctx.kick_model.voltage_for_speed(params.BALL_SPEED_SAFE)
    if ctx.field.game_state == State.DEBUG and params.DEBUG_SCENARIO == "KICK_CALIBRATION":
        max_voltage = 15  # калибровка удара: только в DEBUG, не на матче
    for sub in walk(action):
        if isinstance(sub, KickActions.Kick):
            if sub.is_upper and not params.USE_CHIP_KICK:  # G-12
                ctx.warnings.append(f"G-12: robot {robot.r_id} chip kick disabled")
                sub.is_upper = False
            if sub.pass_pos is not None:  # R-0.5: только модель удара
                sub.voltage = ctx.kick_model.pass_voltage(aux.dist(ctx.world.ball, sub.pass_pos))
                sub.pass_pos = None
        voltage_attr = "one_touch_voltage" if isinstance(sub, StrategyActions.Receive) else "voltage"
        if isinstance(sub, VOLTAGE_ACTIONS + (StrategyActions.Receive,)):
            voltage = int(getattr(sub, voltage_attr))
            if voltage > max_voltage:
                ctx.warnings.append(f"G-7: robot {robot.r_id} voltage {voltage} -> {max_voltage}")
                voltage = max_voltage
                setattr(sub, voltage_attr, voltage)
            if voltage > 0:
                ctx.memory.kick_voltage_issued[robot.r_id] = voltage
        if isinstance(sub, KICK_ACTIONS) and hasattr(sub, "target_pos"):
            target = sub.target_pos
            if not _is_shot_target(ctx, target) and not ctx.world.in_field(target, const.BALL_R):  # G-13
                margin = params.CLEAR_BORDER_MARGIN
                sub.target_pos = aux.Point(
                    aux.minmax(target.x, const.FIELD_DX - margin), aux.minmax(target.y, const.FIELD_DY - margin)
                )


# --- G-11 ---


def _check_crowd(ctx: TickContext, actions: list[Optional[Action]]) -> None:
    if fld.is_ball_not_in_play(ctx.field.game_state):
        return
    ball = ctx.world.ball
    receiver = ctx.memory.pass_in_flight.to_id if ctx.memory.pass_in_flight is not None else None
    for robot in ctx.world.field_allies:
        if ctx.roles.get(robot.r_id) == Role.ATTACKER or robot.r_id == receiver:
            continue
        action = actions[robot.r_id]
        if action is None:
            continue
        for sub in walk(action):
            if not isinstance(sub, (Actions.GoToPoint, Actions.GoToPointIgnore)):
                continue
            if aux.dist(sub.target_pos, ball) < params.NO_CROWD_RADIUS:
                away = sub.target_pos - ball
                if away.mag() < 1:
                    away = ctx.field.ally_goal.center - ball
                sub.target_pos = legal(ctx, robot, ball + away.unity() * (params.NO_CROWD_RADIUS + 10))


# --- Главная функция ---


def apply(ctx: TickContext, actions: list[Optional[Action]]) -> list[Optional[Action]]:
    """Проверить и скорректировать действия всех роботов"""
    state = ctx.field.game_state
    active_ids = {robot.r_id for robot in ctx.world.allies}
    result: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT

    if state in (State.HALT, State.TIMEOUT):  # G-1
        for r_id in active_ids:
            result[r_id] = Actions.Stop()
        return result

    for robot in ctx.world.allies:
        action = actions[robot.r_id]
        if action is None:  # G-2
            ctx.warnings.append(f"G-2: robot {robot.r_id} had no action -> Stop")
            result[robot.r_id] = Actions.Stop()
            continue
        try:
            action = _check_whitelist(ctx, robot, action)
            action = _check_double_touch(ctx, robot, action)
            action = _check_dribble(ctx, robot, action)
            _check_targets(ctx, robot, action)
            _check_kicks(ctx, robot, action)
        except Exception as exc:  # pylint: disable=broad-except
            if not params.SAFE_MODE:
                raise
            ctx.warnings.append(f"guard error robot {robot.r_id}: {exc!r}")
            action = Actions.Stop()
        result[robot.r_id] = action

    _check_crowd(ctx, result)
    return result
