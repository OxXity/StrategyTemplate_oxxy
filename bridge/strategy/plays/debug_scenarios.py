"""
Сценарии калибровки в состоянии DEBUG (ТЗ, 9.9 и раздел 12)

params.DEBUG_SCENARIO: NONE | KICK_CALIBRATION | TURN_TEST | DRIBBLE_TEST | PASS_TEST | SHOOT_TEST | GK_TEST
Результаты печатаются в консоль и в телеметрию готовыми строками для params.py.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, CarryLimits, DribbleActions
from bridge.strategy import evaluation, params
from bridge.strategy.context import (
    TickContext,
    angle_diff,
    angle_to,
    carry_limits,
    unit,
)
from bridge.strategy.guards import go_to, hold_position
from bridge.strategy.plays import open_play
from bridge.strategy.roles import common

KICK_VOLTAGES = list(range(1, 16))
KICK_REPEATS = 3
KICK_TARGET_DIST = 1500
REAL_WAIT_RADIUS = 1000  # на реальном поле робот ждёт мяч ближе этого
TURN_OMEGAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
TURN_ANGLES = [math.pi / 2, -math.pi / 2, math.pi]
TURN_TRIALS = 10
TURN_DONE_TOL = 0.05
TURN_TIMEOUT = 5.0
TURN_RECOMMEND_K = 0.7
DRIBBLE_SQUARE = 250
DRIBBLE_CORNER_TOL = 60
PASS_ARRIVALS = [600.0, 800.0, 1000.0, 1200.0]
PASS_DISTANCES = [1000.0, 1500.0, 2000.0]
PASS_TIMEOUT = 4.0
DECEL_MIN_SPEED = 200
SCENARIOS = ("KICK_CALIBRATION", "TURN_TEST", "DRIBBLE_TEST", "PASS_TEST", "SHOOT_TEST", "GK_TEST")


@dataclass
class ScenarioState:  # pylint: disable=too-many-instance-attributes
    """Состояние сценария между тактами"""

    phase: str = "FETCH"
    phase_time: float = 0.0
    index: int = 0
    trial: int = 0
    start_angle: float = 0.0
    losses: dict[float, int] = field(default_factory=dict)
    max_omega: dict[float, float] = field(default_factory=dict)
    samples: list[tuple[float, float]] = field(default_factory=list)
    results: list[tuple[float, float, float]] = field(default_factory=list)
    stats: dict[tuple[float, float], list[int]] = field(default_factory=dict)
    passer: Optional[int] = None
    corner: int = 0
    lost_since: Optional[float] = None
    done: bool = False
    printed: bool = False


def _print(ctx: TickContext, text: str) -> None:
    print(f"[DEBUG SCENARIO] {text}")
    ctx.draw.tele(text)


class DebugScenarios:
    """Исполнитель сценариев"""

    def __init__(self) -> None:
        self.state = ScenarioState()
        self.name = params.DEBUG_SCENARIO

    def set_phase(self, phase: str, now: float) -> None:
        """Сменить фазу"""
        self.state.phase = phase
        self.state.phase_time = now

    def play(self, ctx: TickContext) -> dict[int, Action]:
        """Действия в DEBUG"""
        if self.name not in SCENARIOS:
            return open_play.play(ctx)
        robots = sorted(ctx.world.field_allies, key=lambda r: r.r_id)
        actions: dict[int, Action] = {r.r_id: hold_position(ctx, r) for r in ctx.world.allies}
        if self.name == "GK_TEST":
            if ctx.world.gk is not None:
                actions[ctx.world.gk.r_id] = open_play.GOALKEEPER.act(ctx.world.gk, ctx)
            return actions
        if not robots:
            return actions
        robot = robots[0]
        ctx.draw.tele(f"SCENARIO {self.name} phase={self.state.phase} idx={self.state.index} trial={self.state.trial}")
        if self.state.done:
            return actions
        if self.name == "KICK_CALIBRATION":
            actions[robot.r_id] = self.kick_calibration(ctx, robot)
        elif self.name == "TURN_TEST":
            actions[robot.r_id] = self.turn_test(ctx, robot)
        elif self.name == "DRIBBLE_TEST":
            actions[robot.r_id] = self.dribble_test(ctx, robot)
        elif self.name == "SHOOT_TEST":
            actions[robot.r_id] = self.shoot_test(ctx, robot)
        elif self.name == "PASS_TEST" and len(robots) >= 2:
            actions.update(self.pass_test(ctx, robots[0], robots[1]))
        return actions

    # --- общее ---

    def _fetch(self, ctx: TickContext, robot: rbt.Robot, angle: float) -> Optional[Action]:
        """Подобрать мяч; None - мяч уже у робота"""
        if ctx.has_ball(robot):
            return None
        ball = ctx.world.ball
        if not const.IS_SIMULATOR_USED and aux.dist(ball, robot.get_pos()) > REAL_WAIT_RADIUS:
            return go_to(ctx, robot, robot.get_pos(), angle_to(robot.get_pos(), ball))
        return Actions.BallGrab(angle, dribbler_speed=params.DRIBBLER_HOLD)

    @staticmethod
    def _free_angle(ctx: TickContext) -> float:
        """Направление в свободную сторону: к центру поля"""
        ball = ctx.world.ball
        center = aux.Point(0, 0)
        if aux.dist(ball, center) < 300:
            return 0.0
        return angle_to(ball, center)

    # --- KICK_CALIBRATION ---

    def kick_calibration(self, ctx: TickContext, robot: rbt.Robot) -> Action:
        """Удары каждым напряжением по KICK_REPEATS раз, скорость мяча и BALL_DECEL"""
        st = self.state
        now = ctx.now
        if st.index >= len(KICK_VOLTAGES) * KICK_REPEATS:
            self._kick_report(ctx)
            st.done = True
            return hold_position(ctx, robot)
        voltage = KICK_VOLTAGES[st.index // KICK_REPEATS]
        angle = self._free_angle(ctx)
        if st.phase == "FETCH":
            action = self._fetch(ctx, robot, angle)
            if action is not None:
                return action
            self.set_phase("KICK", now)
        if st.phase == "KICK":
            log = ctx.memory.kick_log
            if log is not None and log.robot_id == robot.r_id and log.start_time >= st.phase_time:
                st.samples = []
                self.set_phase("MEASURE", now)
            else:
                ctx.memory.kick_voltage_issued[robot.r_id] = voltage
                target = ctx.world.ball + unit(angle) * KICK_TARGET_DIST
                return DribbleActions.DribbleKick(target, voltage, 0.1, params.KICK_MAX_ANG_VEL, carry_limits())
        if st.phase == "MEASURE":
            st.samples.append((now, ctx.world.ball_speed))
            if now - st.phase_time > params.KICK_LOG_TIME * 3 or not ctx.world.ball_moving:
                vmax = max(s for _, s in st.samples) if st.samples else 0.0
                decel = self._estimate_decel(st.samples)
                st.results.append((voltage, vmax, decel))
                _print(ctx, f"kick voltage={voltage} max_speed={vmax: .0f} decel={decel: .0f}")
                st.index += 1
                self.set_phase("FETCH", now)
        return go_to(ctx, robot, robot.get_pos(), robot.get_angle())

    @staticmethod
    def _estimate_decel(samples: list[tuple[float, float]]) -> float:
        """Замедление по спаду скорости на участке качения (МНК)"""
        if not samples:
            return 0.0
        peak = max(range(len(samples)), key=lambda i: samples[i][1])
        tail = [(t, s) for t, s in samples[peak + 2 :] if s > DECEL_MIN_SPEED]
        if len(tail) < 3:
            return 0.0
        t0 = tail[0][0]
        ts = [t - t0 for t, _ in tail]
        ss = [s for _, s in tail]
        mean_t = sum(ts) / len(ts)
        mean_s = sum(ss) / len(ss)
        den = sum((t - mean_t) ** 2 for t in ts)
        if den <= 0:
            return 0.0
        slope = sum((t - mean_t) * (s - mean_s) for t, s in zip(ts, ss)) / den
        return -slope

    def _kick_report(self, ctx: TickContext) -> None:
        st = self.state
        if st.printed:
            return
        st.printed = True
        rows = []
        decels = []
        for voltage in KICK_VOLTAGES:
            speeds = [s for v, s, _ in st.results if v == voltage]
            decels += [d for v, _, d in st.results if v == voltage and d > 0]
            if speeds:
                rows.append(f"({voltage}, {sum(speeds) / len(speeds): .0f})")
        table = "KICK_TABLE_" + ("SIM" if const.IS_SIMULATOR_USED else "REAL")
        _print(ctx, f"{table} = [" + ", ".join(rows) + "]")
        if decels:
            _print(ctx, f"BALL_DECEL = {sum(decels) / len(decels): .0f}")

    # --- TURN_TEST ---

    def turn_test(self, ctx: TickContext, robot: rbt.Robot) -> Action:
        """Развороты с мячом с разными ω; потери мяча и рекомендация BALL_HOLD_MAX_ANG_SPEED"""
        st = self.state
        now = ctx.now
        if st.index >= len(TURN_OMEGAS):
            self._turn_report(ctx)
            st.done = True
            return hold_position(ctx, robot)
        omega = TURN_OMEGAS[st.index]
        if st.phase == "FETCH":
            action = self._fetch(ctx, robot, self._free_angle(ctx))
            if action is not None:
                return action
            st.start_angle = robot.get_angle()
            self.set_phase("TURN", now)
        target = aux.wind_down_angle(st.start_angle + TURN_ANGLES[st.trial % len(TURN_ANGLES)])
        st.max_omega[omega] = max(st.max_omega.get(omega, 0.0), abs(robot.get_anglevel()))
        lost = not ctx.has_ball(robot)
        if lost:
            st.lost_since = st.lost_since if st.lost_since is not None else now
        else:
            st.lost_since = None
        finished = angle_diff(robot.get_angle(), target) < TURN_DONE_TOL or now - st.phase_time > TURN_TIMEOUT
        if st.lost_since is not None and now - st.lost_since > params.BALL_LOST_TIMEOUT:
            st.losses[omega] = st.losses.get(omega, 0) + 1
            finished = True
        if finished:
            st.trial += 1
            st.lost_since = None
            if st.trial >= TURN_TRIALS:
                _print(ctx, f"turn omega={omega} losses={st.losses.get(omega, 0)}/{TURN_TRIALS}")
                st.trial = 0
                st.index += 1
            self.set_phase("FETCH", now)
        limits = CarryLimits(
            params.BALL_HOLD_MAX_SPEED_FWD,
            params.BALL_HOLD_MAX_SPEED_SIDE,
            params.BALL_HOLD_MAX_SPEED_BACK,
            params.BALL_HOLD_MAX_ACC,
            omega,
            omega * 3,
        )
        return DribbleActions.HoldBall(target, limits, dribbler_speed=params.DRIBBLER_HOLD, hold_dist=params.BALL_HOLD_DIST)

    def _turn_report(self, ctx: TickContext) -> None:
        st = self.state
        if st.printed:
            return
        st.printed = True
        safe = [w for w in TURN_OMEGAS if st.losses.get(w, 0) == 0]
        best = max(safe) if safe else 0.0
        measured = st.max_omega.get(best, best)
        _print(ctx, f"max omega without losses: {best} (measured {measured: .2f} rad/s)")
        _print(ctx, f"BALL_HOLD_MAX_ANG_SPEED = {TURN_RECOMMEND_K * min(best, measured or best): .2f}")

    # --- DRIBBLE_TEST ---

    def dribble_test(self, ctx: TickContext, robot: rbt.Robot) -> Action:
        """Ведение по квадрату внутри круга 0.8 м: трекер G-6 и лимиты R-4.3"""
        st = self.state
        track = ctx.memory.dribble.get(robot.r_id)
        if not ctx.has_ball(robot) or track is None:
            action = self._fetch(ctx, robot, self._free_angle(ctx))
            return action if action is not None else hold_position(ctx, robot)
        corners = [aux.Point(1, 1), aux.Point(-1, 1), aux.Point(-1, -1), aux.Point(1, -1)]
        target = track.start_ball_pos + corners[st.corner % 4] * DRIBBLE_SQUARE
        if aux.dist(ctx.world.ball, target) < DRIBBLE_CORNER_TOL:
            st.corner += 1
            disp = ctx.memory.dribble_disp(robot.r_id, ctx.world.ball)
            _print(
                ctx, f"dribble corner {st.corner} disp={disp: .0f} max={ctx.memory.metrics.get('max_dribble_disp', 0): .0f}"
            )
        ctx.draw.circle(track.start_ball_pos, (255, 200, 0), params.DRIBBLE_SOFT_LIMIT)
        return DribbleActions.DribbleTo(
            target,
            angle_to(ctx.world.ball, target),
            track.start_ball_pos,
            params.DRIBBLE_SOFT_LIMIT,
            carry_limits(),
            dribbler_speed=params.DRIBBLER_HOLD,
            hold_dist=params.BALL_HOLD_DIST,
        )

    # --- PASS_TEST ---

    def pass_test(self, ctx: TickContext, first: rbt.Robot, second: rbt.Robot) -> dict[int, Action]:
        """Пасы друг другу при разных PASS_ARRIVAL_SPEED и дистанциях; доля приёмов"""
        st = self.state
        now = ctx.now
        combos = [(a, d) for a in PASS_ARRIVALS for d in PASS_DISTANCES]
        if st.index >= len(combos):
            if not st.printed:
                st.printed = True
                for (arr, dist), (ok, total) in sorted(st.stats.items()):
                    _print(ctx, f"pass arrival={arr: .0f} dist={dist: .0f}: {ok}/{total}")
            st.done = True
            return {}
        arrival, dist = combos[st.index]
        if st.passer is None:
            st.passer = first.r_id
        passer = first if st.passer == first.r_id else second
        receiver = second if passer is first else first
        ball = ctx.world.ball
        recv_point = (
            ball + (aux.Point(0, 0) - ball).unity() * dist
            if aux.dist(ball, aux.Point(0, 0)) > 300
            else ball + aux.Point(dist, 0)
        )
        actions: dict[int, Action] = {}
        if st.phase == "FETCH":
            fetch = self._fetch(ctx, passer, angle_to(ball, recv_point))
            actions[receiver.r_id] = go_to(ctx, receiver, recv_point, angle_to(recv_point, ball))
            if fetch is not None:
                actions[passer.r_id] = fetch
                return actions
            self.set_phase("PASS", now)
        if st.phase == "PASS":
            log = ctx.memory.kick_log
            if log is not None and log.robot_id == passer.r_id and log.start_time >= st.phase_time:
                self.set_phase("FLIGHT", now)
            else:
                speed = math.sqrt(arrival**2 + 2 * params.BALL_DECEL * aux.dist(ball, receiver.get_pos()))
                voltage = ctx.kick_model.kick_voltage(speed)
                ctx.memory.kick_voltage_issued[passer.r_id] = voltage
                actions[passer.r_id] = DribbleActions.DribbleKick(
                    receiver.get_pos(), voltage, params.KICK_ALIGN_TOL_PASS, params.KICK_MAX_ANG_VEL, carry_limits()
                )
                actions[receiver.r_id] = go_to(ctx, receiver, recv_point, angle_to(recv_point, ball))
                return actions
        if st.phase == "FLIGHT":
            actions[receiver.r_id] = common.receive(ctx, receiver, None)
            ok = ctx.has_ball(receiver)
            if ok or now - st.phase_time > PASS_TIMEOUT:
                stat = st.stats.setdefault((arrival, dist), [0, 0])
                stat[0] += int(ok)
                stat[1] += 1
                st.trial += 1
                if st.trial >= KICK_REPEATS:
                    st.trial = 0
                    st.index += 1
                st.passer = receiver.r_id
                self.set_phase("FETCH", now)
        return actions

    # --- SHOOT_TEST ---

    def shoot_test(self, ctx: TickContext, robot: rbt.Robot) -> Action:
        """Удары по воротам: точность и скорость мяча не больше BALL_SPEED_ALERT"""
        st = self.state
        world = ctx.world
        goal = ctx.field.enemy_goal
        if abs(world.ball.x) > const.FIELD_DX and abs(world.ball.y - goal.center.y) < const.GOAL_DY / 2:
            if st.phase != "SCORED":
                st.trial += 1
                speed = ctx.memory.last_kick_max_speed
                _print(ctx, f"shot {st.index}: GOAL ({st.trial} goals), last max speed {speed: .0f}")
            self.set_phase("SCORED", ctx.now)
            return hold_position(ctx, robot)
        if st.phase == "SCORED":
            st.index += 1
            self.set_phase("FETCH", ctx.now)
        shot = evaluation.eval_shot(world, world.ball, robot.r_id)
        voltage = ctx.kick_model.shot_voltage()
        direct = common.kick_on_contact(ctx, robot, shot.target, voltage, params.KICK_ALIGN_TOL_SHOT)
        if direct is not None:
            return direct
        return common.grab(ctx, robot, angle_to(world.ball, shot.target))
