"""BallCarryLimiter, legal_target, ограничитель сближения (ТЗ 15.1, п. 9-10; G-9)"""

import math

import pytest

from bridge import const
from bridge.auxiliary import aux
from bridge.router import legality, safety
from bridge.router.actions import CarryLimits, DribbleActions
from bridge.router.actions import dribble_actions as da
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.strategy import guards, params
from bridge.strategy.memory import Memory
from tests.conftest import make_field

LIMITS = (800.0, 400.0, 300.0, 1500.0, 3.0, 8.0)


def _domain(field, r_id: int = 1) -> ActionDomain:  # type: ignore[no-untyped-def]
    return ActionDomain(field, field.game_state, True, field.allies[r_id])


def test_carry_limiter_ramps_angle_and_limits_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    field = make_field(ball=(90, 0), allies={1: (0, 0, 0.0)}, robot_with_ball=(True, 1))
    domain = _domain(field)
    clock = [1000.0]
    monkeypatch.setattr(da, "time", lambda: clock[0])
    limiter = DribbleActions.BallCarryLimiter(*LIMITS)
    dt = 0.01
    prev_angle = 0.0
    prev_vel = aux.Point(0, 0)
    for _ in range(40):
        clock[0] += dt
        values = ActionValues()
        values.vel = aux.Point(3000, 3000)
        values.angle = math.pi  # просим развернуться мгновенно
        limiter.process(domain, values)
        step = abs(aux.wind_down_angle(values.angle - prev_angle))
        assert step <= LIMITS[4] * dt + 1e-6
        local = aux.rotate(values.vel, -field.allies[1].get_angle())
        assert local.x <= LIMITS[0] + 1e-6 and local.x >= -LIMITS[2] - 1e-6
        assert abs(local.y) <= LIMITS[1] + 1e-6
        assert (values.vel - prev_vel).mag() <= LIMITS[3] * dt + 1e-6
        prev_angle = values.angle
        prev_vel = values.vel


def test_hold_ball_pivots_around_ball(monkeypatch: pytest.MonkeyPatch) -> None:
    field = make_field(ball=(90, 0), allies={1: (0, 0, 0.0)}, robot_with_ball=(True, 1))
    domain = _domain(field)
    clock = [2000.0]
    monkeypatch.setattr(da, "time", lambda: clock[0])
    action = DribbleActions.HoldBall(math.pi / 2, CarryLimits(*LIMITS))
    robot = field.allies[1]
    for _ in range(30):
        clock[0] += 0.01
        values = ActionValues()
        action.process(domain, values)
        robot._angle = values.angle  # робот отслеживает уставку
    assert robot.pivot_point is not None and aux.dist(robot.pivot_point, aux.Point(90, 0)) < 1
    sp = robot.carry_angle_sp or 0.0
    # Разгон с BALL_HOLD_MAX_ANG_ACC: за 0.3 с не меньше половины от a·t²/2 и не больше ω_max·t
    assert LIMITS[5] * 0.3**2 / 4 < sp <= LIMITS[4] * 0.3 + 1e-6


def test_dribble_to_projects_target_into_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    field = make_field(ball=(90, 0), allies={1: (0, 0, 0.0)}, robot_with_ball=(True, 1))
    monkeypatch.setattr(da, "time", lambda: 3000.0)
    action = DribbleActions.DribbleTo(aux.Point(5000, 0), 0.0, aux.Point(0, 0), 700, CarryLimits(*LIMITS))
    sub = action.use_behavior_of(_domain(field), ActionValues())
    target = sub[0].target_pos  # type: ignore[attr-defined]
    assert aux.dist(target + aux.Point(params.BALL_HOLD_DIST, 0), aux.Point(0, 0)) <= 700 - 50 + 1


@pytest.mark.parametrize(
    "state", [const.State.RUN, const.State.STOP, const.State.FREE_KICK, const.State.PREPARE_KICKOFF, const.State.PENALTY]
)
def test_legal_target_zones(state: const.State) -> None:
    field = make_field(ball=(0, 0), allies={0: (-2000, 0, 0), 1: (0, 500, 0), 2: (0, -500, 0)}, state=state)
    memory = Memory()
    enemy_goal = field.enemy_goal
    ally_goal = field.ally_goal
    probes = [
        ally_goal.center + ally_goal.eye_forw * 200,
        enemy_goal.center + enemy_goal.eye_forw * 200,
        enemy_goal.center + enemy_goal.eye_forw * (const.GOAL_PEN_DX + 150),
        aux.Point(5000, 5000),
        aux.Point(-3000, 0),
        aux.Point(50, 50),
        aux.Point(1200, 1400),
    ]
    for we_active in (True, False):
        for point in probes:
            target = guards.legal_target(field, memory, we_active, 1, point)
            assert abs(target.x) <= const.FIELD_DX + const.FIELD_OUT_MARGIN + 1
            assert abs(target.y) <= const.FIELD_DY + const.FIELD_OUT_MARGIN + 1
            assert not legality.is_inside_area(target, ally_goal, legality.ALLY_AREA_MARGIN - 2)
            assert not legality.is_inside_area(target, enemy_goal, legality.enemy_area_margin(state) - 2)
            keep = guards.keep_ball_dist(field, memory, we_active, 1)
            assert aux.dist(target, field.ball.get_pos()) >= keep - 2
            if state in (const.State.PREPARE_KICKOFF,):
                assert target.x * -field.polarity <= -params.KICKOFF_HALF_GAP + 1
            if state in (const.State.STOP,) or (state == const.State.FREE_KICK and not we_active):
                assert aux.dist(target, field.ball.get_pos()) >= params.KEEP_BALL_DIST - 2
            if state == const.State.PENALTY:
                assert aux.dist(target, field.ball.get_pos()) >= params.PENALTY_OTHERS_DIST - 2
        gk_target = guards.legal_target(field, memory, we_active, field.gk_id, ally_goal.center - ally_goal.eye_forw * 300)
        assert legality.area_depth(gk_target, ally_goal) >= -1


def test_collision_limiter() -> None:
    our = aux.Point(0, 0)
    enemy = (aux.Point(250, 0), aux.Point(-500, 0))
    vel = safety.limit_approach_speed(our, aux.Point(1500, 300), [enemy])
    v_rel = aux.scal_mult(vel - enemy[1], aux.Point(1, 0))
    assert v_rel <= const.COLLISION_SAFE_REL_SPEED + 1e-6
    assert abs(vel.y - 300) < 1e-6
    # Отъезд не ограничивается
    away = safety.limit_approach_speed(our, aux.Point(-1000, 0), [enemy])
    assert away.x == -1000
    # Далёкий соперник не мешает
    far = safety.limit_approach_speed(our, aux.Point(1500, 0), [(aux.Point(3000, 0), aux.Point(0, 0))])
    assert far.x == 1500


def test_legal_target_router_level_ball_out() -> None:
    field = make_field(ball=(0, 0), state=const.State.FREE_KICK)
    enemy_goal = field.enemy_goal
    point = enemy_goal.center + enemy_goal.eye_forw * (const.GOAL_PEN_DX + 250)
    target = legality.legal_target(field, 1, point)
    depth = legality.area_depth(target, enemy_goal)
    lateral_out = abs(target.y) >= const.GOAL_PEN_DY / 2 + const.DEF_AREA_MARGIN_BALL_OUT
    assert depth >= const.GOAL_PEN_DX + const.DEF_AREA_MARGIN_BALL_OUT or lateral_out
