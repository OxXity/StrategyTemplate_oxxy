"""Удар, пас, модель удара (ТЗ 15.1, п. 2-4)"""

import math

from bridge import const
from bridge.auxiliary import aux
from bridge.strategy import evaluation, params
from bridge.strategy.kick_model import KickModel
from bridge.strategy.world import WorldModel
from tests.conftest import make_field


def _x(field_x: float) -> float:
    """x в системе «атакуем в +x» -> реальный"""
    probe = make_field()
    return field_x * -probe.polarity


def _world(**kwargs: object) -> WorldModel:
    field = make_field(**kwargs)  # type: ignore[arg-type]
    return WorldModel(field, field.ball.last_update())


def test_shot_empty_goal_wide() -> None:
    world = _world(ball=(_x(1200), 0), allies={1: (_x(1100), 0, 0)})
    shot = evaluation.eval_shot(world, world.ball, 1)
    assert shot.open_width > 0.4
    assert shot.p_goal > 0.9


def test_shot_goalkeeper_center_aims_near_post() -> None:
    gk_x = _x(const.FIELD_DX - 150)
    world = _world(ball=(_x(1200), 0), allies={1: (_x(1100), 0, 0)}, enemies={const.ENEMY_GK: (gk_x, 0, 0)})
    shot = evaluation.eval_shot(world, world.ball, 1)
    goal = world.field.enemy_goal
    assert abs(shot.target.y - goal.center.y) > const.GOAL_DY / 4
    empty = evaluation.eval_shot(_world(ball=(_x(1200), 0), allies={1: (_x(1100), 0, 0)}), aux.Point(_x(1200), 0), 1)
    assert shot.open_width < empty.open_width


def test_shot_blocked_by_robot_on_line() -> None:
    world = _world(
        ball=(_x(1200), 0),
        allies={1: (_x(1100), 0, 0)},
        enemies={3: (_x(1450), 0, 0), 4: (_x(1450), 250, 0), 5: (_x(1450), -250, 0)},
    )
    shot = evaluation.eval_shot(world, world.ball, 1)
    assert shot.p_goal < 0.05


def test_shot_from_own_half_needs_high_p() -> None:
    world = _world(ball=(_x(-300), 0), enemies={const.ENEMY_GK: (_x(const.FIELD_DX - 150), 0, 0)})
    shot = evaluation.eval_shot(world, world.ball)
    assert shot.p_goal == 0.0 or shot.p_goal >= params.SHOT_OWN_HALF_MIN_P


def test_pass_blocked_by_enemy() -> None:
    world = _world(ball=(_x(-500), 0), allies={1: (_x(-600), 0, 0), 2: (_x(700), 0, 0)}, enemies={3: (_x(100), 0, 0)})
    receiver = world.field.allies[2]
    result = evaluation.eval_pass(world, world.ball, receiver, receiver.get_pos())
    assert not result.ok
    assert result.p_success < 0.5


def test_pass_free_line_is_safe() -> None:
    world = _world(ball=(_x(-500), 0), allies={1: (_x(-600), 0, 0), 2: (_x(500), 300, 0)}, enemies={3: (_x(500), -1300, 0)})
    receiver = world.field.allies[2]
    result = evaluation.eval_pass(world, world.ball, receiver, receiver.get_pos())
    assert result.ok
    assert result.p_success >= params.PASS_MIN_SUCCESS


def test_pass_across_own_goal_forbidden() -> None:
    world = _world(ball=(_x(-1500), 900), allies={2: (_x(-1500), -900, 0)})
    receiver = world.field.allies[2]
    result = evaluation.eval_pass(world, world.ball, receiver, receiver.get_pos())
    assert not result.ok and result.reason


def test_kick_model_monotonic_and_capped() -> None:
    model = KickModel()
    speeds = [model.speed_for_voltage(v) for v in range(0, 16)]
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))
    voltages = [model.voltage_for_speed(s) for s in range(0, 4000, 100)]
    assert all(b >= a for a, b in zip(voltages, voltages[1:]))
    for desired in (1000, 2600, 5000):
        voltage = model.kick_voltage(desired)
        assert model.speed_for_voltage(voltage) <= params.BALL_SPEED_SAFE
    assert model.shot_voltage() <= const.VOLTAGE_SHOOT


def test_kick_model_accounts_robot_speed() -> None:
    model = KickModel()
    still = model.shot_voltage(0.0)
    moving = model.shot_voltage(800.0)
    assert moving <= still
    assert model.speed_for_voltage(moving) + 800 <= params.BALL_SPEED_SAFE


def test_kick_model_pass_and_tap() -> None:
    model = KickModel(calibrated=True, table=[(v, 300.0 * v) for v in range(16)])
    near = model.pass_voltage(800)
    far = model.pass_voltage(2500)
    assert far >= near
    assert model.speed_for_voltage(far) <= params.BALL_SPEED_SAFE
    assert model.tap_voltage(500) <= near


def test_feasible_grab_angle_at_sideline() -> None:
    world = _world(ball=(_x(0), const.FIELD_DY - 50), allies={1: (_x(-500), 0, 0)})
    # Подход «с той стороны» из-за поля невозможен: угол должен измениться
    desired = -math.pi / 2
    angle = evaluation.feasible_grab_angle(world, 1, world.ball, desired)
    back = aux.rotate(aux.RIGHT, angle)
    contact = world.ball - back * const.GRAB_ALIGN_DIST
    assert abs(contact.y) <= const.FIELD_DY + const.FIELD_OUT_MARGIN


def test_safe_clear_stays_in_field() -> None:
    world = _world(ball=(_x(-1900), 200), enemies={3: (_x(-800), 0, 0)})
    target, speed = evaluation.safe_clear_target(world, world.ball)
    assert world.in_field(target, params.CLEAR_BORDER_MARGIN - 1)
    assert speed <= params.BALL_SPEED_SAFE
