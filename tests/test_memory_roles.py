"""Трекер ведения, двойное касание, роли, таймер вратаря (ТЗ 15.1, п. 5-8)"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux
from bridge.router.actions import Action, Actions, DribbleActions
from bridge.router.actions.dumb_actions import DumbActions
from bridge.strategy import guards, params
from bridge.strategy.context import TickContext
from bridge.strategy.debug_draw import DebugDraw
from bridge.strategy.kick_model import KickModel
from bridge.strategy.memory import Memory, Role, update_memory
from bridge.strategy.roles.assignment import assign_roles
from bridge.strategy.roles.goalkeeper import Goalkeeper
from bridge.strategy.world import WorldModel
from tests.conftest import make_field


def _x(x: float) -> float:
    return x * -make_field().polarity


def _tick(memory: Memory, field, now: float, we_active: bool = True) -> TickContext:  # type: ignore[no-untyped-def]
    memory.we_active = we_active
    memory.update_state(field.game_state, now)
    world = WorldModel(field, now)
    update_memory(memory, field, world, now, KickModel())
    ctx = TickContext(field, world, memory, now, we_active, KickModel(), DebugDraw(field.strategy_image, False))
    assign_roles(ctx)
    return ctx


def _move_ball(field, x: float, y: float, now: float) -> None:  # type: ignore[no-untyped-def]
    field.ball._pos = aux.Point(x, y)
    field.ball.last_update_ = now


# --- 5. Трекер ведения ---


def test_dribble_tracker_displacement_and_reset() -> None:
    field = make_field(ball=(0, 0), allies={1: (-90, 0, 0)}, robot_with_ball=(True, 1))
    memory = Memory()
    _tick(memory, field, 100.0)
    assert 1 in memory.dribble
    _move_ball(field, _x(600), 0, 100.5)
    field.allies[1]._pos = aux.Point(_x(510), 0)
    _tick(memory, field, 100.5)
    assert abs(memory.dribble_disp(1, field.ball.get_pos()) - 600) < 1
    # Разворот на месте смещение не увеличивает
    field.allies[1]._angle = 1.0
    _tick(memory, field, 100.6)
    assert abs(memory.dribble_disp(1, field.ball.get_pos()) - 600) < 1
    # Зазор дольше BALL_LOST_TIMEOUT - сброс
    field.robot_with_ball = None
    _tick(memory, field, 100.6 + params.BALL_LOST_TIMEOUT + 0.05)
    assert 1 not in memory.dribble


def test_dribble_hard_limit_forces_release() -> None:
    field = make_field(ball=(0, 0), allies={1: (-90, 0, 0)}, enemies={3: (_x(1500), 0, 0)}, robot_with_ball=(True, 1))
    memory = Memory()
    _tick(memory, field, 200.0)
    _move_ball(field, _x(params.DRIBBLE_HARD_LIMIT + 10), 0, 200.4)
    field.allies[1]._pos = aux.Point(_x(params.DRIBBLE_HARD_LIMIT - 80), 0)
    field.allies[1]._angle = 0.0 if _x(1) > 0 else 3.14159
    ctx = _tick(memory, field, 200.4)
    moving = DribbleActions.DribbleTo(
        field.ball.get_pos() + aux.Point(_x(300), 0), 0.0, aux.Point(0, 0), 700, guards.carry_limits()
    )
    actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
    actions[1] = moving
    result = guards.apply(ctx, actions)
    assert isinstance(result[1], (DribbleActions.TapForward, DribbleActions.HoldBall))


# --- 6. Двойное касание ---


def test_double_touch_set_and_cleared_by_other_robot() -> None:
    field = make_field(ball=(0, 0), allies={1: (_x(-110), 0, 0), 2: (_x(-1500), 500, 0)}, state=const.State.FREE_KICK)
    memory = Memory()
    memory.set_piece_kicker_id = 1
    _tick(memory, field, 10.0)
    assert memory.set_piece_armed
    _move_ball(field, _x(300), 0, 10.1)
    field.game_state = const.State.RUN
    _tick(memory, field, 10.1)
    assert memory.double_touch_banned_id == 1
    # Запрещённый робот не получает действий с мячом
    ctx = _tick(memory, field, 10.15)
    actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
    actions[1] = Actions.BallGrab(0.0)
    actions[2] = Actions.GoToPoint(aux.Point(_x(-1500), 500), 0.0, ignore_robots={})
    result = guards.apply(ctx, actions)
    assert result[1] is not None and not guards.has_type(result[1], guards.BALL_ACTIONS)
    # Касание другим роботом снимает запрет
    field.allies[2]._pos = aux.Point(_x(390), 0)
    _tick(memory, field, 10.3)
    assert memory.double_touch_banned_id is None


def test_double_touch_cleared_by_stop_and_not_set_without_kicker() -> None:
    field = make_field(ball=(0, 0), allies={1: (_x(-110), 0, 0)}, state=const.State.KICKOFF)
    memory = Memory()
    memory.set_piece_kicker_id = 1
    _tick(memory, field, 20.0)
    _move_ball(field, _x(300), 0, 20.1)
    field.game_state = const.State.RUN
    _tick(memory, field, 20.1)
    assert memory.double_touch_banned_id == 1
    field.game_state = const.State.STOP
    _tick(memory, field, 20.2)
    assert memory.double_touch_banned_id is None

    # Исполнитель далеко от мяча в момент ввода - запрета нет
    field2 = make_field(ball=(0, 0), allies={1: (_x(-900), 0, 0)}, state=const.State.FREE_KICK)
    memory2 = Memory()
    memory2.set_piece_kicker_id = 1
    _tick(memory2, field2, 30.0)
    _move_ball(field2, _x(300), 0, 30.5)
    field2.game_state = const.State.RUN
    _tick(memory2, field2, 30.5)
    assert memory2.double_touch_banned_id is None


# --- 7. Роли ---


def test_roles_hysteresis_and_no_jitter() -> None:
    allies = {0: (_x(-2000), 0, 0), 1: (_x(-500), 300, 0), 2: (_x(-500), -300, 0)}
    field = make_field(ball=(0, 0), allies=allies)
    memory = Memory()
    first = _tick(memory, field, 50.0).roles
    attacker = [r for r, role in first.items() if role == Role.ATTACKER][0]
    assert first[0] == Role.GOALKEEPER
    # Равные времена - роли не дрожат
    for i in range(10):
        roles = _tick(memory, field, 50.0 + 0.02 * i).roles
        assert roles[attacker] == Role.ATTACKER
    # Небольшое преимущество другого - роль не меняется
    other = 2 if attacker == 1 else 1
    field.allies[other]._pos = aux.Point(_x(-450), field.allies[other].get_pos().y)
    roles = _tick(memory, field, 51.0).roles
    assert roles[attacker] == Role.ATTACKER


def test_roles_exceptions_owner_and_banned() -> None:
    allies = {0: (_x(-2000), 0, 0), 1: (_x(-1200), 0, 0), 2: (_x(-90), 0, 0)}
    field = make_field(ball=(0, 0), allies=allies, robot_with_ball=(True, 1))
    memory = Memory()
    roles = _tick(memory, field, 60.0).roles
    assert roles[1] == Role.ATTACKER  # владеющий мячом - Нападающий
    field.robot_with_ball = None
    memory.double_touch_banned_id = 2
    roles = _tick(memory, field, 61.0).roles
    assert roles[2] != Role.ATTACKER  # запрет двойного касания


def test_single_field_player_is_attacker() -> None:
    field = make_field(ball=(0, 0), allies={0: (_x(-2000), 0, 0), 1: (_x(-1000), 0, 0)})
    roles = _tick(Memory(), field, 70.0).roles
    assert roles == {0: Role.GOALKEEPER, 1: Role.ATTACKER}


# --- 8. Таймер вратаря (G-8) ---


def test_goalkeeper_area_timer_and_hard_clear() -> None:
    ball = (_x(-2000), 200)
    field = make_field(ball=ball, allies={0: (_x(-2100), 0, 0), 1: (_x(0), 0, 0)}, enemies={3: (_x(1000), 0, 0)})
    memory = Memory()
    _tick(memory, field, 80.0)
    assert memory.ball_in_our_area_since == 80.0
    ctx = _tick(memory, field, 80.0 + params.GK_CLEAR_HARD_TIME + 0.5)
    assert ctx.now - (memory.ball_in_our_area_since or 0) > params.GK_CLEAR_HARD_TIME
    action = Goalkeeper().act(field.allies[0], ctx)
    assert guards.has_type(action, guards.BALL_ACTIONS)
    tolerances = [sub.angle_bounds for sub in guards.walk(action) if isinstance(sub, DumbActions.ShootAction)] + [
        sub.align_tol for sub in guards.walk(action) if isinstance(sub, DribbleActions.DribbleKick)
    ]
    assert not tolerances or max(t or 0 for t in tolerances) >= params.PRESSURE_MAX_TURN - 1e-9
    field.ball._pos = aux.Point(_x(0), 0)
    _tick(memory, field, 90.0)
    assert memory.ball_in_our_area_since is None
