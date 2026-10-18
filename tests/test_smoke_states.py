"""Смоук-тест всех состояний (ТЗ 15.1, п. 1)"""

import pytest

from bridge import const
from bridge.auxiliary import aux
from bridge.router import legality
from bridge.router.actions import Action, Actions
from bridge.strategy import guards, params
from bridge.strategy.strategy import Strategy
from tests.conftest import make_field

ALLIES = {0: (-2000.0, 0.0, 0.0), 1: (-600.0, 400.0, 0.0), 2: (-900.0, -500.0, 0.0)}
ENEMIES = {0: (2000.0, 0.0, 3.14), 1: (700.0, 100.0, 3.14), 2: (900.0, -700.0, 3.14)}
# x для BLUE при POLARITY=1: наши ворота на -x
PLACEMENTS = {
    "our_half": (-1000.0, 300.0),
    "their_half": (1000.0, -300.0),
    "near_our_area": (-1550.0, 100.0),
    "near_their_area": (1550.0, -150.0),
    "corner": (2050.0, 1300.0),
}
ALLOWED_IN_STOP = (Actions.GoToPoint, Actions.Stop, Actions.SetDribblerSpeed)


def _place(x: float, y: float, polarity: int) -> tuple[float, float]:
    """Координаты из системы «наши ворота на -x» в реальные"""
    return (x * -polarity, y)


def _targets(action: Action) -> list[aux.Point]:
    return [sub.target_pos for sub in guards.walk(action) if isinstance(sub, guards.TARGET_ACTIONS)]


@pytest.mark.parametrize("state", list(const.State))
@pytest.mark.parametrize("we_active", [True, False])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("color", [const.Color.BLUE, const.Color.YELLOW])
def test_all_states(state: const.State, we_active: bool, placement: str, color: const.Color) -> None:
    probe = make_field(color=color)
    pol = -probe.polarity  # 1, если атакуем в +x
    ball = _place(*PLACEMENTS[placement], -pol)
    allies = {k: (*_place(v[0], v[1], -pol), v[2]) for k, v in ALLIES.items()}
    enemies = {k: (*_place(v[0], v[1], -pol), v[2]) for k, v in ENEMIES.items()}
    ally_color = probe.ally_color
    active = ally_color if we_active else ally_color.reverse()
    field = make_field(ball=ball, allies=allies, enemies=enemies, state=state, active_team=active, color=color)

    strategy = Strategy()
    strategy.we_active = we_active
    for _ in range(3):
        actions = strategy.process(field)

    assert len(actions) == const.TEAM_ROBOTS_MAX_COUNT
    active_ids = {r.r_id for r in field.active_allies(True)}
    for r_id in range(const.TEAM_ROBOTS_MAX_COUNT):
        if r_id in active_ids:
            assert actions[r_id] is not None, f"robot {r_id} has no action"
        else:
            assert actions[r_id] is None

    ball_pos = field.ball.get_pos()
    for r_id in active_ids:
        action = actions[r_id]
        assert action is not None
        if state in (const.State.HALT, const.State.TIMEOUT):
            assert isinstance(action, Actions.Stop)
            continue
        if state == const.State.STOP:
            for sub in guards.walk(action):
                assert isinstance(sub, ALLOWED_IN_STOP + (guards.ComposedAction,)), type(sub)
                if isinstance(sub, Actions.GoToPoint):
                    assert sub.avoid_ball is not False
            for target in _targets(action):
                assert aux.dist(target, ball_pos) >= params.KEEP_BALL_DIST - 1
        if r_id != field.gk_id:
            for target in _targets(action):
                assert not legality.is_inside_area(target, field.ally_goal, const.ROBOT_R), (state, target)
                assert not legality.is_inside_area(target, field.enemy_goal, const.ROBOT_R), (state, target)


def test_halt_after_run_stops_everyone() -> None:
    field = make_field(ball=(0, 0), allies={0: (-2000, 0, 0), 1: (-300, 0, 0)}, enemies={1: (800, 0, 3.14)})
    strategy = Strategy()
    strategy.process(field)
    field.game_state = const.State.HALT
    actions = strategy.process(field)
    assert all(isinstance(actions[r.r_id], Actions.Stop) for r in field.active_allies(True))


def test_tick_budget() -> None:
    """R-5.5: средний такт ≤ 8 мс"""
    import time

    field = make_field(
        ball=(300, 200),
        allies={0: (-2000, 0, 0), 1: (-300, 0, 0), 2: (-800, 600, 0)},
        enemies={0: (2000, 0, 3.14), 1: (900, 0, 3.14), 2: (1200, -500, 3.14)},
    )
    strategy = Strategy()
    start = time.time()
    n = 50
    for _ in range(n):
        strategy.process(field)
    avg = (time.time() - start) / n
    assert avg < 0.008 * 2, f"average tick {avg * 1000: .1f} ms"
