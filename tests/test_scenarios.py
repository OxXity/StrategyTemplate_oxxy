"""
Сценарные тесты в мини-симуляторе (tests/minisim.py): стандарты, STOP, открытая игра.
Физика упрощённая; тесты ловят логические регрессии, а не настраивают параметры.
"""

import pytest

import bridge.router.actions.base_actions as base_actions
import bridge.router.actions.dribble_actions as dribble_actions
import bridge.strategy.strategy as strategy_module
from bridge import const
from bridge.auxiliary import aux
from bridge.strategy import params
from tests.minisim import FakeClock, Sim

ALLIES = {0: (-2100.0, 0.0, 0.0), 1: (-400.0, 300.0, 0.0), 2: (-900.0, -500.0, 0.0)}
ENEMIES = {0: (2100.0, 0.0, 3.14), 1: (500.0, 200.0, 3.14), 2: (900.0, -400.0, 3.14)}


def _x(x: float) -> float:
    """x в системе «атакуем в +x» -> реальный (по POLARITY и цвету)"""
    from tests.conftest import make_field

    return x * -make_field().polarity


def _mirror(poses: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float, float]]:
    flip = _x(1.0)
    return {k: (x * flip, y, a if flip > 0 else 3.14159 - a) for k, (x, y, a) in poses.items()}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Подмена времени в действиях роутера и в стратегии"""
    fake = FakeClock()
    monkeypatch.setattr(base_actions, "time", fake)
    monkeypatch.setattr(dribble_actions, "time", fake)
    monkeypatch.setattr(strategy_module, "time", fake)
    return fake


def _ball_moved(sim: Sim, start: aux.Point) -> bool:
    return aux.dist(sim.field.ball.get_pos(), start) > params.RULE_BALL_IN_PLAY_DIST


def test_free_kick_kicked_in_time_with_double_touch_ban(clock: FakeClock) -> None:
    start = aux.Point(_x(300), 400)
    sim = Sim(clock, (start.x, start.y), _mirror(ALLIES), _mirror(ENEMIES), const.State.FREE_KICK, const.COLOR, False)
    moved_at: list[float] = []

    def watch(s: Sim) -> None:
        if s.field.game_state == const.State.FREE_KICK and _ball_moved(s, start):
            moved_at.append(s.t)
            s.field.game_state = const.State.RUN

    sim.run(params.SET_PIECE_MAX_WAIT + 1.0, on_tick=watch)
    assert moved_at and moved_at[0] <= params.SET_PIECE_MAX_WAIT + 0.2
    assert sim.strategy.memory.metrics.get("double_touch_bans", 0) >= 1
    assert not sim.fouls


def test_kickoff_pass_is_received(clock: FakeClock) -> None:
    allies = _mirror({0: (-2100.0, 0.0, 0.0), 1: (-800.0, 300.0, 0.0), 2: (-900.0, -500.0, 0.0)})
    sim = Sim(clock, (0, 0), allies, _mirror(ENEMIES), const.State.PREPARE_KICKOFF, const.COLOR, False)
    sim.run(3.0)
    for robot in sim.field.active_allies(True):
        is_kicker = robot.r_id == sim.strategy.memory.set_piece_kicker_id
        if not is_kicker:
            assert robot.get_pos().x * -sim.field.polarity < 0  # своя половина
            assert aux.dist(robot.get_pos(), aux.Point(0, 0)) > params.KICKOFF_CIRCLE_R
    sim.field.game_state = const.State.KICKOFF
    start = aux.Point(0, 0)

    def watch(s: Sim) -> None:
        if s.field.game_state == const.State.KICKOFF and _ball_moved(s, start):
            s.field.game_state = const.State.RUN

    sim.run(5.0, on_tick=watch)
    assert sim.field.game_state == const.State.RUN
    assert sim.strategy.memory.metrics.get("passes_received", 0) >= 1
    assert not sim.fouls and not sim.events


def test_penalty_shot_before_deadline(clock: FakeClock) -> None:
    ball = (_x(const.FIELD_DX - 3125), 0.0)
    allies = _mirror({0: (-2100.0, 0.0, 0.0), 1: (-1300.0, 300.0, 0.0), 2: (-1500.0, -500.0, 0.0)})
    enemies = _mirror({0: (2150.0, 0.0, 3.14), 1: (1500.0, 1200.0, 3.14)})
    sim = Sim(clock, ball, allies, enemies, const.State.PREPARE_PENALTY, const.COLOR, True)
    sim.run(3.0)
    for robot in sim.field.active_allies(False):
        if robot.r_id != sim.strategy.memory.set_piece_kicker_id:
            assert aux.dist(robot.get_pos(), sim.field.ball.get_pos()) >= params.RULE_PENALTY_OTHERS_DIST
    sim.field.game_state = const.State.PENALTY
    start_t = sim.t
    sim.run(params.RULE_PENALTY_TIME)
    shots = [k for k in sim.kicks if k[1] and k[3] > params.BALL_SPEED_SAFE * 0.8]
    assert shots and shots[0][0] - start_t <= params.PENALTY_SHOOT_DEADLINE + 0.6
    assert not sim.fouls


def test_stop_keeps_distance(clock: FakeClock) -> None:
    sim = Sim(clock, (_x(500), 300), _mirror(ALLIES), _mirror(ENEMIES), const.State.STOP, const.Color.ALL, False)
    sim.run(4.0)
    for robot in sim.field.active_allies(True):
        assert aux.dist(robot.get_pos(), sim.field.ball.get_pos()) >= params.RULE_STOP_BALL_DIST + const.ROBOT_R


def test_open_play_scores_against_goalkeeper(clock: FakeClock) -> None:
    sim = Sim(clock, (0, 0), _mirror(ALLIES), _mirror({0: (2100.0, 0.0, 3.14)}), const.State.RUN)

    def reset(s: Sim) -> None:
        s.field.ball._pos = aux.Point(0, 0)
        s.field.ball._vel = aux.Point(0, 0)
        s.attached = None

    sim.run(30.0, on_out=reset)
    assert sim.goals["ours"] >= 1
    assert sim.goals["theirs"] == 0
    assert not sim.fouls
    assert sim.max_dribble <= params.DRIBBLE_HARD_LIMIT
