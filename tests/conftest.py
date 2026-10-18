"""
Фабрика синтетического поля для тестов без симулятора (ТЗ, 15.1)
"""

import os
import sys
from time import time
from typing import Mapping, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # bridge.const читает .env относительно текущей папки
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bridge import const  # noqa: E402
from bridge.auxiliary import aux, entity, fld  # noqa: E402

Pose = tuple[float, float, float]
Poses = Mapping[int, tuple[float, float, float]]


def make_field(
    ball: tuple[float, float] = (0.0, 0.0),
    allies: Optional[Poses] = None,
    enemies: Optional[Poses] = None,
    state: const.State = const.State.RUN,
    active_team: const.Color = const.Color.ALL,
    color: const.Color = const.Color.BLUE,
    ball_vel: tuple[float, float] = (0.0, 0.0),
    robot_with_ball: Optional[tuple[bool, int]] = None,
    ball_placement_pos: Optional[aux.Point] = None,
) -> fld.Field:
    """Поле с роботами (id -> (x, y, angle)) и мячом"""
    field = fld.Field(color)
    now = time()
    for team, poses in ((field.allies, allies or {}), (field.enemies, enemies or {})):
        for r_id, (x, y, angle) in poses.items():
            robot = team[r_id]
            robot._pos = aux.Point(x, y)
            robot._angle = angle
            robot._vel = aux.Point(0, 0)
            robot._is_used = 1
            robot.last_update_ = now
    field.update_active_allies([r for r in field.allies if r.is_used() and r.r_id != field.gk_id])
    field.update_active_enemies([r for r in field.enemies if r.is_used() and r.r_id != field.enemy_gk_id])

    field.ball = entity.Entity(aux.Point(*ball), 0, const.BALL_R)
    field.ball._vel = aux.Point(*ball_vel)
    field.ball.last_update_ = now
    field.ball_start_point = aux.Point(ball[0] - ball_vel[0] * 0.2, ball[1] - ball_vel[1] * 0.2)
    field.game_state = state
    field.active_team = active_team
    field.ball_placement_pos = ball_placement_pos
    if robot_with_ball is not None:
        ours, r_id = robot_with_ball
        field.robot_with_ball = (field.allies if ours else field.enemies)[r_id]
    return field


def attack(field: fld.Field) -> float:
    """Знак направления атаки по X"""
    return -field.polarity


@pytest.fixture
def field_factory():  # type: ignore[no-untyped-def]
    """Фабрика поля"""
    return make_field


@pytest.fixture(autouse=True)
def strict_mode():  # type: ignore[no-untyped-def]
    """В тестах исключения не глушатся (SAFE_MODE=False), после теста значение восстанавливается"""
    from bridge.strategy import params

    saved = params.SAFE_MODE
    params.SAFE_MODE = False
    yield
    params.SAFE_MODE = saved
