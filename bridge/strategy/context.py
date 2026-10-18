"""
Контекст такта стратегии и общие помощники для ролей и розыгрышей
"""

import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Optional, Protocol

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.router.actions import Action, CarryLimits
from bridge.strategy import params
from bridge.strategy.debug_draw import DebugDraw
from bridge.strategy.kick_model import KickModel
from bridge.strategy.memory import Memory, Role
from bridge.strategy.world import WorldModel


@dataclass
class TickContext:  # pylint: disable=too-many-instance-attributes
    """Всё, что нужно ролям в одном такте"""

    field: fld.Field
    world: WorldModel
    memory: Memory
    now: float
    we_active: bool
    kick_model: KickModel
    draw: DebugDraw
    roles: dict[int, Role] = dc_field(default_factory=dict)
    warnings: list[str] = dc_field(default_factory=list)
    targets: dict[int, aux.Point] = dc_field(default_factory=dict)

    def ids_with_role(self, role: Role) -> list[int]:
        """id роботов с ролью"""
        return [r_id for r_id, r in self.roles.items() if r == role]

    def robot_with_role(self, role: Role) -> Optional[rbt.Robot]:
        """Первый робот с ролью"""
        ids = self.ids_with_role(role)
        return self.field.allies[ids[0]] if ids else None

    def is_banned(self, robot: rbt.Robot) -> bool:
        """Робот под запретом двойного касания (G-5)"""
        return self.memory.double_touch_banned_id == robot.r_id

    def has_ball(self, robot: rbt.Robot) -> bool:
        """Мяч у этого нашего робота"""
        return self.world.owner_is_ours and self.world.owner is not None and self.world.owner.r_id == robot.r_id

    def is_gk(self, robot: rbt.Robot) -> bool:
        """Робот - вратарь"""
        return robot.r_id == self.field.gk_id


class RoleBehavior(Protocol):  # pylint: disable=too-few-public-methods
    """Поведение роли"""

    def act(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Действие робота в такте"""


def carry_limits() -> CarryLimits:
    """Лимиты движения с мячом из параметров (R-4.1, R-4.3)"""
    return CarryLimits(
        params.BALL_HOLD_MAX_SPEED_FWD,
        params.BALL_HOLD_MAX_SPEED_SIDE,
        params.BALL_HOLD_MAX_SPEED_BACK,
        params.BALL_HOLD_MAX_ACC,
        params.BALL_HOLD_MAX_ANG_SPEED,
        params.BALL_HOLD_MAX_ANG_ACC,
    )


def angle_to(src: aux.Point, dst: aux.Point) -> float:
    """Угол направления src -> dst"""
    return aux.angle_to_point(src, dst)


def unit(angle: float) -> aux.Point:
    """Единичный вектор"""
    return aux.Point(math.cos(angle), math.sin(angle))


def angle_diff(a: float, b: float) -> float:
    """|a - b| по кругу"""
    return abs(aux.wind_down_angle(a - b))


def clamp_angle_towards(base: float, target: float, max_dev: float) -> float:
    """Угол base, довёрнутый к target не больше чем на max_dev"""
    return base + aux.minmax(aux.wind_down_angle(target - base), -max_dev, max_dev)


def rotation_time(robot: rbt.Robot, angle: float) -> float:
    """R-4.5: время поворота с мячом"""
    return angle_diff(robot.get_angle(), angle) / params.BALL_HOLD_MAX_ANG_SPEED


def ball_near_sideline(ball: aux.Point, margin: float) -> bool:
    """Мяч у боковой линии"""
    return abs(ball.y) > const.FIELD_DY - margin
