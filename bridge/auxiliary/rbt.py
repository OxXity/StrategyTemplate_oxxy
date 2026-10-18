"""
Описание полей и интерфейсов взаимодействия с роботом
"""

import typing
from time import time

from bridge import const
from bridge.auxiliary import aux, entity, tau


class Robot(entity.Entity):
    """
    Описание робота
    """

    def __init__(
        self,
        pos: aux.Point,
        angle: float,
        R: float,
        color: const.Color,
        r_id: int,
    ) -> None:
        super().__init__(pos, angle, R)

        self.r_id = r_id
        self._is_used = 0
        self.color = color
        self.last_update_ = 0.0
        self.live_time_: typing.Optional[float] = None

        self.speed_x = 0.0
        self.speed_y = 0.0
        self.speed_r = 0.0
        self.delta_angle = 0.0

        # v! SIM
        if const.IS_SIMULATOR_USED:
            self.k_wy = -0.001
            self.t_wy = 0.15
            self.r_comp_f_dy = tau.FOD(self.t_wy, const.Ts)
            self.r_comp_f_fy = tau.FOLP(self.t_wy, const.Ts)

        # v! REAL
        else:
            self.k_wy = 0
            self.t_wy = 0.15
            self.r_comp_f_dy = tau.FOD(self.t_wy, const.Ts)
            self.r_comp_f_fy = tau.FOLP(self.t_wy, const.Ts)

        self.xx_t = 0.1
        self.xx_flp = tau.FOLP(self.xx_t, const.Ts)
        self.yy_t = 0.1
        self.yy_flp = tau.FOLP(self.yy_t, const.Ts)

        # !v REAL
        gains_full = [2, 0.01, 0.3]  # 0.9 for average moving, 1.7 for rickochet and ball catching
        gains_catch = [2.8, 0.01, 0.87]  # for ricochet and pass receiving
        if color == const.COLOR and self.r_id == const.GK:
            gains_catch = [2.8, 0.01, 0.5]
        a_gains_full = [15, 0.5]
        if const.IS_SIMULATOR_USED:
            gains_full = [1.8, 0.06, 0.0]
            gains_catch = gains_full
            a_gains_full = [8, 0.1]
        a_gains_catch = a_gains_full

        self.pos_reg_x = tau.AdaptivePDController(
            [gains_full[0], gains_catch[0]],
            [gains_full[1], gains_catch[1]],
            [gains_full[2], gains_catch[2]],
        )
        self.pos_reg_y = tau.AdaptivePDController(
            [gains_full[0], gains_catch[0]],
            [gains_full[1], gains_catch[1]],
            [gains_full[2], gains_catch[2]],
        )
        self.angle_reg = tau.PDController(
            [a_gains_full[0], a_gains_catch[0]],
            [a_gains_full[1], a_gains_catch[1]],
        )

        self.is_kick_committed = False

        self.prev_sended_vel = aux.Point(0, 0)
        self.prev_sended_time = time()
        self.prev_sended_angle = 0.0
        self.prev_ricochet_target: typing.Optional[aux.Point] = None

        # Состояние действий для слабого дриблера (живёт в роутере между тактами, R-4.1, R-4.2, R-4.3)
        self.carry_angle_sp: typing.Optional[float] = None  # уставка угла рампы
        self.carry_ang_vel = 0.0  # текущая угловая скорость рампы
        self.carry_ramp_time = 0.0  # время последнего обновления рампы
        self.carry_prev_vel: typing.Optional[aux.Point] = None  # прошлая скорость после ограничителя
        self.carry_vel_time = 0.0
        self.pivot_point: typing.Optional[aux.Point] = None  # точка мяча для разворота вокруг мяча
        self.pivot_time = 0.0

    def __eq__(self, robo: typing.Any) -> bool:
        if not isinstance(robo, Robot):
            return False
        return self.r_id == robo.r_id and self.color == robo.color

    def to_entity(self) -> entity.Entity:
        """convert to entity"""
        ent = entity.Entity(self._pos, self._angle, self._radius)
        ent._vel = self._vel
        # ent._acc = self._acc
        return ent

    def used(self, a: int) -> None:
        """
        Выставить флаг использования робота
        """
        self._is_used = a

        if a == 0:
            self.live_time_ = None

    def is_used(self) -> int:
        """returns true if the robot is used"""
        return self._is_used

    def last_update(self) -> float:
        """get the robot's last update time"""
        return self.last_update_

    def live_time(self) -> typing.Optional[float]:
        """get the robot's lifetime"""
        return self.live_time_

    def update(self, pos: aux.Point, angle: float, t: float) -> None:
        """
        Обновить состояние робота согласно SSL Vision
        """
        super().update(pos, angle, t)
        self.kick_forward_ = 0
        self.kick_up_ = 0
        self.last_update_ = t

        if self.live_time_ is None:
            self.live_time_ = t

    def update_(self, lite_robot: "LiteRobot") -> None:
        """
        Обновить состояние робота используя готовые данные
        """
        self._pos = lite_robot.pos
        self._vel = lite_robot.vel
        self._angle = lite_robot.angle
        self._anglevel = lite_robot.anglevel

        self._is_used = lite_robot.is_used
        self.last_update_ = lite_robot.last_update

    def kick_forward(self) -> None:
        """
        Ударить вперед
        """
        self.kick_forward_ = 1

    def kick_up(self) -> None:
        """
        Ударить вверх
        """
        self.kick_up_ = 1

    def set_dribbler_speed(self, speed: float) -> None:
        """
        Включить дриблер и задать его скорость
        """
        self.dribbler_enable_ = True
        self.dribbler_speed_ = round(aux.minmax(speed, 0.0, 15.0))

    def clear_fields(self) -> None:
        """
        Очистить поля управления
        """
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.speed_r = 0.0
        self.delta_angle = 0.0

    def is_kick_aligned(self, pos: aux.Point, angle: float) -> bool:
        """
        Определить, выровнен ли робот относительно путевой точки target
        """

        commit_scale = 1.2 if self.is_kick_committed else 1
        is_dist = (self.get_pos() - pos).mag() < const.KICK_ALIGN_DIST * const.KICK_ALIGN_DIST_MULT * commit_scale
        is_angle = self.is_kick_aligned_by_angle(angle)
        is_offset = (
            aux.dist(
                aux.closest_point_on_line(
                    pos,
                    pos - aux.rotate(aux.RIGHT, angle) * const.KICK_ALIGN_DIST,
                    self._pos,
                ),
                self._pos,
            )
            < const.KICK_ALIGN_OFFSET * commit_scale
        )
        is_aligned = is_dist and is_angle and is_offset

        if is_aligned:
            self.is_kick_committed = True
        else:
            self.is_kick_committed = False

        return is_aligned

    def is_kick_aligned_by_angle(self, angle: float, *, angle_bounds: float = const.KICK_ALIGN_ANGLE) -> bool:
        """
        Определить, выровнен ли робот относительно путевой точки target
        """
        commit_scale = 1.2 if self.is_kick_committed else 1
        return abs(aux.wind_down_angle(self._angle - angle)) < angle_bounds * commit_scale

    def update_vel_xy(self, vel: aux.Point) -> None:
        """
        Выполнить тик низкоуровневых регуляторов скорости робота (no)

        vel - требуемый вектор скорости [мм/с]
        """
        speed = -aux.rotate(aux.Point(vel.x, vel.y), -self._angle)
        self.speed_x = -speed.x
        self.speed_y = speed.y

    def update_vel_w(self, wvel: float) -> None:
        """Update robot angle vel"""
        self.speed_r = wvel

    def __str__(self) -> str:
        return (
            str(
                str(self.color)
                + " "
                + str(self.r_id)
                + " "
                + str(self.get_pos())
                + " "
                + str(self.speed_x)
                + " "
                + str(self.speed_y)
            )
            + " "
            + str(self.speed_r)
        )


class LiteRobot:
    """Lite class, to moving information about robot between processes"""

    def __init__(self, robot: Robot) -> None:
        self.r_id = robot.r_id

        self.pos = robot.get_pos()
        self.vel = robot.get_vel()
        self.angle = robot.get_angle()
        self.anglevel = robot.get_anglevel()

        self.is_used = robot.is_used()
        self.last_update = robot.last_update()
