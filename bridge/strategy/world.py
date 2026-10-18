"""
Модель мира: анализ кадра, считается один раз за такт (ТЗ, раздел 6)
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.strategy import params
from bridge.strategy.memory import Memory, Possession

INF = math.inf


@dataclass
class Intercept:
    """Перехват мяча роботом: время и точка"""

    t: float
    point: aux.Point


def travel_time(dist: float, v0: float, vmax: float, amax: float) -> float:
    """Время на путь dist по трапециевидному профилю с начальной скоростью v0 вдоль пути"""
    if dist <= 0:
        return 0.0
    v0 = min(max(v0, 0.0), vmax)
    d_acc = (vmax**2 - v0**2) / (2 * amax)
    if dist <= d_acc:
        return (-v0 + math.sqrt(v0**2 + 2 * amax * dist)) / amax
    return (vmax - v0) / amax + (dist - d_acc) / vmax


def travel_dist(t: float, vmax: float, amax: float) -> float:
    """Путь из покоя за время t по трапециевидному профилю"""
    if t <= 0:
        return 0.0
    t_acc = vmax / amax
    if t <= t_acc:
        return amax * t**2 / 2
    return vmax**2 / (2 * amax) + vmax * (t - t_acc)


def robot_time_to_point(
    robot: rbt.Robot, target: aux.Point, ours: bool, reach: float = const.ROBOT_R + const.BALL_R
) -> float:
    """R-6.2: время робота до точки (минус reach), с разворотом и реакцией"""
    vec = target - robot.get_pos()
    dist = max(0.0, vec.mag() - reach)
    if ours:
        vmax, amax, react = float(const.MAX_SPEED), float(const.MAX_ACCELERATION), params.REACTION_TIME
    else:
        vmax, amax, react = params.ENEMY_MAX_SPEED, params.ENEMY_MAX_ACC, params.ENEMY_REACTION_TIME
    v0 = aux.scal_mult(robot.get_vel(), vec.unity()) if vec.mag() > 0 else 0.0
    turn = abs(aux.wind_down_angle(vec.arg() - robot.get_angle())) if vec.mag() > 0 else 0.0
    return travel_time(dist, v0, vmax, amax) + 0.5 * turn / params.OMEGA_FREE + react


def _times_vec(dist: np.ndarray, v0: np.ndarray, vmax: float, amax: float) -> np.ndarray:
    """Векторный travel_time"""
    v0 = np.clip(v0, 0.0, vmax)
    d_acc = (vmax**2 - v0**2) / (2 * amax)
    t_short = (-v0 + np.sqrt(v0**2 + 2 * amax * dist)) / amax
    t_long = (vmax - v0) / amax + (dist - d_acc) / vmax
    return np.where(dist <= d_acc, t_short, t_long)


class WorldModel:  # pylint: disable=too-many-instance-attributes
    """Анализ кадра"""

    def __init__(self, field: fld.Field, now: float) -> None:
        self.field = field
        self.now = now
        self.attack_sign = -field.polarity  # атака по X в сторону attack_sign

        self.ball = field.ball.get_pos()
        self.ball_vel = field.ball.get_vel()
        self.ball_speed = self.ball_vel.mag()
        self.ball_dir = self.ball_vel.unity() if self.ball_speed > 1e-6 else aux.Point(0, 0)
        self.ball_moving = self.ball_speed > params.BALL_STOPPED_SPEED

        self.gk: Optional[rbt.Robot] = field.allies[field.gk_id] if field.allies[field.gk_id].is_used() else None
        self.field_allies: list[rbt.Robot] = field.active_allies(False)
        self.allies: list[rbt.Robot] = field.active_allies(True)
        self.enemies: list[rbt.Robot] = field.active_enemies(True)
        self.enemy_gk: Optional[rbt.Robot] = self._find_enemy_gk()

        owner = field.robot_with_ball
        self.owner: Optional[rbt.Robot] = owner
        self.owner_is_ours = owner is not None and owner.color == field.ally_color
        self.owner_is_theirs = owner is not None and owner.color != field.ally_color

        self.ball_lost_vision = now - field.ball.last_update() > params.BALL_LOST_VISION and owner is None
        lim_x = const.FIELD_DX + const.BALL_R
        lim_y = const.FIELD_DY + const.BALL_R
        self.ball_out_of_field = abs(self.ball.x) > lim_x or abs(self.ball.y) > lim_y

        self._build_trajectory()
        self.our_intercepts: dict[int, Intercept] = {}
        self.their_intercepts: dict[int, Intercept] = {}
        for robot in self.allies:
            self.our_intercepts[robot.r_id] = self._intercept(robot, ours=True)
        for robot in self.enemies:
            self.their_intercepts[robot.r_id] = self._intercept(robot, ours=False)

        self.our_best_id, self.our_best_t = self._best(self.our_intercepts, exclude_gk=field.gk_id)
        self.their_best_id, self.their_best_t = self._best(self.their_intercepts, exclude_gk=None)
        self.ball_race = self.their_best_t - self.our_best_t
        if math.isinf(self.their_best_t) and math.isinf(self.our_best_t):
            self.ball_race = 0.0

        self.raw_possession = self._raw_possession()
        self.possession = self.raw_possession

    @staticmethod
    def build(field: fld.Field, memory: Memory, now: float) -> "WorldModel":
        """Построить модель мира (память пока не нужна, параметр - по интерфейсу ТЗ)"""
        del memory
        return WorldModel(field, now)

    # --- Геометрия ---

    def x_att(self, point: aux.Point) -> float:
        """R-6.6: координата в системе атаки (растёт к чужим воротам)"""
        return point.x * self.attack_sign

    def from_att(self, x_att: float, y: float) -> aux.Point:
        """Точка по координате атаки"""
        return aux.Point(x_att * self.attack_sign, y)

    def zone(self, point: aux.Point) -> str:
        """Зона поля по трети: DEF / MID / ATT"""
        x = self.x_att(point)
        third = const.FIELD_DX * 2 / 3
        if x < -const.FIELD_DX + third:
            return "DEF"
        if x > const.FIELD_DX - third:
            return "ATT"
        return "MID"

    def in_our_half(self, point: aux.Point) -> bool:
        """Точка на нашей половине"""
        return self.x_att(point) < 0

    @staticmethod
    def in_field(point: aux.Point, margin: float = 0.0) -> bool:
        """Точка в поле с отступом margin внутрь"""
        return abs(point.x) <= const.FIELD_DX - margin and abs(point.y) <= const.FIELD_DY - margin

    def in_area(self, point: aux.Point, ally: bool, margin: float = 0.0) -> bool:
        """Точка в штрафной (своей при ally=True), расширенной на margin"""
        goal = self.field.ally_goal if ally else self.field.enemy_goal
        depth = (point.x - goal.center.x) * goal.eye_forw.x
        return depth < const.GOAL_PEN_DX + margin and abs(point.y) < const.GOAL_PEN_DY / 2 + margin

    # --- Мяч ---

    def ball_pos_at(self, t: float) -> aux.Point:
        """R-6.1: положение мяча через t с (без отскоков)"""
        if not self.ball_moving:
            return self.ball
        t_stop = self.ball_speed / params.BALL_DECEL
        t = min(max(t, 0.0), t_stop)
        s = self.ball_speed * t - params.BALL_DECEL * t**2 / 2
        return self.ball + self.ball_dir * s

    def ball_stop_point(self) -> aux.Point:
        """Точка остановки мяча"""
        return self.ball_pos_at(INF)

    def _build_trajectory(self) -> None:
        times = np.arange(0.0, params.INTERCEPT_HORIZON + 1e-9, params.INTERCEPT_STEP)
        if self.ball_moving:
            t_stop = self.ball_speed / params.BALL_DECEL
            tt = np.minimum(times, t_stop)
            s = self.ball_speed * tt - params.BALL_DECEL * tt**2 / 2
        else:
            s = np.zeros_like(times)
        self.traj_t = times
        self.traj_x = self.ball.x + self.ball_dir.x * s
        self.traj_y = self.ball.y + self.ball_dir.y * s
        in_field = (np.abs(self.traj_x) <= const.FIELD_DX) & (np.abs(self.traj_y) <= const.FIELD_DY)
        self.traj_valid_gk = in_field
        self.traj_valid_field = in_field & ~self._in_any_area(self.traj_x, self.traj_y)

    def _in_any_area(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        result = np.zeros_like(xs, dtype=bool)
        for goal in (self.field.ally_goal, self.field.enemy_goal):
            depth = (xs - goal.center.x) * goal.eye_forw.x
            inside = (depth < const.GOAL_PEN_DX + const.ROBOT_R) & (np.abs(ys) < const.GOAL_PEN_DY / 2 + const.ROBOT_R)
            result |= inside
        return result

    def _intercept(self, robot: rbt.Robot, ours: bool) -> Intercept:
        """R-6.3: минимальное t, при котором робот успевает в точку мяча"""
        if self.field.robot_with_ball == robot:
            return Intercept(0.0, self.ball)
        is_gk = robot.r_id == (self.field.gk_id if ours else self.field.enemy_gk_id)
        valid = self.traj_valid_gk if is_gk else self.traj_valid_field

        if ours:
            vmax, amax, react = float(const.MAX_SPEED), float(const.MAX_ACCELERATION), params.REACTION_TIME
        else:
            vmax, amax, react = params.ENEMY_MAX_SPEED, params.ENEMY_MAX_ACC, params.ENEMY_REACTION_TIME
        pos = robot.get_pos()
        vel = robot.get_vel()
        dx = self.traj_x - pos.x
        dy = self.traj_y - pos.y
        dist_full = np.hypot(dx, dy)
        dist = np.maximum(dist_full - (const.ROBOT_R + const.BALL_R), 0.0)
        safe = np.maximum(dist_full, 1e-6)
        v0 = (vel.x * dx + vel.y * dy) / safe
        turn = np.abs(np.angle(np.exp(1j * (np.arctan2(dy, dx) - robot.get_angle()))))
        t_robot = _times_vec(dist, v0, vmax, amax) + 0.5 * turn / params.OMEGA_FREE + react

        ok = (t_robot <= self.traj_t) & valid
        idx = np.flatnonzero(ok)
        if idx.size > 0:
            i = int(idx[0])
            return Intercept(float(self.traj_t[i]), aux.Point(float(self.traj_x[i]), float(self.traj_y[i])))
        if bool(valid[-1]):
            return Intercept(
                max(float(t_robot[-1]), float(self.traj_t[-1])), aux.Point(float(self.traj_x[-1]), float(self.traj_y[-1]))
            )
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size > 0:
            i = int(valid_idx[-1])
            return Intercept(INF, aux.Point(float(self.traj_x[i]), float(self.traj_y[i])))
        return Intercept(INF, aux.Point(aux.minmax(self.ball.x, const.FIELD_DX), aux.minmax(self.ball.y, const.FIELD_DY)))

    @staticmethod
    def _best(intercepts: dict[int, Intercept], exclude_gk: Optional[int]) -> tuple[Optional[int], float]:
        best_id: Optional[int] = None
        best_t = INF
        for r_id, inter in intercepts.items():
            if exclude_gk is not None and r_id == exclude_gk:
                continue
            if inter.t < best_t:
                best_t = inter.t
                best_id = r_id
        return best_id, best_t

    # --- Роботы ---

    def _find_enemy_gk(self) -> Optional[rbt.Robot]:
        """R-6.7: вратарь соперника"""
        gk = self.field.enemies[self.field.enemy_gk_id]
        if gk.is_used():
            return gk
        best: Optional[rbt.Robot] = None
        best_dist = INF
        for enemy in self.enemies:
            if self.in_area(enemy.get_pos(), ally=False, margin=200):
                dist = aux.dist(enemy.get_pos(), self.field.enemy_goal.center)
                if dist < best_dist:
                    best_dist = dist
                    best = enemy
        return best

    def time_to_point(self, robot: rbt.Robot, target: aux.Point, reach: float = const.ROBOT_R + const.BALL_R) -> float:
        """Время робота (нашего или соперника) до точки"""
        return robot_time_to_point(robot, target, robot.color == self.field.ally_color, reach)

    def nearest_enemy(self, point: aux.Point, exclude_gk: bool = False) -> Optional[rbt.Robot]:
        """Ближайший к точке соперник"""
        best: Optional[rbt.Robot] = None
        best_dist = INF
        for enemy in self.enemies:
            if exclude_gk and self.enemy_gk is not None and enemy.r_id == self.enemy_gk.r_id:
                continue
            dist = aux.dist(enemy.get_pos(), point)
            if dist < best_dist:
                best_dist = dist
                best = enemy
        return best

    def enemy_time_to(self, point: aux.Point) -> float:
        """Лучшее время соперников до точки"""
        best = INF
        for enemy in self.enemies:
            best = min(best, self.time_to_point(enemy, point))
        return best

    def _raw_possession(self) -> Possession:
        """R-6.4: владение без гистерезиса"""
        if self.owner_is_ours:
            return Possession.OURS
        if self.owner_is_theirs:
            return Possession.THEIRS
        ours_close = any(aux.dist(r.get_pos(), self.ball) < params.CONTEST_RADIUS for r in self.allies)
        theirs_close = any(aux.dist(r.get_pos(), self.ball) < params.CONTEST_RADIUS for r in self.enemies)
        if ours_close and theirs_close:
            return Possession.CONTESTED
        return Possession.LOOSE
