"""
Легальность целей роботов по Регламенту (R-0.4, R-0.8, G-4)

Роутерная часть: модуль не импортирует стратегию. Стратегия (bridge/strategy/guards.py)
использует эти же функции и добавляет свои ограничения.
"""

import math
from typing import Callable, Optional

from bridge import const
from bridge.auxiliary import aux, fld

# Сколько раз прогонять набор ограничений: выталкивание из одной зоны может завести в другую
LEGAL_ITERATIONS = 4
# Отступ за границу зоны при выталкивании, мм
PUSH_EPS = 5.0
# Отступ цели полевого от своей штрафной (как fld.Goal.big_hull)
ALLY_AREA_MARGIN = const.ROBOT_R * 1.2
# Запасной поиск допустимой точки: шаг радиуса, число радиусов и направлений
SEARCH_STEP = 50.0
SEARCH_RINGS = 40
SEARCH_DIRS = 24


def enemy_area_margin(state: const.State) -> float:
    """Отступ центра полевого робота от чужой штрафной (как big_* оболочки в fld.Goal)"""
    if state == const.State.STOP:
        return const.STOP_AREA_DELTA + const.ROBOT_R * 1.2
    if fld.is_ball_not_in_play(state):
        return const.DEF_AREA_MARGIN_BALL_OUT + const.ROBOT_R * 1.2
    return const.ROBOT_R * 1.2


def clamp_to_field(point: aux.Point, margin: float = const.FIELD_OUT_MARGIN) -> aux.Point:
    """Цель внутри поля, расширенного на margin (margin < 0 - внутрь поля)"""
    lim_x = const.FIELD_DX + margin
    lim_y = const.FIELD_DY + margin
    return aux.Point(aux.minmax(point.x, -lim_x, lim_x), aux.minmax(point.y, -lim_y, lim_y))


def area_depth(point: aux.Point, goal: fld.Goal) -> float:
    """Расстояние от линии ворот goal в сторону поля"""
    return (point.x - goal.center.x) * goal.eye_forw.x


def is_inside_area(point: aux.Point, goal: fld.Goal, margin: float) -> bool:
    """Точка внутри штрафной goal, расширенной на margin (включая всё, что за линией ворот)"""
    half_w = const.GOAL_PEN_DY / 2 + margin
    front = const.GOAL_PEN_DX + margin
    return area_depth(point, goal) < front and abs(point.y - goal.center.y) < half_w


def push_out_of_area(point: aux.Point, goal: fld.Goal, margin: float) -> aux.Point:
    """Вытолкнуть точку из штрафной goal, расширенной на margin, кратчайшим путём"""
    if not is_inside_area(point, goal, margin):
        return point
    half_w = const.GOAL_PEN_DY / 2 + margin
    front = const.GOAL_PEN_DX + margin
    depth = area_depth(point, goal)
    lat = point.y - goal.center.y
    side = 1.0 if lat >= 0 else -1.0

    to_front = front - depth
    to_side = half_w - abs(lat)
    side_y = goal.center.y + side * (half_w + PUSH_EPS)
    side_fits = abs(side_y) <= const.FIELD_DY + const.FIELD_OUT_MARGIN and depth >= 0
    if side_fits and to_side < to_front:
        return aux.Point(point.x, side_y)
    return aux.Point(goal.center.x + goal.eye_forw.x * (front + PUSH_EPS), point.y)


def push_out_of_circle(point: aux.Point, center: aux.Point, radius: float, away_dir: aux.Point) -> aux.Point:
    """Вытолкнуть точку из круга; away_dir - направление выталкивания, если точка в центре"""
    vec = point - center
    if vec.mag() >= radius:
        return point
    if vec.mag() < 1:
        vec = away_dir
    return center + vec.unity() * (radius + PUSH_EPS)


def is_legal_target(
    field: fld.Field,
    robot_id: int,
    point: aux.Point,
    *,
    keep_ball_dist: float = 0.0,
    field_margin: float = const.FIELD_OUT_MARGIN,
) -> bool:
    """Проверить цель робота без коррекции"""
    lim_x = const.FIELD_DX + field_margin + 1
    lim_y = const.FIELD_DY + field_margin + 1
    if abs(point.x) > lim_x or abs(point.y) > lim_y:
        return False
    if robot_id != field.gk_id:
        if is_inside_area(point, field.ally_goal, ALLY_AREA_MARGIN - 1):
            return False
        if is_inside_area(point, field.enemy_goal, enemy_area_margin(field.game_state) - 1):
            return False
    elif area_depth(point, field.ally_goal) < -1:
        return False
    if keep_ball_dist > 0 and aux.dist(point, field.ball.get_pos()) < keep_ball_dist - 1:
        return False
    return True


def legal_target(
    field: fld.Field,
    robot_id: int,
    point: aux.Point,
    *,
    keep_ball_dist: float = 0.0,
    field_margin: float = const.FIELD_OUT_MARGIN,
    extra_circles: Optional[list[tuple[aux.Point, float]]] = None,
) -> aux.Point:
    """
    Спроецировать цель робота в допустимую область:
    - в поле с зоной выбега (R-0.8);
    - полевой вне своей и чужой штрафных, у чужой - с отступом по состоянию игры (R-0.4);
    - вратарь не за линией своих ворот;
    - не ближе keep_ball_dist к мячу и вне extra_circles.
    """
    is_gk = robot_id == field.gk_id
    ball = field.ball.get_pos()
    enemy_margin = enemy_area_margin(field.game_state)
    circles: list[tuple[aux.Point, float]] = []
    if keep_ball_dist > 0:
        circles.append((ball, keep_ball_dist))
    if extra_circles is not None:
        circles.extend(extra_circles)

    p = aux.Point(point.x, point.y)
    for _ in range(LEGAL_ITERATIONS):
        p = clamp_to_field(p, field_margin)
        if is_gk:
            if area_depth(p, field.ally_goal) < 0:
                p = aux.Point(field.ally_goal.center.x, p.y)
        else:
            p = push_out_of_area(p, field.ally_goal, ALLY_AREA_MARGIN)
            p = push_out_of_area(p, field.enemy_goal, enemy_margin)
        for center, radius in circles:
            away = field.ally_goal.center - center
            if away.mag() < 1:
                away = field.ally_goal.eye_forw
            p = push_out_of_circle(p, center, radius, away)

        if is_legal_target(field, robot_id, p, field_margin=field_margin) and all(
            aux.dist(p, c) >= r - 1 for c, r in circles
        ):
            return p

    # Ограничения конфликтуют (например, мяч у штрафной): ближайшая точка, где выполнены все
    def ok(q: aux.Point) -> bool:
        return is_legal_target(field, robot_id, q, field_margin=field_margin) and all(
            aux.dist(q, c) >= r + PUSH_EPS for c, r in circles
        )

    found = nearest_valid(point, ok)
    return found if found is not None else p


def nearest_valid(point: aux.Point, ok: Callable[[aux.Point], bool]) -> Optional[aux.Point]:
    """Ближайшая к point точка (на кольцах вокруг неё), для которой ok() истинно"""
    if ok(point):
        return point
    for ring in range(1, SEARCH_RINGS + 1):
        radius = ring * SEARCH_STEP
        best: Optional[aux.Point] = None
        for k in range(SEARCH_DIRS):
            angle = 2 * math.pi * k / SEARCH_DIRS
            candidate = point + aux.Point(math.cos(angle), math.sin(angle)) * radius
            if ok(candidate) and (best is None or aux.dist(candidate, point) < aux.dist(best, point)):
                best = candidate
        if best is not None:
            return best
    return None


def angle_diff(a: float, b: float) -> float:
    """Модуль разности углов"""
    return abs(aux.wind_down_angle(a - b))


def unit(angle: float) -> aux.Point:
    """Единичный вектор по углу"""
    return aux.Point(math.cos(angle), math.sin(angle))
