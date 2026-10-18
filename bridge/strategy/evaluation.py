"""
Оценочные функции (ТЗ, раздел 7)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.router import legality
from bridge.strategy import kick_model as km
from bridge.strategy import params
from bridge.strategy.memory import Memory
from bridge.strategy.world import WorldModel, travel_dist, travel_time

HIT_R = const.ROBOT_R + const.BALL_R


def clip01(x: float) -> float:
    """Ограничить [0, 1]"""
    return max(0.0, min(1.0, x))


def logistic(x: float) -> float:
    """Гладкая ступенька"""
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def dist_factor(dist: float) -> float:
    """Фактор дистанции удара: 1 до SHOT_FULL_DIST, линейно до SHOT_FAR_FACTOR к SHOT_MAX_DIST, дальше 0"""
    if dist > params.SHOT_MAX_DIST:
        return 0.0
    if dist <= params.SHOT_FULL_DIST:
        return 1.0
    k = (dist - params.SHOT_FULL_DIST) / (params.SHOT_MAX_DIST - params.SHOT_FULL_DIST)
    return 1.0 - k * (1.0 - params.SHOT_FAR_FACTOR)


def ball_time(dist: float, speed: float) -> float:
    """Время мяча на дистанцию (если не доходит - оценка по средней скорости)"""
    t = km.ball_travel_time(dist, speed)
    if math.isinf(t):
        return dist / max(speed, 1.0) * 2
    return t


# --- 7.1 Удар ---


@dataclass
class ShotEval:
    """Оценка удара"""

    target: aux.Point
    open_width: float  # рад
    p_goal: float
    gk_margin: float = math.inf
    intervals: list[tuple[float, float]] = field(default_factory=list)


def _goal_ends(goal: fld.Goal) -> tuple[aux.Point, aux.Point]:
    """Створ, сужённый на BALL_R + SHOT_POST_MARGIN с каждой стороны"""
    half = const.GOAL_DY / 2 - const.BALL_R - params.SHOT_POST_MARGIN
    return aux.Point(goal.center.x, goal.center.y + half), aux.Point(goal.center.x, goal.center.y - half)


def subtract_intervals(lo: float, hi: float, blocks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = [(lo, hi)]
    for b_lo, b_hi in sorted(blocks):
        new_free: list[tuple[float, float]] = []
        for f_lo, f_hi in free:
            if b_hi <= f_lo or b_lo >= f_hi:
                new_free.append((f_lo, f_hi))
                continue
            if b_lo > f_lo:
                new_free.append((f_lo, b_lo))
            if b_hi < f_hi:
                new_free.append((b_hi, f_hi))
        free = new_free
    return [(a, b) for a, b in free if b - a > 1e-4]


def enemy_gk_expected(goal: fld.Goal, ball: aux.Point) -> aux.Point:
    """Где встанет вратарь соперника: биссектриса угла «мяч -> штанги» на линии вратаря"""
    direction = ((goal.up - ball).unity() + (goal.down - ball).unity()).unity()
    depth = const.ROBOT_R + params.SHOT_ENEMY_GK_LINE_GAP
    a = goal.center + goal.eye_forw * depth + aux.Point(0, -const.FIELD_DY)
    b = goal.center + goal.eye_forw * depth + aux.Point(0, const.FIELD_DY)
    point = aux.get_line_intersection(ball, ball + direction, a, b, "RL")
    if point is None:
        return goal.center + goal.eye_forw * depth
    limit = const.GOAL_DY / 2
    return aux.Point(point.x, aux.minmax(point.y, goal.center.y - limit, goal.center.y + limit))


def eval_shot(
    world: WorldModel,
    from_point: aux.Point,
    shooter_id: Optional[int] = None,
    speed: float = params.SHOT_SPEED,
    extra_time: float = 0.0,
    ignore_ids: Optional[list[int]] = None,
) -> ShotEval:
    """
    Удар из from_point (7.1). extra_time - задержка до удара (приём паса и т.п.): соперники
    успевают сместиться дальше. ignore_ids - наши роботы, которых не считать помехой.
    """
    goal = world.field.enemy_goal
    end_a, end_b = _goal_ends(goal)
    empty = ShotEval(goal.center, 0.0, 0.0)
    d_goal = aux.dist(from_point, goal.center)
    if d_goal > params.SHOT_MAX_DIST or world.x_att(from_point) >= world.x_att(goal.center) - const.BALL_R:
        return empty

    ref = aux.angle_to_point(from_point, goal.center)
    ra = aux.wind_down_angle(aux.angle_to_point(from_point, end_a) - ref)
    rb = aux.wind_down_angle(aux.angle_to_point(from_point, end_b) - ref)
    lo, hi = min(ra, rb), max(ra, rb)
    goal_dir = (goal.center - from_point).unity()

    blocks: list[tuple[float, float]] = []
    gk = world.enemy_gk
    gk_rel: Optional[float] = None
    gk_pos = gk.get_pos() if gk is not None else None
    if gk is not None and extra_time > 0:
        # Удар позже: вратарь соперника успеет встать на биссектрису угла «мяч -> штанги»
        gk_pos = enemy_gk_expected(goal, from_point)
    ignore = set(ignore_ids or [])
    robots: list[tuple[rbt.Robot, bool]] = [(e, True) for e in world.enemies]
    robots += [(a, False) for a in world.allies if a.r_id != shooter_id and a.r_id not in ignore]
    goal_line_x = world.x_att(goal.center)
    for robot, is_enemy in robots:
        pos = robot.get_pos()
        if gk is not None and gk_pos is not None and is_enemy and robot.r_id == gk.r_id:
            pos = gk_pos
        d = aux.dist(from_point, pos)
        if d < 1:
            continue
        if aux.scal_mult(pos - from_point, goal_dir) < -HIT_R or world.x_att(pos) > goal_line_x + const.ROBOT_R:
            continue
        reach = 0.0
        is_gk = gk is not None and is_enemy and robot.r_id == gk.r_id
        if is_enemy and not is_gk:
            t_avail = ball_time(d, speed) + extra_time - params.ENEMY_REACTION_TIME
            reach = travel_dist(t_avail, params.ENEMY_MAX_SPEED, params.ENEMY_MAX_ACC) * params.SHOT_REACH_K
            reach = min(reach, params.SHOT_REACH_CAP)
        half = math.asin(min(1.0, (HIT_R + reach) / d))
        center = aux.wind_down_angle(aux.angle_to_point(from_point, pos) - ref)
        blocks.append((center - half, center + half))
        if gk is not None and robot.r_id == gk.r_id and is_enemy:
            gk_rel = center

    free = subtract_intervals(lo, hi, blocks)
    if not free:
        # Створ закрыт: запасная цель - штанга, дальняя от вратаря (а не центр ворот)
        far_end = end_a
        if gk_rel is not None:
            far_end = end_a if abs(ra - gk_rel) > abs(rb - gk_rel) else end_b
        return ShotEval(far_end, 0.0, 0.0, intervals=[])

    f_lo, f_hi = max(free, key=lambda iv: iv[1] - iv[0])
    width = f_hi - f_lo
    aim = (f_lo + f_hi) / 2
    if gk_rel is not None:
        shift = params.SHOT_GK_SHIFT * width * (-1.0 if gk_rel > aim else 1.0)
        aim = aux.minmax(aim + shift, f_lo + width * 0.1, f_hi - width * 0.1)

    direction = aux.rotate(aux.RIGHT, ref + aim)
    target = aux.get_line_intersection(from_point, from_point + direction, goal.down, goal.up, "RL")
    if target is None:
        target = goal.center

    gk_margin = math.inf
    if gk is not None and gk_pos is not None:
        t_ball = ball_time(aux.dist(from_point, target), speed)
        on_line = aux.closest_point_on_line(from_point, target, gk_pos, "S")
        d_gk = max(0.0, aux.dist(on_line, gk_pos) - HIT_R)
        t_gk = travel_time(d_gk, 0.0, params.ENEMY_MAX_SPEED, params.ENEMY_MAX_ACC) + params.ENEMY_REACTION_TIME
        gk_margin = t_gk - t_ball

    p_width = clip01(width / params.SHOT_WIDTH_FULL)
    p_gk = clip01((gk_margin + params.SHOT_GK_MARGIN_OFFSET) / params.SHOT_GK_MARGIN_SCALE)
    p_goal = p_width * p_gk * dist_factor(d_goal)
    if world.in_our_half(from_point) and p_goal < params.SHOT_OWN_HALF_MIN_P:
        p_goal = 0.0  # 6.2.3: из своей половины только верный удар
    return ShotEval(target, width, p_goal, gk_margin, free)


# --- 7.2 Пас ---


@dataclass
class PassEval:
    """Оценка паса"""

    speed: float
    p_success: float
    value: float
    min_margin: float
    ok: bool
    reason: str = ""


def _crosses_own_goal_front(world: WorldModel, a: aux.Point, b: aux.Point) -> bool:
    """Пас поперёк нашей штрафной перед воротами"""
    goal = world.field.ally_goal
    closest = aux.closest_point_on_line(a, b, goal.center, "S")
    if aux.dist(closest, goal.center) > params.PASS_OWN_GOAL_FRONT:
        return False
    direction = (b - a).unity()
    lateral = abs(direction.y) > abs(direction.x)
    side_change = a.y * b.y < 0 and min(abs(a.y), abs(b.y)) > const.GOAL_DY / 2
    return lateral or side_change


def field_travel_limit(from_point: aux.Point, direction: aux.Point, margin: float = const.BALL_R * 2) -> float:
    """Сколько мяч может прокатиться от from_point по direction, оставаясь в поле"""
    limits = []
    for coord, vel, half in ((from_point.x, direction.x, const.FIELD_DX), (from_point.y, direction.y, const.FIELD_DY)):
        if abs(vel) < 1e-9:
            continue
        bound = (half - margin) if vel > 0 else (-half + margin)
        limits.append((bound - coord) / vel)
    return max(0.0, min(limits)) if limits else math.inf


def pass_speed_in_field(from_point: aux.Point, receive_point: aux.Point) -> tuple[float, bool]:
    """
    Скорость паса (7.2) с остановкой мяча в поле (G-13): если мяч после приёма укатился бы за поле,
    скорость снижается. False - мяч тогда придёт слишком медленно (пас недопустим).
    """
    d = aux.dist(from_point, receive_point)
    speed = km.pass_speed(d)
    limit = field_travel_limit(from_point, (receive_point - from_point).unity())
    if km.stop_distance(speed) > limit:
        speed = km.stop_speed(limit)
        if km.ball_speed_after(d, speed) < params.PASS_MIN_ARRIVAL_SPEED:
            return speed, False
    return speed, True


def pass_forbidden(world: WorldModel, from_point: aux.Point, receive_point: aux.Point, speed: float) -> str:
    """Запрещённые пасы (7.2, G-13); пустая строка - пас разрешён"""
    fld_ = world.field
    if not world.in_field(receive_point, const.ROBOT_R):
        return "receiver out"
    if aux.segment_poly_intersect(from_point, receive_point, fld_.enemy_goal.hull) is not None:
        return "enemy area"
    if aux.is_point_inside_poly(receive_point, fld_.ally_goal.big_hull) or aux.is_point_inside_poly(
        receive_point, fld_.enemy_goal.big_hull
    ):
        return "receiver in area"
    from_in_area = aux.is_point_inside_poly(from_point, fld_.ally_goal.hull)
    if not from_in_area and aux.segment_poly_intersect(from_point, receive_point, fld_.ally_goal.hull) is not None:
        return "ally area"
    if _crosses_own_goal_front(world, from_point, receive_point):
        return "own goal front"
    direction = (receive_point - from_point).unity()
    stop = from_point + direction * km.stop_distance(speed)
    if not world.in_field(stop):
        return "stops out"
    return ""


def eval_pass(
    world: WorldModel,
    from_point: aux.Point,
    receiver: Optional[rbt.Robot],
    receive_point: aux.Point,
    t_rot: float = 0.0,
    with_value: bool = True,
) -> PassEval:
    """Пас из from_point в receive_point (7.2)"""
    d = aux.dist(from_point, receive_point)
    raw_speed = math.sqrt(params.PASS_ARRIVAL_SPEED**2 + 2 * params.BALL_DECEL * d)
    speed, in_field = pass_speed_in_field(from_point, receive_point)
    capped = raw_speed > params.BALL_SPEED_SAFE

    reason = pass_forbidden(world, from_point, receive_point, speed) if in_field else "stops out"
    if reason:
        return PassEval(speed, 0.0, -1.0, -math.inf, False, reason)

    # Самая рискованная точка для каждого соперника
    n = params.PASS_SAMPLES
    fractions = np.linspace(1.0 / n, 1.0, n)
    xs = from_point.x + (receive_point.x - from_point.x) * fractions
    ys = from_point.y + (receive_point.y - from_point.y) * fractions
    dists = d * fractions
    disc = np.maximum(speed**2 - 2 * params.BALL_DECEL * dists, 0.0)
    t_ball = (speed - np.sqrt(disc)) / params.BALL_DECEL
    min_margin = math.inf
    for enemy in world.enemies:
        pos = enemy.get_pos()
        dd = np.maximum(np.hypot(xs - pos.x, ys - pos.y) - HIT_R, 0.0)
        t_en = np.array([travel_time(float(x), 0.0, params.ENEMY_MAX_SPEED, params.ENEMY_MAX_ACC) for x in dd])
        t_en += params.ENEMY_REACTION_TIME
        min_margin = min(min_margin, float(np.min(t_en - t_ball)))

    p_success = logistic(min_margin / params.PASS_P_SCALE) if not math.isinf(min_margin) else 1.0
    t_arrive = float(t_ball[-1])
    if receiver is not None:
        t_recv = world.time_to_point(receiver, receive_point, reach=0.0)
        if t_recv > t_arrive + params.PASS_RECEIVER_SLACK:
            p_success *= clip01(1.0 - (t_recv - t_arrive - params.PASS_RECEIVER_SLACK))
    ok = min_margin >= params.PASS_SAFETY_MARGIN and p_success >= params.PASS_MIN_SUCCESS * 0.5

    value = 0.0
    if with_value:
        shot = eval_shot(world, receive_point, receiver.r_id if receiver else None, extra_time=t_arrive)
        progress = (world.x_att(receive_point) - world.x_att(from_point)) / (2 * const.FIELD_DX)
        value = p_success * (shot.p_goal + params.PROGRESS_W * progress) - params.ROT_W * t_rot
        if capped:
            value *= params.PASS_LONG_PENALTY
    return PassEval(speed, p_success, value, min_margin, ok)


# --- 7.3 Открывание Опорного ---


def _support_candidates(
    world: WorldModel, attacker_pos: Optional[aux.Point], ball_in_play: bool
) -> tuple[np.ndarray, np.ndarray]:
    step = params.SUPPORT_GRID_STEP
    border = params.SUPPORT_BORDER_MARGIN
    xs = np.arange(-const.FIELD_DX + border, const.FIELD_DX - border + 1, step)
    ys = np.arange(-const.FIELD_DY + border, const.FIELD_DY - border + 1, step)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel().astype(float)
    py = gy.ravel().astype(float)

    mask = np.ones_like(px, dtype=bool)
    for goal, ally in ((world.field.ally_goal, True), (world.field.enemy_goal, False)):
        margin = float(params.SUPPORT_AREA_MARGIN)
        if not ally and not ball_in_play:
            margin = max(margin, legality.enemy_area_margin(world.field.game_state))
        depth = (px - goal.center.x) * goal.eye_forw.x
        mask &= ~((depth < const.GOAL_PEN_DX + margin) & (np.abs(py) < const.GOAL_PEN_DY / 2 + margin))
    ball = world.ball
    mask &= np.hypot(px - ball.x, py - ball.y) >= params.SUPPORT_MIN_DIST_BALL
    if attacker_pos is not None:
        mask &= np.hypot(px - attacker_pos.x, py - attacker_pos.y) >= params.SUPPORT_MIN_DIST_MATE
    for enemy in world.enemies:
        pos = enemy.get_pos()
        mask &= np.hypot(px - pos.x, py - pos.y) >= params.SUPPORT_MIN_DIST_ENEMY
    return px[mask], py[mask]


def _cheap_scores(world: WorldModel, px: np.ndarray, py: np.ndarray, support: rbt.Robot) -> np.ndarray:
    goal = world.field.enemy_goal
    a_x, a_y = goal.center.x, goal.center.y + const.GOAL_DY / 2
    b_x, b_y = goal.center.x, goal.center.y - const.GOAL_DY / 2
    ang = np.abs(np.angle(np.exp(1j * (np.arctan2(a_y - py, a_x - px) - np.arctan2(b_y - py, b_x - px)))))
    d_goal = np.hypot(goal.center.x - px, goal.center.y - py)
    dfac = np.where(
        d_goal <= params.SHOT_FULL_DIST,
        1.0,
        1.0
        - np.clip((d_goal - params.SHOT_FULL_DIST) / (params.SHOT_MAX_DIST - params.SHOT_FULL_DIST), 0, 1)
        * (1 - params.SHOT_FAR_FACTOR),
    )
    dfac = np.where(d_goal > params.SHOT_MAX_DIST, 0.0, dfac)
    p_goal = np.clip(ang / params.SHOT_WIDTH_FULL, 0, 1) * dfac * 0.5

    free = np.full_like(px, 1000.0)
    safety = np.ones_like(px)
    ball = world.ball
    for enemy in world.enemies:
        pos = enemy.get_pos()
        free = np.minimum(free, np.hypot(px - pos.x, py - pos.y))
        # расстояние соперника до отрезка мяч -> точка
        vx, vy = px - ball.x, py - ball.y
        length2 = np.maximum(vx**2 + vy**2, 1e-6)
        t = np.clip(((pos.x - ball.x) * vx + (pos.y - ball.y) * vy) / length2, 0, 1)
        dseg = np.hypot(ball.x + t * vx - pos.x, ball.y + t * vy - pos.y)
        safety = np.minimum(safety, np.clip((dseg - 200) / 600, 0, 1))
    sp = support.get_pos()
    t_reach = np.hypot(px - sp.x, py - sp.y) / const.MAX_SPEED
    x_att = px * world.attack_sign
    front = const.FIELD_DX - const.GOAL_PEN_DX
    rebound = ((x_att > front - 700) & (x_att < front - 200)).astype(float)
    progress = np.clip((x_att - world.x_att(ball)) / const.FIELD_DX, -1, 1)
    return (
        params.SUPPORT_W_GOAL * p_goal
        + params.SUPPORT_W_SAFETY * safety
        + params.SUPPORT_W_FREE * free / 1000
        - params.SUPPORT_W_TIME * t_reach
        + params.SUPPORT_W_REBOUND * rebound
        + params.SUPPORT_W_PROGRESS * progress
    )


def support_point_score(world: WorldModel, point: aux.Point, support: rbt.Robot, pass_from: aux.Point) -> float:
    """Полная оценка точки открывания (7.3)"""
    shot = eval_shot(world, point, support.r_id, extra_time=0.5)
    # Пас гипотетический (после прихода Опорного): время подхода учтено слагаемым t_reach
    pas = eval_pass(world, pass_from, None, point, with_value=False)
    safety = pas.p_success if pas.ok or pas.p_success > 0 else 0.0
    free = 1000.0
    for enemy in world.enemies:
        free = min(free, aux.dist(enemy.get_pos(), point))
    t_reach = aux.dist(support.get_pos(), point) / const.MAX_SPEED
    front = const.FIELD_DX - const.GOAL_PEN_DX
    x_att = world.x_att(point)
    rebound = 1.0 if front - 700 < x_att < front - 200 else 0.0
    progress = aux.minmax((x_att - world.x_att(pass_from)) / const.FIELD_DX, -1, 1)
    return (
        params.SUPPORT_W_GOAL * shot.p_goal
        + params.SUPPORT_W_SAFETY * safety
        + params.SUPPORT_W_FREE * free / 1000
        - params.SUPPORT_W_TIME * t_reach
        + params.SUPPORT_W_REBOUND * rebound
        + params.SUPPORT_W_PROGRESS * progress
    )


def best_support_point(
    world: WorldModel,
    memory: Memory,
    support: rbt.Robot,
    attacker_pos: Optional[aux.Point],
    ball_in_play: bool = True,
) -> aux.Point:
    """Лучшая точка открывания Опорного с гистерезисом (7.3)"""
    now = world.now
    old = memory.support_target
    if old is not None and now - memory.support_update_time < params.SUPPORT_UPDATE_PERIOD:
        return old

    px, py = _support_candidates(world, attacker_pos, ball_in_play)
    if px.size == 0:
        fallback = world.ball + (world.field.ally_goal.center - world.ball).unity() * params.SUPPORT_MIN_DIST_BALL
        memory.support_target = fallback
        memory.support_update_time = now
        return fallback

    cheap = _cheap_scores(world, px, py, support)
    if px.size > params.SUPPORT_MAX_CANDIDATES:
        keep = np.argsort(-cheap)[: params.SUPPORT_MAX_CANDIDATES]
        px, py, cheap = px[keep], py[keep], cheap[keep]
    top = np.argsort(-cheap)[: params.SUPPORT_TOP_K]

    best_point = aux.Point(float(px[top[0]]), float(py[top[0]]))
    best_score = -math.inf
    for i in top:
        point = aux.Point(float(px[i]), float(py[i]))
        score = support_point_score(world, point, support, world.ball)
        if score > best_score:
            best_score = score
            best_point = point

    old_valid = old is not None and _support_point_valid(world, old, attacker_pos, ball_in_play)
    if old is not None and old_valid:
        old_score = support_point_score(world, old, support, world.ball)
        if best_score < old_score + params.SUPPORT_SWITCH_GAIN:
            best_point, best_score = old, old_score

    memory.support_target = best_point
    memory.support_target_score = best_score
    memory.support_update_time = now
    return best_point


def _support_point_valid(world: WorldModel, point: aux.Point, attacker_pos: Optional[aux.Point], ball_in_play: bool) -> bool:
    if not world.in_field(point, params.SUPPORT_BORDER_MARGIN - 1):
        return False
    if world.in_area(point, True, params.SUPPORT_AREA_MARGIN) or world.in_area(point, False, params.SUPPORT_AREA_MARGIN):
        return False
    if not ball_in_play and world.in_area(point, False, legality.enemy_area_margin(world.field.game_state)):
        return False
    if aux.dist(point, world.ball) < params.SUPPORT_MIN_DIST_BALL:
        return False
    if attacker_pos is not None and aux.dist(point, attacker_pos) < params.SUPPORT_MIN_DIST_MATE:
        return False
    return all(aux.dist(point, e.get_pos()) >= params.SUPPORT_MIN_DIST_ENEMY for e in world.enemies)


# --- 7.4 Безопасный вынос ---


def safe_clear_target(
    world: WorldModel, from_point: aux.Point, cone: Optional[tuple[float, float]] = None
) -> tuple[aux.Point, float]:
    """
    Точка выноса в поле подальше от соперников и скорость, при которой мяч остановится там (7.4).
    cone = (курс, максимальный поворот) - только направления в этом конусе (R-4.6).
    """
    margin = params.CLEAR_BORDER_MARGIN
    step = params.CLEAR_GRID_STEP
    best: Optional[aux.Point] = None
    best_score = -math.inf
    x_from = world.x_att(from_point)
    max_dist = km.stop_distance(params.BALL_SPEED_SAFE)
    x = -const.FIELD_DX + margin
    while x <= const.FIELD_DX - margin:
        y = -const.FIELD_DY + margin
        while y <= const.FIELD_DY - margin:
            point = aux.Point(x, y)
            y += step
            d = aux.dist(point, from_point)
            if d < 2 * step or d > max_dist:
                continue
            if world.x_att(point) < x_from - const.ROBOT_R and cone is None:
                continue
            if cone is not None and abs(aux.wind_down_angle(aux.angle_to_point(from_point, point) - cone[0])) > cone[1]:
                continue
            if world.in_area(point, True, params.SUPPORT_AREA_MARGIN) or world.in_area(
                point, False, params.SUPPORT_AREA_MARGIN
            ):
                continue
            if pass_forbidden(world, from_point, point, km.stop_speed(d)) not in ("", "stops out"):
                continue
            t_enemy = min(world.enemy_time_to(point), 2.0)
            t_enemy_line = math.inf
            for enemy in world.enemies:
                on_line = aux.closest_point_on_line(from_point, point, enemy.get_pos(), "S")
                t_enemy_line = min(t_enemy_line, aux.dist(on_line, enemy.get_pos()) / params.ENEMY_MAX_SPEED)
            progress = (world.x_att(point) - x_from) / (2 * const.FIELD_DX)
            away_goal = aux.dist(point, world.field.ally_goal.center) / (2 * const.FIELD_DX)
            score = t_enemy + 0.5 * min(t_enemy_line, 1.0) + 0.5 * progress + 0.3 * away_goal
            if score > best_score:
                best_score = score
                best = point
        x += step
    if best is None and cone is not None:
        direction = aux.rotate(aux.RIGHT, cone[0])
        best = from_point + direction * step * 2
        best = aux.Point(aux.minmax(best.x, const.FIELD_DX - margin), aux.minmax(best.y, const.FIELD_DY - margin))
    if best is None:
        best = world.from_att(min(x_from + 1000, const.FIELD_DX - margin), aux.minmax(from_point.y, const.FIELD_DY - margin))
    return best, km.stop_speed(aux.dist(best, from_point))


# --- 7.5 Допустимый угол захвата ---

GRAB_ANGLE_STEP = math.radians(10)
GRAB_PREP_EXTRA = 150


def feasible_grab_angle(world: WorldModel, robot_id: int, ball: aux.Point, desired_angle: float) -> float:
    """Ближайший к желаемому угол, при котором точки подъезда за мячом допустимы (7.5, R-4.4)"""
    fld_ = world.field
    for k in range(0, 19):
        for sign in (1, -1):
            if k == 0 and sign < 0:
                continue
            angle = desired_angle + sign * k * GRAB_ANGLE_STEP
            back = aux.rotate(aux.RIGHT, angle)
            contact = ball - back * const.GRAB_ALIGN_DIST
            prep = ball - back * (const.GRAB_ALIGN_DIST + GRAB_PREP_EXTRA)
            if legality.is_legal_target(fld_, robot_id, contact) and legality.is_legal_target(fld_, robot_id, prep):
                return aux.wind_down_angle(angle)
    return desired_angle
