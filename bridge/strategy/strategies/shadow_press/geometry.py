"""Geometry helpers of the «shadow, press, goalkeeper» strategy"""

import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt

from . import config as cfg


def face(src: aux.Point, dst: aux.Point, fallback: float) -> float:
    """Angle from src to dst; fallback when the points are too close (the direction is noisy)"""
    if aux.dist(src, dst) < cfg.FACE_MIN_DIST:
        return fallback
    return aux.angle_to_point(src, dst)


def receive_angle(point: aux.Point, ball_from: aux.Point, target: aux.Point) -> float:
    """
    Angle of the robot receiving the ball at point: facing the incoming ball,
    turned to the next kick target as much as the dribbler can still catch the ball
    """
    a_in = aux.angle_to_point(point, ball_from)
    turn = aux.wind_down_angle(aux.angle_to_point(point, target) - a_in)
    return a_in + aux.minmax(turn, -cfg.RECEIVE_ANGLE_OFFSET, cfg.RECEIVE_ANGLE_OFFSET)


def turn_after_receive(point: aux.Point, ball_from: aux.Point, target: aux.Point) -> float:
    """Turn (rad) left to do with the ball after receiving it at point"""
    a_in = aux.angle_to_point(point, ball_from)
    turn = abs(aux.wind_down_angle(aux.angle_to_point(point, target) - a_in))
    return max(turn - cfg.RECEIVE_ANGLE_OFFSET, 0)


def ball_pos_at(field: fld.Field, t: float) -> aux.Point:
    """Predicted ball position after t seconds (uniform slowdown)"""
    ball = field.ball.get_pos()
    vel = field.ball.get_vel()
    speed = vel.mag()
    if speed < 1:
        return ball
    t = min(t, speed / cfg.BALL_DECEL)
    return ball + vel.unity() * (speed * t - cfg.BALL_DECEL * t * t / 2)


def ball_stop_point(field: fld.Field) -> aux.Point:
    """Predicted point where the ball stops"""
    return ball_pos_at(field, 1e3)


def intercept_point(field: fld.Field, robot_pos: aux.Point, max_path: Optional[float] = None) -> aux.Point:
    """
    The first point of the ball trajectory that the robot can reach in time (inside the field)
    max_path: limit of the ball path (to intercept the ball before some point)
    """
    return aux.nearest_point_in_poly(_intercept_point(field, robot_pos, max_path), field.small_hull)


def _intercept_point(field: fld.Field, robot_pos: aux.Point, max_path: Optional[float]) -> aux.Point:
    ball = field.ball.get_pos()
    t = 0.0
    while t < 4:
        point = ball_pos_at(field, t)
        if max_path is not None and aux.dist(ball, point) > max_path:
            break
        if aux.dist(robot_pos, point) - const.ROBOT_R <= cfg.ROBOT_SPEED * t:
            return point
        t += 0.05

    end = ball_stop_point(field)
    if max_path is not None and aux.dist(ball, end) > max_path:
        # can't reach in time: go to the closest point of the trajectory before the limit
        limit = aux.point_on_line(ball, end, max_path)
        return aux.closest_point_on_line(ball, limit, robot_pos, "S")
    return end


def ball_cross_x(field: fld.Field, x_line: float) -> Optional[aux.Point]:
    """Point where the moving ball crosses the line x = x_line"""
    ball = field.ball.get_pos()
    vel = field.ball.get_vel()
    if vel.mag() < 1:
        return None
    return aux.get_line_intersection(ball, ball + vel, aux.Point(x_line, -1), aux.Point(x_line, 1), "RL")


def is_line_free(src: aux.Point, dst: aux.Point, robots: list[rbt.Robot], margin: float = cfg.PASS_LINE_MARGIN) -> bool:
    """No robots near the segment src -> dst"""
    return all(aux.dist_to_line(src, dst, robot.get_pos(), "S") >= margin for robot in robots)


def goal_windows(src: aux.Point, goal: fld.Goal, obstacles: list[rbt.Robot]) -> list[tuple[float, float]]:
    """Free intervals (y_min, y_max) on the goal line, visible from src"""
    lx = abs(goal.center.x - src.x)  # distance to the goal line
    direction = aux.sign(goal.center.x - src.x)
    lo = min(goal.up.y, goal.down.y) + const.BALL_R
    hi = max(goal.up.y, goal.down.y) - const.BALL_R
    if lx < const.ROBOT_R:
        return []

    blocked: list[tuple[float, float]] = []
    rad = const.ROBOT_R + const.BALL_R
    for robot in obstacles:
        pos = robot.get_pos()
        # local frame: the goal line is x = lx
        rx = (pos.x - src.x) * direction
        ry = pos.y - src.y
        if rx <= -rad or rx >= lx + rad:
            continue
        d = math.hypot(rx, ry)
        if d <= rad:
            return []
        theta = math.atan2(ry, rx)
        alpha = math.asin(rad / d)
        border = math.pi / 2 - 1e-3
        y1 = src.y + lx * math.tan(aux.minmax(theta - alpha, -border, border))
        y2 = src.y + lx * math.tan(aux.minmax(theta + alpha, -border, border))
        if abs(theta) - alpha >= math.pi / 2:
            continue  # robot doesn't cover the direction to the goal
        blocked.append((min(y1, y2), max(y1, y2)))

    windows: list[tuple[float, float]] = []
    cur = lo
    for b_lo, b_hi in sorted(blocked):
        if b_lo > cur:
            windows.append((cur, min(b_lo, hi)))
        cur = max(cur, b_hi)
        if cur >= hi:
            break
    if cur < hi:
        windows.append((cur, hi))
    return [w for w in windows if w[1] > w[0]]


def widest_window(src: aux.Point, goal: fld.Goal, obstacles: list[rbt.Robot]) -> Optional[tuple[aux.Point, float]]:
    """Center and width of the widest free window in the goal, None if there is no window"""
    windows = goal_windows(src, goal, obstacles)
    if not windows:
        return None
    lo, hi = max(windows, key=lambda w: w[1] - w[0])
    if hi - lo < cfg.MIN_WINDOW:
        return None
    return aux.Point(goal.center.x, (lo + hi) / 2), hi - lo


def out_of_zone(point: aux.Point, goal: fld.Goal) -> aux.Point:
    """Move the point out of the goal zone (with a robot radius margin)"""
    if aux.is_point_inside_poly(point, goal.big_hull):
        return aux.nearest_point_on_poly(point, goal.big_hull)
    return point


def in_ally_half(field: fld.Field, point: aux.Point) -> bool:
    """The point is on our half of the field"""
    return point.x * field.polarity > 0


def to_own_half(field: fld.Field, point: aux.Point) -> aux.Point:
    """Move the point to our half of the field (for kickoff)"""
    min_x = const.ROBOT_R + 50
    if point.x * field.polarity < min_x:
        return aux.Point(min_x * field.polarity, point.y)
    return point


def clearance_point(field: fld.Field, src: aux.Point, away_from: Optional[aux.Point]) -> aux.Point:
    """Point on the flank of their half for a clearance: free line, far from away_from"""
    x = field.enemy_goal.center.x * cfg.CLEAR_X_K
    y = const.FIELD_DY - cfg.CLEAR_Y_OFFSET
    candidates = [aux.Point(x, y), aux.Point(x, -y)]
    enemies = field.active_enemies(True)

    def score(point: aux.Point) -> float:
        value = 0.0
        if away_from is not None:
            value += aux.dist(point, away_from)
        if not is_line_free(src, point, enemies):
            value -= 2000
        return value

    return max(candidates, key=score)


def free_flank_point(field: fld.Field, src: aux.Point) -> aux.Point:
    """Point in front of the corner of their zone on the freer flank (for dribbling)"""
    goal = field.enemy_goal
    candidates = [corner + goal.eye_forw * (2 * const.ROBOT_R) for corner in [goal.frw_up, goal.frw_down]]
    enemies = field.active_enemies(True)

    def freedom(point: aux.Point) -> float:
        return min((aux.dist_to_line(src, point, e.get_pos(), "S") for e in enemies), default=1e5)

    return max(candidates, key=freedom)


def free_corner(field: fld.Field, src: aux.Point) -> aux.Point:
    """The goal corner farther from their goalkeeper"""
    goal = field.enemy_goal
    corners = [
        goal.up - goal.eye_up * cfg.CORNER_OFFSET,
        goal.down + goal.eye_up * cfg.CORNER_OFFSET,
    ]
    enemy_gk = field.enemies[field.enemy_gk_id]
    if not enemy_gk.is_used():
        return corners[0]
    gk_pos = enemy_gk.get_pos()
    return max(corners, key=lambda c: aux.dist_to_line(src, c, gk_pos, "S"))
