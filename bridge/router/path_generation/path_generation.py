import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.auxiliary import quickhull as qh
from bridge.auxiliary import rbt
from bridge.auxiliary.entity import Entity
from bridge.router.actions.action import ActionDomain


def calc_passthrough_point(
    domain: ActionDomain,
    target: aux.Point,
    *,
    avoid_ball: bool = False,
    ignore_ball: bool = False,
    ball_placement_target: Optional[aux.Point] = None,
    perform_ball_placement: bool = False,
    ignore_robots: dict[const.Color, list[int]] = {}
) -> Optional[aux.Point]:
    """
    Рассчитать ближайшую промежуточную путевую точку
    """
    robot = domain.robot
    field = domain.field

    obstacles_dist: list[tuple[Entity, float]] = []

    if avoid_ball:
        if aux.is_point_inside_circle(robot.get_pos(), field.ball.get_pos(), const.KEEP_BALL_DIST):
            mb_target = _is_we_stuck_in_ballplacement(field, robot.get_pos())
            if mb_target is not None:
                return mb_target
            else:
                return aux.nearest_point_on_circle(robot.get_pos(), field.ball.get_pos(), const.KEEP_BALL_DIST + 50)
        ball = Entity(
            field.ball.get_pos(),
            field.ball.get_angle(),
            const.KEEP_BALL_DIST - const.ROBOT_R - 50,
        )
        obstacles_dist.append((ball, aux.dist(ball.get_pos(), robot.get_pos())))
    elif not ignore_ball:
        ball = field.ball

        if (
            len(aux.line_circle_intersect(robot.get_pos(), target, ball.get_pos(), const.ROBOT_R + ball.get_radius(), "S"))
            > 0
        ):
            obstacles_dist.append((ball, aux.dist(ball.get_pos(), robot.get_pos())))

    if ball_placement_target is not None:
        point_on_ball_line = aux.closest_point_on_line(field.ball.get_pos(), ball_placement_target, robot.get_pos(), "S")
        if aux.dist(robot.get_pos(), point_on_ball_line) < const.KEEP_BALL_DIST:
            return (robot.get_pos() - point_on_ball_line).unity() * (const.KEEP_BALL_DIST + 50) + point_on_ball_line
        # if aux.is_point_inside_circle(robot.get_pos(), ball_placement_target, const.KEEP_BALL_DIST):
        #     return aux.nearest_point_on_circle(robot.get_pos(), ball_placement_target, const.KEEP_BALL_DIST + 50)

        ball_target = field.ball
        obstacles_dist.append((ball_target, aux.dist(ball_target.get_pos(), robot.get_pos())))

    for obstacle in field.allies:
        if field.ally_color in ignore_robots and obstacle.r_id in ignore_robots[field.ally_color]:
            continue
        dist = (obstacle.get_pos() - robot.get_pos()).mag()
        if obstacle.is_used() and obstacle.get_radius() + robot.get_radius() < dist < const.VIEW_DIST:
            obstacles_dist.append((obstacle.to_entity(), dist))

    for obstacle in field.enemies:
        if field.ally_color.reverse() in ignore_robots and obstacle.r_id in ignore_robots[field.ally_color.reverse()]:
            continue
        dist = (obstacle.get_pos() - robot.get_pos()).mag()
        if obstacle.is_used() and obstacle.get_radius() + robot.get_radius() < dist < const.VIEW_DIST:
            obstacles_dist.append((obstacle.to_entity(), dist))

    sorted_obstacles = sorted(obstacles_dist, key=lambda x: x[1])

    obstacles = []
    for obst in sorted_obstacles:
        obstacles.append(obst[0])

    pth_point = calc_next_point(
        field,
        robot.get_pos(),
        target,
        domain.robot,
        obstacles,
        ball_placement_target,
        perform_ball_placement,
    )

    if pth_point is None:
        return None
    field.path_image.draw_line(robot.get_pos(), pth_point[0], color=(0, 0, 0))
    if pth_point[0] == target:
        return None

    return pth_point[0]


def calc_next_point(
    field: fld.Field,
    position: aux.Point,
    target: aux.Point,
    robot: rbt.Robot,
    obstacles: list[Entity],
    ball_placement_target: Optional[aux.Point],
    perform_ball_placement: bool,
) -> Optional[tuple[aux.Point, float]]:
    """Calculate next point for robot"""
    remaining_obstacles: list[Entity] = obstacles.copy()
    skipped_obstacles: list[Entity] = []
    while len(remaining_obstacles) > 0:
        obstacle = remaining_obstacles.pop(0)
        skipped_obstacles.append(obstacle)

        t_horizon = 0.4
        pred_pos = obstacle.get_pos() + obstacle.get_vel() * t_horizon

        # Effective radius calculation (mostly original logic)
        margin_max = 250.0
        v_max = const.MAX_SPEED
        v_curr = robot.get_vel().mag()
        r_base = const.ROBOT_R + obstacle.get_radius()

        # Use distance to obstacle to estimate dynamic radius if needed,
        # but the capsule already handles the "moving" part.
        # r_obst_dyn = time_to_reach * obstacle.get_vel().mag() * 0.2

        radius = r_base + ((min(v_curr, v_max) / v_max) ** 2) * margin_max

        # Collision check using Time Tube (shortest distance between robot path and obstacle path)
        if aux.dist_between_segments(position, target, obstacle.get_pos(), pred_pos) < radius:
            # For tangent points, we need a stable "center".
            # We use the point on the obstacle's path that is closest to our CURRENT position.
            effective_center = aux.closest_point_on_line(obstacle.get_pos(), pred_pos, position, "S")

            # Draw the Time Tube (capsule)
            tube_color = (200, 200, 200)  # Light gray
            field.path_image.draw_circle(obstacle.get_pos(), tube_color, radius)
            field.path_image.draw_circle(pred_pos, tube_color, radius)
            field.path_image.draw_line(obstacle.get_pos(), pred_pos, color=tube_color, size_in_pixels=int(radius * 2))

            field.path_image.draw_circle(effective_center, (127, 127, 127), radius)

            tangents = aux.get_tangent_points(effective_center, position, radius)
            if tangents is None or len(tangents) < 2:
                # If we are already inside the obstacle, just return None or try to get out
                return None

            tangents[0] = aux.point_on_line(
                effective_center,
                tangents[0],
                radius + const.ROBOT_R * 0.5,
            )
            tangents[1] = aux.point_on_line(
                effective_center,
                tangents[1],
                radius + const.ROBOT_R * 0.5,
            )

            path_before0 = calc_next_point(
                field, position, tangents[0], robot, skipped_obstacles[:-1], ball_placement_target, perform_ball_placement
            )
            path_before1 = calc_next_point(
                field, position, tangents[1], robot, skipped_obstacles[:-1], ball_placement_target, perform_ball_placement
            )
            path_after0 = calc_next_point(
                field, tangents[0], target, robot, remaining_obstacles, ball_placement_target, perform_ball_placement
            )
            path_after1 = calc_next_point(
                field, tangents[1], target, robot, remaining_obstacles, ball_placement_target, perform_ball_placement
            )

            if (path_before0 is None or path_after0 is None) and (path_before1 is not None and path_after1 is not None):
                pth_point = path_before1[0]
                length = path_before1[1] + path_after1[1]
                return pth_point, length
            if (path_before0 is not None and path_after0 is not None) and (path_before1 is None or path_after1 is None):
                if path_after0 is None:
                    return None
                pth_point = path_before0[0]
                length = path_before0[1] + path_after0[1]
                return pth_point, length
            if (path_before0 is not None and path_after0 is not None) and (
                path_before1 is not None and path_after1 is not None
            ):

                length0 = path_before0[1] + path_after0[1]
                length1 = path_before1[1] + path_after1[1]

                def is_path_blocked(
                    robot_pos: aux.Point, point_before: aux.Point, tangent: aux.Point, point_after: aux.Point
                ) -> bool:
                    if perform_ball_placement:
                        return False
                    if robot.r_id != field.gk_id:
                        # goal zone interference
                        if (
                            aux.segment_poly_intersect(robot_pos, point_before, field.ally_goal.hull) is not None
                            or aux.segment_poly_intersect(robot_pos, point_before, field.enemy_goal.hull) is not None
                        ):
                            return True
                        if (
                            aux.segment_poly_intersect(tangent, point_after, field.ally_goal.hull) is not None
                            or aux.segment_poly_intersect(tangent, point_after, field.enemy_goal.hull) is not None
                        ):
                            return True

                    # ball placement zone interference
                    if ball_placement_target is not None:
                        ball_placement_hull_delta = (
                            aux.rotate(field.ball.get_pos() - ball_placement_target, math.pi / 2).unity()
                            * const.KEEP_BALL_DIST
                        )
                        ball_placement_hull = [
                            field.ball.get_pos() + ball_placement_hull_delta,
                            field.ball.get_pos() - ball_placement_hull_delta,
                            ball_placement_target - ball_placement_hull_delta,
                            ball_placement_target + ball_placement_hull_delta,
                        ]
                        if (
                            aux.segment_poly_intersect(robot_pos, point_before, ball_placement_hull) is not None
                            or aux.segment_poly_intersect(robot_pos, point_before, ball_placement_hull) is not None
                        ):
                            return True

                    return False

                blocked_path0 = is_path_blocked(robot.get_pos(), path_before0[0], tangents[0], path_after0[0])
                blocked_path1 = is_path_blocked(robot.get_pos(), path_before1[0], tangents[1], path_after1[0])

                if (length0 < length1 or blocked_path1) and not blocked_path0:
                    pth_point = path_before0[0]
                    length = path_before0[1] + path_after0[1]
                    field.path_image.draw_line(position, pth_point, color=(255, 0, 255))
                    return pth_point, length
                if (length1 < length0 or blocked_path0) and not blocked_path1:
                    pth_point = path_before1[0]
                    length = path_before1[1] + path_after1[1]
                    field.path_image.draw_line(position, pth_point, color=(255, 0, 255))
                    return pth_point, length

            return None

    field.path_image.draw_line(position, target, color=(255, 0, 127))
    return target, aux.dist(position, target)


def correct_target_pos(
    field: fld.Field,
    robot: rbt.Robot,
    target: aux.Point,
    target_vel: aux.Point,
    avoid_ball: bool,
    ball_placement_target: Optional[aux.Point],
) -> tuple[aux.Point, aux.Point]:
    """Correct target position"""
    ball_pos = field.ball.get_pos()
    field_hull = field.target_hull
    if not aux.is_point_inside_poly(target, field_hull):
        target = aux.nearest_point_on_poly(target, field_hull)
        target_vel = aux.Point(0, 0)

    if ball_placement_target is not None:
        delta_vec = aux.rotate((ball_pos - ball_placement_target).unity(), math.pi / 2) * const.KEEP_BALL_DIST
        line1: tuple[aux.Point, aux.Point] = (ball_pos + delta_vec, ball_pos - delta_vec)
        line2: tuple[aux.Point, aux.Point] = (ball_placement_target - delta_vec, ball_placement_target + delta_vec)

        poly = list(line1) + list(line2)
        field.router_image.draw_poly(poly, (255, 255, 255))
        pnt = aux.closest_point_on_line(ball_pos, ball_placement_target, target)
        if aux.dist(pnt, target) < const.KEEP_BALL_DIST:
            mb_target = _is_we_stuck_in_ballplacement(field, robot.get_pos())
            if mb_target is None:
                target = pnt + (ball_pos - pnt).unity() * (const.KEEP_BALL_DIST)
            else:
                target = mb_target

        field.router_image.draw_circle(ball_placement_target, (255, 255, 255), const.KEEP_BALL_DIST)
        # if aux.is_point_inside_circle(target, ball_placement_target, const.KEEP_BALL_DIST):
        #     target = aux.nearest_point_on_circle(target, ball_placement_target, const.KEEP_BALL_DIST)
        #     target_vel = aux.Point(0, 0)

        if aux.segment_poly_intersect(robot.get_pos(), target, poly):
            if aux.dist(target, ball_placement_target) > aux.dist(target, ball_pos):
                tangents_ball = aux.get_tangent_points(ball_pos, robot.get_pos(), const.KEEP_BALL_DIST)
                if len(tangents_ball) > 0:
                    if aux.dist(tangents_ball[0], ball_placement_target) > aux.dist(tangents_ball[1], ball_placement_target):
                        target = tangents_ball[0]
                    else:
                        target = tangents_ball[1]
                target += (target - ball_pos).unity() * 150
            else:
                tangents_ball_targ = aux.get_tangent_points(ball_placement_target, robot.get_pos(), const.KEEP_BALL_DIST)
                if len(tangents_ball_targ) > 0:
                    if aux.dist(tangents_ball_targ[0], ball_pos) > aux.dist(tangents_ball_targ[1], ball_pos):
                        target = tangents_ball_targ[0]
                    else:
                        target = tangents_ball_targ[1]
                target += (target - ball_placement_target).unity() * 150

            target_vel = aux.Point(0, 0)

    if avoid_ball:
        field.router_image.draw_circle(ball_pos, (255, 255, 255), const.KEEP_BALL_DIST)
        if aux.is_point_inside_circle(target, ball_pos, const.KEEP_BALL_DIST):
            mb_target = _is_we_stuck_in_ballplacement(field, robot.get_pos())
            if mb_target is not None:
                target = mb_target
            else:
                target = aux.nearest_point_on_circle(target, ball_pos, const.KEEP_BALL_DIST)
            target_vel = aux.Point(0, 0)

    if robot.r_id != field.gk_id:
        if aux.is_point_inside_poly(target, field.ally_goal.big_hull):
            target = aux.nearest_point_on_poly(target, field.ally_goal.big_hull)
            target_vel = aux.Point(0, 0)
        elif aux.is_point_inside_poly(target, field.enemy_goal.big_hull):
            target = aux.nearest_point_on_poly(target, field.enemy_goal.big_hull)
            target_vel = aux.Point(0, 0)

        # R-0.4: пока мяч не в игре, держим цели дальше от чужой штрафной
        _, enemy_big_hull_avoid = fld.enemy_area_hulls(field.enemy_goal, field.game_state)
        if aux.is_point_inside_poly(target, enemy_big_hull_avoid):
            target = aux.nearest_point_on_poly(target, enemy_big_hull_avoid)
            target_vel = aux.Point(0, 0)

    if not aux.is_point_inside_poly(target, field.target_hull):
        target = aux.nearest_point_on_poly(target, field.target_hull)
        target_vel = aux.Point(0, 0)

    return target, target_vel


def avoid_goal_zone(field: fld.Field, robot: rbt.Robot, next_point: aux.Point) -> Optional[aux.Point]:
    enemy_hull_avoid, enemy_big_hull_avoid = fld.enemy_area_hulls(field.enemy_goal, field.game_state)

    if aux.is_point_inside_poly(robot.get_pos(), field.ally_goal.hull):
        next_point = aux.nearest_point_on_poly(robot.get_pos(), field.ally_goal.big_hull)
        return next_point
    elif aux.is_point_inside_poly(robot.get_pos(), enemy_hull_avoid):
        next_point = aux.nearest_point_on_poly(robot.get_pos(), enemy_big_hull_avoid)
        return next_point

    new_next_point: Optional[aux.Point] = None

    def move_zone_angle(point: aux.Point) -> aux.Point:
        new_x = point.x - math.copysign(const.ROBOT_R * 1.5, point.x)
        new_y = point.y + math.copysign(const.ROBOT_R * 1.5, point.y)
        return aux.Point(new_x, new_y)

    pint = aux.segment_poly_intersect(robot.get_pos(), next_point, field.ally_goal.hull)
    if pint is not None:
        convex_hull = qh.shortesthull(robot.get_pos(), next_point, field.ally_goal.hull)
        if len(convex_hull) > 1:
            new_next_point = move_zone_angle(convex_hull[1])

    pint = aux.segment_poly_intersect(robot.get_pos(), next_point, enemy_hull_avoid)
    if pint is not None:
        convex_hull = qh.shortesthull(robot.get_pos(), next_point, enemy_hull_avoid)
        if len(convex_hull) > 1:
            new_next_point = move_zone_angle(convex_hull[1])

    if new_next_point is not None:
        next_point.x = new_next_point.x
        next_point.y = new_next_point.y
    return None


def get_ball_radius(field: fld.Field, robot: rbt.Robot) -> float:
    if field.game_state == const.State.FREE_KICK:
        return const.GRAB_ALIGN_DIST + const.ROBOT_R
    return (
        const.GRAB_ALIGN_DIST
        + const.ROBOT_R * (robot.get_vel().mag() / const.MAX_SPEED) * 2  # <-- coefficient of fear [0; 1] for fast speed
    )


def _is_we_stuck_in_ballplacement(field: fld.Field, robot_pos: aux.Point) -> Optional[aux.Point]:
    ball_pos = field.ball.get_pos()
    closest_corner = aux.find_nearest_point(ball_pos, field.hull + field.enemy_goal.stop_hull)

    field.router_image.draw_circle(ball_pos, (255, 0, 0), 700)

    ball_to_corner = closest_corner - ball_pos
    robot_to_corner = closest_corner - robot_pos
    dist_ball_to_corner = ball_to_corner.mag()
    dist_robot_to_corner = robot_to_corner.mag()

    if dist_robot_to_corner < dist_ball_to_corner < 700:
        for point in field.hull + field.enemy_goal.hull:
            field.router_image.draw_circle(point, (255, 0, 0), 700)
        if abs(ball_to_corner.x) > abs(ball_to_corner.y):
            return robot_pos - aux.Point(ball_to_corner.x, 0).unity() * 150
        return robot_pos - aux.Point(0, ball_to_corner.y).unity() * 150
    return None
