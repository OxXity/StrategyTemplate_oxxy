import math

from bridge import const
from bridge.auxiliary import aux, fld


def get_pass_voltage(length: float, is_upper: bool) -> int:
    """Calc voltage for pass by length"""
    if is_upper:
        # print(int(aux.minmax(2.73 * length / 1000 + 2.09, 5, 15)))
        return int(aux.minmax(length / 735 + 3.8, 5, 15))
    else:
        # print(round(aux.minmax(1.13 + math.sqrt(7.87 * length / 1000 + 1.07), 2, 15)))
        return round(aux.minmax(-1.8 + math.sqrt(0.015 * length + 19.16) + 1, 2, 15))


def get_grab_speed(
    robot_pos: aux.Point,
    transl_vel: aux.Point,
    field: fld.Field,
    grab_point: aux.Point,
    grab_angle: float,
    target_speed: float = 0,
) -> aux.Point:
    """Calculate speed for carefully grabbing a ball"""
    GRAB_DIST, GRAB_MULT, GRAB_OFFSET_ANGLE = (
        const.GRAB_DIST,
        const.GRAB_MULT,
        const.GRAB_OFFSET_ANGLE,
    )

    ball = field.ball.get_pos()

    grab_delta = 100 if (field.robot_with_ball in field.enemies) else 200
    start_point = ball + (grab_point - ball).unity() * grab_delta

    point_on_center_line = aux.closest_point_on_line(ball, start_point, robot_pos, "S")
    dist_to_center_line = aux.dist(robot_pos, point_on_center_line)
    ball_dist_center_line = aux.dist(point_on_center_line, ball)

    if ball_dist_center_line == 0:
        offset_angle = GRAB_OFFSET_ANGLE
    else:
        offset_angle = math.atan(dist_to_center_line / ball_dist_center_line)

    dist_to_catch = (ball - aux.rotate(aux.RIGHT, grab_angle) * GRAB_DIST) - robot_pos

    vel_to_catch = dist_to_catch * GRAB_MULT

    vel_to_catch_r = (
        aux.scal_mult(
            vel_to_catch,
            aux.rotate(aux.RIGHT, grab_angle),
        )
        + target_speed
    )

    vel_to_align_r = aux.scal_mult(
        transl_vel,
        aux.rotate(aux.RIGHT, grab_angle),
    )

    vel_to_align = transl_vel - aux.rotate(aux.RIGHT, grab_angle) * vel_to_align_r

    board = min(offset_angle / const.GRAB_OFFSET_ANGLE, 1)  # 0 - go to ball; 1 - go to grab_point

    vel_r = vel_to_catch_r * (1 - board) + vel_to_align_r * board
    vel = vel_to_align + aux.rotate(aux.RIGHT, grab_angle) * vel_r

    return vel
