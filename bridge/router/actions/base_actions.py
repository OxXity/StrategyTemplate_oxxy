"""
Class with robot actions
"""

from time import time
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, tau
from bridge.const import State as GameStates
from bridge.router.actions.action import Action, ActionDomain, ActionValues
from bridge.router.path_generation.path_generation import (
    avoid_goal_zone,
    calc_passthrough_point,
    correct_target_pos,
    get_ball_radius,
)

from .dumb_actions import DumbActions
from .extra_functions import get_grab_speed


class Actions:
    """Class with all user-available actions (except kicks)"""

    class Unused(Action):
        is_used = False

    class Stop(Action):
        """Stop the robot"""

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            """Behavior"""
            current_action.vel = aux.Point(0, 0)
            current_action.angle = 0.0
            current_action.beep = 1

    class GoToPointIgnore(Action):
        """Go to point ignore obstacles"""

        def __init__(
            self,
            target_pos: aux.Point,
            target_angle: float,
            ball_catching: bool = False,
            target_vel: aux.Point = aux.Point(0, 0),
        ) -> None:
            self.target_pos = target_pos
            self.target_angle = target_angle
            self.ball_catching = ball_catching
            self.target_vel = target_vel

            self.use_dribbler = False

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            cur_robot = domain.robot
            vec_err = self.target_pos - cur_robot.get_pos()
            cur_vel = cur_robot.get_vel()
            now = time()

            if self.ball_catching:
                cur_robot.pos_reg_x.select_mode(tau.Mode.CATCH)
                cur_robot.pos_reg_y.select_mode(tau.Mode.CATCH)
            else:
                cur_robot.pos_reg_x.select_mode(tau.Mode.NORMAL)
                cur_robot.pos_reg_y.select_mode(tau.Mode.NORMAL)

            dist_err = vec_err.mag()
            u_x = cur_robot.pos_reg_x.process(vec_err.x, -cur_vel.x, total_dist=dist_err)
            u_y = cur_robot.pos_reg_y.process(vec_err.y, -cur_vel.y, total_dist=dist_err)
            current_action.vel = aux.Point(u_x, u_y)

            if not (self.ball_catching and domain.robot.r_id == domain.field.gk_id):
                cur_vel_abs = current_action.vel  # aux.rotate(current_action.vel, cur_robot.get_angle())
                prev_vel_abs = (
                    cur_robot.prev_sended_vel
                )  # aux.rotate(cur_robot.prev_sended_vel, -cur_robot.prev_sended_angle)
                dt = max(now - cur_robot.prev_sended_time, 1e-3)  # два вызова в один момент времени - без деления на 0
                if (
                    cur_vel_abs - prev_vel_abs
                ).mag() / dt > const.MAX_ACCELERATION and cur_vel_abs.mag() > prev_vel_abs.mag():
                    # domain.field.router_image.draw_circle(aux.Point(0, 1000), size_in_mms=200)
                    current_action.vel = prev_vel_abs + (cur_vel_abs - prev_vel_abs).unity() * const.MAX_ACCELERATION * dt

            # current_action.vel = aux.Point(0,500)
            cur_robot.prev_sended_vel = current_action.vel
            cur_robot.prev_sended_angle = cur_robot.get_angle()
            cur_robot.prev_sended_time = now
            current_action.angle = self.target_angle

            if self.use_dribbler:
                current_action.dribbler_speed = 15

            DumbActions.AddFinalVelocityAction(self.target_pos, self.target_vel).process(domain, current_action)

    class GoToPoint(Action):
        """Go to point and avoid obstacles"""

        def __init__(
            self,
            target_pos: aux.Point,
            target_angle: float,
            *,
            ball_catch: bool = False,
            ignore_ball: bool = False,
            perform_ball_placement: bool = False,
            target_vel: aux.Point = aux.Point(0, 0),
            ignore_robots: dict[const.Color, list[int]] = {},
            avoid_ball: Optional[bool] = None,
        ) -> None:
            self.target_pos = target_pos
            self.target_angle = target_angle
            self.ball_catch = ball_catch
            self.ignore_ball = ignore_ball
            self.target_vel = target_vel
            self.perform_ball_placement = perform_ball_placement
            self.ignore_robots = ignore_robots
            self.avoid_ball = avoid_ball  # None - автоматически по состоянию игры, True/False - явно

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:
            avoid_ball = (
                domain.game_state in [GameStates.STOP, GameStates.PREPARE_KICKOFF]
                or (domain.game_state in [GameStates.FREE_KICK, GameStates.KICKOFF] and not domain.we_active)
                or (domain.game_state == GameStates.BALL_PLACEMENT and not self.perform_ball_placement)
            )
            if self.avoid_ball is not None:
                avoid_ball = self.avoid_ball
            ball_placement_target = domain.field.ball_placement_pos if (not self.perform_ball_placement) else None
            if not self.perform_ball_placement:
                self.target_pos, self.target_vel = correct_target_pos(
                    domain.field, domain.robot, self.target_pos, self.target_vel, avoid_ball, ball_placement_target
                )

            angle0 = self.target_angle
            next_point = self.target_pos

            if domain.robot.r_id != domain.field.gk_id and not self.perform_ball_placement:
                avoid_point = avoid_goal_zone(domain.field, domain.robot, next_point)
                if avoid_point is not None:
                    vel = (avoid_point - domain.robot.get_pos()).unity() * 250
                    domain.field.router_image.draw_circle(avoid_point, size_in_mms=30)

                    return [Actions.GoToPointIgnore(avoid_point, angle0, target_vel=vel)]

            pth_wp = calc_passthrough_point(
                domain,
                next_point,
                avoid_ball=avoid_ball,
                ignore_ball=self.ignore_ball,
                ball_placement_target=ball_placement_target,
                perform_ball_placement=self.perform_ball_placement,
                ignore_robots=self.ignore_robots,
            )
            if pth_wp is not None:
                target_speed = min(const.MAX_SPEED, aux.dist(pth_wp, next_point))
                target_vel = (pth_wp - domain.robot.get_pos()).unity() * target_speed
                return [Actions.GoToPointIgnore(pth_wp, angle0, target_vel=target_vel)]

            if next_point != self.target_pos:
                target_speed = min(const.MAX_SPEED * 0.7, aux.dist(self.target_pos, next_point))
                target_vel = (next_point - domain.robot.get_pos()).unity() * target_speed
                return [Actions.GoToPointIgnore(next_point, angle0, target_vel=target_vel)]

            return [Actions.GoToPointIgnore(self.target_pos, angle0, self.ball_catch, self.target_vel)]

    class BallGrab(Action):
        """Grab ball in a given direction"""

        def __init__(self, target_angle: float, *, perform_ball_placement: bool = False, dribbler_speed: int = 15) -> None:
            self.target_angle = target_angle
            self.perform_ball_placement = perform_ball_placement
            self.dribbler_speed = dribbler_speed

        def is_defined(self, domain: ActionDomain) -> bool:
            return (
                aux.dist(domain.robot.get_pos(), domain.field.ball.get_pos()) < 3000
                and (
                    domain.robot.r_id == domain.field.gk_id
                    or self.perform_ball_placement
                    or (
                        not aux.is_point_inside_poly(domain.field.ball.get_pos(), domain.field.enemy_goal.hull)
                        and not aux.is_point_inside_poly(domain.field.ball.get_pos(), domain.field.ally_goal.hull)
                    )
                )
                and domain.game_state not in [GameStates.STOP, GameStates.PREPARE_KICKOFF]
                and (domain.game_state not in [GameStates.FREE_KICK, GameStates.KICKOFF] or domain.we_active)
            )

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            ball_pos = domain.field.ball.get_pos()
            robot_pos = domain.robot.get_pos()
            align_pos = ball_pos - aux.rotate(aux.RIGHT, self.target_angle) * const.GRAB_ALIGN_DIST
            transl_vel = get_grab_speed(
                robot_pos,
                current_action.vel,
                domain.field,
                align_pos,
                self.target_angle,
            )
            current_action.vel = transl_vel
            current_action.angle = self.target_angle

            current_action.dribbler_speed = self.dribbler_speed

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:
            ball_pos = domain.field.ball.get_pos()
            align_pos = ball_pos - aux.rotate(aux.RIGHT, self.target_angle) * (
                get_ball_radius(domain.field, domain.robot) + 10
            )
            ignore_ball = len(aux.line_circle_intersect(domain.robot.get_pos(), align_pos, ball_pos, const.ROBOT_R, "S")) < 2
            actions: list[Action] = [
                Actions.GoToPoint(
                    align_pos,
                    self.target_angle,
                    ignore_ball=ignore_ball,
                    perform_ball_placement=self.perform_ball_placement,
                ),
            ]
            return actions

    class Velocity(Action):
        """Move robot with velocity and angle_speed"""

        def __init__(self, velocity: aux.Point, angle: float, control_angle_by_speed: bool = False) -> None:
            self.velocity = velocity
            self.angle = angle  # angle to turn / angle speed

            self.control_angle_by_speed = control_angle_by_speed

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:

            current_action.vel = self.velocity
            current_action.angle = self.angle

            if self.control_angle_by_speed:
                current_action.beep = 1

    class SetDribblerSpeed(Action):
        def __init__(self, speed: int = 15):
            self.speed = speed

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            current_action.dribbler_speed = self.speed
