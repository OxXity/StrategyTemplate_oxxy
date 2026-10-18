"""
Class with strategy actions
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux
from bridge.router import legality
from bridge.router.actions import Actions
from bridge.router.actions.action import Action, ActionDomain, ActionValues

# Мяч считается катящимся при скорости выше, мм/с
RECEIVE_MIN_BALL_SPEED = 100


class StrategyActions:
    """Class with strategy actions"""

    class CatchBall(Action):
        """Catch ball or avoid it, if it is moving to goal"""

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            eye_angle = aux.angle_to_point(domain.field.ball.get_pos(), domain.field.ball_start_point)

            # ball_vec = domain.field.ball.get_pos() - domain.field.ball_start_point
            # for enemy in domain.field.active_enemies(False):
            #     pnt = aux.closest_point_on_line(
            #         domain.field.ball.get_pos(),
            #         domain.field.ball.get_pos() + ball_vec,
            #         enemy.get_pos(),
            #         "R",
            #     )
            #     if aux.dist(pnt, enemy.get_pos()) < 2 * const.ROBOT_R:
            #         break

            # if ( NOTE
            #     domain.field.is_ball_moves_to_enemy_goal()
            #     and domain.field.ball.get_vel().mag() > 500
            #     and not can_enemy_catch_ball
            # ):
            #     # avoid the ball, because it may be a goal
            #     vec = domain.robot.get_pos() - aux.closest_point_on_line(
            #         domain.field.ball_start_point, domain.field.ball.get_pos(), domain.robot.get_pos()
            #     )
            #     return [Actions.GoToPointIgnore(domain.robot.get_pos() + vec.unity() * 500, eye_angle)]

            target = aux.closest_point_on_line(
                domain.field.ball_start_point,
                domain.field.ball.get_pos(),
                domain.robot.get_pos(),
                "R",
            )
            if aux.is_point_inside_poly(target, domain.field.enemy_goal.stop_hull):
                return [Actions.GoToPoint(target, eye_angle)]

            current_action.dribbler_speed = 15
            domain.field.router_image.draw_line(target, domain.robot.get_pos(), (255, 127, 0), 2)
            domain.field.router_image.draw_circle(target, (128, 128, 255), const.ROBOT_R)

            return [Actions.GoToPointIgnore(target, eye_angle, ball_catching=True)]

    class Receive(Action):
        """
        Приём паса (R-4.7): встать на линию мяча лицом к мячу с доворотом к face_target
        не больше max_bias; дриблер, мягкие коэффициенты, «гашение» мяча у робота.
        При one_touch_voltage > 0 - удар в одно касание на face_target (автоудар).
        """

        def __init__(
            self,
            face_target: Optional[aux.Point],
            *,
            max_bias: float,
            dribbler_speed: int,
            cushion_speed: float,
            cushion_dist: float = 300,
            cushion_shift: float = 40,
            hold_dist: float = 90,
            one_touch_voltage: int = 0,
        ) -> None:
            self.face_target = face_target
            self.max_bias = max_bias
            self.dribbler_speed = dribbler_speed
            self.cushion_speed = cushion_speed
            self.cushion_dist = cushion_dist
            self.cushion_shift = cushion_shift
            self.hold_dist = hold_dist
            self.one_touch_voltage = one_touch_voltage

        def face_angle(self, domain: ActionDomain) -> float:
            """Угол робота: на мяч с доворотом к face_target"""
            robot_pos = domain.robot.get_pos()
            ang_ball = aux.angle_to_point(robot_pos, domain.field.ball.get_pos())
            if self.face_target is None:
                return ang_ball
            ang_target = aux.angle_to_point(robot_pos, self.face_target)
            if self.one_touch_voltage > 0:
                return ang_target
            bias = aux.minmax(aux.wind_down_angle(ang_target - ang_ball), -self.max_bias, self.max_bias)
            return ang_ball + bias

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            field = domain.field
            robot = domain.robot
            ball = field.ball.get_pos()
            ball_vel = field.ball.get_vel()
            face = self.face_angle(domain)
            face_dir = aux.rotate(aux.RIGHT, face)

            target = robot.get_pos()
            target_vel = aux.Point(0, 0)
            if ball_vel.mag() > RECEIVE_MIN_BALL_SPEED:
                ball_dir = ball_vel.unity()
                front = robot.get_pos() + face_dir * self.hold_dist
                meet = aux.closest_point_on_line(ball, ball + ball_dir, front, "R")
                target = meet - face_dir * self.hold_dist
                approaching = aux.scal_mult(robot.get_pos() - ball, ball_dir) > 0
                if approaching and aux.dist(ball, robot.get_pos()) < self.cushion_dist:
                    target = target + ball_dir * self.cushion_shift
                    target_vel = ball_dir * self.cushion_speed

            target = legality.legal_target(field, robot.r_id, target)
            field.router_image.draw_line(ball, target, (255, 127, 0), 2)

            actions: list[Action] = [
                Actions.GoToPointIgnore(target, face, ball_catching=True, target_vel=target_vel),
                Actions.SetDribblerSpeed(self.dribbler_speed),
            ]
            return actions

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            if self.one_touch_voltage > 0:
                current_action.kicker_voltage = self.one_touch_voltage
                current_action.auto_kick = 1
