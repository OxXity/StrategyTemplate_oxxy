"""Reusable receiving action with legal, forward-looking interception."""

from bridge import const
from bridge.auxiliary import aux
from bridge.router.actions import Actions
from bridge.router.actions.action import Action, ActionDomain, ActionValues


class StrategyActions:
    """Class with strategy actions."""

    class CatchBall(Action):
        """Face the incoming ball and put the dribbler on its future trajectory."""

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            ball = domain.field.ball.get_pos()
            velocity = domain.field.ball.get_vel()
            if velocity.mag() < 100:
                return [Actions.BallGrab(aux.angle_to_point(domain.robot.get_pos(), ball))]
            direction = velocity.unity()
            target = aux.closest_point_on_line(ball, ball + velocity, domain.robot.get_pos(), "R")
            stopping_distance = velocity.mag() ** 2 / (2 * 500)
            target = ball + direction * min(aux.dist(ball, target), stopping_distance)
            # The ball touches the front of the robot, not its centre.
            target += direction * const.GRAB_DIST
            # Use GoToPoint so receiving cannot bypass penalty areas or other robots.
            current_action.dribbler_speed = 15
            domain.field.router_image.draw_line(target, domain.robot.get_pos(), (255, 127, 0), 2)
            return [Actions.GoToPoint(target, (-direction).arg(), ball_catch=True, ignore_ball=True)]
