from typing import Optional

from bridge import const
from bridge.auxiliary import aux

from . import Actions
from .action import Action, ActionDomain, ActionValues
from .dumb_actions import DumbActions


class KickActions:
    """Class with available types of ball kicks"""

    class Kick(Action):
        """Base class"""

        def __init__(
            self,
            target_pos: aux.Point,
            voltage: int = 15,
            is_pass: bool = False,
            is_upper: bool = False,
        ) -> None:
            self.target_pos = target_pos
            self.voltage = voltage  # ignore if is_pass
            self.is_upper = is_upper

            if self.voltage > const.VOLTAGE_SHOOT:
                self.voltage = const.VOLTAGE_SHOOT

            if self.is_upper:
                self.voltage = const.VOLTAGE_UP

            self.pass_pos: Optional[aux.Point] = None
            if is_pass:
                self.pass_pos = self.target_pos

    class Straight(Kick):
        """Grab the ball and kick it straight"""

        def __init__(
            self,
            target_pos: aux.Point,
            voltage: int = 15,
            is_pass: bool = False,
            is_upper: bool = False,
            *,
            perform_ball_placement: bool = False,
            max_angular_speed: Optional[float] = None,
        ):
            super().__init__(target_pos, voltage, is_pass, is_upper)

            self.perform_ball_placement = perform_ball_placement
            self.max_angular_speed = max_angular_speed

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:

            kick_angle = aux.angle_to_point(domain.field.ball.get_pos(), self.target_pos)

            actions = [
                Actions.BallGrab(
                    kick_angle,
                    perform_ball_placement=self.perform_ball_placement,
                    max_angular_speed=self.max_angular_speed,
                ),
                DumbActions.ShootAction(self.target_pos, self.is_upper),
                DumbActions.ControlVoltageAction(self.voltage, self.pass_pos, self.is_upper),
            ]

            return actions
