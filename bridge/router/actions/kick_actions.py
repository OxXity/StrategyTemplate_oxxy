import math
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
            dribbler_speed: int = 15,
        ):
            super().__init__(target_pos, voltage, is_pass, is_upper)

            self.perform_ball_placement = perform_ball_placement
            self.dribbler_speed = dribbler_speed

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:

            kick_angle = aux.angle_to_point(domain.field.ball.get_pos(), self.target_pos)

            actions = [
                Actions.BallGrab(
                    kick_angle, perform_ball_placement=self.perform_ball_placement, dribbler_speed=self.dribbler_speed
                ),
                DumbActions.ShootAction(self.target_pos, self.is_upper),
                DumbActions.ControlVoltageAction(self.voltage, self.pass_pos, self.is_upper),
            ]

            return actions

    class SetPieceKick(Action):
        """
        Удар со стандарта без двойного касания (Регламент 8.2):
        подъезд к точке за мячом, медленный финальный подъезд с дриблером 0,
        автоудар только при довороте не хуже align_tol.
        """

        def __init__(
            self,
            target_pos: aux.Point,
            voltage: int,
            approach_speed: float,
            align_tol: float,
            *,
            prep_gap: float = 100,
            lateral_tol: float = 30,
        ) -> None:
            self.target_pos = target_pos
            self.voltage = min(voltage, const.VOLTAGE_SHOOT)
            self.approach_speed = approach_speed
            self.align_tol = align_tol
            self.prep_gap = prep_gap
            self.lateral_tol = lateral_tol

        def kick_angle(self, domain: ActionDomain) -> float:
            """Направление удара от мяча на цель"""
            return aux.angle_to_point(domain.field.ball.get_pos(), self.target_pos)

        def is_lined_up(self, domain: ActionDomain) -> bool:
            """Робот за мячом на линии удара и довёрнут"""
            ball = domain.field.ball.get_pos()
            robot = domain.robot
            angle = self.kick_angle(domain)
            direction = aux.rotate(aux.RIGHT, angle)
            rel = robot.get_pos() - ball
            along = aux.scal_mult(rel, direction)
            lateral = abs(aux.vec_mult(direction, rel))
            angle_err = abs(aux.wind_down_angle(robot.get_angle() - angle))
            return along < -const.ROBOT_R * 0.5 and lateral < self.lateral_tol and angle_err < self.align_tol

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:
            ball = domain.field.ball.get_pos()
            angle = self.kick_angle(domain)
            direction = aux.rotate(aux.RIGHT, angle)
            if not self.is_lined_up(domain):
                radius = const.GRAB_ALIGN_DIST + self.prep_gap
                prep = ball - direction * radius
                rel = domain.robot.get_pos() - ball
                if aux.scal_mult(rel, direction) > const.ROBOT_R / 2:
                    # Робот не позади мяча: обойти мяч по дуге через боковую точку, не задевая его
                    side = 1.0 if aux.vec_mult(direction, rel) >= 0 else -1.0
                    prep = ball + aux.rotate(direction, side * math.pi / 2) * (radius + const.ROBOT_R)
                return [
                    Actions.GoToPoint(prep, angle, ignore_ball=False, ignore_robots={}, avoid_ball=False),
                    Actions.SetDribblerSpeed(0),
                ]

            contact = ball - direction * (const.ROBOT_R * 0.5)
            return [
                Actions.GoToPointIgnore(contact, angle),
                DumbActions.LimitSpeed(self.approach_speed),
                Actions.SetDribblerSpeed(0),
            ]

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            current_action.kicker_voltage = self.voltage
            current_action.dribbler_speed = 0
            if self.is_lined_up(domain):
                current_action.auto_kick = 1
