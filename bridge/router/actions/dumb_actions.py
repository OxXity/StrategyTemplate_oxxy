from typing import Optional

from bridge import const
from bridge.auxiliary import aux

from .action import Action, ActionDomain, ActionValues, limit_action
from .extra_functions import get_pass_voltage


class DumbActions:
    """User-unavailable actions, are used in Actions"""

    class ShootAction(Action):
        """Shoot the target when kick is aligned"""

        def __init__(
            self,
            target_pos: aux.Point,
            is_upper: bool = False,
            angle_bounds: Optional[float] = None,
            autokick_fast: bool = False,
        ) -> None:
            self.target_pos = target_pos
            self.autokick = 2 if is_upper else 1
            if autokick_fast and not is_upper:
                self.autokick = 3
            self.angle_bounds = angle_bounds

        def is_defined(self, domain: ActionDomain) -> bool:
            if domain.game_state in (
                const.State.STOP,
                const.State.HALT,
                const.State.TIMEOUT,
                const.State.PREPARE_KICKOFF,
                const.State.PREPARE_PENALTY,
            ):
                return False
            if (
                domain.game_state in (const.State.KICKOFF, const.State.FREE_KICK, const.State.PENALTY)
                and not domain.we_active
            ):
                return False
            ball = domain.field.ball.get_pos()
            kick_angle = aux.angle_to_point(ball, self.target_pos)
            is_aligned = (
                domain.robot.is_kick_aligned_by_angle(kick_angle, angle_bounds=self.angle_bounds)
                if self.angle_bounds is not None
                else domain.robot.is_kick_aligned_by_angle(kick_angle, angle_bounds=0.04)
            )
            # Enabling the kicker while still orbiting the ball sends it sideways.
            behind_ball = ball - aux.rotate(aux.RIGHT, kick_angle) * const.KICK_ALIGN_DIST
            centered = aux.dist_to_line(ball, behind_ball, domain.robot.get_pos(), "S") < const.KICK_ALIGN_OFFSET
            return is_aligned and centered and domain.field._is_ball_in(domain.robot)

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            current_action.auto_kick = self.autokick

    class ControlVoltageAction(Action):
        """Control voltage before shooting"""

        def __init__(self, voltage: int = 15, pass_pos: Optional[aux.Point] = None, is_upper: bool = False) -> None:
            self.voltage = voltage
            self.pass_pos = pass_pos
            self.is_upper = is_upper

        def is_defined(self, domain: ActionDomain) -> bool:
            return aux.dist(domain.robot.get_pos(), domain.field.ball.get_pos()) < 1000

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            if self.pass_pos is not None:
                self.voltage = get_pass_voltage(aux.dist(domain.field.ball.get_pos(), self.pass_pos), self.is_upper)

            # Pass calibration must not bypass the division's kick limit.
            limit = const.VOLTAGE_UP if self.is_upper else const.VOLTAGE_SHOOT
            current_action.kicker_voltage = min(self.voltage, limit)

    class AddFinalVelocityAction(Action):
        """Add velocity in final target"""

        def __init__(
            self, target: aux.Point, final_velocity: aux.Point, max_dist: float = 300, min_dist: float = 100
        ) -> None:
            self.target = target
            self.final_velocity = final_velocity
            self.min_dist = min_dist
            self.max_dist = max_dist

        def is_defined(self, domain: ActionDomain) -> bool:
            return aux.dist(self.target, domain.robot.get_pos()) < self.max_dist

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            vec_to_target = self.target - domain.robot.get_pos()
            cur_speed = self.final_velocity * aux.minmax(
                (self.max_dist - vec_to_target.mag()) / (self.max_dist - self.min_dist), 0, 1
            )

            current_action.vel += cur_speed

    class LimitSpeed(Action):
        """Limit robot speed"""

        def __init__(self, limit: float = const.MAX_SPEED) -> None:
            self.limit = limit

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            limit_action(domain, current_action, self.limit)
