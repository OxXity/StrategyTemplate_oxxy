"""Stabilization of kick targets: the robot doesn't spin around the ball when the target jumps"""

from typing import Optional

from bridge.auxiliary import aux

from . import config as cfg


class AimLatch:
    """
    Keeps the kick target of one robot.
    Small changes of the kick direction are followed at once,
    a big change (another window, flank, pass instead of clearance) is accepted
    not more often than once in AIM_HOLD_TIME
    """

    def __init__(self) -> None:
        self.target: Optional[aux.Point] = None
        self.since = 0.0

    def get(self, ball: aux.Point, target: aux.Point, now: float) -> aux.Point:
        """Stabilized target"""
        if self.target is None:
            self.target, self.since = target, now
            return target

        new_angle = aux.angle_to_point(ball, target)
        old_angle = aux.angle_to_point(ball, self.target)
        if abs(aux.wind_down_angle(new_angle - old_angle)) < cfg.AIM_SMALL_ANGLE:
            self.target = target
        elif now - self.since > cfg.AIM_HOLD_TIME:
            self.target, self.since = target, now
        return self.target

    def reset(self) -> None:
        """Forget the target (the robot doesn't kick anymore)"""
        self.target = None
