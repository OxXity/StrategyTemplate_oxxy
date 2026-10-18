"""
Description of the base action class
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt


class ActionValues:
    """Return values of the action"""

    vel = aux.Point(0, 0)
    angle = 0.0
    max_angular_speed: Optional[float] = None
    kick_up = False
    kick_forward = False
    auto_kick = 0  # 0-lower, 1-upper
    kicker_voltage = const.VOLTAGE_ZERO
    dribbler_speed = 0
    beep = 0  # MOST IMPORTANT


class ActionDomain:
    """Data to perform an action"""

    def __init__(
        self,
        field: fld.Field,
        game_state: const.State,
        we_active: bool,
        robot: rbt.Robot,
    ) -> None:
        self.field = field
        self.game_state = game_state
        self.we_active = we_active
        self.robot = robot

        self.united_obstacles: Optional[list] = None
        # list of obstacles WITHOUT BALL; None -> the list hasn't been generated yet


class Action:
    """Base class of Action"""

    is_used: bool = True

    def is_defined(self, domain: ActionDomain) -> bool:
        """Scope"""
        return True

    def compose(self, action: "Action") -> "Action":
        return ComposedAction(self, action)

    def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
        """Behavior"""

    def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list["Action"]:
        """Condition for performing an action"""
        return []

    def process(self, domain: ActionDomain, current_action: ActionValues) -> None:
        """Process of Action"""
        for action in self.use_behavior_of(domain, current_action):
            action.process(domain, current_action)

        if self.is_defined(domain):
            self.behavior(domain, current_action)
            limit_action(domain, current_action)


def limit_action(domain: ActionDomain, current_action: ActionValues, speed: float = const.MAX_SPEED) -> None:
    """Limit robot speed"""
    if domain.game_state == const.State.STOP:
        speed = min(speed, const.STOP_SPEED)

    if current_action.vel.mag() > speed:
        current_action.vel = current_action.vel.unity() * speed


class ComposedAction(Action):
    def __init__(self, first: Action, second: Action):
        self.first = first
        self.second = second

    def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
        """Condition for performing an action"""
        return [self.first, self.second]
