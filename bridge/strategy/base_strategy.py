"""Base class for all strategies"""

from typing import Optional

from bridge import const
from bridge.auxiliary import fld
from bridge.const import State as GameStates
from bridge.router.actions import Action, Actions


class BaseStrategy:
    """
    Base class of a strategy

    How to add a new strategy:
    1. Create a file (or a package) in bridge/strategy/strategies/
    2. Inherit from BaseStrategy and override the handlers of the needed game states
    3. Register the class in STRATEGIES (bridge/strategy/strategies/__init__.py)
    4. Select it with the ACTIVE_STRATEGY flag in bridge/strategy/config.py
    """

    def __init__(self) -> None:
        self.we_active = False

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            self.we_active = field.active_team in [const.Color.ALL, field.ally_color]

        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT

        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.TIMEOUT:
                self.timeout(field, actions)
            case GameStates.HALT:
                return [Actions.Stop()] * const.TEAM_ROBOTS_MAX_COUNT
            case GameStates.PREPARE_PENALTY:
                self.prepare_penalty(field, actions)
            case GameStates.PENALTY:
                self.penalty(field, actions)
            case GameStates.PREPARE_KICKOFF:
                self.prepare_kickoff(field, actions)
            case GameStates.KICKOFF:
                self.kickoff(field, actions)
            case GameStates.FREE_KICK:
                self.free_kick(field, actions)
            case GameStates.STOP:
                self.stop(field, actions)
            case GameStates.BALL_PLACEMENT:
                self.ball_placement(field, actions)
            case GameStates.DEBUG:
                self.debug(field, actions)

        return actions

    # Handlers of game states. self.we_active tells whether the set piece is ours.
    # actions[r_id] = Action -> command for the robot r_id

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """RUN: one iteration of the normal game"""

    def stop(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """STOP: the router will automatically prevent robots from getting too close to the ball"""
        self.run(field, actions)

    def timeout(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """TIMEOUT"""

    def prepare_kickoff(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """PREPARE_KICKOFF"""

    def kickoff(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """KICKOFF"""

    def prepare_penalty(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """PREPARE_PENALTY"""

    def penalty(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """PENALTY"""

    def free_kick(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """FREE_KICK"""

    def ball_placement(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """BALL_PLACEMENT"""

    def debug(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """DEBUG"""
