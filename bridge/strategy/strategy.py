"""
High-level strategy code

Strategy is a facade: it creates the strategy selected by the flags
in bridge/strategy/config.py and delegates all work to it
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import fld
from bridge.router.actions import Action
from bridge.strategy import config
from bridge.strategy.base_strategy import BaseStrategy
from bridge.strategy.strategies import STRATEGIES


def create_strategy(name: str) -> BaseStrategy:
    """Create a strategy by its name from STRATEGIES"""
    if name not in STRATEGIES:
        raise RuntimeError(f"Unknown strategy '{name}', available: {', '.join(STRATEGIES)}")
    return STRATEGIES[name]()


class Strategy:
    """Main class of strategy"""

    def __init__(self, name: Optional[str] = None) -> None:
        """name: force the strategy name, otherwise it is taken from bridge/strategy/config.py"""
        self.name = name
        self.impl: Optional[BaseStrategy] = None

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""
        if self.impl is None:
            if self.name is None:
                self.name = config.ACTIVE_STRATEGY if field.ally_color == const.COLOR else config.SECOND_TEAM_STRATEGY
            self.impl = create_strategy(self.name)
            print(f"Strategy for {field.ally_color}: {self.name}")

        return self.impl.process(field)
