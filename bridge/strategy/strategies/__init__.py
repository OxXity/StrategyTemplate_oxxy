"""
Registry of available strategies

To add a strategy: import its class here and add it to STRATEGIES,
then select it by name with ACTIVE_STRATEGY in bridge/strategy/config.py
"""

from bridge.strategy.base_strategy import BaseStrategy

from .example import ExampleStrategy
from .shadow_press import ShadowPressStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "shadow_press": ShadowPressStrategy,  # 1. «Тень, прессинг, вратарь» (Div. C, 3 × 3)
    "example": ExampleStrategy,  # template example: attacker + goalkeeper
}
