"""Risk management layer — immutable risk shell for the XAU/USD Ichimoku agent.

All limits in this package are HARD-CODED. The learning loop can suggest
parameter adjustments but cannot override the absolute risk caps defined here.
"""

from src.risk.circuit_breaker import DailyCircuitBreaker
from src.risk.exit_manager import ActiveTrade, ExitDecision, HybridExitManager
from src.risk.position_sizer import AdaptivePositionSizer, PositionSize
from src.risk.trade_manager import TradeManager

__all__ = [
    "AdaptivePositionSizer",
    "PositionSize",
    "DailyCircuitBreaker",
    "HybridExitManager",
    "ExitDecision",
    "ActiveTrade",
    "TradeManager",
]
