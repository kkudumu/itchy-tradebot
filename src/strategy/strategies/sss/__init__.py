"""SSS (Shit Show Sequence) strategy package.

Public API
----------
BreathingRoomDetector : vectorized swing high/low detector
SwingPoint            : dataclass representing a single confirmed swing
CBCDetector           : 3-candle entry pattern detector
CBCSignal             : dataclass for CBC detection results
CBCType               : CBC pattern type constants
FiftyTapCalculator    : 50% retracement entry confirmation
FiftyTapLevel         : dataclass for 50% tap levels
"""

from .breathing_room import BreathingRoomDetector, SwingPoint
from .cbc_detector import CBCDetector, CBCSignal, CBCType
from .fifty_tap import FiftyTapCalculator, FiftyTapLevel

__all__ = [
    "BreathingRoomDetector",
    "SwingPoint",
    "CBCDetector",
    "CBCSignal",
    "CBCType",
    "FiftyTapCalculator",
    "FiftyTapLevel",
]
