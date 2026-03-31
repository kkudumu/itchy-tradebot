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
SequenceExitMode      : breathing room trailing exit for SSS trades
SSLevelManager        : tracks SS levels (POIs) from inefficient sequences
SSLevel               : dataclass for a single SS level
"""

from .breathing_room import BreathingRoomDetector, SwingPoint
from .cbc_detector import CBCDetector, CBCSignal, CBCType
from .fifty_tap import FiftyTapCalculator, FiftyTapLevel
from .sequence_exit import SequenceExitMode
from .sequence_tracker import SequenceTracker, SequenceEvent
from .ss_level_manager import SSLevelManager, SSLevel

__all__ = [
    "BreathingRoomDetector",
    "SwingPoint",
    "CBCDetector",
    "CBCSignal",
    "CBCType",
    "FiftyTapCalculator",
    "FiftyTapLevel",
    "SequenceExitMode",
    "SequenceTracker",
    "SequenceEvent",
    "SSLevelManager",
    "SSLevel",
]
