"""SSS (Shit Show Sequence) strategy package.

Public API
----------
SSSStrategy           : main strategy orchestrator (on_bar -> Signal)
BreathingRoomDetector : vectorized swing high/low detector
SwingPoint            : dataclass representing a single confirmed swing
SequenceTracker       : state machine for sequence progression
SequenceEvent         : dataclass for state transition events
SSLevelManager        : tracks SS levels (POIs) from inefficient sequences
SSLevel               : dataclass for a single SS level
CBCDetector           : 3-candle entry pattern detector
CBCSignal             : dataclass for CBC detection results
CBCType               : CBC pattern type constants
FiftyTapCalculator    : 50% retracement entry confirmation
FiftyTapLevel         : dataclass for 50% tap levels
SequenceExitMode      : breathing room trailing exit for SSS trades
"""

from .strategy import SSSStrategy
from .breathing_room import BreathingRoomDetector, SwingPoint
from .sequence_tracker import SequenceTracker, SequenceEvent
from .ss_level_manager import SSLevelManager, SSLevel
from .cbc_detector import CBCDetector, CBCSignal, CBCType
from .fifty_tap import FiftyTapCalculator, FiftyTapLevel
from .sequence_exit import SequenceExitMode

__all__ = [
    "SSSStrategy",
    "BreathingRoomDetector",
    "SwingPoint",
    "SequenceTracker",
    "SequenceEvent",
    "SSLevelManager",
    "SSLevel",
    "CBCDetector",
    "CBCSignal",
    "CBCType",
    "FiftyTapCalculator",
    "FiftyTapLevel",
    "SequenceExitMode",
]
