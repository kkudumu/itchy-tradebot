"""SSS (Swing Structure Strategy) — package init.

Public API
----------
BreathingRoomDetector : vectorized swing high/low detector
SwingPoint            : dataclass representing a single confirmed swing
"""

from .breathing_room import BreathingRoomDetector, SwingPoint

__all__ = ["BreathingRoomDetector", "SwingPoint"]
