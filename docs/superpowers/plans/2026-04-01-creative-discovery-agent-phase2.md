# Creative Pattern Discovery Agent (Phase 2: Chart Pattern Detection + Visual Analysis) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add chart pattern detection (double tops/bottoms, head & shoulders, triangles, wedges) to the discovery engine, wire it into the backtest loop, selectively capture trade screenshots for the worst/best trades, send those screenshots to Claude for visual analysis, and feed detected patterns back as confluence scoring adjustments.

**Architecture:** Chart pattern detection runs inline during the bar-by-bar backtest via a lightweight `PatternDetector` that consumes swing points from the existing `BreathingRoomDetector`. After each 30-day window, a `ScreenshotSelector` picks the 5 worst losers + 5 best winners + near-SL exits, generates mplfinance charts with pattern overlays, and a `VisualAnalyzer` sends them to Claude via the CLI for entry/exit quality assessment. Detected patterns contribute +/- confluence points via a `PatternConfluenceEdge` filter, and all findings persist to the Phase 1 knowledge base.

**Tech Stack:** Custom chart pattern detection (no external PatternPy -- not available on PyPI), mplfinance for chart rendering, Claude Code CLI (`claude -p`) for visual analysis, existing BreathingRoomDetector/SwingPoint infrastructure, Phase 1 KnowledgeBase for persistence.

**Phases overview (this is Phase 2 of 5):**
- Phase 1 (done): XGBoost/SHAP analysis + hypothesis loop + knowledge base
- **Phase 2 (this plan):** Chart pattern detection + selective screenshots + Claude visual analysis
- Phase 3: Macro regime (DXY synthesis, SPX, US10Y, econ calendar)
- Phase 4: LLM-generated EdgeFilter code with AST/test/backtest safety
- Phase 5: Full orchestrator tying phases 1-4 into the 30-day rolling challenge loop

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/discovery/chart_patterns.py` | Chart pattern detection from swing points (double top/bottom, H&S, triangles, wedges) |
| `src/discovery/screenshot_selector.py` | Selects worst/best/near-SL trades per window, generates annotated mplfinance charts |
| `src/discovery/visual_analyzer.py` | Sends trade screenshots to Claude CLI for visual pattern analysis |
| `src/discovery/pattern_confluence.py` | EdgeFilter that adjusts confluence score based on detected chart patterns |
| `tests/test_chart_patterns.py` | Tests for all pattern detection algorithms |
| `tests/test_screenshot_selector.py` | Tests for trade selection and chart generation |
| `tests/test_visual_analyzer.py` | Tests for prompt construction and response parsing |
| `tests/test_pattern_confluence.py` | Tests for pattern-based confluence scoring |
| `tests/test_phase2_integration.py` | Integration test: patterns -> screenshots -> analysis -> confluence |

---

### Task 1: Chart Pattern Data Structures

**Files:**
- Create: `src/discovery/chart_patterns.py`
- Test: `tests/test_chart_patterns.py`

- [ ] **Step 1: Write failing test for pattern data structures**

```python
# tests/test_chart_patterns.py
"""Tests for chart pattern detection from swing points."""

from datetime import datetime, timezone

import numpy as np
import pytest


def _make_swing(price: float, swing_type: str, index: int = 0, ts: datetime = None) -> "SwingPoint":
    """Helper: build a SwingPoint for testing."""
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=ts or datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestChartPatternDataclass:
    def test_pattern_has_required_fields(self):
        from src.discovery.chart_patterns import ChartPattern

        pattern = ChartPattern(
            pattern_type="double_top",
            direction="bearish",
            confidence=0.85,
            key_prices=[2050.0, 2048.0, 2050.5],
            start_index=100,
            end_index=200,
            description="Double top near 2050 resistance",
        )
        assert pattern.pattern_type == "double_top"
        assert pattern.direction == "bearish"
        assert pattern.confidence == 0.85
        assert len(pattern.key_prices) == 3

    def test_pattern_to_dict(self):
        from src.discovery.chart_patterns import ChartPattern

        pattern = ChartPattern(
            pattern_type="head_and_shoulders",
            direction="bearish",
            confidence=0.75,
            key_prices=[2040.0, 2060.0, 2040.0],
            start_index=50,
            end_index=150,
            description="H&S with neckline at 2030",
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "head_and_shoulders"
        assert isinstance(d["key_prices"], list)
        assert "confidence" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chart_patterns.py::TestChartPatternDataclass -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.discovery.chart_patterns'`

- [ ] **Step 3: Implement ChartPattern dataclass**

```python
# src/discovery/chart_patterns.py
"""Chart pattern detection from swing points.

Detects classical chart patterns (double top/bottom, head & shoulders,
triangles, wedges) from a sequence of SwingPoints produced by the
existing BreathingRoomDetector. Strategy-agnostic — works with any
OHLCV data that produces swing points.

Pattern detection uses geometric relationships between consecutive swing
highs and lows, with tolerance thresholds scaled to ATR for volatility
adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartPattern:
    """A detected chart pattern with metadata.

    Attributes
    ----------
    pattern_type:
        One of: double_top, double_bottom, head_and_shoulders,
        inverse_head_and_shoulders, ascending_triangle,
        descending_triangle, symmetrical_triangle,
        rising_wedge, falling_wedge.
    direction:
        'bullish' or 'bearish' — the implied directional bias.
    confidence:
        0.0 to 1.0 — how well the swing points match the ideal pattern.
    key_prices:
        List of significant price levels (e.g. peaks, neckline).
    start_index:
        Bar index where the pattern begins.
    end_index:
        Bar index where the pattern completes.
    description:
        Human-readable explanation.
    swing_indices:
        Bar indices of the swing points forming this pattern.
    """

    pattern_type: str
    direction: str
    confidence: float
    key_prices: List[float]
    start_index: int
    end_index: int
    description: str
    swing_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "key_prices": [round(p, 2) for p in self.key_prices],
            "start_index": self.start_index,
            "end_index": self.end_index,
            "description": self.description,
            "swing_indices": self.swing_indices,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chart_patterns.py::TestChartPatternDataclass -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/chart_patterns.py tests/test_chart_patterns.py
git commit -m "feat: add ChartPattern dataclass for pattern detection (Phase 2 Task 1)"
```

---

### Task 2: Double Top / Double Bottom Detection

**Files:**
- Modify: `src/discovery/chart_patterns.py`
- Test: `tests/test_chart_patterns.py`

- [ ] **Step 1: Write failing test for double top/bottom detection**

```python
# Append to tests/test_chart_patterns.py

class TestDoubleTopBottom:
    def test_detects_double_top(self):
        from src.discovery.chart_patterns import PatternDetector, ChartPattern

        # Classic double top: high, low, high (roughly equal peaks)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),  # second peak ~equal
            _make_swing(2025.0, "low", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_tops(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "double_top"
        assert p.direction == "bearish"
        assert p.confidence > 0.5

    def test_detects_double_bottom(self):
        from src.discovery.chart_patterns import PatternDetector

        # Classic double bottom: low, high, low (roughly equal troughs)
        swings = [
            _make_swing(2020.0, "low", index=10),
            _make_swing(2040.0, "high", index=20),
            _make_swing(2019.5, "low", index=30),  # second trough ~equal
            _make_swing(2045.0, "high", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_bottoms(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "double_bottom"
        assert p.direction == "bullish"

    def test_rejects_unequal_peaks(self):
        from src.discovery.chart_patterns import PatternDetector

        # Peaks too far apart to be a double top
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2070.0, "high", index=30),  # 20 points higher -- not double top
            _make_swing(2025.0, "low", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_tops(swings)

        assert len(patterns) == 0

    def test_empty_swings_returns_empty(self):
        from src.discovery.chart_patterns import PatternDetector

        detector = PatternDetector(atr=5.0)
        assert detector.detect_double_tops([]) == []
        assert detector.detect_double_bottoms([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chart_patterns.py::TestDoubleTopBottom -v`
Expected: FAIL with `ImportError: cannot import name 'PatternDetector'`

- [ ] **Step 3: Implement PatternDetector with double top/bottom detection**

Append to `src/discovery/chart_patterns.py`:

```python
# Type alias for external SwingPoint
try:
    from src.strategy.strategies.sss.breathing_room import SwingPoint
except ImportError:
    # Fallback for standalone testing
    SwingPoint = Any  # type: ignore


class PatternDetector:
    """Detects classical chart patterns from a sequence of swing points.

    All distance thresholds are scaled to ATR for volatility adaptation.
    This makes the detector work across different market regimes.

    Parameters
    ----------
    atr:
        Current Average True Range for the instrument/timeframe.
    tolerance_atr_mult:
        How many ATRs apart two peaks/troughs can be and still be
        considered "equal" for pattern matching. Default 1.5.
    min_pattern_bars:
        Minimum bar span for a pattern to be valid. Default 10.
    """

    def __init__(
        self,
        atr: float,
        tolerance_atr_mult: float = 1.5,
        min_pattern_bars: int = 10,
    ) -> None:
        self._atr = max(atr, 0.01)  # guard against zero
        self._tolerance = atr * tolerance_atr_mult
        self._min_bars = min_pattern_bars

    def _prices_equal(self, a: float, b: float) -> bool:
        """Check if two prices are within ATR tolerance."""
        return abs(a - b) <= self._tolerance

    def _confidence_from_diff(self, a: float, b: float) -> float:
        """Higher confidence when peaks/troughs are closer together."""
        diff = abs(a - b)
        if diff <= 0.001:
            return 1.0
        ratio = diff / self._tolerance
        return max(0.0, 1.0 - ratio * 0.5)

    # ------------------------------------------------------------------
    # Double Top
    # ------------------------------------------------------------------

    def detect_double_tops(self, swings: List) -> List[ChartPattern]:
        """Detect double top patterns from swing point sequence.

        A double top requires:
        1. Two swing highs at roughly equal price (within ATR tolerance)
        2. A swing low between them (the neckline)
        3. Minimum bar span

        Returns list of detected ChartPattern objects.
        """
        if len(swings) < 3:
            return []

        patterns: List[ChartPattern] = []
        highs = [(i, s) for i, s in enumerate(swings) if s.swing_type == "high"]

        for a_idx in range(len(highs) - 1):
            for b_idx in range(a_idx + 1, min(a_idx + 4, len(highs))):
                i_a, sw_a = highs[a_idx]
                i_b, sw_b = highs[b_idx]

                if not self._prices_equal(sw_a.price, sw_b.price):
                    continue

                bar_span = abs(sw_b.index - sw_a.index)
                if bar_span < self._min_bars:
                    continue

                # Find the lowest low between the two highs
                lows_between = [
                    s for s in swings[i_a + 1:i_b]
                    if s.swing_type == "low"
                ]
                if not lows_between:
                    continue

                neckline_swing = min(lows_between, key=lambda s: s.price)
                neckline = neckline_swing.price

                # The pullback between peaks must be meaningful (> 0.5 ATR)
                pullback = max(sw_a.price, sw_b.price) - neckline
                if pullback < self._atr * 0.5:
                    continue

                conf = self._confidence_from_diff(sw_a.price, sw_b.price)

                patterns.append(ChartPattern(
                    pattern_type="double_top",
                    direction="bearish",
                    confidence=conf,
                    key_prices=[sw_a.price, neckline, sw_b.price],
                    start_index=sw_a.index,
                    end_index=sw_b.index,
                    description=(
                        f"Double top at {sw_a.price:.1f}/{sw_b.price:.1f}, "
                        f"neckline {neckline:.1f}"
                    ),
                    swing_indices=[sw_a.index, neckline_swing.index, sw_b.index],
                ))

        return patterns

    # ------------------------------------------------------------------
    # Double Bottom
    # ------------------------------------------------------------------

    def detect_double_bottoms(self, swings: List) -> List[ChartPattern]:
        """Detect double bottom patterns from swing point sequence.

        Mirror of double top: two roughly equal lows with a high between.
        """
        if len(swings) < 3:
            return []

        patterns: List[ChartPattern] = []
        lows = [(i, s) for i, s in enumerate(swings) if s.swing_type == "low"]

        for a_idx in range(len(lows) - 1):
            for b_idx in range(a_idx + 1, min(a_idx + 4, len(lows))):
                i_a, sw_a = lows[a_idx]
                i_b, sw_b = lows[b_idx]

                if not self._prices_equal(sw_a.price, sw_b.price):
                    continue

                bar_span = abs(sw_b.index - sw_a.index)
                if bar_span < self._min_bars:
                    continue

                highs_between = [
                    s for s in swings[i_a + 1:i_b]
                    if s.swing_type == "high"
                ]
                if not highs_between:
                    continue

                neckline_swing = max(highs_between, key=lambda s: s.price)
                neckline = neckline_swing.price

                pullback = neckline - min(sw_a.price, sw_b.price)
                if pullback < self._atr * 0.5:
                    continue

                conf = self._confidence_from_diff(sw_a.price, sw_b.price)

                patterns.append(ChartPattern(
                    pattern_type="double_bottom",
                    direction="bullish",
                    confidence=conf,
                    key_prices=[sw_a.price, neckline, sw_b.price],
                    start_index=sw_a.index,
                    end_index=sw_b.index,
                    description=(
                        f"Double bottom at {sw_a.price:.1f}/{sw_b.price:.1f}, "
                        f"neckline {neckline:.1f}"
                    ),
                    swing_indices=[sw_a.index, neckline_swing.index, sw_b.index],
                ))

        return patterns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chart_patterns.py::TestDoubleTopBottom -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/chart_patterns.py tests/test_chart_patterns.py
git commit -m "feat: add double top/bottom pattern detection (Phase 2 Task 2)"
```

---

### Task 3: Head & Shoulders Detection

**Files:**
- Modify: `src/discovery/chart_patterns.py`
- Test: `tests/test_chart_patterns.py`

- [ ] **Step 1: Write failing test for H&S detection**

```python
# Append to tests/test_chart_patterns.py

class TestHeadAndShoulders:
    def test_detects_head_and_shoulders(self):
        from src.discovery.chart_patterns import PatternDetector

        # Classic H&S: left shoulder, head (highest), right shoulder (roughly equal to left)
        swings = [
            _make_swing(2040.0, "high", index=10),   # left shoulder
            _make_swing(2025.0, "low", index=15),     # left neckline
            _make_swing(2055.0, "high", index=25),    # head (highest)
            _make_swing(2024.0, "low", index=35),     # right neckline
            _make_swing(2041.0, "high", index=45),    # right shoulder (~left)
            _make_swing(2020.0, "low", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_head_and_shoulders(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "head_and_shoulders"
        assert p.direction == "bearish"
        assert p.confidence > 0.5

    def test_detects_inverse_head_and_shoulders(self):
        from src.discovery.chart_patterns import PatternDetector

        # Inverse H&S: left shoulder, head (lowest), right shoulder (~left)
        swings = [
            _make_swing(2030.0, "low", index=10),     # left shoulder
            _make_swing(2045.0, "high", index=15),     # left neckline
            _make_swing(2015.0, "low", index=25),      # head (lowest)
            _make_swing(2046.0, "high", index=35),     # right neckline
            _make_swing(2029.0, "low", index=45),      # right shoulder (~left)
            _make_swing(2050.0, "high", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_inverse_head_and_shoulders(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "inverse_head_and_shoulders"
        assert p.direction == "bullish"

    def test_rejects_when_head_not_highest(self):
        from src.discovery.chart_patterns import PatternDetector

        # "Head" is not the highest peak — not valid H&S
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2025.0, "low", index=15),
            _make_swing(2045.0, "high", index=25),    # head LOWER than left shoulder
            _make_swing(2024.0, "low", index=35),
            _make_swing(2048.0, "high", index=45),
            _make_swing(2020.0, "low", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_head_and_shoulders(swings)
        assert len(patterns) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chart_patterns.py::TestHeadAndShoulders -v`
Expected: FAIL with `AttributeError: 'PatternDetector' object has no attribute 'detect_head_and_shoulders'`

- [ ] **Step 3: Implement H&S and inverse H&S detection**

Append to `PatternDetector` class in `src/discovery/chart_patterns.py`:

```python
    # ------------------------------------------------------------------
    # Head and Shoulders
    # ------------------------------------------------------------------

    def detect_head_and_shoulders(self, swings: List) -> List[ChartPattern]:
        """Detect head & shoulders patterns.

        Requires 5 swing points: H-L-H-L-H where the middle H is highest
        and the two outer H's are roughly equal (within tolerance).
        The two lows form the neckline.
        """
        if len(swings) < 5:
            return []

        patterns: List[ChartPattern] = []
        highs = [(i, s) for i, s in enumerate(swings) if s.swing_type == "high"]

        for h_idx in range(len(highs) - 2):
            # Need 3 consecutive highs
            i_ls, ls = highs[h_idx]       # left shoulder
            i_hd, hd = highs[h_idx + 1]   # head
            i_rs, rs = highs[h_idx + 2]   # right shoulder

            # Head must be highest
            if hd.price <= ls.price or hd.price <= rs.price:
                continue

            # Shoulders must be roughly equal
            if not self._prices_equal(ls.price, rs.price):
                continue

            # Head must be meaningfully higher than shoulders (> 0.5 ATR)
            head_prominence = hd.price - max(ls.price, rs.price)
            if head_prominence < self._atr * 0.5:
                continue

            # Find neckline lows between shoulders and head
            lows_left = [s for s in swings[i_ls + 1:i_hd] if s.swing_type == "low"]
            lows_right = [s for s in swings[i_hd + 1:i_rs] if s.swing_type == "low"]

            if not lows_left or not lows_right:
                continue

            nl_left = min(lows_left, key=lambda s: s.price)
            nl_right = min(lows_right, key=lambda s: s.price)
            neckline = (nl_left.price + nl_right.price) / 2

            shoulder_conf = self._confidence_from_diff(ls.price, rs.price)
            neckline_conf = self._confidence_from_diff(nl_left.price, nl_right.price)
            conf = (shoulder_conf + neckline_conf) / 2

            patterns.append(ChartPattern(
                pattern_type="head_and_shoulders",
                direction="bearish",
                confidence=conf,
                key_prices=[ls.price, hd.price, rs.price, neckline],
                start_index=ls.index,
                end_index=rs.index,
                description=(
                    f"H&S: shoulders {ls.price:.1f}/{rs.price:.1f}, "
                    f"head {hd.price:.1f}, neckline {neckline:.1f}"
                ),
                swing_indices=[ls.index, nl_left.index, hd.index, nl_right.index, rs.index],
            ))

        return patterns

    # ------------------------------------------------------------------
    # Inverse Head and Shoulders
    # ------------------------------------------------------------------

    def detect_inverse_head_and_shoulders(self, swings: List) -> List[ChartPattern]:
        """Detect inverse head & shoulders patterns.

        Mirror of H&S: L-H-L-H-L where the middle L is lowest and
        the two outer L's are roughly equal.
        """
        if len(swings) < 5:
            return []

        patterns: List[ChartPattern] = []
        lows = [(i, s) for i, s in enumerate(swings) if s.swing_type == "low"]

        for l_idx in range(len(lows) - 2):
            i_ls, ls = lows[l_idx]         # left shoulder
            i_hd, hd = lows[l_idx + 1]     # head (lowest)
            i_rs, rs = lows[l_idx + 2]     # right shoulder

            # Head must be lowest
            if hd.price >= ls.price or hd.price >= rs.price:
                continue

            # Shoulders must be roughly equal
            if not self._prices_equal(ls.price, rs.price):
                continue

            # Head must be meaningfully lower than shoulders
            head_prominence = min(ls.price, rs.price) - hd.price
            if head_prominence < self._atr * 0.5:
                continue

            # Find neckline highs between shoulders and head
            highs_left = [s for s in swings[i_ls + 1:i_hd] if s.swing_type == "high"]
            highs_right = [s for s in swings[i_hd + 1:i_rs] if s.swing_type == "high"]

            if not highs_left or not highs_right:
                continue

            nl_left = max(highs_left, key=lambda s: s.price)
            nl_right = max(highs_right, key=lambda s: s.price)
            neckline = (nl_left.price + nl_right.price) / 2

            shoulder_conf = self._confidence_from_diff(ls.price, rs.price)
            neckline_conf = self._confidence_from_diff(nl_left.price, nl_right.price)
            conf = (shoulder_conf + neckline_conf) / 2

            patterns.append(ChartPattern(
                pattern_type="inverse_head_and_shoulders",
                direction="bullish",
                confidence=conf,
                key_prices=[ls.price, hd.price, rs.price, neckline],
                start_index=ls.index,
                end_index=rs.index,
                description=(
                    f"Inverse H&S: shoulders {ls.price:.1f}/{rs.price:.1f}, "
                    f"head {hd.price:.1f}, neckline {neckline:.1f}"
                ),
                swing_indices=[ls.index, nl_left.index, hd.index, nl_right.index, rs.index],
            ))

        return patterns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chart_patterns.py::TestHeadAndShoulders -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/chart_patterns.py tests/test_chart_patterns.py
git commit -m "feat: add head & shoulders pattern detection (Phase 2 Task 3)"
```

---

### Task 4: Triangle and Wedge Detection

**Files:**
- Modify: `src/discovery/chart_patterns.py`
- Test: `tests/test_chart_patterns.py`

- [ ] **Step 1: Write failing tests for triangle/wedge detection**

```python
# Append to tests/test_chart_patterns.py

class TestTriangles:
    def test_detects_ascending_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Ascending triangle: flat highs, rising lows
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=15),
            _make_swing(2050.5, "high", index=25),
            _make_swing(2035.0, "low", index=30),     # higher low
            _make_swing(2051.0, "high", index=40),
            _make_swing(2040.0, "low", index=45),     # higher low
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "ascending_triangle" for p in patterns)

    def test_detects_descending_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Descending triangle: flat lows, falling highs
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2020.0, "low", index=15),
            _make_swing(2045.0, "high", index=25),     # lower high
            _make_swing(2020.5, "low", index=30),
            _make_swing(2040.0, "high", index=40),     # lower high
            _make_swing(2019.5, "low", index=45),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "descending_triangle" for p in patterns)

    def test_detects_symmetrical_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Symmetrical: lower highs AND higher lows
        swings = [
            _make_swing(2055.0, "high", index=10),
            _make_swing(2025.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # lower high
            _make_swing(2030.0, "low", index=30),      # higher low
            _make_swing(2045.0, "high", index=40),     # lower high
            _make_swing(2035.0, "low", index=45),      # higher low
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "symmetrical_triangle" for p in patterns)


class TestWedges:
    def test_detects_rising_wedge(self):
        from src.discovery.chart_patterns import PatternDetector

        # Rising wedge: higher highs AND higher lows, converging
        swings = [
            _make_swing(2040.0, "high", index=10),
            _make_swing(2020.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # higher high
            _make_swing(2035.0, "low", index=30),      # higher low (bigger move)
            _make_swing(2055.0, "high", index=40),     # higher high
            _make_swing(2045.0, "low", index=45),      # higher low (bigger move)
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_wedges(swings)

        assert any(p.pattern_type == "rising_wedge" for p in patterns)

    def test_detects_falling_wedge(self):
        from src.discovery.chart_patterns import PatternDetector

        # Falling wedge: lower highs AND lower lows, converging
        swings = [
            _make_swing(2060.0, "high", index=10),
            _make_swing(2040.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # lower high
            _make_swing(2035.0, "low", index=30),      # lower low (smaller drop)
            _make_swing(2045.0, "high", index=40),     # lower high
            _make_swing(2032.0, "low", index=45),      # lower low (smaller drop)
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_wedges(swings)

        assert any(p.pattern_type == "falling_wedge" for p in patterns)

    def test_needs_minimum_swings(self):
        from src.discovery.chart_patterns import PatternDetector

        detector = PatternDetector(atr=5.0)
        swings = [_make_swing(2050.0, "high", index=10)]
        assert detector.detect_wedges(swings) == []
        assert detector.detect_triangles(swings) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chart_patterns.py::TestTriangles tests/test_chart_patterns.py::TestWedges -v`
Expected: FAIL with `AttributeError: 'PatternDetector' object has no attribute 'detect_triangles'`

- [ ] **Step 3: Implement triangle and wedge detection**

Append to `PatternDetector` class in `src/discovery/chart_patterns.py`:

```python
    # ------------------------------------------------------------------
    # Triangles
    # ------------------------------------------------------------------

    def detect_triangles(self, swings: List) -> List[ChartPattern]:
        """Detect ascending, descending, and symmetrical triangles.

        Requires at least 3 highs and 3 lows to establish trendlines.
        """
        if len(swings) < 6:
            return []

        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        if len(highs) < 3 or len(lows) < 3:
            return []

        patterns: List[ChartPattern] = []

        # Use last 3 highs and last 3 lows
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        # Classify high slope: flat, falling, rising
        high_slope = self._classify_slope(
            [h.price for h in recent_highs],
            [h.index for h in recent_highs],
        )

        # Classify low slope: flat, falling, rising
        low_slope = self._classify_slope(
            [l.price for l in recent_lows],
            [l.index for l in recent_lows],
        )

        start_idx = min(recent_highs[0].index, recent_lows[0].index)
        end_idx = max(recent_highs[-1].index, recent_lows[-1].index)
        all_indices = [s.index for s in recent_highs + recent_lows]

        # Ascending triangle: flat highs + rising lows
        if high_slope == "flat" and low_slope == "rising":
            resistance = np.mean([h.price for h in recent_highs])
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="ascending_triangle",
                direction="bullish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description=f"Ascending triangle, resistance ~{resistance:.1f}",
                swing_indices=sorted(all_indices),
            ))

        # Descending triangle: falling highs + flat lows
        elif high_slope == "falling" and low_slope == "flat":
            support = np.mean([l.price for l in recent_lows])
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="descending_triangle",
                direction="bearish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description=f"Descending triangle, support ~{support:.1f}",
                swing_indices=sorted(all_indices),
            ))

        # Symmetrical triangle: falling highs + rising lows
        elif high_slope == "falling" and low_slope == "rising":
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="symmetrical_triangle",
                direction="neutral",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Symmetrical triangle — converging trendlines",
                swing_indices=sorted(all_indices),
            ))

        return patterns

    def _classify_slope(self, prices: List[float], indices: List[int]) -> str:
        """Classify a price sequence as flat, rising, or falling.

        'flat' = all prices within ATR tolerance of each other.
        'rising' = each price higher than the previous (within noise).
        'falling' = each price lower than the previous.
        """
        if len(prices) < 2:
            return "flat"

        # Check if all roughly equal (flat)
        price_range = max(prices) - min(prices)
        if price_range <= self._tolerance:
            return "flat"

        # Check monotonic direction
        diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        if all(d > -self._atr * 0.2 for d in diffs) and sum(d > 0 for d in diffs) >= len(diffs) * 0.5:
            return "rising"
        if all(d < self._atr * 0.2 for d in diffs) and sum(d < 0 for d in diffs) >= len(diffs) * 0.5:
            return "falling"

        return "mixed"

    def _triangle_confidence(self, highs: List, lows: List) -> float:
        """Calculate triangle pattern confidence based on convergence."""
        if len(highs) < 2 or len(lows) < 2:
            return 0.5

        # Range between highs and lows should be narrowing
        first_range = highs[0].price - lows[0].price
        last_range = highs[-1].price - lows[-1].price

        if first_range <= 0:
            return 0.3

        convergence_ratio = last_range / first_range
        # Perfect triangle converges; ratio < 1 is good
        if convergence_ratio < 0.3:
            return 0.9
        elif convergence_ratio < 0.6:
            return 0.75
        elif convergence_ratio < 0.85:
            return 0.6
        else:
            return 0.4

    # ------------------------------------------------------------------
    # Wedges
    # ------------------------------------------------------------------

    def detect_wedges(self, swings: List) -> List[ChartPattern]:
        """Detect rising and falling wedge patterns.

        Rising wedge: higher highs + higher lows, but range narrowing (bearish).
        Falling wedge: lower highs + lower lows, but range narrowing (bullish).
        """
        if len(swings) < 6:
            return []

        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        if len(highs) < 3 or len(lows) < 3:
            return []

        patterns: List[ChartPattern] = []
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        high_slope = self._classify_slope(
            [h.price for h in recent_highs],
            [h.index for h in recent_highs],
        )
        low_slope = self._classify_slope(
            [l.price for l in recent_lows],
            [l.index for l in recent_lows],
        )

        # Check convergence: range narrowing
        first_range = recent_highs[0].price - recent_lows[0].price
        last_range = recent_highs[-1].price - recent_lows[-1].price
        is_converging = last_range < first_range * 0.85 if first_range > 0 else False

        start_idx = min(recent_highs[0].index, recent_lows[0].index)
        end_idx = max(recent_highs[-1].index, recent_lows[-1].index)
        all_indices = [s.index for s in recent_highs + recent_lows]

        # Rising wedge: both rising + converging
        if high_slope == "rising" and low_slope == "rising" and is_converging:
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="rising_wedge",
                direction="bearish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Rising wedge — bearish reversal pattern",
                swing_indices=sorted(all_indices),
            ))

        # Falling wedge: both falling + converging
        elif high_slope == "falling" and low_slope == "falling" and is_converging:
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="falling_wedge",
                direction="bullish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Falling wedge — bullish reversal pattern",
                swing_indices=sorted(all_indices),
            ))

        return patterns

    # ------------------------------------------------------------------
    # Unified scan
    # ------------------------------------------------------------------

    def detect_all(self, swings: List) -> List[ChartPattern]:
        """Run all pattern detectors and return combined results.

        Deduplicates overlapping patterns, keeping the highest confidence.
        """
        all_patterns: List[ChartPattern] = []
        all_patterns.extend(self.detect_double_tops(swings))
        all_patterns.extend(self.detect_double_bottoms(swings))
        all_patterns.extend(self.detect_head_and_shoulders(swings))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(swings))
        all_patterns.extend(self.detect_triangles(swings))
        all_patterns.extend(self.detect_wedges(swings))

        # Sort by confidence descending
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)

        # Deduplicate: if two patterns overlap by >50% of bar range, keep higher confidence
        unique: List[ChartPattern] = []
        for p in all_patterns:
            overlaps = False
            for q in unique:
                overlap_start = max(p.start_index, q.start_index)
                overlap_end = min(p.end_index, q.end_index)
                p_span = max(1, p.end_index - p.start_index)
                overlap_pct = max(0, overlap_end - overlap_start) / p_span
                if overlap_pct > 0.5 and p.direction == q.direction:
                    overlaps = True
                    break
            if not overlaps:
                unique.append(p)

        return unique
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chart_patterns.py::TestTriangles tests/test_chart_patterns.py::TestWedges -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/chart_patterns.py tests/test_chart_patterns.py
git commit -m "feat: add triangle and wedge pattern detection (Phase 2 Task 4)"
```

---

### Task 5: Screenshot Selector (Trade Selection + Chart Generation)

**Files:**
- Create: `src/discovery/screenshot_selector.py`
- Test: `tests/test_screenshot_selector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_screenshot_selector.py
"""Tests for selective trade screenshot capture."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


def _make_trade_dict(r_multiple: float, entry_bar: int = 50, exit_bar: int = 80, **kwargs) -> dict:
    """Build a minimal trade dict for testing."""
    base = {
        "r_multiple": r_multiple,
        "entry_bar_idx": entry_bar,
        "exit_bar_idx": exit_bar,
        "entry_price": 2050.0,
        "exit_price": 2050.0 + r_multiple * 5.0,
        "stop_loss": 2045.0,
        "direction": "long",
        "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
    }
    base.update(kwargs)
    return base


class TestTradeSelection:
    def test_selects_worst_losers(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=3, n_best=2, near_sl_threshold=0.2)
        trades = [
            _make_trade_dict(-2.0),
            _make_trade_dict(-1.5),
            _make_trade_dict(-1.0),
            _make_trade_dict(-0.5),
            _make_trade_dict(1.0),
            _make_trade_dict(1.5),
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)

        worst = [t for t in selected if t["_selection_reason"] == "worst_loser"]
        assert len(worst) == 3
        # Should be the 3 worst by R
        worst_rs = [t["r_multiple"] for t in worst]
        assert sorted(worst_rs) == [-2.0, -1.5, -1.0]

    def test_selects_best_winners(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=2, n_best=3)
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(0.5),
            _make_trade_dict(1.0),
            _make_trade_dict(1.5),
            _make_trade_dict(2.5),
        ]
        selected = selector.select_trades(trades)

        best = [t for t in selected if t["_selection_reason"] == "best_winner"]
        assert len(best) == 3
        best_rs = [t["r_multiple"] for t in best]
        assert sorted(best_rs) == [1.0, 1.5, 2.5]

    def test_selects_near_sl_exits(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=1, n_best=1, near_sl_threshold=0.3)
        # Near-SL: trade that hit within 0.3R of the stop before winning
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(1.0, min_r_during_trade=-0.85),  # near SL
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)

        near_sl = [t for t in selected if t.get("_selection_reason") == "near_sl_exit"]
        # min_r_during_trade is -0.85, threshold is -1.0 + 0.3 = -0.7. -0.85 < -0.7 so this qualifies
        assert len(near_sl) >= 1

    def test_no_duplicates_in_selection(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=5, n_best=5)
        # Only 3 trades — no duplication
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(0.5),
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)
        assert len(selected) == 3  # all selected, no duplicates

    def test_empty_trades_returns_empty(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector()
        assert selector.select_trades([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_screenshot_selector.py::TestTradeSelection -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ScreenshotSelector**

```python
# src/discovery/screenshot_selector.py
"""Selective trade screenshot capture for visual analysis.

After each 30-day backtest window, selects the most informative trades
for Claude visual analysis: worst losers (what went wrong?), best winners
(what pattern to repeat?), and near-SL survivors (lucky or skilled?).

Generates annotated mplfinance charts with entry/exit markers and
optional pattern overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScreenshotSelector:
    """Selects and captures trade screenshots for visual analysis.

    Parameters
    ----------
    n_worst:
        Number of worst losers to capture. Default 5.
    n_best:
        Number of best winners to capture. Default 5.
    near_sl_threshold:
        R-multiple threshold for "near stop-loss" detection. A winning
        trade that dipped below -(1.0 - threshold)R during its life is
        considered a near-SL exit. Default 0.2 (dipped below -0.8R).
    bars_before:
        Number of bars before entry to include in chart. Default 60.
    bars_after:
        Number of bars after exit to include in chart. Default 20.
    """

    def __init__(
        self,
        n_worst: int = 5,
        n_best: int = 5,
        near_sl_threshold: float = 0.2,
        bars_before: int = 60,
        bars_after: int = 20,
    ) -> None:
        self._n_worst = n_worst
        self._n_best = n_best
        self._near_sl_thresh = near_sl_threshold
        self._bars_before = bars_before
        self._bars_after = bars_after

    def select_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the most informative trades for screenshot capture.

        Returns copies of trade dicts with a '_selection_reason' field
        indicating why each trade was selected.
        """
        if not trades:
            return []

        # Sort for worst losers (lowest R first)
        sorted_by_r = sorted(trades, key=lambda t: float(t.get("r_multiple", 0)))
        losers = [t for t in sorted_by_r if float(t.get("r_multiple", 0)) < 0]
        winners = [t for t in reversed(sorted_by_r) if float(t.get("r_multiple", 0)) > 0]

        selected: Dict[int, Dict[str, Any]] = {}  # keyed by id(trade) to prevent dupes

        # Worst losers
        for t in losers[:self._n_worst]:
            entry = {**t, "_selection_reason": "worst_loser"}
            selected[id(t)] = entry

        # Best winners
        for t in winners[:self._n_best]:
            if id(t) not in selected:
                selected[id(t)] = {**t, "_selection_reason": "best_winner"}

        # Near-SL exits: winning trades that nearly hit their stop
        sl_cutoff = -(1.0 - self._near_sl_thresh)
        for t in trades:
            if id(t) in selected:
                continue
            r = float(t.get("r_multiple", 0))
            min_r = float(t.get("min_r_during_trade", 0))
            if r > 0 and min_r < sl_cutoff:
                selected[id(t)] = {**t, "_selection_reason": "near_sl_exit"}

        return list(selected.values())

    def generate_chart(
        self,
        trade: Dict[str, Any],
        df_5m: pd.DataFrame,
        output_dir: str,
        patterns: Optional[List] = None,
        instrument: str = "XAUUSD",
    ) -> Optional[str]:
        """Generate an annotated mplfinance chart for a selected trade.

        Parameters
        ----------
        trade: Trade dict with entry_bar_idx, exit_bar_idx, direction, etc.
        df_5m: Full 5M OHLCV DataFrame from the backtest.
        output_dir: Directory to save PNG files.
        patterns: Optional list of ChartPattern objects to overlay.
        instrument: Instrument symbol for the chart title.

        Returns
        -------
        Path to saved PNG, or None on failure.
        """
        try:
            import mplfinance as mpf
        except ImportError:
            logger.warning("mplfinance not installed — skipping chart generation")
            return None

        entry_idx = int(trade.get("entry_bar_idx", 0))
        exit_idx = int(trade.get("exit_bar_idx", entry_idx + 30))

        start = max(0, entry_idx - self._bars_before)
        end = min(len(df_5m), exit_idx + self._bars_after)
        chunk = df_5m.iloc[start:end].copy()

        if chunk.empty or len(chunk) < 5:
            return None

        # Ensure DatetimeIndex
        if not isinstance(chunk.index, pd.DatetimeIndex):
            chunk.index = pd.DatetimeIndex(chunk.index)

        # Rename tick_volume if needed
        if "tick_volume" in chunk.columns and "volume" not in chunk.columns:
            chunk = chunk.rename(columns={"tick_volume": "volume"})

        # Entry/exit markers
        entry_local = entry_idx - start
        exit_local = exit_idx - start
        entry_local = max(0, min(entry_local, len(chunk) - 1))
        exit_local = max(0, min(exit_local, len(chunk) - 1))

        entry_marker = np.full(len(chunk), np.nan)
        exit_marker = np.full(len(chunk), np.nan)
        entry_marker[entry_local] = chunk.iloc[entry_local]["low"] * 0.999
        exit_marker[exit_local] = chunk.iloc[exit_local]["high"] * 1.001

        apds = [
            mpf.make_addplot(entry_marker, type="scatter", markersize=120, marker="^", color="green"),
            mpf.make_addplot(exit_marker, type="scatter", markersize=120, marker="v", color="red"),
        ]

        # Build title
        r = float(trade.get("r_multiple", 0))
        reason = trade.get("_selection_reason", "")
        direction = trade.get("direction", "")
        title = f"{instrument} 5M — {direction.upper()} ({r:+.2f}R) [{reason}]"

        # Output path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        entry_time = trade.get("entry_time")
        ts_str = str(entry_time).replace(" ", "_").replace(":", "")[:15] if entry_time else "unknown"
        filename = f"{reason}_{r:+.2f}R_{ts_str}.png"
        file_path = out_path / filename

        try:
            mpf.plot(
                chunk,
                type="candle",
                style="charles",
                title=title,
                volume=("volume" in chunk.columns),
                addplot=apds,
                savefig=str(file_path),
                figsize=(14.0, 8.0),
            )
            import matplotlib.pyplot as plt
            plt.close("all")
            return str(file_path)
        except Exception as exc:
            logger.debug("Chart generation failed: %s", exc)
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_screenshot_selector.py::TestTradeSelection -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/screenshot_selector.py tests/test_screenshot_selector.py
git commit -m "feat: add screenshot selector for worst/best/near-SL trades (Phase 2 Task 5)"
```

---

### Task 6: Visual Analyzer (Claude Screenshot Analysis)

**Files:**
- Create: `src/discovery/visual_analyzer.py`
- Test: `tests/test_visual_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_visual_analyzer.py
"""Tests for Claude visual analysis of trade screenshots."""

import pytest


class TestBuildVisualPrompt:
    def test_prompt_contains_trade_metadata(self):
        from src.discovery.visual_analyzer import build_visual_prompt

        trade = {
            "r_multiple": -1.5,
            "direction": "long",
            "entry_price": 2050.0,
            "exit_price": 2042.5,
            "stop_loss": 2045.0,
            "_selection_reason": "worst_loser",
            "exit_reason": "stop_loss",
        }
        prompt = build_visual_prompt(trade, strategy_name="sss")

        assert "long" in prompt.lower()
        assert "2050" in prompt
        assert "-1.5" in prompt or "-1.50" in prompt
        assert "worst_loser" in prompt
        assert "sss" in prompt

    def test_prompt_asks_for_pattern_identification(self):
        from src.discovery.visual_analyzer import build_visual_prompt

        trade = {"r_multiple": 2.0, "direction": "short", "_selection_reason": "best_winner"}
        prompt = build_visual_prompt(trade, strategy_name="sss")

        assert "pattern" in prompt.lower()
        assert "entry" in prompt.lower()
        assert "json" in prompt.lower()

    def test_prompt_includes_detected_patterns(self):
        from src.discovery.visual_analyzer import build_visual_prompt
        from src.discovery.chart_patterns import ChartPattern

        trade = {"r_multiple": 1.0, "direction": "long", "_selection_reason": "best_winner"}
        patterns = [
            ChartPattern(
                pattern_type="double_bottom",
                direction="bullish",
                confidence=0.8,
                key_prices=[2020.0, 2040.0, 2021.0],
                start_index=10,
                end_index=40,
                description="Double bottom near 2020",
            )
        ]
        prompt = build_visual_prompt(trade, strategy_name="sss", patterns=patterns)
        assert "double_bottom" in prompt
        assert "2020" in prompt


class TestParseVisualAnalysis:
    def test_parses_json_response(self):
        from src.discovery.visual_analyzer import parse_visual_response

        response = '''Based on my analysis:

```json
{
  "patterns_at_entry": ["double_bottom", "support_bounce"],
  "patterns_at_exit": ["resistance_rejection"],
  "entry_quality": "good",
  "exit_quality": "poor",
  "confluence_adjustment": 1,
  "reasoning": "Entry was well-timed at double bottom support. Exit was premature.",
  "improvement_suggestion": "Hold through minor resistance when trend is strong."
}
```
'''
        result = parse_visual_response(response)

        assert result is not None
        assert "double_bottom" in result["patterns_at_entry"]
        assert result["entry_quality"] == "good"
        assert result["confluence_adjustment"] == 1

    def test_handles_empty_response(self):
        from src.discovery.visual_analyzer import parse_visual_response

        result = parse_visual_response("")
        assert result is None

    def test_handles_malformed_json(self):
        from src.discovery.visual_analyzer import parse_visual_response

        result = parse_visual_response("This response has no JSON block at all.")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement visual_analyzer**

```python
# src/discovery/visual_analyzer.py
"""Claude visual analysis of trade screenshots.

Sends annotated trade charts to Claude via the CLI for pattern
identification at entry/exit points. Returns structured feedback
including detected patterns, entry/exit quality assessment, and
confluence scoring adjustments.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_visual_prompt(
    trade: Dict[str, Any],
    strategy_name: str,
    patterns: Optional[List] = None,
    extra_context: str = "",
) -> str:
    """Build a structured prompt for Claude visual analysis of a trade chart.

    Parameters
    ----------
    trade: Trade metadata dict.
    strategy_name: Name of the strategy that generated the trade.
    patterns: Optional list of ChartPattern objects detected near the trade.
    extra_context: Additional context to include in the prompt.
    """
    r = trade.get("r_multiple", 0)
    direction = trade.get("direction", "unknown")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    stop_loss = trade.get("stop_loss", 0)
    reason = trade.get("_selection_reason", "")
    exit_reason = trade.get("exit_reason", "")

    pattern_section = ""
    if patterns:
        pattern_section = "\n## Pre-Detected Chart Patterns\n"
        for p in patterns:
            d = p.to_dict() if hasattr(p, "to_dict") else p
            pattern_section += (
                f"- {d.get('pattern_type', 'unknown')}: {d.get('description', '')}"
                f" (confidence: {d.get('confidence', 0):.0%})\n"
            )

    prompt = f"""You are a professional gold (XAU/USD) chart analyst reviewing a trade screenshot
from the {strategy_name} strategy. Analyze the attached candlestick chart.

## Trade Metadata
- **Direction:** {direction}
- **Entry Price:** {entry_price}
- **Exit Price:** {exit_price}
- **Stop Loss:** {stop_loss}
- **R-Multiple:** {r:+.2f}R
- **Selection Reason:** {reason}
- **Exit Reason:** {exit_reason}
{pattern_section}
{extra_context}

## Your Task

1. **Identify chart patterns** visible at the entry point (e.g., double bottom, head & shoulders,
   triangle breakout, wedge, flag, channel, support/resistance bounce)
2. **Assess entry quality** — was this a good entry location given the visible pattern?
3. **Assess exit quality** — could the exit have been better?
4. **Suggest confluence adjustment** — should this pattern type add or subtract confluence points?

Respond with a JSON block:

```json
{{
  "patterns_at_entry": ["list of pattern names visible at entry"],
  "patterns_at_exit": ["list of pattern names visible at exit"],
  "entry_quality": "excellent|good|fair|poor",
  "exit_quality": "excellent|good|fair|poor",
  "confluence_adjustment": 0,
  "reasoning": "One paragraph explaining your analysis",
  "improvement_suggestion": "One actionable suggestion for the strategy"
}}
```

Be specific about what you see in the chart. The green triangle marks entry, red triangle marks exit."""

    return prompt


def parse_visual_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse Claude's visual analysis response into a structured dict.

    Extracts the JSON block from the response. Returns None if parsing fails.
    """
    if not response or not response.strip():
        return None

    # Try ```json block first
    match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if not match:
        # Try raw JSON object
        match = re.search(r"\{[^{}]*\"patterns_at_entry\"[^{}]*\}", response, re.DOTALL)
        if not match:
            logger.warning("Could not parse visual analysis response")
            return None

    try:
        raw = match.group(1) if match.lastindex else match.group(0)
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error in visual response: %s", e)
        return None

    # Validate required fields
    required = {"patterns_at_entry", "entry_quality", "confluence_adjustment"}
    if not required.issubset(result.keys()):
        logger.warning("Missing required fields: %s", required - result.keys())
        return None

    return result


def analyze_screenshot(
    screenshot_path: str,
    trade: Dict[str, Any],
    strategy_name: str,
    patterns: Optional[List] = None,
    cli_command: Optional[List[str]] = None,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Send a trade screenshot to Claude for visual analysis.

    Parameters
    ----------
    screenshot_path: Path to the PNG screenshot file.
    trade: Trade metadata dict.
    strategy_name: Strategy that generated the trade.
    patterns: Optional detected ChartPattern objects near this trade.
    cli_command: CLI command list. Defaults to ["claude", "-p"].
    timeout: Subprocess timeout in seconds.

    Returns
    -------
    Parsed analysis dict, or None on failure.
    """
    path = Path(screenshot_path)
    if not path.exists():
        logger.warning("Screenshot not found: %s", screenshot_path)
        return None

    prompt = build_visual_prompt(trade, strategy_name, patterns)

    # Claude Code CLI supports image input via file attachment
    cmd = cli_command or ["claude", "-p"]
    full_prompt = f"[Image: {screenshot_path}]\n\n{prompt}"

    logger.info("Analyzing screenshot: %s", path.name)

    try:
        result = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        response = result.stdout
        if not response.strip():
            logger.warning("Empty response from visual analysis CLI")
            return None
    except subprocess.TimeoutExpired:
        logger.error("Visual analysis CLI timed out after %ds", timeout)
        return None
    except FileNotFoundError:
        logger.error("CLI command not found: %s", cmd)
        return None

    analysis = parse_visual_response(response)
    if analysis:
        analysis["screenshot_path"] = str(path)
        analysis["trade_r"] = float(trade.get("r_multiple", 0))
        logger.info(
            "Visual analysis complete: entry=%s, exit=%s, adjustment=%+d",
            analysis.get("entry_quality"),
            analysis.get("exit_quality"),
            analysis.get("confluence_adjustment", 0),
        )
    return analysis
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_analyzer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/visual_analyzer.py tests/test_visual_analyzer.py
git commit -m "feat: add Claude visual analysis for trade screenshots (Phase 2 Task 6)"
```

---

### Task 7: Pattern Confluence Edge Filter

**Files:**
- Create: `src/discovery/pattern_confluence.py`
- Test: `tests/test_pattern_confluence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pattern_confluence.py
"""Tests for pattern-based confluence scoring adjustments."""

from datetime import datetime, timezone

import pytest


def _make_edge_context(**overrides):
    """Build a minimal EdgeContext for testing."""
    from src.edges.base import EdgeContext

    defaults = {
        "timestamp": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        "day_of_week": 3,
        "close_price": 2050.0,
        "high_price": 2052.0,
        "low_price": 2048.0,
        "spread": 0.3,
        "session": "london",
        "adx": 30.0,
        "atr": 5.0,
    }
    defaults.update(overrides)
    return EdgeContext(**defaults)


class TestPatternConfluenceScoring:
    def test_bullish_pattern_adds_confluence_for_long(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="double_bottom",
                direction="bullish",
                confidence=0.8,
                key_prices=[2020.0, 2040.0, 2021.0],
                start_index=10,
                end_index=40,
                description="Double bottom",
            )
        ]
        adjustment = scorer.score(patterns, trade_direction="long")
        assert adjustment > 0

    def test_bearish_pattern_subtracts_confluence_for_long(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="head_and_shoulders",
                direction="bearish",
                confidence=0.75,
                key_prices=[2040.0, 2060.0, 2040.0, 2030.0],
                start_index=10,
                end_index=50,
                description="H&S",
            )
        ]
        adjustment = scorer.score(patterns, trade_direction="long")
        assert adjustment < 0

    def test_aligned_pattern_boosts_score(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="falling_wedge",
                direction="bullish",
                confidence=0.9,
                key_prices=[2050.0, 2040.0, 2045.0, 2035.0],
                start_index=10,
                end_index=50,
                description="Falling wedge",
            )
        ]
        # Bullish pattern + long trade = aligned
        aligned = scorer.score(patterns, trade_direction="long")
        # Bullish pattern + short trade = conflicting
        conflicting = scorer.score(patterns, trade_direction="short")
        assert aligned > conflicting

    def test_low_confidence_patterns_contribute_less(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()

        high_conf = [ChartPattern(
            pattern_type="double_bottom", direction="bullish", confidence=0.95,
            key_prices=[], start_index=0, end_index=50, description="",
        )]
        low_conf = [ChartPattern(
            pattern_type="double_bottom", direction="bullish", confidence=0.3,
            key_prices=[], start_index=0, end_index=50, description="",
        )]

        high_adj = scorer.score(high_conf, trade_direction="long")
        low_adj = scorer.score(low_conf, trade_direction="long")
        assert high_adj > low_adj

    def test_no_patterns_returns_zero(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer

        scorer = PatternConfluenceScorer()
        assert scorer.score([], trade_direction="long") == 0

    def test_multiple_patterns_aggregate(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="double_bottom", direction="bullish", confidence=0.8,
                key_prices=[], start_index=0, end_index=30, description="",
            ),
            ChartPattern(
                pattern_type="ascending_triangle", direction="bullish", confidence=0.7,
                key_prices=[], start_index=0, end_index=30, description="",
            ),
        ]
        adj = scorer.score(patterns, trade_direction="long")
        # Two aligned bullish patterns should give more than one
        single = scorer.score(patterns[:1], trade_direction="long")
        assert adj > single


class TestPatternConfluenceEdge:
    def test_edge_filter_allows_when_disabled(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": False},
        )
        ctx = _make_edge_context()
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_edge_filter_allows_with_no_patterns(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": True},
        )
        ctx = _make_edge_context()
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_edge_filter_blocks_on_conflicting_pattern(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge
        from src.discovery.chart_patterns import ChartPattern

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": True, "min_pattern_score": 0, "block_on_conflict": True},
        )
        ctx = _make_edge_context(
            indicator_values={
                "_active_patterns": [
                    ChartPattern(
                        pattern_type="head_and_shoulders",
                        direction="bearish",
                        confidence=0.9,
                        key_prices=[],
                        start_index=0,
                        end_index=50,
                        description="Strong H&S",
                    ).to_dict()
                ],
                "_trade_direction": "long",
            }
        )
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "conflicting" in result.reason.lower() or "bearish" in result.reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pattern_confluence.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PatternConfluenceScorer and PatternConfluenceEdge**

```python
# src/discovery/pattern_confluence.py
"""Pattern-based confluence scoring for the edge filter pipeline.

Detected chart patterns add or subtract confluence points based on:
1. Pattern direction vs trade direction (aligned = bonus, conflicting = penalty)
2. Pattern confidence (high confidence = stronger adjustment)
3. Pattern type importance (H&S > double top > triangle > wedge)

Integrates with the existing EdgeFilter ABC via PatternConfluenceEdge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.edges.base import EdgeContext, EdgeFilter, EdgeResult

logger = logging.getLogger(__name__)

# Pattern type weights: how strongly each pattern type affects confluence
_PATTERN_WEIGHTS: Dict[str, float] = {
    "head_and_shoulders": 2.0,
    "inverse_head_and_shoulders": 2.0,
    "double_top": 1.5,
    "double_bottom": 1.5,
    "ascending_triangle": 1.2,
    "descending_triangle": 1.2,
    "symmetrical_triangle": 0.8,
    "rising_wedge": 1.3,
    "falling_wedge": 1.3,
}

# Direction mapping: what direction each pattern implies
_PATTERN_DIRECTION: Dict[str, str] = {
    "double_top": "bearish",
    "double_bottom": "bullish",
    "head_and_shoulders": "bearish",
    "inverse_head_and_shoulders": "bullish",
    "ascending_triangle": "bullish",
    "descending_triangle": "bearish",
    "symmetrical_triangle": "neutral",
    "rising_wedge": "bearish",
    "falling_wedge": "bullish",
}


class PatternConfluenceScorer:
    """Scores confluence adjustment from detected chart patterns.

    Parameters
    ----------
    max_adjustment:
        Maximum absolute confluence adjustment. Default 3 (on a 0-8 scale).
    confidence_threshold:
        Minimum confidence for a pattern to contribute. Default 0.4.
    """

    def __init__(
        self,
        max_adjustment: int = 3,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._max_adj = max_adjustment
        self._conf_threshold = confidence_threshold

    def score(
        self,
        patterns: List,
        trade_direction: str,
    ) -> int:
        """Calculate confluence adjustment from detected patterns.

        Parameters
        ----------
        patterns: List of ChartPattern objects or dicts with to_dict() output.
        trade_direction: 'long' or 'short'.

        Returns
        -------
        Integer confluence adjustment (can be negative for conflicts).
        """
        if not patterns:
            return 0

        total_score = 0.0
        trade_bias = "bullish" if trade_direction == "long" else "bearish"

        for p in patterns:
            # Support both ChartPattern objects and dicts
            if hasattr(p, "to_dict"):
                d = p.to_dict()
            elif isinstance(p, dict):
                d = p
            else:
                continue

            conf = float(d.get("confidence", 0))
            if conf < self._conf_threshold:
                continue

            pattern_type = d.get("pattern_type", "")
            pattern_dir = d.get("direction", _PATTERN_DIRECTION.get(pattern_type, "neutral"))
            weight = _PATTERN_WEIGHTS.get(pattern_type, 1.0)

            # Aligned: pattern direction matches trade direction
            if pattern_dir == trade_bias:
                total_score += weight * conf
            elif pattern_dir == "neutral":
                total_score += 0.2 * conf  # neutral patterns give a tiny boost
            else:
                # Conflicting: pattern opposes trade direction
                total_score -= weight * conf

        # Round and clamp
        adjustment = int(round(total_score))
        return max(-self._max_adj, min(self._max_adj, adjustment))


class PatternConfluenceEdge(EdgeFilter):
    """EdgeFilter that adjusts or blocks trades based on chart patterns.

    Reads active patterns from EdgeContext.indicator_values['_active_patterns']
    (list of pattern dicts) and the intended trade direction from
    EdgeContext.indicator_values['_trade_direction'].

    Config keys:
    - enabled: bool (default True)
    - block_on_conflict: bool — if True, block entry when patterns strongly
      oppose the trade direction (default False)
    - min_pattern_score: int — minimum net pattern score to allow entry (default -2)
    """

    def __init__(self, name: str, config: dict) -> None:
        super().__init__(name, config)
        self._scorer = PatternConfluenceScorer(
            max_adjustment=config.get("max_adjustment", 3),
            confidence_threshold=config.get("confidence_threshold", 0.4),
        )
        self._block_on_conflict = config.get("block_on_conflict", False)
        self._min_score = config.get("min_pattern_score", -2)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        """Evaluate chart pattern confluence for the proposed trade."""
        if not self.enabled:
            return self._disabled_result()

        iv = context.indicator_values or {}
        patterns_raw = iv.get("_active_patterns", [])
        trade_dir = iv.get("_trade_direction", "long")

        if not patterns_raw:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="No active chart patterns — allowing",
            )

        # Import here to avoid circular dependency
        from src.discovery.chart_patterns import ChartPattern

        # Convert dicts back to ChartPattern if needed
        patterns = []
        for p in patterns_raw:
            if isinstance(p, dict):
                patterns.append(p)
            elif hasattr(p, "to_dict"):
                patterns.append(p.to_dict())
            else:
                patterns.append(p)

        score = self._scorer.score(patterns, trade_direction=trade_dir)

        if self._block_on_conflict and score < self._min_score:
            # Find the strongest conflicting pattern for the reason
            trade_bias = "bullish" if trade_dir == "long" else "bearish"
            conflicting = [
                p for p in patterns
                if p.get("direction", "") != trade_bias and p.get("direction", "") != "neutral"
            ]
            conflict_desc = conflicting[0].get("pattern_type", "unknown") if conflicting else "unknown"
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Conflicting chart pattern: {conflict_desc} "
                    f"({score:+d} score, trade={trade_dir})"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=f"Pattern confluence: {score:+d} (patterns: {len(patterns_raw)})",
            modifier=float(score),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pattern_confluence.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/pattern_confluence.py tests/test_pattern_confluence.py
git commit -m "feat: add pattern confluence edge filter (Phase 2 Task 7)"
```

---

### Task 8: Wire Pattern Detection into Backtest Loop Hook

**Files:**
- Modify: `src/discovery/runner.py` (or create new `src/discovery/pattern_hook.py`)
- Test: `tests/test_pattern_hook.py`

- [ ] **Step 1: Write failing tests for the pattern detection hook**

```python
# tests/test_pattern_hook.py
"""Tests for pattern detection hook that wires into the backtest loop."""

from datetime import datetime, timezone

import pytest


def _make_swing(price, swing_type, index=0):
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestPatternHook:
    def test_detects_patterns_from_swing_history(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),
            _make_swing(2025.0, "low", index=40),
        ]
        patterns = hook.detect(swings)
        assert isinstance(patterns, list)
        # Should find the double top
        assert any(p.pattern_type == "double_top" for p in patterns)

    def test_returns_empty_with_insufficient_swings(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        patterns = hook.detect([_make_swing(2050.0, "high", index=10)])
        assert patterns == []

    def test_updates_atr(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        hook.update_atr(10.0)
        # With higher ATR, the tolerance is wider
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2058.0, "high", index=30),  # 8 points apart — within 10*1.5=15 tol
            _make_swing(2025.0, "low", index=40),
        ]
        patterns = hook.detect(swings)
        assert any(p.pattern_type == "double_top" for p in patterns)

    def test_get_confluence_adjustment(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),
            _make_swing(2025.0, "low", index=40),
        ]
        patterns = hook.detect(swings)
        # Double top is bearish — should penalize longs
        adj = hook.get_confluence_adjustment(patterns, trade_direction="long")
        assert adj < 0

        # Double top is bearish — should boost shorts
        adj_short = hook.get_confluence_adjustment(patterns, trade_direction="short")
        assert adj_short > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pattern_hook.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PatternHook**

```python
# src/discovery/pattern_hook.py
"""Pattern detection hook for the backtest loop.

Provides a simple interface for the backtester to call on each bar
(or on each new swing point) to detect chart patterns and compute
confluence adjustments. Strategy-agnostic — works with any strategy
that produces swing points.

Usage in the backtest loop:
    pattern_hook = PatternHook(atr=current_atr)
    # After swing detection:
    patterns = pattern_hook.detect(swing_history)
    adjustment = pattern_hook.get_confluence_adjustment(patterns, "long")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.discovery.chart_patterns import ChartPattern, PatternDetector
from src.discovery.pattern_confluence import PatternConfluenceScorer

logger = logging.getLogger(__name__)


class PatternHook:
    """Lightweight hook for wiring pattern detection into any backtest loop.

    Parameters
    ----------
    atr:
        Current ATR for the instrument. Updated via update_atr().
    tolerance_atr_mult:
        Tolerance multiplier for pattern detection. Default 1.5.
    min_swings_for_scan:
        Minimum swing points before attempting pattern detection. Default 4.
    """

    def __init__(
        self,
        atr: float = 5.0,
        tolerance_atr_mult: float = 1.5,
        min_swings_for_scan: int = 4,
    ) -> None:
        self._atr = max(atr, 0.01)
        self._tolerance_mult = tolerance_atr_mult
        self._min_swings = min_swings_for_scan
        self._detector = PatternDetector(
            atr=self._atr,
            tolerance_atr_mult=self._tolerance_mult,
        )
        self._scorer = PatternConfluenceScorer()
        self._last_patterns: List[ChartPattern] = []

    def update_atr(self, atr: float) -> None:
        """Update ATR and rebuild the detector with new tolerance."""
        self._atr = max(atr, 0.01)
        self._detector = PatternDetector(
            atr=self._atr,
            tolerance_atr_mult=self._tolerance_mult,
        )

    def detect(self, swings: List) -> List[ChartPattern]:
        """Run pattern detection on the current swing history.

        Parameters
        ----------
        swings: List of SwingPoint objects (from BreathingRoomDetector).

        Returns
        -------
        List of detected ChartPattern objects, sorted by confidence.
        """
        if len(swings) < self._min_swings:
            return []

        # Only scan the most recent 20 swings to avoid stale pattern matches
        recent = swings[-20:] if len(swings) > 20 else swings
        self._last_patterns = self._detector.detect_all(recent)
        return self._last_patterns

    def get_confluence_adjustment(
        self,
        patterns: Optional[List[ChartPattern]] = None,
        trade_direction: str = "long",
    ) -> int:
        """Get confluence score adjustment from detected patterns.

        Parameters
        ----------
        patterns: Patterns to score. Defaults to last detect() result.
        trade_direction: 'long' or 'short'.

        Returns
        -------
        Integer confluence adjustment (-3 to +3).
        """
        pts = patterns if patterns is not None else self._last_patterns
        return self._scorer.score(pts, trade_direction=trade_direction)

    @property
    def last_patterns(self) -> List[ChartPattern]:
        """Most recently detected patterns."""
        return self._last_patterns

    def patterns_as_dicts(self) -> List[Dict[str, Any]]:
        """Serialize last detected patterns to dicts for EdgeContext injection."""
        return [p.to_dict() for p in self._last_patterns]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pattern_hook.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/pattern_hook.py tests/test_pattern_hook.py
git commit -m "feat: add pattern detection hook for backtest loop (Phase 2 Task 8)"
```

---

### Task 9: Integration Test — Full Phase 2 Pipeline

**Files:**
- Test: `tests/test_phase2_integration.py`
- Modify: `src/discovery/__init__.py` (update exports)

- [ ] **Step 1: Write integration test**

```python
# tests/test_phase2_integration.py
"""Integration test: pattern detection -> screenshot selection -> visual analysis -> confluence."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _make_swing(price, swing_type, index=0):
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestPhase2Pipeline:
    def test_pattern_detection_to_confluence(self):
        """Patterns detected -> confluence scored -> EdgeContext injectable."""
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)

        # Build a swing history with a clear double bottom
        swings = [
            _make_swing(2020.0, "low", index=10),
            _make_swing(2040.0, "high", index=20),
            _make_swing(2019.5, "low", index=30),
            _make_swing(2045.0, "high", index=40),
        ]

        patterns = hook.detect(swings)
        assert len(patterns) >= 1
        assert any(p.pattern_type == "double_bottom" for p in patterns)

        # Confluence should boost longs (double bottom is bullish)
        adj_long = hook.get_confluence_adjustment(patterns, "long")
        assert adj_long > 0

        # Confluence should penalize shorts
        adj_short = hook.get_confluence_adjustment(patterns, "short")
        assert adj_short < 0

        # Serializable for EdgeContext
        dicts = hook.patterns_as_dicts()
        assert isinstance(dicts, list)
        assert all(isinstance(d, dict) for d in dicts)

    def test_screenshot_selection_with_trades(self):
        """Trade list -> selector picks worst/best -> no crashes."""
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=3, n_best=3)
        trades = []
        for i in range(20):
            r = float(np.random.default_rng(i).uniform(-2.0, 3.0))
            trades.append({
                "r_multiple": r,
                "entry_bar_idx": i * 50,
                "exit_bar_idx": i * 50 + 30,
                "entry_price": 2050.0,
                "exit_price": 2050.0 + r * 5.0,
                "stop_loss": 2045.0,
                "direction": "long",
                "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
            })

        selected = selector.select_trades(trades)
        assert 3 <= len(selected) <= 10  # at most n_worst + n_best + near_SL
        reasons = {t["_selection_reason"] for t in selected}
        assert "worst_loser" in reasons
        assert "best_winner" in reasons

    def test_visual_prompt_and_response_round_trip(self):
        """Build prompt -> mock response -> parse -> valid result."""
        from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

        trade = {
            "r_multiple": -1.5,
            "direction": "long",
            "entry_price": 2050.0,
            "exit_price": 2042.5,
            "stop_loss": 2045.0,
            "_selection_reason": "worst_loser",
            "exit_reason": "stop_loss",
        }
        prompt = build_visual_prompt(trade, strategy_name="sss")
        assert len(prompt) > 100

        # Simulate Claude response
        mock_response = '''Analysis:

```json
{
  "patterns_at_entry": ["rising_wedge"],
  "patterns_at_exit": ["support_break"],
  "entry_quality": "poor",
  "exit_quality": "fair",
  "confluence_adjustment": -1,
  "reasoning": "Entry was against the rising wedge pattern. Should have waited for breakout.",
  "improvement_suggestion": "Add wedge detection to filter entries against wedge direction."
}
```
'''
        result = parse_visual_response(mock_response)
        assert result is not None
        assert result["entry_quality"] == "poor"
        assert result["confluence_adjustment"] == -1

    def test_knowledge_base_stores_pattern_findings(self):
        """Detected patterns persist to knowledge base for future reference."""
        from src.discovery.knowledge_base import KnowledgeBase

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        kb = KnowledgeBase(base_dir=kb_dir)

        # Save pattern-derived hypothesis
        hyp = {
            "id": "hyp_pattern_001",
            "description": "Double bottom at 2020 support consistently leads to 2R+ winners",
            "source": "chart_pattern_visual",
            "strategy": "sss",
            "status": "proposed",
            "evidence": {
                "pattern_type": "double_bottom",
                "occurrences": 5,
                "avg_r_when_aligned": 1.8,
                "avg_r_when_opposed": -0.7,
            },
        }
        kb.save_hypothesis(hyp)
        loaded = kb.load_hypothesis("hyp_pattern_001")
        assert loaded["source"] == "chart_pattern_visual"
        assert loaded["evidence"]["pattern_type"] == "double_bottom"

    def test_full_pipeline_no_crashes(self):
        """End-to-end: swings -> patterns -> selection -> prompt -> parse.

        Verifies the entire pipeline runs without exceptions,
        even without actual Claude CLI calls.
        """
        from src.discovery.pattern_hook import PatternHook
        from src.discovery.screenshot_selector import ScreenshotSelector
        from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

        # 1. Detect patterns
        hook = PatternHook(atr=5.0)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2055.0, "high", index=30),
            _make_swing(2025.0, "low", index=35),
            _make_swing(2041.0, "high", index=45),
            _make_swing(2020.0, "low", index=55),
        ]
        patterns = hook.detect(swings)

        # 2. Select trades
        selector = ScreenshotSelector(n_worst=2, n_best=2)
        trades = [
            {
                "r_multiple": -1.5, "entry_bar_idx": 20, "exit_bar_idx": 50,
                "entry_price": 2050.0, "exit_price": 2042.5, "stop_loss": 2045.0,
                "direction": "long", "_selection_reason": "worst_loser",
                "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
            },
            {
                "r_multiple": 2.0, "entry_bar_idx": 30, "exit_bar_idx": 55,
                "entry_price": 2030.0, "exit_price": 2040.0, "stop_loss": 2025.0,
                "direction": "long", "_selection_reason": "best_winner",
                "entry_time": datetime(2026, 1, 16, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 16, 14, 0, tzinfo=timezone.utc),
            },
        ]
        selected = selector.select_trades(trades)
        assert len(selected) >= 1

        # 3. Build prompts
        for trade in selected:
            prompt = build_visual_prompt(trade, strategy_name="sss", patterns=patterns)
            assert len(prompt) > 50

        # 4. Get confluence
        adj = hook.get_confluence_adjustment(patterns, "long")
        assert isinstance(adj, int)
```

- [ ] **Step 2: Update `src/discovery/__init__.py` with Phase 2 exports**

```python
# src/discovery/__init__.py
"""Creative Pattern Discovery Agent.

Strategy-agnostic edge discovery via XGBoost/SHAP analysis,
chart pattern detection, visual analysis, and validated rule absorption.
"""

from src.discovery.xgb_analyzer import build_training_data
from src.discovery.chart_patterns import ChartPattern, PatternDetector
from src.discovery.pattern_hook import PatternHook
from src.discovery.pattern_confluence import PatternConfluenceScorer, PatternConfluenceEdge
from src.discovery.screenshot_selector import ScreenshotSelector
from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

__all__ = [
    "build_training_data",
    "ChartPattern",
    "PatternDetector",
    "PatternHook",
    "PatternConfluenceScorer",
    "PatternConfluenceEdge",
    "ScreenshotSelector",
    "build_visual_prompt",
    "parse_visual_response",
]
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/test_chart_patterns.py tests/test_screenshot_selector.py tests/test_visual_analyzer.py tests/test_pattern_confluence.py tests/test_pattern_hook.py tests/test_phase2_integration.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/discovery/__init__.py src/discovery/chart_patterns.py src/discovery/screenshot_selector.py src/discovery/visual_analyzer.py src/discovery/pattern_confluence.py src/discovery/pattern_hook.py tests/test_chart_patterns.py tests/test_screenshot_selector.py tests/test_visual_analyzer.py tests/test_pattern_confluence.py tests/test_pattern_hook.py tests/test_phase2_integration.py
git commit -m "feat: complete Phase 2 — chart pattern detection + visual analysis pipeline"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | ChartPattern dataclass | `chart_patterns.py` | 2 tests |
| 2 | Double top/bottom detection | `chart_patterns.py` | 4 tests |
| 3 | Head & shoulders detection | `chart_patterns.py` | 3 tests |
| 4 | Triangle and wedge detection | `chart_patterns.py` | 5 tests |
| 5 | Screenshot selector | `screenshot_selector.py` | 5 tests |
| 6 | Visual analyzer (Claude) | `visual_analyzer.py` | 6 tests |
| 7 | Pattern confluence edge | `pattern_confluence.py` | 9 tests |
| 8 | Pattern detection hook | `pattern_hook.py` | 4 tests |
| 9 | Integration test | `__init__.py` update | 5 tests |

**Total: 9 tasks, 43 tests, 5 new source files, 1 modified source file.**

Phase 3 plan (Macro Regime) will be written separately once Phase 2 is validated.
