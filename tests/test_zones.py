"""
Unit tests for the zone detection system.

Covers:
- Swing point detection (fractal algorithm)
- DBSCAN S/R clustering with ATR-based epsilon
- Supply/demand zone detection from impulse moves
- Standard and Fibonacci pivot point calculations
- Confluence density scoring
- Zone lifecycle (test → invalidate flow)
- ZoneManager: add, query, update status, maintenance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.zones.detector import SwingPointDetector, SwingPoints
from src.zones.sr_clusters import SRClusterDetector, SRZone
from src.zones.supply_demand import SupplyDemandDetector, SDZone
from src.zones.pivots import PivotCalculator
from src.zones.confluence_density import ConfluenceDensityScorer, ConfluenceScore
from src.zones.manager import ZoneManager


# =============================================================================
# Shared helpers
# =============================================================================

RNG = np.random.default_rng(42)


def _make_simple_ohlc(n: int = 100, base: float = 1800.0) -> pd.DataFrame:
    """Flat-ish OHLC with small random noise."""
    close = base + RNG.uniform(-5, 5, n).cumsum() * 0.1
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "timestamp": ts})


def _make_candle(open_=1800.0, high=1805.0, low=1795.0, close=1802.0, ts=None):
    """Create a simple candle dict."""
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "timestamp": ts or datetime(2024, 1, 1),
    }


@dataclass
class SimpleZone:
    """Minimal zone object compatible with ZoneManager and ConfluenceDensityScorer."""
    price_high: float
    price_low: float
    zone_type: str
    center: float = 0.0
    timeframe: str = "H1"
    status: str = "active"
    touch_count: int = 1
    strength: float = 1.0
    last_touched: datetime = None
    last_tested: datetime = None

    def __post_init__(self):
        if self.center == 0.0:
            self.center = (self.price_high + self.price_low) / 2.0
        if self.last_touched is None:
            self.last_touched = datetime(2024, 1, 1)
        if self.last_tested is None:
            self.last_tested = datetime(2024, 1, 1)


# =============================================================================
# 1. Swing Point Detection
# =============================================================================

class TestSwingPointDetector:
    """Validate fractal swing high/low detection on synthetic data."""

    def _make_known_peaks(self):
        """Build a price series with known swing highs and lows.

        Structure (indices):
            - Swing high at bar 10 (value 1820)
            - Swing low  at bar 20 (value 1780)
            - Swing high at bar 30 (value 1815)
            - Swing low  at bar 40 (value 1785)
        """
        n = 60
        high = np.full(n, 1800.0)
        low = np.full(n, 1800.0)

        # Swing high at bar 10
        for i in range(5, 16):
            dist = abs(i - 10)
            high[i] = 1800.0 + (10.0 - dist) * 2  # peak at 1820
            low[i] = high[i] - 2.0

        # Swing low at bar 20
        for i in range(15, 26):
            dist = abs(i - 20)
            low[i] = 1800.0 - (10.0 - dist) * 2   # trough at 1780
            high[i] = low[i] + 2.0

        # Swing high at bar 30
        for i in range(25, 36):
            dist = abs(i - 30)
            high[i] = 1800.0 + (10.0 - dist) * 1.5  # peak at 1815
            low[i] = high[i] - 2.0

        # Swing low at bar 40
        for i in range(35, 46):
            dist = abs(i - 40)
            low[i] = 1800.0 - (10.0 - dist) * 1.5   # trough at 1785
            high[i] = low[i] + 2.0

        return high, low

    def test_detects_known_swing_high(self):
        high, low = self._make_known_peaks()
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)

        sh_indices = [sp[0] for sp in result.swing_highs]
        assert 10 in sh_indices, f"Expected swing high at bar 10, got {sh_indices}"

    def test_detects_known_swing_low(self):
        high, low = self._make_known_peaks()
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)

        sl_indices = [sp[0] for sp in result.swing_lows]
        assert 20 in sl_indices, f"Expected swing low at bar 20, got {sl_indices}"

    def test_detects_multiple_swing_highs(self):
        high, low = self._make_known_peaks()
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)
        assert len(result.swing_highs) >= 2, "Expected at least 2 swing highs"

    def test_detects_multiple_swing_lows(self):
        high, low = self._make_known_peaks()
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)
        assert len(result.swing_lows) >= 2, "Expected at least 2 swing lows"

    def test_no_swing_on_monotone_up(self):
        """A strictly rising series has no swing highs (no bar higher than all neighbours)."""
        n = 50
        high = np.arange(1800.0, 1800.0 + n, dtype=float)
        low = high - 1.0
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)
        # No bar can be a swing high on a strictly monotone series
        assert len(result.swing_highs) == 0

    def test_timestamps_attached(self):
        high, low = self._make_known_peaks()
        ts = np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "h") for i in range(len(high))])
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low, timestamps=ts)

        # Each swing high tuple should have 3 elements: (index, price, timestamp)
        for sp in result.swing_highs:
            assert len(sp) == 3, f"Expected 3-tuple with timestamp, got {sp}"

    def test_vectorized_matches_detect(self):
        high, low = self._make_known_peaks()
        detector = SwingPointDetector(lookback=5)
        sh_mask, sl_mask = detector.detect_vectorized(high, low)
        result = detector.detect(high, low)

        sh_indices_mask = set(np.where(sh_mask)[0])
        sh_indices_result = {sp[0] for sp in result.swing_highs}
        assert sh_indices_mask == sh_indices_result

    def test_lookback_1_more_swing_points(self):
        """lookback=1 should detect more swing points than lookback=10."""
        high, low = self._make_known_peaks()
        det1 = SwingPointDetector(lookback=1)
        det10 = SwingPointDetector(lookback=10)
        r1 = det1.detect(high, low)
        r10 = det10.detect(high, low)
        assert len(r1.swing_highs) >= len(r10.swing_highs)

    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            SwingPointDetector(lookback=0)

    def test_swing_points_dataclass_defaults(self):
        sp = SwingPoints()
        assert sp.swing_highs == []
        assert sp.swing_lows == []

    def test_output_indices_within_bounds(self):
        n = 40
        high = 1800.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        low = high - 2.0
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)
        for idx, price in result.swing_highs:
            assert 0 <= idx < n
        for idx, price in result.swing_lows:
            assert 0 <= idx < n

    def test_short_series_no_crash(self):
        """Series shorter than 2×lookback+1 should return empty swing points."""
        high = np.array([1800.0, 1810.0, 1805.0])
        low = high - 2.0
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)
        assert len(result.swing_highs) == 0
        assert len(result.swing_lows) == 0


# =============================================================================
# 2. S/R Cluster Detection (DBSCAN)
# =============================================================================

class TestSRClusterDetector:
    """DBSCAN-based S/R zone clustering with ATR-based epsilon."""

    def test_nearby_points_cluster_together(self):
        """Points at 1000, 1000.5, 1001 should form one cluster with ATR=5."""
        prices = np.array([1000.0, 1000.5, 1001.0])
        atr = 5.0  # epsilon = 5.0 * 1.0 = 5.0 → all points within epsilon
        detector = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
        zones = detector.cluster(prices, atr, zone_type="support")
        assert len(zones) == 1, f"Expected 1 cluster, got {len(zones)}"
        assert zones[0].touch_count == 3

    def test_distant_points_separate_clusters(self):
        """Points at 1000 and 1050 with ATR=5 should not cluster (distance 50 > epsilon 5)."""
        prices = np.array([1000.0, 1050.0])
        atr = 5.0
        # Need min_touches=1 so each isolated point forms its own cluster
        detector = SRClusterDetector(atr_multiplier=1.0, min_touches=1)
        zones = detector.cluster(prices, atr, zone_type="resistance")
        assert len(zones) == 2, f"Expected 2 separate clusters, got {len(zones)}"

    def test_noise_points_discarded(self):
        """Isolated points not meeting min_touches are discarded as noise."""
        # 3 tight points + 1 isolated outlier
        prices = np.array([1000.0, 1000.3, 1000.6, 1500.0])
        atr = 2.0
        detector = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
        zones = detector.cluster(prices, atr, zone_type="support")
        centers = [z.center for z in zones]
        # The cluster near 1000 should be found; the isolated 1500 should not
        assert any(abs(c - 1000.3) < 1.0 for c in centers), "Expected cluster near 1000"
        assert not any(abs(c - 1500.0) < 1.0 for c in centers), "Isolated point should be noise"

    def test_zone_boundaries(self):
        """Zone price_high and price_low should match cluster min/max."""
        prices = np.array([1000.0, 1001.0, 1002.0])
        atr = 5.0
        detector = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
        zones = detector.cluster(prices, atr, zone_type="support")
        assert len(zones) == 1
        assert zones[0].price_low == pytest.approx(1000.0)
        assert zones[0].price_high == pytest.approx(1002.0)

    def test_empty_input_returns_empty(self):
        detector = SRClusterDetector()
        zones = detector.cluster(np.array([]), atr=5.0)
        assert zones == []

    def test_timestamps_assigned(self):
        """Timestamps should be attached when provided."""
        prices = np.array([1000.0, 1001.0, 1002.0])
        ts = [datetime(2024, 1, i + 1) for i in range(3)]
        atr = 5.0
        detector = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
        zones = detector.cluster(prices, atr, timestamps=ts, zone_type="support")
        assert zones[0].first_seen == datetime(2024, 1, 1)
        assert zones[0].last_touched == datetime(2024, 1, 3)

    def test_score_zone_increases_with_touch_count(self):
        """Higher touch count should produce higher strength."""
        detector = SRClusterDetector()
        now = datetime(2024, 6, 1)
        z2 = SRZone(1800.0, 1799.0, 1799.5, 2, "H1", now, now, "support")
        z5 = SRZone(1800.0, 1799.0, 1799.5, 5, "H1", now, now, "support")
        score2 = detector.score_zone(z2, 1800.0, now, 5.0)
        score5 = detector.score_zone(z5, 1800.0, now, 5.0)
        assert score5 > score2

    def test_score_zone_decreases_with_distance(self):
        """Zones farther from current price should score lower."""
        detector = SRClusterDetector()
        now = datetime(2024, 6, 1)
        z_near = SRZone(1801.0, 1799.0, 1800.0, 3, "H1", now, now, "support")
        z_far = SRZone(1850.0, 1848.0, 1849.0, 3, "H1", now, now, "support")
        score_near = detector.score_zone(z_near, 1800.0, now, 5.0)
        score_far = detector.score_zone(z_far, 1800.0, now, 5.0)
        assert score_near > score_far

    def test_invalid_multiplier_raises(self):
        with pytest.raises(ValueError):
            SRClusterDetector(atr_multiplier=0)

    def test_invalid_min_touches_raises(self):
        with pytest.raises(ValueError):
            SRClusterDetector(min_touches=0)


# =============================================================================
# 3. Supply/Demand Zone Detection
# =============================================================================

class TestSupplyDemandDetector:
    """Supply/demand zone detection from impulse moves."""

    def _make_demand_setup(self, n: int = 50, base: float = 1800.0) -> tuple[pd.DataFrame, np.ndarray]:
        """Flat consolidation followed by a strong bullish impulse."""
        opens = np.full(n, base)
        highs = np.full(n, base + 1.0)
        lows = np.full(n, base - 1.0)
        closes = np.full(n, base)

        # Place a 3-candle base at bars 20-22
        # followed by a large bullish candle at bar 23
        impulse_size = 15.0  # 3× ATR of 5
        opens[23] = base
        highs[23] = base + impulse_size + 1.0
        lows[23] = base - 1.0
        closes[23] = base + impulse_size  # big bullish candle

        ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "timestamp": ts
        })
        atr = np.full(n, 5.0)
        return df, atr

    def _make_supply_setup(self, n: int = 50, base: float = 1800.0) -> tuple[pd.DataFrame, np.ndarray]:
        """Flat consolidation followed by a strong bearish impulse."""
        opens = np.full(n, base)
        highs = np.full(n, base + 1.0)
        lows = np.full(n, base - 1.0)
        closes = np.full(n, base)

        impulse_size = 15.0
        opens[23] = base
        highs[23] = base + 1.0
        lows[23] = base - impulse_size - 1.0
        closes[23] = base - impulse_size  # big bearish candle

        ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "timestamp": ts
        })
        atr = np.full(n, 5.0)
        return df, atr

    def test_detects_demand_zone(self):
        df, atr = self._make_demand_setup()
        detector = SupplyDemandDetector(impulse_threshold=2.5, consolidation_bars=3)
        zones = detector.detect(df, atr)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1, f"Expected at least 1 demand zone, got {len(demand_zones)}"

    def test_detects_supply_zone(self):
        df, atr = self._make_supply_setup()
        detector = SupplyDemandDetector(impulse_threshold=2.5, consolidation_bars=3)
        zones = detector.detect(df, atr)
        supply_zones = [z for z in zones if z.zone_type == "supply"]
        assert len(supply_zones) >= 1, f"Expected at least 1 supply zone, got {len(supply_zones)}"

    def test_zone_boundaries_within_base(self):
        """Zone boundaries should enclose the base candles, not the impulse."""
        df, atr = self._make_demand_setup()
        detector = SupplyDemandDetector(impulse_threshold=2.5, consolidation_bars=3)
        zones = detector.detect(df, atr)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        if demand_zones:
            zone = demand_zones[0]
            # Zone should not extend into the large impulse candle
            assert zone.price_high < 1810.0, (
                f"Zone high ({zone.price_high}) should not include the impulse bar"
            )

    def test_default_status_is_active(self):
        df, atr = self._make_demand_setup()
        detector = SupplyDemandDetector()
        zones = detector.detect(df, atr)
        for zone in zones:
            assert zone.status == "active"

    def test_impulse_size_recorded(self):
        df, atr = self._make_demand_setup()
        detector = SupplyDemandDetector(impulse_threshold=2.5)
        zones = detector.detect(df, atr)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        if demand_zones:
            # Impulse = 15, ATR = 5 → impulse_size = 3.0
            assert demand_zones[0].impulse_size >= 2.5

    def test_nan_atr_skipped(self):
        """Bars with NaN ATR should be skipped without crash."""
        df, atr = self._make_demand_setup()
        atr[:15] = np.nan  # NaN warmup
        detector = SupplyDemandDetector()
        zones = detector.detect(df, atr)  # should not raise

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1800.0, 1801.0]})
        atr = np.array([5.0, 5.0])
        detector = SupplyDemandDetector()
        with pytest.raises(ValueError, match="ohlc must contain"):
            detector.detect(df, atr)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SupplyDemandDetector(impulse_threshold=0)

    def test_sdzone_fields(self):
        """SDZone dataclass fields should be accessible."""
        z = SDZone(
            price_high=1810.0,
            price_low=1800.0,
            zone_type="demand",
            impulse_size=3.0,
            base_bars=2,
            timestamp=datetime(2024, 1, 1),
        )
        assert z.status == "active"
        assert z.zone_type == "demand"
        assert z.base_bars == 2


# =============================================================================
# 4. Pivot Points
# =============================================================================

class TestPivotCalculator:
    """Standard and Fibonacci pivot point calculations."""

    H = 1900.0
    L = 1850.0
    C = 1880.0
    PP = (H + L + C) / 3.0  # 1876.666...

    def test_standard_pp(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        assert pivots["PP"] == pytest.approx(self.PP, rel=1e-9)

    def test_standard_r1(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_r1 = 2.0 * self.PP - self.L
        assert pivots["R1"] == pytest.approx(expected_r1, rel=1e-9)

    def test_standard_s1(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_s1 = 2.0 * self.PP - self.H
        assert pivots["S1"] == pytest.approx(expected_s1, rel=1e-9)

    def test_standard_r2(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_r2 = self.PP + (self.H - self.L)
        assert pivots["R2"] == pytest.approx(expected_r2, rel=1e-9)

    def test_standard_s2(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_s2 = self.PP - (self.H - self.L)
        assert pivots["S2"] == pytest.approx(expected_s2, rel=1e-9)

    def test_standard_r3(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_r3 = self.H + 2.0 * (self.PP - self.L)
        assert pivots["R3"] == pytest.approx(expected_r3, rel=1e-9)

    def test_standard_s3(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        expected_s3 = self.L - 2.0 * (self.H - self.PP)
        assert pivots["S3"] == pytest.approx(expected_s3, rel=1e-9)

    def test_standard_level_ordering(self):
        """R levels should be above PP; S levels below PP."""
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        pp = pivots["PP"]
        assert pivots["R1"] > pp
        assert pivots["R2"] > pivots["R1"]
        assert pivots["R3"] > pivots["R2"]
        assert pivots["S1"] < pp
        assert pivots["S2"] < pivots["S1"]
        assert pivots["S3"] < pivots["S2"]

    def test_fibonacci_pp(self):
        calc = PivotCalculator()
        pivots = calc.fibonacci_pivots(self.H, self.L, self.C)
        assert pivots["PP"] == pytest.approx(self.PP, rel=1e-9)

    def test_fibonacci_r1(self):
        calc = PivotCalculator()
        pivots = calc.fibonacci_pivots(self.H, self.L, self.C)
        expected = self.PP + 0.382 * (self.H - self.L)
        assert pivots["R1"] == pytest.approx(expected, rel=1e-9)

    def test_fibonacci_s2(self):
        calc = PivotCalculator()
        pivots = calc.fibonacci_pivots(self.H, self.L, self.C)
        expected = self.PP - 0.618 * (self.H - self.L)
        assert pivots["S2"] == pytest.approx(expected, rel=1e-9)

    def test_fibonacci_r3_equals_pp_plus_range(self):
        """Fib R3 = PP + 1.0 × (H − L), which equals PP + full range."""
        calc = PivotCalculator()
        pivots = calc.fibonacci_pivots(self.H, self.L, self.C)
        expected = self.PP + 1.0 * (self.H - self.L)
        assert pivots["R3"] == pytest.approx(expected, rel=1e-9)

    def test_standard_output_keys(self):
        calc = PivotCalculator()
        pivots = calc.standard_pivots(self.H, self.L, self.C)
        assert set(pivots.keys()) == {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}

    def test_fibonacci_output_keys(self):
        calc = PivotCalculator()
        pivots = calc.fibonacci_pivots(self.H, self.L, self.C)
        assert set(pivots.keys()) == {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}

    def test_from_daily_ohlc(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "open": [1800.0] * 5,
            "high": [1900.0, 1910.0, 1920.0, 1930.0, 1940.0],
            "low":  [1850.0, 1860.0, 1870.0, 1880.0, 1890.0],
            "close": [1880.0, 1890.0, 1900.0, 1910.0, 1920.0],
        }, index=dates)
        calc = PivotCalculator()
        result = calc.from_daily_ohlc(df)
        assert len(result) == 5
        assert set(result.columns) == {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}
        # Verify first row matches manual calculation
        expected_pp = (1900.0 + 1850.0 + 1880.0) / 3.0
        assert result.iloc[0]["PP"] == pytest.approx(expected_pp, rel=1e-9)

    def test_from_weekly_ohlc(self):
        df = pd.DataFrame({
            "high": [2000.0, 2050.0],
            "low": [1900.0, 1950.0],
            "close": [1980.0, 2020.0],
        })
        calc = PivotCalculator()
        result = calc.from_weekly_ohlc(df)
        assert len(result) == 2

    def test_from_monthly_ohlc(self):
        df = pd.DataFrame({
            "high": [2100.0],
            "low": [1900.0],
            "close": [2000.0],
        })
        calc = PivotCalculator()
        result = calc.from_monthly_ohlc(df)
        assert len(result) == 1
        expected_pp = (2100.0 + 1900.0 + 2000.0) / 3.0
        assert result.iloc[0]["PP"] == pytest.approx(expected_pp)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1800.0]})
        calc = PivotCalculator()
        with pytest.raises(ValueError):
            calc.from_daily_ohlc(df)


# =============================================================================
# 5. Confluence Density Scoring
# =============================================================================

class TestConfluenceDensityScorer:
    """Count and score zone confluence near a reference price."""

    def _make_sr_zone(self, center: float, tf: str = "H1") -> SimpleZone:
        return SimpleZone(
            price_high=center + 2.0,
            price_low=center - 2.0,
            zone_type="support",
            center=center,
            timeframe=tf,
        )

    def _make_sd_zone(self, center: float, tf: str = "H4") -> SimpleZone:
        return SimpleZone(
            price_high=center + 3.0,
            price_low=center - 3.0,
            zone_type="demand",
            center=center,
            timeframe=tf,
        )

    def test_counts_sr_zone_within_proximity(self):
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        zone = self._make_sr_zone(1800.0)
        result = scorer.score(1800.0, [zone], {}, atr=5.0)
        assert result.sr_zones == 1
        assert result.total == 1

    def test_counts_sd_zone_within_proximity(self):
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        zone = self._make_sd_zone(1800.0)
        result = scorer.score(1800.0, [zone], {}, atr=5.0)
        assert result.sd_zones == 1

    def test_counts_pivot_level_within_proximity(self):
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        pivots = {"PP": 1800.0, "R1": 1810.0}
        # With ATR=5 and proximity=1.0, window = 5 points. PP at 1800 is exact match.
        result = scorer.score(1800.0, [], pivots, atr=5.0)
        assert result.pivot_levels >= 1  # PP at 1800 should be counted

    def test_distant_zone_excluded(self):
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        zone = self._make_sr_zone(2000.0)  # 200 points away
        result = scorer.score(1800.0, [zone], {}, atr=5.0)
        assert result.sr_zones == 0

    def test_multi_tf_count(self):
        scorer = ConfluenceDensityScorer(proximity_atr=2.0)
        zones = [
            self._make_sr_zone(1800.0, tf="H1"),
            self._make_sr_zone(1802.0, tf="H4"),
            self._make_sr_zone(1801.0, tf="D1"),
        ]
        result = scorer.score(1800.0, zones, {}, atr=10.0)
        assert result.multi_tf_count == 3

    def test_details_populated(self):
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        zone = self._make_sr_zone(1800.0)
        pivots = {"S1": 1800.5}
        result = scorer.score(1800.0, [zone, ], pivots, atr=5.0)
        types = [d["type"] for d in result.details]
        assert "sr_zone" in types

    def test_price_inside_zone_counts(self):
        """Price inside a zone boundary should count as within proximity."""
        scorer = ConfluenceDensityScorer(proximity_atr=0.0001)  # very tight window
        zone = SimpleZone(price_high=1810.0, price_low=1790.0, zone_type="support", center=1800.0)
        result = scorer.score(1800.0, [zone], {}, atr=5.0)
        assert result.sr_zones == 1

    def test_invalid_proximity_raises(self):
        with pytest.raises(ValueError):
            ConfluenceDensityScorer(proximity_atr=0)

    def test_empty_inputs(self):
        scorer = ConfluenceDensityScorer()
        result = scorer.score(1800.0, [], {}, atr=5.0)
        assert result.total == 0
        assert result.sr_zones == 0
        assert result.sd_zones == 0
        assert result.pivot_levels == 0

    def test_total_equals_sum_of_types(self):
        scorer = ConfluenceDensityScorer(proximity_atr=2.0)
        zones = [
            self._make_sr_zone(1800.0),
            self._make_sd_zone(1800.5),
        ]
        pivots = {"PP": 1801.0}
        result = scorer.score(1800.0, zones, pivots, atr=5.0)
        assert result.total == result.sr_zones + result.sd_zones + result.pivot_levels

    def test_confluence_score_dataclass_defaults(self):
        cs = ConfluenceScore()
        assert cs.total == 0
        assert cs.details == []


# =============================================================================
# 6. Zone Lifecycle (test → invalidate flow)
# =============================================================================

class TestZoneLifecycle:
    """Verify zone status transitions through the test/invalidate flow."""

    def _make_zone(self, zone_type: str = "support", center: float = 1800.0) -> SimpleZone:
        return SimpleZone(
            price_high=center + 5.0,
            price_low=center - 5.0,
            zone_type=zone_type,
            center=center,
        )

    def test_initial_status_active(self):
        zone = self._make_zone()
        assert zone.status == "active"

    def test_wick_into_zone_marks_tested(self):
        """A candle whose wick enters the zone marks it as 'tested'."""
        zone = self._make_zone("support", center=1800.0)
        # Zone: 1795 – 1805.  Candle wicks into zone but closes above.
        candle = _make_candle(open_=1810.0, high=1815.0, low=1797.0, close=1808.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.update_zone(zone_id, candle)
        assert zone.status == "tested"

    def test_body_close_below_support_invalidates(self):
        """Body closing below a support zone invalidates it."""
        zone = self._make_zone("support", center=1800.0)
        # Zone: 1795 – 1805.  Candle opens inside zone, closes below zone_low.
        candle = _make_candle(open_=1798.0, high=1800.0, low=1790.0, close=1791.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.update_zone(zone_id, candle)
        assert zone.status == "invalidated"

    def test_body_close_above_resistance_invalidates(self):
        """Body closing above a resistance zone invalidates it."""
        zone = self._make_zone("resistance", center=1820.0)
        # Zone: 1815 – 1825.  Candle opens inside, closes above zone_high.
        candle = _make_candle(open_=1820.0, high=1830.0, low=1818.0, close=1828.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.update_zone(zone_id, candle)
        assert zone.status == "invalidated"

    def test_candle_missing_zone_leaves_active(self):
        """Candle that doesn't touch zone leaves status unchanged."""
        zone = self._make_zone("support", center=1800.0)  # zone: 1795–1805
        candle = _make_candle(open_=1850.0, high=1855.0, low=1845.0, close=1852.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.update_zone(zone_id, candle)
        assert zone.status == "active"

    def test_touch_count_increments_on_test(self):
        zone = self._make_zone("support", center=1800.0)
        initial_count = zone.touch_count
        candle = _make_candle(open_=1810.0, high=1815.0, low=1797.0, close=1808.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.update_zone(zone_id, candle)
        assert zone.touch_count > initial_count

    def test_invalidated_zone_not_updated_again(self):
        """An already-invalidated zone should not change status on further candles."""
        zone = self._make_zone("support", center=1800.0)
        manager = ZoneManager()
        zone_id = manager.add_zone(zone)
        manager.invalidate(zone_id, reason="manual")
        assert zone.status == "invalidated"
        # Additional candle should not change the status
        candle = _make_candle(open_=1798.0, high=1800.0, low=1790.0, close=1791.0)
        manager.update_zone(zone_id, candle)
        assert zone.status == "invalidated"


# =============================================================================
# 7. Zone Manager
# =============================================================================

class TestZoneManager:
    """ZoneManager: add, query nearby, update status, maintenance."""

    def _support_zone(self, center: float = 1800.0, tf: str = "H1") -> SimpleZone:
        return SimpleZone(
            price_high=center + 5.0,
            price_low=center - 5.0,
            zone_type="support",
            center=center,
            timeframe=tf,
            strength=2.0,
        )

    def _resistance_zone(self, center: float = 1850.0, tf: str = "H1") -> SimpleZone:
        return SimpleZone(
            price_high=center + 5.0,
            price_low=center - 5.0,
            zone_type="resistance",
            center=center,
            timeframe=tf,
            strength=1.5,
        )

    def test_add_zone_returns_id(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone()
        zone_id = manager.add_zone(zone)
        assert isinstance(zone_id, int)
        assert zone_id >= 1

    def test_add_multiple_zones_unique_ids(self):
        manager = ZoneManager(merge_overlap=False)
        z1 = self._support_zone(1800.0)
        z2 = self._support_zone(1900.0)
        id1 = manager.add_zone(z1)
        id2 = manager.add_zone(z2)
        assert id1 != id2

    def test_zones_property(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone()
        manager.add_zone(zone)
        assert zone in manager.zones

    def test_active_zones_excludes_invalidated(self):
        manager = ZoneManager(merge_overlap=False)
        z1 = self._support_zone(1800.0)
        z2 = self._support_zone(1900.0)
        id1 = manager.add_zone(z1)
        manager.add_zone(z2)
        manager.invalidate(id1)
        assert z1 not in manager.active_zones
        assert z2 in manager.active_zones

    def test_get_nearby_zones_finds_close_zone(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone(center=1800.0)
        manager.add_zone(zone)
        nearby = manager.get_nearby_zones(price=1800.0, atr=5.0, max_distance_atr=2.0)
        assert zone in nearby

    def test_get_nearby_zones_excludes_far_zone(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone(center=1900.0)  # 100 pts away
        manager.add_zone(zone)
        nearby = manager.get_nearby_zones(price=1800.0, atr=5.0, max_distance_atr=2.0)
        assert zone not in nearby

    def test_get_nearby_zones_excludes_invalidated(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone(center=1800.0)
        zone_id = manager.add_zone(zone)
        manager.invalidate(zone_id)
        nearby = manager.get_nearby_zones(price=1800.0, atr=5.0, max_distance_atr=2.0)
        assert zone not in nearby

    def test_get_nearby_zones_sorted_by_proximity(self):
        manager = ZoneManager(merge_overlap=False)
        z_near = self._support_zone(center=1801.0)
        z_far = self._support_zone(center=1815.0)
        manager.add_zone(z_near)
        manager.add_zone(z_far)
        nearby = manager.get_nearby_zones(price=1800.0, atr=5.0, max_distance_atr=10.0)
        # First result should be the nearer zone
        assert nearby[0] is z_near

    def test_get_strongest_zone_long(self):
        manager = ZoneManager(merge_overlap=False)
        weak_sup = self._support_zone(1800.0)
        weak_sup.strength = 1.0
        strong_sup = self._support_zone(1798.0)
        strong_sup.strength = 5.0
        manager.add_zone(weak_sup)
        manager.add_zone(strong_sup)
        result = manager.get_strongest_zone(1800.0, atr=5.0, direction="long")
        assert result is strong_sup

    def test_get_strongest_zone_short(self):
        manager = ZoneManager(merge_overlap=False)
        res = self._resistance_zone(1820.0)
        res.strength = 3.0
        manager.add_zone(res)
        result = manager.get_strongest_zone(1820.0, atr=5.0, direction="short")
        assert result is res

    def test_get_strongest_zone_no_match_returns_none(self):
        manager = ZoneManager(merge_overlap=False)
        result = manager.get_strongest_zone(1800.0, atr=5.0, direction="long")
        assert result is None

    def test_maintenance_updates_zones(self):
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone(center=1800.0)  # zone: 1795–1805
        manager.add_zone(zone)
        # Candle that wicks into the zone → should be marked as tested
        candle = _make_candle(open_=1810.0, high=1815.0, low=1797.0, close=1808.0)
        manager.maintenance(candle)
        assert zone.status == "tested"

    def test_maintenance_prunes_old_invalidated(self):
        manager = ZoneManager(merge_overlap=False, max_invalidated_age_bars=3)
        zone = self._support_zone()
        zone_id = manager.add_zone(zone)
        manager.invalidate(zone_id)

        candle = _make_candle()
        for _ in range(4):  # exceed max_invalidated_age_bars
            manager.maintenance(candle)

        # Zone should have been pruned from internal list
        assert zone not in manager.zones

    def test_merge_overlapping_zones(self):
        """Two overlapping zones of the same type should merge into one."""
        manager = ZoneManager(merge_overlap=True)
        z1 = SimpleZone(price_high=1810.0, price_low=1795.0, zone_type="support", center=1802.5)
        z2 = SimpleZone(price_high=1820.0, price_low=1805.0, zone_type="support", center=1812.5)
        manager.add_zone(z1)
        manager.add_zone(z2)
        # After merge, there should only be 1 active zone
        assert len(manager.active_zones) == 1

    def test_no_merge_different_zone_types(self):
        """Overlapping zones of different types should NOT merge."""
        manager = ZoneManager(merge_overlap=True)
        z1 = SimpleZone(price_high=1810.0, price_low=1795.0, zone_type="support", center=1802.5)
        z2 = SimpleZone(price_high=1808.0, price_low=1793.0, zone_type="resistance", center=1800.5)
        manager.add_zone(z1)
        manager.add_zone(z2)
        assert len(manager.active_zones) == 2

    def test_invalidate_nonexistent_zone_no_crash(self):
        manager = ZoneManager()
        manager.invalidate(99999)  # should not raise

    def test_update_nonexistent_zone_no_crash(self):
        manager = ZoneManager()
        candle = _make_candle()
        manager.update_zone(99999, candle)  # should not raise

    def test_dict_candle_supported(self):
        """Manager should accept candle as a dict as well as an object."""
        manager = ZoneManager(merge_overlap=False)
        zone = self._support_zone(1800.0)
        zone_id = manager.add_zone(zone)
        candle = {"open": 1810.0, "high": 1815.0, "low": 1797.0, "close": 1808.0}
        manager.update_zone(zone_id, candle)  # should not raise


# =============================================================================
# 8. Integration test
# =============================================================================

class TestZoneIntegration:
    """End-to-end: detect swings → cluster → score confluence."""

    def test_full_pipeline_no_crash(self):
        """Run the full pipeline on synthetic data without error."""
        n = 200
        np.random.seed(7)
        base = 1800.0
        close = base + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        atr_val = 5.0

        # Step 1: detect swings
        detector = SwingPointDetector(lookback=5)
        result = detector.detect(high, low)

        swing_high_prices = np.array([sp[1] for sp in result.swing_highs])
        swing_low_prices = np.array([sp[1] for sp in result.swing_lows])

        # Step 2: cluster into S/R zones
        cluster_det = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
        res_zones = cluster_det.cluster(swing_high_prices, atr=atr_val, zone_type="resistance")
        sup_zones = cluster_det.cluster(swing_low_prices, atr=atr_val, zone_type="support")

        # Step 3: pivot calculation
        pivot_calc = PivotCalculator()
        pivots = pivot_calc.standard_pivots(float(np.max(high)), float(np.min(low)), float(close[-1]))

        # Step 4: confluence density
        scorer = ConfluenceDensityScorer(proximity_atr=1.0)
        all_zones = res_zones + sup_zones
        score = scorer.score(close[-1], all_zones, pivots, atr=atr_val)

        assert isinstance(score, ConfluenceScore)
        assert score.total >= 0

    def test_swing_to_cluster_consistency(self):
        """Swing points near the same price level should collapse into one cluster."""
        # Build a price series that bounces off 1800 multiple times
        n = 100
        high = np.full(n, 1805.0)
        low = np.full(n, 1795.0)
        close = np.full(n, 1800.0)

        # Create clear swing lows at 1795 every 15 bars
        for i in range(10, n - 10, 15):
            for j in range(i - 3, i + 4):
                if 0 <= j < n:
                    low[j] = 1795.0 - (3.0 - abs(j - i))
            low[i] = 1792.0  # pronounced low

        detector = SwingPointDetector(lookback=3)
        result = detector.detect(high, low)
        sl_prices = np.array([sp[1] for sp in result.swing_lows])

        if len(sl_prices) >= 2:
            det = SRClusterDetector(atr_multiplier=1.0, min_touches=2)
            zones = det.cluster(sl_prices, atr=5.0, zone_type="support")
            # Multiple touches near 1792 should form at least one cluster
            assert len(zones) >= 1
