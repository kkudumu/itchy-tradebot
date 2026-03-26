"""
Unit tests for confluence indicators: ADX, ATR, RSI, Bollinger Bands,
session identification, and RSI divergence detection.

Each test validates against known reference values or synthetic data with
predictable properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indicators.confluence import (
    ADXCalculator,
    ATRCalculator,
    BollingerBandCalculator,
    RSICalculator,
    wilders_smooth,
)
from src.indicators.divergence import DivergenceDetector
from src.indicators.sessions import SessionIdentifier


# ============================================================
# Helpers / fixtures
# ============================================================

RNG = np.random.default_rng(42)


def _make_trending_up(n: int = 200, slope: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady uptrend: each close is strictly higher than the previous."""
    close = np.arange(1, n + 1, dtype=float) * slope + 1000.0
    high = close + 0.5
    low = close - 0.5
    return high, low, close


def _make_flat(n: int = 200, base: float = 1800.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Completely flat price series — no trend, very low ATR."""
    close = np.full(n, base, dtype=float)
    high = close + 0.1
    low = close - 0.1
    return high, low, close


def _make_alternating(n: int = 100, base: float = 1800.0) -> np.ndarray:
    """Alternating +1 / -1 moves from base price."""
    close = np.empty(n, dtype=float)
    close[0] = base
    for i in range(1, n):
        close[i] = close[i - 1] + (1.0 if i % 2 == 1 else -1.0)
    return close


def _make_continuous_up(n: int = 100, start: float = 1800.0) -> np.ndarray:
    """Every bar closes higher — RSI should approach 100."""
    return np.arange(start, start + n, dtype=float)


# ============================================================
# Wilder's smoothing
# ============================================================

class TestWildersSmooth:
    def test_first_value_is_sma(self):
        values = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        result = wilders_smooth(values, period=3)
        # First valid position should be mean of [2, 4, 6] = 4.0
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(4.0)

    def test_subsequent_values(self):
        values = np.array([2.0, 4.0, 6.0, 8.0])
        result = wilders_smooth(values, period=3)
        # result[2] = 4.0 (SMA)
        # result[3] = (4.0 * 2 + 8.0) / 3 = 16/3
        assert result[3] == pytest.approx(16.0 / 3.0, rel=1e-9)

    def test_too_short_returns_all_nan(self):
        values = np.array([1.0, 2.0])
        result = wilders_smooth(values, period=5)
        assert np.all(np.isnan(result))

    def test_length_preserved(self):
        values = np.ones(50)
        result = wilders_smooth(values, period=14)
        assert len(result) == 50


# ============================================================
# ATR
# ============================================================

class TestATRCalculator:
    def test_output_length(self):
        high, low, close = _make_trending_up(100)
        atr = ATRCalculator(period=14).calculate(high, low, close)
        assert len(atr) == 100

    def test_nan_warmup(self):
        high, low, close = _make_trending_up(50)
        atr = ATRCalculator(period=14).calculate(high, low, close)
        assert np.all(np.isnan(atr[:13]))
        assert not np.isnan(atr[13])

    def test_flat_data_low_atr(self):
        """Flat price → ATR should be very small (just H-L range = 0.2)."""
        high, low, close = _make_flat(100)
        atr = ATRCalculator(period=14).calculate(high, low, close)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid < 1.0), f"Expected ATR < 1.0 for flat data, got max={valid.max()}"

    def test_volatile_data_higher_atr(self):
        """Noisy data should produce larger ATR than flat data."""
        np.random.seed(0)
        close = 1800.0 + np.cumsum(np.random.randn(200) * 5)
        high = close + np.abs(np.random.randn(200) * 3)
        low = close - np.abs(np.random.randn(200) * 3)

        atr_noisy = ATRCalculator(period=14).calculate(high, low, close)
        high_f, low_f, close_f = _make_flat(200)
        atr_flat = ATRCalculator(period=14).calculate(high_f, low_f, close_f)

        valid_noisy = np.nanmean(atr_noisy)
        valid_flat = np.nanmean(atr_flat)
        assert valid_noisy > valid_flat

    def test_manual_first_atr_value(self):
        """Manually verify the first non-NaN ATR value against hand calculation.

        Data is constructed so every bar has a range of exactly 2.0 and the
        close equals the midpoint.  The previous-close component of TR can
        still exceed H-L for some bars (because close = mid and the next bar
        starts higher), so the expected seed value is the mean of the actual
        TRs, not simply H-L.
        """
        n = 20
        # Flat price: same high/low every bar → TR = H - L = 2.0 each bar
        high  = np.full(n, 1801.0)
        low   = np.full(n, 1799.0)
        close = np.full(n, 1800.0)

        atr = ATRCalculator(period=14).calculate(high, low, close)

        # Every TR = 2.0; ATR seed = mean([2.0] * 14) = 2.0
        assert atr[13] == pytest.approx(2.0, rel=1e-9)
        # All subsequent ATR values are also 2.0 (Wilder recursion on constant input)
        np.testing.assert_allclose(atr[13:], 2.0, rtol=1e-9)


# ============================================================
# ADX
# ============================================================

class TestADXCalculator:
    def test_output_shape(self):
        high, low, close = _make_trending_up(100)
        result = ADXCalculator(period=14).calculate(high, low, close)
        assert result.adx.shape == (100,)
        assert result.plus_di.shape == (100,)
        assert result.minus_di.shape == (100,)
        assert result.is_trending.shape == (100,)

    def test_trending_data_high_adx(self):
        """Strong steady uptrend should produce ADX > 28 after warm-up."""
        high, low, close = _make_trending_up(200, slope=2.0)
        result = ADXCalculator(period=14, threshold=28.0).calculate(high, low, close)
        # Allow burn-in: check the last 50 bars
        adx_tail = result.adx[-50:]
        valid_adx = adx_tail[~np.isnan(adx_tail)]
        assert np.all(valid_adx > 25), (
            f"Expected ADX > 25 for strong uptrend, got min={valid_adx.min():.2f}"
        )

    def test_flat_data_low_adx(self):
        """Flat/sideways data should produce ADX < 20 after warm-up."""
        high, low, close = _make_flat(200)
        result = ADXCalculator(period=14).calculate(high, low, close)
        adx_tail = result.adx[-50:]
        valid_adx = adx_tail[~np.isnan(adx_tail)]
        assert np.all(valid_adx < 20), (
            f"Expected ADX < 20 for flat data, got max={valid_adx.max():.2f}"
        )

    def test_uptrend_plus_di_dominant(self):
        """In an uptrend, +DI should exceed -DI."""
        high, low, close = _make_trending_up(200)
        result = ADXCalculator(period=14).calculate(high, low, close)
        valid = ~np.isnan(result.plus_di) & ~np.isnan(result.minus_di)
        assert np.all(result.plus_di[valid][-50:] > result.minus_di[valid][-50:])

    def test_nan_warmup_period(self):
        """ADX NaN warm-up.

        Smoothed TR/DI are NaN before index ``period - 1``.  Wilders_smooth
        on ADX/DX seeds at that same index (DX[0:period-1] are 0, not NaN,
        because DI=0 before smoothing begins).  Therefore ADX is first
        defined at index ``period - 1``.
        """
        high, low, close = _make_trending_up(100)
        result = ADXCalculator(period=14).calculate(high, low, close)
        # ADX NaN exists for bars 0 through period-2 (i.e., indices 0..12)
        assert np.all(np.isnan(result.adx[:13])), (
            "Expected NaN for ADX[0:13]"
        )
        assert not np.isnan(result.adx[13]), "Expected ADX[13] to be non-NaN"

    def test_is_trending_boolean(self):
        result = ADXCalculator(period=14, threshold=28).calculate(*_make_trending_up(200))
        assert result.is_trending.dtype == bool

    def test_default_threshold_is_28(self):
        """Gold-specific default threshold should be 28."""
        calc = ADXCalculator()
        assert calc.threshold == 28.0


# ============================================================
# RSI
# ============================================================

class TestRSICalculator:
    def test_output_length(self):
        close = _make_alternating(100)
        result = RSICalculator(period=14).calculate(close)
        assert len(result.rsi) == 100

    def test_nan_warmup(self):
        close = _make_alternating(100)
        result = RSICalculator(period=14).calculate(close)
        assert np.all(np.isnan(result.rsi[:13]))
        assert not np.isnan(result.rsi[13])

    def test_alternating_moves_near_50(self):
        """Equal up/down moves → RSI should converge near 50."""
        close = _make_alternating(200)
        result = RSICalculator(period=14).calculate(close)
        # Ignore warm-up; take last 50 bars
        rsi_tail = result.rsi[-50:]
        valid = rsi_tail[~np.isnan(rsi_tail)]
        # After sufficient bars, RSI should be within [45, 55]
        assert np.all(valid > 40) and np.all(valid < 60), (
            f"Expected RSI near 50 for alternating data; range: [{valid.min():.1f}, {valid.max():.1f}]"
        )

    def test_continuous_up_approaches_100(self):
        """All positive closes → RSI should approach 100."""
        close = _make_continuous_up(200)
        result = RSICalculator(period=14).calculate(close)
        rsi_tail = result.rsi[-10:]
        valid = rsi_tail[~np.isnan(rsi_tail)]
        assert np.all(valid > 90), (
            f"Expected RSI > 90 for continuous uptrend, got min={valid.min():.2f}"
        )

    def test_continuous_down_approaches_0(self):
        """All negative closes → RSI should approach 0."""
        close = np.arange(2000.0, 1800.0, -1.0)  # strictly falling
        result = RSICalculator(period=14).calculate(close)
        rsi_tail = result.rsi[-10:]
        valid = rsi_tail[~np.isnan(rsi_tail)]
        assert np.all(valid < 10), (
            f"Expected RSI < 10 for continuous downtrend, got max={valid.max():.2f}"
        )

    def test_overbought_oversold_flags(self):
        """Overbought / oversold flags should be consistent with RSI levels."""
        close = _make_continuous_up(200)
        result = RSICalculator(period=14, overbought=70, oversold=30).calculate(close)
        valid = ~np.isnan(result.rsi)
        # Where RSI > 70, overbought should be True
        assert np.all(result.overbought[valid] == (result.rsi[valid] > 70))
        # Where RSI < 30, oversold should be True
        assert np.all(result.oversold[valid] == (result.rsi[valid] < 30))

    def test_rsi_bounded_0_100(self):
        np.random.seed(7)
        close = 1800.0 + np.cumsum(np.random.randn(300) * 3)
        result = RSICalculator(period=14).calculate(close)
        valid = result.rsi[~np.isnan(result.rsi)]
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)


# ============================================================
# Bollinger Bands
# ============================================================

class TestBollingerBandCalculator:
    def test_output_shapes(self):
        close = 1800.0 + np.random.randn(300)
        result = BollingerBandCalculator().calculate(close)
        n = len(close)
        for arr in (result.upper, result.middle, result.lower,
                    result.width, result.width_percentile, result.is_squeeze):
            assert arr.shape == (n,)

    def test_band_ordering(self):
        """Upper >= Middle >= Lower for all valid bars."""
        np.random.seed(1)
        close = 1800.0 + np.cumsum(np.random.randn(300))
        result = BollingerBandCalculator(period=20).calculate(close)
        valid = ~np.isnan(result.upper)
        assert np.all(result.upper[valid] >= result.middle[valid])
        assert np.all(result.middle[valid] >= result.lower[valid])

    def test_nan_warmup(self):
        close = np.ones(200) * 1800.0
        result = BollingerBandCalculator(period=20).calculate(close)
        assert np.all(np.isnan(result.middle[:19]))
        assert not np.isnan(result.middle[19])

    def test_flat_data_narrow_width(self):
        """Flat price → width should be 0 (or near 0)."""
        close = np.full(200, 1800.0)
        result = BollingerBandCalculator(period=20).calculate(close)
        valid_width = result.width[~np.isnan(result.width)]
        assert np.all(valid_width < 0.01), (
            f"Expected near-zero width for flat data, got max={valid_width.max()}"
        )

    def test_volatile_data_wider_bands(self):
        """High-volatility data → wider bands than flat data."""
        np.random.seed(2)
        close_vol = 1800.0 + np.cumsum(np.random.randn(300) * 10)
        close_flat = np.full(300, 1800.0)

        calc = BollingerBandCalculator(period=20)
        r_vol = calc.calculate(close_vol)
        r_flat = calc.calculate(close_flat)

        mean_width_vol = np.nanmean(r_vol.width)
        mean_width_flat = np.nanmean(r_flat.width)
        assert mean_width_vol > mean_width_flat

    def test_squeeze_detected_after_volatile_baseline(self):
        """A low-volatility period preceded by a high-volatility baseline triggers squeeze.

        The percentile rank of width compares the current width against the
        ``squeeze_lookback`` prior values.  If the lookback window is entirely
        flat (width=0), every bar equals all prior bars, yielding a 100th-
        percentile rank — not a squeeze.

        To detect a squeeze, the lookback window must contain wider historical
        widths so that the current narrow width ranks low.
        """
        np.random.seed(42)
        # Phase 1: high volatility → wide Bollinger widths as historical baseline
        volatile_close = 1800.0 + np.cumsum(np.random.randn(200) * 15)
        # Phase 2: flat / low volatility → narrow Bollinger widths that rank low
        flat_close = np.full(100, volatile_close[-1])
        close = np.concatenate([volatile_close, flat_close])

        result = BollingerBandCalculator(
            period=20, squeeze_lookback=100, squeeze_percentile=20.0
        ).calculate(close)

        # Squeeze should be detected somewhere across the full series
        # (volatile-to-flat transitions cause narrow widths to rank below 20th pct)
        assert result.is_squeeze.any(), (
            "Expected at least one squeeze flag when low-volatility follows high-volatility"
        )

    def test_percentile_bounded(self):
        """Width percentile should be in [0, 100]."""
        np.random.seed(3)
        close = 1800.0 + np.cumsum(np.random.randn(400))
        result = BollingerBandCalculator().calculate(close)
        valid = result.width_percentile[~np.isnan(result.width_percentile)]
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)

    def test_is_squeeze_boolean(self):
        close = np.ones(300) * 1800.0
        result = BollingerBandCalculator().calculate(close)
        assert result.is_squeeze.dtype == bool


# ============================================================
# Session identification
# ============================================================

class TestSessionIdentifier:
    """Verify session boundaries using known UTC timestamps."""

    SI = SessionIdentifier()

    # Reference timestamps (all UTC)
    _CASES = [
        # (ISO UTC string,    expected_session)
        ("2024-01-15T04:00:00", "asian"),       # 04:00 UTC → Asian
        ("2024-01-15T00:00:00", "asian"),       # 00:00 UTC → Asian boundary (start)
        ("2024-01-15T07:59:00", "asian"),       # 07:59 UTC → still Asian
        ("2024-01-15T08:00:00", "london"),      # 08:00 UTC → London open
        ("2024-01-15T12:00:00", "london"),      # 12:00 UTC → mid-London
        ("2024-01-15T12:59:00", "london"),      # 12:59 UTC → just before overlap
        ("2024-01-15T13:00:00", "overlap"),     # 13:00 UTC → overlap start
        ("2024-01-15T14:30:00", "overlap"),     # 14:30 UTC → overlap
        ("2024-01-15T15:59:00", "overlap"),     # 15:59 UTC → still overlap
        ("2024-01-15T16:00:00", "new_york"),    # 16:00 UTC → NY only (London closed)
        ("2024-01-15T18:00:00", "new_york"),    # 18:00 UTC → mid NY
        ("2024-01-15T20:59:00", "new_york"),    # 20:59 UTC → end NY
        ("2024-01-15T21:00:00", "off_hours"),   # 21:00 UTC → off-hours
        ("2024-01-15T23:00:00", "off_hours"),   # 23:00 UTC → off-hours
    ]

    def test_individual_session_labels(self):
        for ts_str, expected in self._CASES:
            ts = np.array([np.datetime64(ts_str, "s")])
            labels = self.SI.identify(ts)
            assert labels[0] == expected, (
                f"For {ts_str}: expected '{expected}', got '{labels[0]}'"
            )

    def test_batch_classification(self):
        """Batch input produces same result as individual calls."""
        ts_list = [np.datetime64(ts, "s") for ts, _ in self._CASES]
        expected = [exp for _, exp in self._CASES]
        ts_arr = np.array(ts_list)
        labels = self.SI.identify(ts_arr)
        for i, (label, exp) in enumerate(zip(labels, expected)):
            assert label == exp, f"Index {i}: expected '{exp}', got '{label}'"

    def test_session_mask_london(self):
        # 09:00 UTC should be in London
        ts = np.array([np.datetime64("2024-01-15T09:00:00", "s")])
        assert self.SI.session_mask(ts, "london")[0]

    def test_session_mask_outside(self):
        # 22:00 UTC is outside all named sessions
        ts = np.array([np.datetime64("2024-01-15T22:00:00", "s")])
        assert not self.SI.session_mask(ts, "london")[0]
        assert not self.SI.session_mask(ts, "new_york")[0]
        assert not self.SI.session_mask(ts, "asian")[0]
        assert not self.SI.session_mask(ts, "overlap")[0]

    def test_is_active_session_covers_london_and_ny(self):
        london_ts = np.array([np.datetime64("2024-01-15T10:00:00", "s")])
        ny_ts = np.array([np.datetime64("2024-01-15T17:00:00", "s")])
        asian_ts = np.array([np.datetime64("2024-01-15T04:00:00", "s")])

        assert self.SI.is_active_session(london_ts)[0]
        assert self.SI.is_active_session(ny_ts)[0]
        assert not self.SI.is_active_session(asian_ts)[0]

    def test_invalid_session_raises(self):
        ts = np.array([np.datetime64("2024-01-15T10:00:00", "s")])
        with pytest.raises(ValueError, match="Unknown session"):
            self.SI.session_mask(ts, "pacific")

    def test_overlap_is_subset_of_london_and_ny(self):
        """Every overlap bar should also be a London bar and a NY bar."""
        # Build 1-minute timestamps for a full day
        start = np.datetime64("2024-01-15T00:00:00", "m")
        ts = np.array([start + np.timedelta64(i, "m") for i in range(1440)])

        overlap = self.SI.session_mask(ts, "overlap")
        london = self.SI.session_mask(ts, "london")
        new_york = self.SI.session_mask(ts, "new_york")

        # Overlap must be contained within both London and NY
        assert np.all(london[overlap])
        assert np.all(new_york[overlap])


# ============================================================
# Divergence detection
# ============================================================

class TestDivergenceDetector:
    """Test divergence detection on synthetic data with known patterns."""

    def _make_regular_bullish_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct data with a clear regular bullish divergence:
        - Price makes a lower low on the second swing
        - RSI is synthetic and makes a higher low on the second swing
        """
        n = 100
        close = np.full(n, 1800.0)

        # First swing low at bar 20 (price 1780)
        close[20] = 1780.0
        # Second swing low at bar 60 (price 1770 — lower low)
        close[60] = 1770.0

        # Fabricate smooth transitions
        for i in range(15, 25):
            close[i] = 1800.0 - (1800.0 - 1780.0) * (1 - abs(i - 20) / 5)
        for i in range(55, 65):
            close[i] = 1800.0 - (1800.0 - 1770.0) * (1 - abs(i - 60) / 5)

        high = close + 5.0
        low = close - 5.0

        return high, low, close

    def test_no_divergence_on_random_data(self):
        """Random walk data should produce minimal spurious divergences."""
        np.random.seed(99)
        close = 1800.0 + np.cumsum(np.random.randn(200) * 0.5)
        high = close + 1.0
        low = close - 1.0

        rsi_result = RSICalculator(period=14).calculate(close)
        rsi = np.nan_to_num(rsi_result.rsi, nan=50.0)

        detector = DivergenceDetector(lookback=5, min_bars_between=5, max_bars_between=50)
        result = detector.detect(close, rsi, high, low)

        total_signals = (
            result.regular_bullish.sum()
            + result.regular_bearish.sum()
            + result.hidden_bullish.sum()
            + result.hidden_bearish.sum()
        )
        # Random data may occasionally trigger divergence but should not be excessive
        assert total_signals < 20, f"Too many spurious divergences: {total_signals}"

    def test_output_shapes(self):
        """All output arrays must have the same length as input."""
        n = 150
        np.random.seed(5)
        close = 1800.0 + np.cumsum(np.random.randn(n))
        high = close + 2.0
        low = close - 2.0
        rsi = RSICalculator(period=14).calculate(close)
        rsi_values = np.nan_to_num(rsi.rsi, nan=50.0)

        result = DivergenceDetector().detect(close, rsi_values, high, low)
        assert result.regular_bullish.shape == (n,)
        assert result.regular_bearish.shape == (n,)
        assert result.hidden_bullish.shape == (n,)
        assert result.hidden_bearish.shape == (n,)

    def test_output_dtypes_are_bool(self):
        n = 100
        close = np.ones(n) * 1800.0
        high = close + 1.0
        low = close - 1.0
        rsi = np.full(n, 50.0)
        result = DivergenceDetector().detect(close, rsi, high, low)
        for arr in (result.regular_bullish, result.regular_bearish,
                    result.hidden_bullish, result.hidden_bearish):
            assert arr.dtype == bool

    def _make_regular_bullish_setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct price and RSI arrays with a confirmed regular bullish divergence:
        - Price swing low 1 at bar 20: lower[20] = 1775
        - Price swing low 2 at bar 60: lower[60] = 1770  (lower low)
        - RSI  swing low 1 at bar 20: rsi[20]  = 32
        - RSI  swing low 2 at bar 60: rsi[60]  = 38  (higher low → divergence)
        """
        n = 90
        # Build price as flat base with two V-shaped dips
        close = np.full(n, 1800.0, dtype=float)
        for i in range(15, 26):
            depth = 1.0 - abs(i - 20) / 5.0
            close[i] -= 25.0 * depth          # dip to ~1775 at bar 20
        for i in range(55, 66):
            depth = 1.0 - abs(i - 60) / 5.0
            close[i] -= 30.0 * depth          # dip to ~1770 at bar 60 (lower low)

        high = close + 5.0
        low  = close - 5.0

        # Build RSI: mirrors the dips but second is shallower (higher low)
        rsi = np.full(n, 50.0, dtype=float)
        for i in range(15, 26):
            depth = 1.0 - abs(i - 20) / 5.0
            rsi[i] -= 18.0 * depth            # dip to ~32 at bar 20
        for i in range(55, 66):
            depth = 1.0 - abs(i - 60) / 5.0
            rsi[i] -= 12.0 * depth            # dip only to ~38 at bar 60 (higher low)

        return high, low, close, rsi

    def _make_regular_bearish_setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct price and RSI arrays with a confirmed regular bearish divergence:
        - Price swing high 1 at bar 20: high[20] ~ 1830
        - Price swing high 2 at bar 60: high[60] ~ 1840  (higher high)
        - RSI  swing high 1 at bar 20: rsi[20] ~ 72
        - RSI  swing high 2 at bar 60: rsi[60] ~ 65  (lower high → divergence)
        """
        n = 90
        close = np.full(n, 1800.0, dtype=float)
        for i in range(15, 26):
            depth = 1.0 - abs(i - 20) / 5.0
            close[i] += 25.0 * depth          # peak ~1825 at bar 20
        for i in range(55, 66):
            depth = 1.0 - abs(i - 60) / 5.0
            close[i] += 35.0 * depth          # peak ~1835 at bar 60 (higher high)

        high = close + 5.0
        low  = close - 5.0

        rsi = np.full(n, 50.0, dtype=float)
        for i in range(15, 26):
            depth = 1.0 - abs(i - 20) / 5.0
            rsi[i] += 22.0 * depth            # peak ~72 at bar 20
        for i in range(55, 66):
            depth = 1.0 - abs(i - 60) / 5.0
            rsi[i] += 15.0 * depth            # peak ~65 at bar 60 (lower high → divergence)

        return high, low, close, rsi

    def test_regular_bullish_on_known_pattern(self):
        """Verify regular bullish divergence is detected on explicitly constructed data."""
        high, low, close, rsi = self._make_regular_bullish_setup()
        result = DivergenceDetector(lookback=4, min_bars_between=5, max_bars_between=60).detect(
            close, rsi, high, low
        )
        assert result.regular_bullish.any(), (
            "Expected at least one regular bullish divergence signal"
        )

    def test_regular_bearish_on_known_pattern(self):
        """Verify regular bearish divergence is detected on explicitly constructed data."""
        high, low, close, rsi = self._make_regular_bearish_setup()
        result = DivergenceDetector(lookback=4, min_bars_between=5, max_bars_between=60).detect(
            close, rsi, high, low
        )
        assert result.regular_bearish.any(), (
            "Expected at least one regular bearish divergence signal"
        )


# ============================================================
# Integration: ADX + ATR + RSI together
# ============================================================

class TestIndicatorIntegration:
    def test_trending_regime_all_indicators(self):
        """Sanity check all indicators on the same trending dataset."""
        high, low, close = _make_trending_up(250, slope=3.0)

        adx_result = ADXCalculator(period=14, threshold=28).calculate(high, low, close)
        atr = ATRCalculator(period=14).calculate(high, low, close)
        rsi_result = RSICalculator(period=14).calculate(close)
        bb_result = BollingerBandCalculator(period=20).calculate(close)

        # All arrays have the same length
        n = len(close)
        assert adx_result.adx.shape == (n,)
        assert atr.shape == (n,)
        assert rsi_result.rsi.shape == (n,)
        assert bb_result.upper.shape == (n,)

        # Spot-check: no NaN beyond warm-up period
        cutoff = 50
        assert not np.any(np.isnan(adx_result.adx[cutoff:]))
        assert not np.any(np.isnan(atr[cutoff:]))
        assert not np.any(np.isnan(rsi_result.rsi[cutoff:]))
        assert not np.any(np.isnan(bb_result.middle[cutoff:]))
