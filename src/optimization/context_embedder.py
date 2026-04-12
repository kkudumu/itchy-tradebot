"""64-dim context embedder for adaptive strategy optimization.

Produces three embedding layers, all normalized to [0, 1]:
  - Market context  (20 dims): trend, volatility, price action, instrument identity
  - Strategy params (24 dims): normalized parameter snapshot
  - Outcome fingerprint (20 dims): performance metrics

Stored in pgvector as VECTOR(20), VECTOR(24), VECTOR(20) columns and used
for cosine similarity search to find past optimization runs with similar
market conditions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Optional scipy for skew/kurtosis — fall back to manual if unavailable.
try:
    from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurtosis
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── helper functions ─────────────────────────────────────────────────


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def _safe(val: Any, default: float = 0.0) -> float:
    """Return *val* as a float, replacing None / NaN / inf with *default*."""
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _normalize(val: float, lo: float, hi: float) -> float:
    """Linear normalize *val* from [lo, hi] to [0, 1], clamped."""
    if hi <= lo:
        return 0.5
    return _clamp((val - lo) / (hi - lo))


def _log_normalize(val: float, lo: float, hi: float) -> float:
    """Log-scale normalize *val* from [lo, hi] to [0, 1], clamped.

    lo and hi are the raw (non-log) bounds.  We take log1p of all three
    so that val=0 maps cleanly to 0.
    """
    if val <= 0:
        val = 1e-12
    if lo <= 0:
        lo = 1e-12
    if hi <= 0:
        hi = 1e-12
    log_val = math.log(val)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    if log_hi <= log_lo:
        return 0.5
    return _clamp((log_val - log_lo) / (log_hi - log_lo))


def _manual_skew(arr: np.ndarray) -> float:
    """Compute skewness without scipy."""
    n = len(arr)
    if n < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _manual_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis without scipy (Fisher definition)."""
    n = len(arr)
    if n < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _compute_skew(arr: np.ndarray) -> float:
    if _HAS_SCIPY:
        val = _scipy_skew(arr, nan_policy="omit")
        return _safe(float(val))
    return _manual_skew(arr)


def _compute_kurtosis(arr: np.ndarray) -> float:
    if _HAS_SCIPY:
        val = _scipy_kurtosis(arr, nan_policy="omit", fisher=True)
        return _safe(float(val))
    return _manual_kurtosis(arr)


def _autocorrelation(series: np.ndarray, lag: int) -> float:
    """Compute lag-*lag* autocorrelation of a 1-D array."""
    n = len(series)
    if n <= lag + 1:
        return 0.0
    s = series - np.mean(series)
    var = np.sum(s ** 2)
    if var == 0:
        return 0.0
    return float(np.sum(s[lag:] * s[:-lag]) / var)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Compute EMA (exponential moving average) using pandas."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


# ── market context embedding (20 dims) ──────────────────────────────


def embed_market(
    df: pd.DataFrame,
    *,
    tick_size: float,
    tick_value_usd: float,
    contract_size: float,
    point_value: float,
) -> np.ndarray:
    """Build a 20-dim market context vector, all values in [0, 1].

    Parameters
    ----------
    df : DataFrame with columns (open, high, low, close, volume) and a
         DatetimeIndex.  At least ~60 rows required; 240+ recommended.
    tick_size, tick_value_usd, contract_size, point_value :
         Instrument metadata floats.

    Returns
    -------
    np.ndarray of shape (20,).
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)
    mean_price = np.mean(close)
    if mean_price == 0:
        mean_price = 1.0  # safety

    vec = np.zeros(20, dtype=np.float64)

    # ── dims 0-3: trend structure ────────────────────────────────────
    # EMA slopes normalized by mean price
    ema60 = _ema(close, min(60, n))
    ema240 = _ema(close, min(240, n))

    # slope = (ema[-1] - ema[-2]) / mean_price  (handle short data)
    def _slope(ema_arr: np.ndarray) -> float:
        if len(ema_arr) < 2:
            return 0.0
        return (ema_arr[-1] - ema_arr[-2]) / mean_price

    slope_1h = _slope(ema60)
    slope_4h = _slope(ema240)
    price_vs_ema60 = (close[-1] - ema60[-1]) / mean_price
    price_vs_ema240 = (close[-1] - ema240[-1]) / mean_price

    # Normalize slopes: typical range ~ [-0.01, 0.01]
    vec[0] = _normalize(slope_1h, -0.01, 0.01)
    vec[1] = _normalize(slope_4h, -0.01, 0.01)
    vec[2] = _normalize(price_vs_ema60, -0.05, 0.05)
    vec[3] = _normalize(price_vs_ema240, -0.10, 0.10)

    # ── dims 4-7: volatility ────────────────────────────────────────
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # fix first bar
    atr_period = min(14, n)
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    atr_mean = np.mean(atr)
    atr_std = np.std(atr)
    atr_ratio = atr_std / atr_mean if atr_mean > 0 else 0.0
    hl_range_mean = np.mean(high - low)

    # ATR trend: expanding if last quarter ATR > first quarter ATR
    quarter = max(n // 4, 1)
    atr_last_q = np.mean(atr[-quarter:])
    atr_first_q = np.mean(atr[:quarter])
    atr_expanding = 1.0 if atr_last_q > atr_first_q else 0.0

    vec[4] = _normalize(atr_mean / mean_price, 0.0, 0.02)
    vec[5] = _normalize(atr_ratio, 0.0, 2.0)
    vec[6] = atr_expanding
    vec[7] = _normalize(hl_range_mean / mean_price, 0.0, 0.02)

    # ── dims 8-11: price action (returns) ───────────────────────────
    returns = np.diff(close) / close[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        returns = np.array([0.0, 0.0])

    ret_mean = np.mean(returns)
    ret_std = np.std(returns)
    ret_skew = _compute_skew(returns)
    ret_kurt = _compute_kurtosis(returns)

    vec[8] = _normalize(ret_mean, -0.005, 0.005)
    vec[9] = _normalize(ret_std, 0.0, 0.01)
    vec[10] = _normalize(ret_skew, -3.0, 3.0)
    vec[11] = _normalize(ret_kurt, -3.0, 10.0)

    # ── dims 12-15: drawdown/runup + autocorrelation ────────────────
    cum_returns = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = abs(float(np.min(drawdowns)))

    running_min = np.minimum.accumulate(cum_returns)
    runups = (cum_returns - running_min) / np.maximum(running_min, 1e-12)
    max_ru = float(np.max(runups))

    acf1 = _autocorrelation(returns, 1)
    acf5 = _autocorrelation(returns, 5)

    vec[12] = _normalize(max_dd, 0.0, 0.20)
    vec[13] = _normalize(max_ru, 0.0, 0.20)
    vec[14] = _normalize(acf1, -1.0, 1.0)
    vec[15] = _normalize(acf5, -1.0, 1.0)

    # ── dims 16-19: instrument identity (log-normalized) ────────────
    vec[16] = _log_normalize(_safe(tick_size, 0.01), 0.001, 100.0)
    vec[17] = _log_normalize(_safe(tick_value_usd, 1.0), 0.01, 100.0)
    vec[18] = _log_normalize(_safe(contract_size, 1.0), 0.1, 10000.0)
    vec[19] = _log_normalize(_safe(point_value, 1.0), 0.01, 100.0)

    # Final safety: clamp everything
    np.clip(vec, 0.0, 1.0, out=vec)
    return vec


# ── strategy params embedding (24 dims) ─────────────────────────────


def embed_params(params: dict[str, Any]) -> np.ndarray:
    """Build a 24-dim strategy params vector, all values in [0, 1].

    Parameters
    ----------
    params : dict with keys "risk", "exit", "strategies" containing
             nested SSS, ichimoku, asian_breakout, ema_pullback params.

    Returns
    -------
    np.ndarray of shape (24,).
    """
    vec = np.zeros(24, dtype=np.float64)

    risk = params.get("risk", {})
    exit_ = params.get("exit", {})
    strats = params.get("strategies", {})

    # ── dims 0-4: Risk / Exit ───────────────────────────────────────
    vec[0] = _normalize(_safe(risk.get("initial_risk_pct"), 0.5), 0.0, 5.0)
    vec[1] = _normalize(_safe(risk.get("reduced_risk_pct"), 0.75), 0.0, 5.0)
    vec[2] = _normalize(_safe(risk.get("daily_circuit_breaker_pct"), 4.5), 0.0, 10.0)
    vec[3] = _normalize(_safe(risk.get("max_concurrent_positions"), 3), 1, 10)
    vec[4] = _normalize(_safe(exit_.get("tp_r_multiple"), 1.5), 0.5, 5.0)

    # ── dims 5-11: SSS ─────────────────────────────────────────────
    sss = strats.get("sss", {})
    vec[5] = _normalize(_safe(sss.get("swing_lookback_n"), 2), 1, 20)
    vec[6] = _normalize(_safe(sss.get("min_swing_pips"), 0.5), 0.0, 50.0)
    vec[7] = _normalize(_safe(sss.get("min_stop_pips"), 10.0), 0.0, 100.0)
    vec[8] = _normalize(_safe(sss.get("min_confluence_score"), 2), 0, 10)
    vec[9] = _normalize(_safe(sss.get("rr_ratio"), 2.0), 0.5, 5.0)
    # entry_mode: cbc_only=0, fifty_tap=0.5, combined=1
    entry_mode_str = sss.get("entry_mode", "cbc_only")
    entry_mode_map = {"cbc_only": 0.0, "fifty_tap": 0.5, "combined": 1.0}
    vec[10] = _clamp(entry_mode_map.get(entry_mode_str, 0.0))
    vec[11] = _normalize(_safe(sss.get("spread_multiplier"), 2.0), 0.5, 5.0)

    # ── dims 12-17: Ichimoku ───────────────────────────────────────
    ichi = strats.get("ichimoku", {})
    vec[12] = _normalize(_safe(ichi.get("tenkan_period"), 9) / 9.0, 0.5, 3.0)
    vec[13] = _normalize(_safe(ichi.get("adx_threshold"), 20), 10, 40)
    vec[14] = _normalize(_safe(ichi.get("atr_stop_multiplier"), 2.5), 0.5, 5.0)
    vec[15] = _normalize(_safe(ichi.get("min_confluence_score"), 1), 0, 10)
    vec[16] = _normalize(_safe(ichi.get("tier_c"), 1), 0, 10)
    vec[17] = 0.0  # reserved

    # ── dims 18-20: Asian Breakout ─────────────────────────────────
    ab = strats.get("asian_breakout", {})
    vec[18] = _normalize(_safe(ab.get("min_range_pips"), 3), 0.0, 50.0)
    vec[19] = _normalize(_safe(ab.get("max_range_pips"), 80), 10.0, 200.0)
    vec[20] = _normalize(_safe(ab.get("rr_ratio"), 2.0), 0.5, 5.0)

    # ── dims 21-23: EMA Pullback ───────────────────────────────────
    ep = strats.get("ema_pullback", {})
    vec[21] = _normalize(_safe(ep.get("min_ema_angle_deg"), 2), 0.0, 90.0)
    vec[22] = _normalize(_safe(ep.get("pullback_candles_max"), 20), 1, 50)
    vec[23] = _normalize(_safe(ep.get("rr_ratio"), 2.0), 0.5, 5.0)

    # Final safety: clamp everything
    np.clip(vec, 0.0, 1.0, out=vec)
    return vec


# ── outcome fingerprint embedding (20 dims) ─────────────────────────


def embed_outcome(outcome: dict[str, Any]) -> np.ndarray:
    """Build a 20-dim outcome fingerprint vector, all values in [0, 1].

    Parameters
    ----------
    outcome : dict with outcome metric keys (win_rate, profit_factor, etc.).

    Returns
    -------
    np.ndarray of shape (20,).
    """
    vec = np.zeros(20, dtype=np.float64)

    def g(key: str, default: float = 0.0) -> float:
        return _safe(outcome.get(key), default)

    # ── dims 0-4: Core ─────────────────────────────────────────────
    vec[0] = _clamp(g("win_rate", 0.0))  # already [0, 1]
    vec[1] = _normalize(g("profit_factor", 1.0), 0.0, 5.0)
    vec[2] = _normalize(g("total_return_pct", 0.0), -20.0, 30.0)
    vec[3] = _normalize(g("sharpe_ratio", 0.0), -2.0, 5.0)
    vec[4] = _normalize(g("max_drawdown_pct", 0.0), 0.0, 20.0)

    # ── dims 5-9: Trades ───────────────────────────────────────────
    total_trades = g("total_trades", 0)
    vec[5] = _log_normalize(max(total_trades, 1), 1, 10000)
    vec[6] = _normalize(g("avg_r_multiple", 0.0), -3.0, 5.0)
    vec[7] = _normalize(g("best_trade_r", 0.0), 0.0, 10.0)
    vec[8] = _normalize(g("worst_trade_r", -1.0), -5.0, 0.0)
    vec[9] = _log_normalize(max(g("avg_trade_duration_bars", 1), 1), 1, 5000)

    # ── dims 10-14: Combine ────────────────────────────────────────
    passed = g("passed", 0.0)
    vec[10] = 1.0 if passed else 0.0
    vec[11] = _normalize(g("final_balance", 50000), 40000, 60000)
    vec[12] = _normalize(g("distance_to_target", 0.0), 0.0, 5000)
    vec[13] = _normalize(g("best_day_profit", 0.0), 0.0, 3000)
    vec[14] = _clamp(g("consistency_ratio", 0.0))  # already [0, 1]

    # ── dims 15-19: Robustness ─────────────────────────────────────
    vec[15] = _clamp(g("p_value", 1.0))  # [0, 1]
    vec[16] = _normalize(g("n_permutations_beaten", 0), 0, 1000)
    vec[17] = _clamp(g("edge_filtered_pct", 0.0))  # [0, 1]
    vec[18] = _clamp(g("signals_entered_pct", 0.0))  # [0, 1]
    # win_rate_long - win_rate_short  →  directional bias in [-1, 1]
    wr_long = g("win_rate_long", 0.5)
    wr_short = g("win_rate_short", 0.5)
    vec[19] = _normalize(wr_long - wr_short, -1.0, 1.0)

    # Final safety: clamp everything
    np.clip(vec, 0.0, 1.0, out=vec)
    return vec
