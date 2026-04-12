"""Guardrail checks for strategy robustness validation.

Provides two independent guardrails that sit between the optimizer and
the final go/no-go gate:

1. **Consecutive passes** — run the strategy on N offset data windows to
   confirm it doesn't depend on a lucky starting bar.
2. **Permutation significance** — shuffle candle returns and compare the
   real strategy return against the distribution of random returns.  If the
   real return is not statistically better than chance, the strategy is
   likely curve-fit.

Both functions accept a generic ``backtest_fn`` callable so they stay
decoupled from any specific backtesting engine.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bars per trading day: 23 hours * 60 minutes (futures session)
_BARS_PER_DAY = 60 * 23

# Window offsets in days
_OFFSET_DAYS = [0, 3, 7]

# Minimum bars before we fall back to the full dataset
_MIN_BARS = 5000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_consecutive_passes(
    data: pd.DataFrame,
    config: dict,
    instrument: str,
    backtest_fn: Callable[[pd.DataFrame, dict, str, float], dict | None],
    n_required: int = 3,
    initial_balance: float = 50_000.0,
) -> dict:
    """Run the backtest on N offset data windows and check pass/fail.

    Parameters
    ----------
    data:
        Full OHLCV DataFrame.
    config:
        Strategy configuration dict.
    instrument:
        Instrument symbol (e.g. ``"MGC"``).
    backtest_fn:
        ``(data, config, instrument, initial_balance) -> dict | None``.
        Must return a dict with at least ``{"passed": bool}`` or ``None``
        on failure.
    n_required:
        Minimum number of windows that must pass for the overall check
        to succeed.
    initial_balance:
        Starting account balance forwarded to *backtest_fn*.

    Returns
    -------
    dict
        ``{"all_passed": bool, "attempts": list[dict]}``
    """
    offsets = [d * _BARS_PER_DAY for d in _OFFSET_DAYS]
    attempts: list[dict] = []

    for offset in offsets:
        window = data.iloc[offset:]
        if len(window) < _MIN_BARS:
            window = data  # fall back to full dataset

        result = backtest_fn(window, config, instrument, initial_balance)
        passed = result is not None and result.get("passed", False)

        attempts.append(
            {
                "offset": offset,
                "passed": passed,
                "result": result,
            }
        )
        logger.info(
            "consecutive-pass offset=%d  passed=%s  bars=%d",
            offset,
            passed,
            len(window),
        )

    n_passed = sum(1 for a in attempts if a["passed"])
    all_passed = n_passed >= n_required

    logger.info(
        "consecutive-pass check: %d/%d passed (need %d) → %s",
        n_passed,
        len(attempts),
        n_required,
        "PASS" if all_passed else "FAIL",
    )

    return {"all_passed": all_passed, "attempts": attempts}


def check_permutation_significance(
    real_return: float,
    data: pd.DataFrame,
    config: dict,
    instrument: str,
    backtest_fn: Callable[[pd.DataFrame, dict, str, float], dict | None],
    n_permutations: int = 20,
    p_threshold: float = 0.05,
    initial_balance: float = 50_000.0,
) -> dict:
    """Compare real return against a distribution of permuted-data returns.

    Parameters
    ----------
    real_return:
        The strategy's return (%) on the real data.
    data:
        Full OHLCV DataFrame.
    config:
        Strategy configuration dict.
    instrument:
        Instrument symbol.
    backtest_fn:
        ``(data, config, instrument, initial_balance) -> dict | None``.
    n_permutations:
        Number of random permutations to run.
    p_threshold:
        Significance threshold (one-tailed).
    initial_balance:
        Starting account balance.

    Returns
    -------
    dict
        ``{"p_value": float, "significant": bool, "real_return": float,
          "mean_permuted": float, "permuted_returns": list[float]}``
    """
    permuted_returns: list[float] = []

    for i in range(n_permutations):
        perm_data = _permute_candles(data, seed=42 + i)
        result = backtest_fn(perm_data, config, instrument, initial_balance)
        ret = 0.0 if result is None else result.get("total_return_pct", 0.0)
        permuted_returns.append(ret)
        logger.debug("permutation %d/%d  return=%.2f%%", i + 1, n_permutations, ret)

    n_ge = sum(1 for r in permuted_returns if r >= real_return)
    p_value = n_ge / n_permutations
    significant = p_value < p_threshold
    mean_permuted = float(np.mean(permuted_returns)) if permuted_returns else 0.0

    logger.info(
        "permutation test: p=%.4f (threshold=%.2f) real=%.2f%% mean_perm=%.2f%% → %s",
        p_value,
        p_threshold,
        real_return,
        mean_permuted,
        "SIGNIFICANT" if significant else "NOT SIGNIFICANT",
    )

    return {
        "p_value": p_value,
        "significant": significant,
        "real_return": real_return,
        "mean_permuted": mean_permuted,
        "permuted_returns": permuted_returns,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _permute_candles(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Shuffle candle log-returns and reconstruct a synthetic OHLCV series.

    Algorithm
    ---------
    1. Compute log-returns of the close series.
    2. Shuffle the log-returns (deterministic via *seed*).
    3. Reconstruct a new close series from the shuffled returns,
       anchored at the original first close.
    4. Scale open/high/low by the ratio ``new_close / old_close``.
    5. Fix OHLC consistency: ``high >= max(open, close)``,
       ``low <= min(open, close)``.
    6. Volume is left unchanged.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns ``open, high, low, close, volume``.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Permuted OHLCV DataFrame with the same shape, columns, and index.
    """
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    if n < 2:
        return df.copy()

    # Step 1: log returns
    log_returns = np.diff(np.log(closes))

    # Step 2: shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(log_returns)

    # Step 3: reconstruct close series
    new_closes = np.empty(n, dtype=np.float64)
    new_closes[0] = closes[0]
    for i in range(len(log_returns)):
        new_closes[i + 1] = new_closes[i] * np.exp(log_returns[i])

    # Ensure positive prices
    new_closes = np.maximum(new_closes, 1e-8)

    # Step 4: scale OHLC by ratio
    ratio = new_closes / closes
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].values * ratio

    # Step 5: fix OHLC consistency
    o = out["open"].values
    h = out["high"].values
    l = out["low"].values  # noqa: E741
    c = out["close"].values

    out["high"] = np.maximum(h, np.maximum(o, c))
    out["low"] = np.minimum(l, np.minimum(o, c))

    return out
