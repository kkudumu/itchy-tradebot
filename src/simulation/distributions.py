"""
Fat-tailed trade R-multiple distributions and block bootstrapping.

TradeDistribution fits a skewed-t distribution to observed R-multiples
from backtest data, then samples from it to generate synthetic trade
sequences with realistic tail behaviour.

BlockBootstrapper groups trades by calendar day and resamples whole
day-blocks with replacement, preserving intra-day trade correlations
that exist due to session regimes, news events, and volatility clusters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Minimum sample size required before fitting a distribution.
# Below this, fall back to empirical resampling.
_MIN_FIT_SAMPLES: int = 10


# =============================================================================
# FittedDistribution
# =============================================================================

@dataclass
class FittedDistribution:
    """Parameters of a fitted skewed-t distribution for one side of trades."""

    # scipy.stats.nct parameters (non-central t)
    df: float          # degrees of freedom — controls tail heaviness
    nc: float          # non-centrality — controls skew
    loc: float         # location shift
    scale: float       # scale factor

    win_probability: float  # fraction of trades that are wins

    # Empirical fallback arrays (used when fit has too few samples)
    empirical_wins: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_losses: np.ndarray = field(default_factory=lambda: np.array([]))

    fit_quality_wins: float = 0.0    # K-S statistic vs fitted (lower = better)
    fit_quality_losses: float = 0.0


# =============================================================================
# TradeDistribution
# =============================================================================

class TradeDistribution:
    """Model trade R-multiple distributions from backtest data.

    Fits separate skewed non-central t-distributions to the winning and
    losing sides.  The non-central t has heavier tails than a normal
    distribution and the non-centrality parameter captures the asymmetry
    often observed in real trade outcomes.

    Parameters
    ----------
    tail_df_cap:
        Upper bound on fitted degrees of freedom.  Lower values force
        heavier tails.  Default: 30.0 (above this the tails are nearly
        normal).
    min_df:
        Minimum degrees of freedom — prevents degenerate fits on sparse
        data.  Default: 2.1 (keeps finite variance).
    """

    def __init__(
        self,
        tail_df_cap: float = 30.0,
        min_df: float = 2.1,
    ) -> None:
        self._tail_df_cap = tail_df_cap
        self._min_df = min_df
        self._fitted: Optional[FittedDistribution] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, r_multiples: List[float]) -> FittedDistribution:
        """Fit distribution to observed R-multiples from a backtest.

        Parameters
        ----------
        r_multiples:
            List of R-multiple outcomes.  Positive values are wins,
            non-positive are losses.

        Returns
        -------
        FittedDistribution with all parameters populated.
        """
        arr = np.array(r_multiples, dtype=float)
        wins = arr[arr > 0.0]
        losses = arr[arr <= 0.0]

        win_probability = len(wins) / len(arr) if len(arr) > 0 else 0.5

        # --- fit winning side ---
        emp_wins = wins.copy()
        df_w, nc_w, loc_w, scale_w, ks_w = self._fit_nct(wins, side="win")

        # --- fit losing side (negate, fit, negate back) ---
        emp_losses = losses.copy()
        neg_losses = -losses  # flip to positive domain for fitting
        df_l, nc_l, loc_l, scale_l, ks_l = self._fit_nct(neg_losses, side="loss")

        fitted = FittedDistribution(
            df=df_w,
            nc=nc_w,
            loc=loc_w,
            scale=scale_w,
            win_probability=win_probability,
            empirical_wins=emp_wins,
            empirical_losses=emp_losses,
            fit_quality_wins=ks_w,
            fit_quality_losses=ks_l,
        )

        # Store loss parameters as secondary attributes for sampling
        fitted._loss_df = df_l       # type: ignore[attr-defined]
        fitted._loss_nc = nc_l       # type: ignore[attr-defined]
        fitted._loss_loc = loc_l     # type: ignore[attr-defined]
        fitted._loss_scale = scale_l # type: ignore[attr-defined]

        self._fitted = fitted
        logger.debug(
            "TradeDistribution fitted: win_prob=%.3f df_win=%.2f df_loss=%.2f "
            "ks_win=%.4f ks_loss=%.4f",
            win_probability, df_w, df_l, ks_w, ks_l,
        )
        return fitted

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n trade R-multiples from the fitted distribution.

        Parameters
        ----------
        n:
            Number of trades to sample.
        rng:
            NumPy Generator instance for reproducible randomness.

        Returns
        -------
        Array of R-multiple values (positive = win, negative = loss).

        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._fitted is None:
            raise RuntimeError("Call fit() before sample().")

        results = np.empty(n, dtype=float)
        is_win = rng.random(n) < self._fitted.win_probability

        n_wins = int(is_win.sum())
        n_losses = n - n_wins

        if n_wins > 0:
            results[is_win] = self._sample_side(
                n_wins, rng,
                self._fitted.df, self._fitted.nc,
                self._fitted.loc, self._fitted.scale,
                self._fitted.empirical_wins,
                negate=False,
            )

        if n_losses > 0:
            results[~is_win] = self._sample_side(
                n_losses, rng,
                self._fitted._loss_df, self._fitted._loss_nc,  # type: ignore[attr-defined]
                self._fitted._loss_loc, self._fitted._loss_scale,  # type: ignore[attr-defined]
                self._fitted.empirical_losses,
                negate=True,   # flip back to negative domain
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_nct(
        self,
        data: np.ndarray,
        side: str,
    ) -> tuple[float, float, float, float, float]:
        """Fit a non-central t-distribution to one-sided positive data.

        Returns (df, nc, loc, scale, ks_statistic).
        Falls back to normal approximation when data is too sparse.
        """
        if len(data) < _MIN_FIT_SAMPLES:
            logger.debug(
                "Insufficient %s samples (%d) — using normal approximation.",
                side, len(data),
            )
            # Fall back to normal approximation with fixed low df
            mu = float(data.mean()) if len(data) > 0 else 1.0
            sigma = float(data.std(ddof=1)) if len(data) > 1 else 0.5
            return self._min_df, 0.0, mu, max(sigma, 0.01), 0.0

        mu = float(data.mean())
        sigma = float(data.std(ddof=1)) if len(data) > 1 else 0.0

        # Minimum meaningful scale: at least 5% of the mean, or 0.1R absolute.
        # Near-zero variance (e.g. all trades exactly the same R) would make
        # NCT fit collapse to a degenerate scale, producing microscopic samples.
        min_scale = max(abs(mu) * 0.05, 0.1)

        try:
            # scipy nct.fit returns (df, nc, loc, scale)
            df, nc, loc, scale = stats.nct.fit(data)

            # Clamp df to prevent degenerate distributions
            df = float(np.clip(df, self._min_df, self._tail_df_cap))
            nc = float(np.clip(nc, -10.0, 10.0))
            loc = float(loc)
            scale = float(max(scale, min_scale))

            # Sanity-check: if fitted scale is implausibly small compared with the
            # data spread, the optimiser has found a bad local minimum.  Fall back
            # to a plain t-fit in this case.
            if sigma > 0.0 and scale < sigma * 0.01:
                raise ValueError(
                    f"NCT scale={scale:.6f} << data σ={sigma:.4f}; fit collapsed."
                )

            # K-S test for fit quality (lower = better fit)
            ks_stat, _ = stats.kstest(data, "nct", args=(df, nc, loc, scale))

        except Exception as exc:
            logger.warning("NCT fit failed for %s side: %s — using t-dist.", side, exc)
            # Fallback to plain t-distribution
            try:
                df_t, loc_t, scale_t = stats.t.fit(data)
                df = float(np.clip(df_t, self._min_df, self._tail_df_cap))
                nc = 0.0
                loc = float(loc_t)
                scale = float(max(scale_t, min_scale))
            except Exception:
                # Last resort: use normal approximation
                df = self._min_df
                nc = 0.0
                loc = mu
                scale = max(sigma, min_scale)
            ks_stat = 0.0

        return df, nc, loc, scale, float(ks_stat)

    def _sample_side(
        self,
        n: int,
        rng: np.random.Generator,
        df: float,
        nc: float,
        loc: float,
        scale: float,
        empirical: np.ndarray,
        negate: bool,
    ) -> np.ndarray:
        """Draw n samples from one side (win or loss).

        Uses empirical resampling as a safety fallback when the fitted
        distribution would produce implausible extreme values.
        """
        # Draw from fitted non-central t
        samples = stats.nct.rvs(
            df=df, nc=nc, loc=loc, scale=scale,
            size=n, random_state=rng,
        )

        # Clip to [0, 20R] on the positive domain — removes unphysical extremes
        samples = np.clip(samples, 0.0, 20.0)

        # If majority of samples collapsed to 0, fall back to empirical
        if len(empirical) >= _MIN_FIT_SAMPLES and float((samples == 0).mean()) > 0.5:
            logger.debug("NCT collapsed — falling back to empirical resampling.")
            samples = rng.choice(np.abs(empirical), size=n, replace=True)

        if negate:
            samples = -samples

        return samples


# =============================================================================
# BlockBootstrapper
# =============================================================================

class BlockBootstrapper:
    """Block bootstrap that preserves intra-day trade groups.

    Standard bootstrap samples individual trades independently, destroying
    any correlation structure within a single trading day (e.g., two trades
    on the same volatile London session tend to both win or both lose).
    Block bootstrap instead resamples whole calendar-day blocks, keeping
    intra-day correlations intact.

    Parameters
    ----------
    trades:
        List of trade dicts.  Each dict must have an 'entry_time' key
        with a datetime (or ISO-8601 string) value so trades can be
        assigned to calendar days.
    date_key:
        Dict key used to extract the trade date.  Default: 'entry_time'.
    """

    def __init__(
        self,
        trades: List[dict],
        date_key: str = "entry_time",
    ) -> None:
        self._date_key = date_key
        self._day_blocks: Dict[str, List[dict]] = self._group_by_day(trades)
        self._day_list: List[str] = sorted(self._day_blocks.keys())

        logger.debug(
            "BlockBootstrapper: %d trades across %d day-blocks.",
            len(trades), len(self._day_list),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_days(self) -> int:
        """Number of distinct trading days in the dataset."""
        return len(self._day_list)

    def resample(self, n_days: int, rng: np.random.Generator) -> List[dict]:
        """Resample day-blocks with replacement to produce n_days of trades.

        Parameters
        ----------
        n_days:
            Target number of trading days to generate.  If the source has
            fewer unique days, days will be reused multiple times.
        rng:
            NumPy Generator for reproducibility.

        Returns
        -------
        Flat list of trade dicts from n_days sampled day-blocks.
        """
        if not self._day_list:
            return []

        chosen_days = rng.choice(self._day_list, size=n_days, replace=True)
        result: List[dict] = []
        for day in chosen_days:
            result.extend(self._day_blocks[day])

        return result

    def resample_n_trades(self, n_trades: int, rng: np.random.Generator) -> List[dict]:
        """Resample until at least n_trades have been collected.

        Useful when the simulation needs a fixed trade count rather than
        a fixed number of days.
        """
        result: List[dict] = []
        while len(result) < n_trades:
            day = rng.choice(self._day_list)
            result.extend(self._day_blocks[day])
        return result[:n_trades]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_by_day(self, trades: List[dict]) -> Dict[str, List[dict]]:
        """Partition trades into calendar-day buckets."""
        blocks: Dict[str, List[dict]] = {}
        no_date_count = 0

        for trade in trades:
            raw = trade.get(self._date_key)
            date_str = self._extract_date_str(raw)

            if date_str is None:
                no_date_count += 1
                date_str = f"unknown_{no_date_count}"

            if date_str not in blocks:
                blocks[date_str] = []
            blocks[date_str].append(trade)

        if no_date_count > 0:
            logger.warning(
                "%d trades had no parseable '%s' — placed in unique buckets.",
                no_date_count, self._date_key,
            )

        return blocks

    @staticmethod
    def _extract_date_str(value: object) -> Optional[str]:
        """Convert a datetime, date, or ISO string to a YYYY-MM-DD string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")  # type: ignore[union-attr]
        if isinstance(value, str):
            # Accept 'YYYY-MM-DD...' — take the first 10 chars
            if len(value) >= 10:
                return value[:10]
        return None
