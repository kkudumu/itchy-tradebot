"""Daily macro regime classifier.

Classifies each trading day into one of 5 regimes based on the
daily percentage moves of DXY, SPX, and US10Y:

    risk_on:        SPX rallying, DXY weakening (risk appetite)
    risk_off:       SPX falling, DXY strengthening (flight to safety)
    dollar_driven:  DXY moving strongly, SPX relatively flat
    inflation_fear: US10Y spiking, SPX weak (rate shock)
    mixed:          No clear macro theme (small moves)

Thresholds are calibrated for daily percentage changes:
- "Strong" move: |pct_change| > 0.5% for DXY/SPX, > 1.5% for US10Y
- "Spike" move: |pct_change| > 2.0% for US10Y

Gold correlation patterns:
- risk_off + dollar_driven -> bearish for gold (strong USD headwind)
- risk_off + DXY_down -> bullish for gold (safe haven bid)
- inflation_fear -> bullish for gold (inflation hedge)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RegimeLabel(Enum):
    """Macro regime classification labels."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    DOLLAR_DRIVEN = "dollar_driven"
    INFLATION_FEAR = "inflation_fear"
    MIXED = "mixed"


# Thresholds for regime classification (daily pct change)
_DXY_STRONG = 0.5     # |DXY pct change| > 0.5% = strong dollar move
_SPX_STRONG = 0.5     # |SPX pct change| > 0.5% = strong equity move
_US10Y_STRONG = 1.5   # |US10Y pct change| > 1.5% = notable yield move
_US10Y_SPIKE = 2.0    # |US10Y pct change| > 2.0% = yield spike


def classify_single_day(
    dxy_pct: float,
    spx_pct: float,
    us10y_pct: float,
) -> RegimeLabel:
    """Classify a single day's macro regime.

    Parameters
    ----------
    dxy_pct: DXY daily percentage change.
    spx_pct: SPX daily percentage change.
    us10y_pct: US10Y daily percentage change.

    Returns
    -------
    RegimeLabel for the day.
    """
    dxy_abs = abs(dxy_pct)
    spx_abs = abs(spx_pct)
    us10y_abs = abs(us10y_pct)

    # Inflation fear: yields spike sharply while equities drop
    if us10y_abs >= _US10Y_SPIKE and spx_pct < 0:
        return RegimeLabel.INFLATION_FEAR

    # Risk-on: equities rising, dollar weakening
    if spx_pct > _SPX_STRONG and dxy_pct < 0:
        return RegimeLabel.RISK_ON

    # Risk-off: equities falling, dollar strengthening
    if spx_pct < -_SPX_STRONG and dxy_pct > 0:
        return RegimeLabel.RISK_OFF

    # Dollar-driven: DXY moving strongly, equities relatively flat
    if dxy_abs >= _DXY_STRONG and spx_abs < _SPX_STRONG:
        return RegimeLabel.DOLLAR_DRIVEN

    # Inflation fear (secondary): yields notable + equities weak
    if us10y_abs >= _US10Y_STRONG and spx_pct < -_SPX_STRONG:
        return RegimeLabel.INFLATION_FEAR

    # Mixed: no clear dominant theme
    return RegimeLabel.MIXED


class RegimeClassifier:
    """Classify daily macro regimes from a macro panel DataFrame.

    Parameters
    ----------
    dxy_strong: Threshold for strong DXY daily move (default 0.5%).
    spx_strong: Threshold for strong SPX daily move (default 0.5%).
    us10y_spike: Threshold for US10Y yield spike (default 2.0%).
    """

    def __init__(
        self,
        dxy_strong: float = _DXY_STRONG,
        spx_strong: float = _SPX_STRONG,
        us10y_spike: float = _US10Y_SPIKE,
    ) -> None:
        self._dxy_strong = dxy_strong
        self._spx_strong = spx_strong
        self._us10y_spike = us10y_spike

    def classify(self, panel: pd.DataFrame) -> pd.Series:
        """Classify each day in the macro panel.

        Parameters
        ----------
        panel:
            DataFrame with columns: dxy_pct_change, spx_pct_change,
            us10y_pct_change.

        Returns
        -------
        pd.Series of regime label strings with name="regime".
        """
        regimes = []
        for _, row in panel.iterrows():
            dxy_pct = float(row.get("dxy_pct_change", 0) or 0)
            spx_pct = float(row.get("spx_pct_change", 0) or 0)
            us10y_pct = float(row.get("us10y_pct_change", 0) or 0)

            label = classify_single_day(dxy_pct, spx_pct, us10y_pct)
            regimes.append(label.value)

        result = pd.Series(regimes, index=panel.index, name="regime")
        logger.info(
            "Classified %d days: %s",
            len(result),
            result.value_counts().to_dict(),
        )
        return result

    @staticmethod
    def get_regime_for_date(
        regime_series: pd.Series,
        target_date: pd.Timestamp,
    ) -> str:
        """Look up the regime for a specific date.

        If the exact date is missing, uses the most recent prior date
        (forward-fill semantics). Falls back to "mixed" if no data exists.

        Parameters
        ----------
        regime_series: Output of classify().
        target_date: The date to look up.

        Returns
        -------
        Regime label string.
        """
        if target_date in regime_series.index:
            return str(regime_series.loc[target_date])

        # Find nearest prior date
        prior = regime_series.index[regime_series.index <= target_date]
        if len(prior) > 0:
            return str(regime_series.loc[prior[-1]])

        return RegimeLabel.MIXED.value
