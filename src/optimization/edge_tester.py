"""Edge isolation testing — measure each edge's marginal OOS contribution.

Design
------
For each named edge in the strategy:

1. Run a baseline backtest with the edge **disabled**.
2. Run a comparison backtest with the edge **enabled**.
3. Compute the marginal impact: delta Sharpe = (with_edge) - (base).
4. Recommend keeping the edge when its marginal OOS Sharpe is positive.

The tester operates on the OOS (out-of-sample) slice only so that the
results are not contaminated by in-sample optimization.

Edge names correspond to keys under the ``edges`` sub-dict of the strategy
configuration.  Each edge is toggled by setting ``enabled=True/False`` in
its sub-dict.

Usage
-----
::

    tester = EdgeIsolationTester(initial_balance=10_000)
    impacts = tester.test_all_edges(oos_data, base_config=best_params)
    for name, impact in impacts.items():
        print(f"{name}: marginal={impact.marginal_impact:+.3f}, "
              f"keep={impact.recommended}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Edge keys that the tester iterates over.  Each maps to a sub-dict key
# under ``config["edges"]`` that contains at least ``{"enabled": bool}``.
_EDGE_NAMES = [
    "time_of_day",
    "day_of_week",
    "london_open_delay",
    "candle_close_confirmation",
    "spread_filter",
    "friday_close",
    "regime_filter",
    "time_stop",
    "bb_squeeze",
    "confluence_scoring",
    "equity_curve",
]


# =============================================================================
# EdgeImpact
# =============================================================================


@dataclass
class EdgeImpact:
    """Marginal performance impact of a single edge.

    Attributes
    ----------
    edge_name:
        Identifier of the edge being tested.
    base_sharpe:
        Sharpe ratio with the edge disabled (baseline).
    with_edge_sharpe:
        Sharpe ratio with the edge enabled.
    marginal_impact:
        ``with_edge_sharpe - base_sharpe``.  Positive values indicate the
        edge adds value on OOS data.
    base_trades:
        Number of trades executed in the baseline backtest.
    with_edge_trades:
        Number of trades executed with the edge enabled.
    recommended:
        ``True`` when ``marginal_impact > 0`` (edge improves OOS Sharpe).
    """

    edge_name: str
    base_sharpe: float
    with_edge_sharpe: float
    marginal_impact: float
    base_trades: int
    with_edge_trades: int
    recommended: bool


# =============================================================================
# EdgeIsolationTester
# =============================================================================


class EdgeIsolationTester:
    """Test each strategy edge's marginal contribution on OOS data.

    Parameters
    ----------
    initial_balance:
        Starting account equity for every backtest run.  Default: 10 000.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self._initial_balance = initial_balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_all_edges(
        self, data: pd.DataFrame, base_config: dict
    ) -> Dict[str, EdgeImpact]:
        """Evaluate every named edge's marginal OOS contribution.

        For each edge the method runs two backtests:
        - **Baseline**: edge disabled, all other edges as in *base_config*.
        - **With edge**: edge enabled, all other edges as in *base_config*.

        Parameters
        ----------
        data:
            Out-of-sample OHLCV ``DataFrame`` (1-minute UTC).
        base_config:
            Parameter dict from the IS optimization phase.  ``edges`` sub-
            dict values from this config are used as the starting state for
            all other edges.

        Returns
        -------
        dict[str, EdgeImpact]
            One ``EdgeImpact`` entry per edge name.
        """
        results: Dict[str, EdgeImpact] = {}

        for edge_name in _EDGE_NAMES:
            impact = self._test_single_edge(data, base_config, edge_name)
            results[edge_name] = impact
            logger.info(
                "Edge '%s': base_sharpe=%.3f, with_edge_sharpe=%.3f, "
                "marginal=%.3f, recommended=%s",
                edge_name,
                impact.base_sharpe,
                impact.with_edge_sharpe,
                impact.marginal_impact,
                impact.recommended,
            )

        return results

    def test_single_edge(
        self, data: pd.DataFrame, base_config: dict, edge_name: str
    ) -> EdgeImpact:
        """Evaluate the marginal impact of one specific edge.

        This is the public single-edge entry point — delegates to the
        internal implementation.

        Parameters
        ----------
        data:
            OOS backtest data.
        base_config:
            Optimised parameter dict.
        edge_name:
            Key of the edge to isolate (e.g. ``"regime_filter"``).
        """
        return self._test_single_edge(data, base_config, edge_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _test_single_edge(
        self, data: pd.DataFrame, base_config: dict, edge_name: str
    ) -> EdgeImpact:
        """Run baseline (disabled) and enabled backtests for *edge_name*."""
        # Baseline: disable the target edge.
        config_base = self._set_edge_enabled(base_config, edge_name, enabled=False)
        base_sharpe, base_trades = self._run(data, config_base)

        # With-edge: enable the target edge.
        config_with = self._set_edge_enabled(base_config, edge_name, enabled=True)
        with_sharpe, with_trades = self._run(data, config_with)

        marginal = with_sharpe - base_sharpe
        recommended = marginal > 0.0

        return EdgeImpact(
            edge_name=edge_name,
            base_sharpe=base_sharpe,
            with_edge_sharpe=with_sharpe,
            marginal_impact=marginal,
            base_trades=base_trades,
            with_edge_trades=with_trades,
            recommended=recommended,
        )

    def _run(self, data: pd.DataFrame, config: dict) -> tuple[float, int]:
        """Execute one backtest and return (sharpe, n_trades)."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        try:
            bt = IchimokuBacktester(
                config=config,
                initial_balance=self._initial_balance,
            )
            result = bt.run(data, log_trades=False)
            sharpe = float(result.metrics.get("sharpe_ratio") or 0.0)
            if not math.isfinite(sharpe):
                sharpe = 0.0
            n_trades = int(result.metrics.get("total_trades") or 0)
            return sharpe, n_trades
        except Exception as exc:
            logger.debug("Backtest failed in edge tester: %s", exc)
            return 0.0, 0

    @staticmethod
    def _set_edge_enabled(
        base_config: dict, edge_name: str, enabled: bool
    ) -> dict:
        """Return a deep-copied config with *edge_name* toggled.

        Avoids mutating the caller's config dict.  The ``edges`` sub-dict
        is expected to follow the ``EdgeConfig`` model where each edge is a
        dict with at least ``{"enabled": bool, "params": {...}}``.
        """
        import copy

        config = copy.deepcopy(base_config)

        edges = config.setdefault("edges", {})
        if edge_name not in edges:
            edges[edge_name] = {}
        edges[edge_name]["enabled"] = enabled

        return config
