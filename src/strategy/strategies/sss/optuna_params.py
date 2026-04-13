"""SSS Optuna parameter space registration.

Registers a thin adapter in ``STRATEGY_REGISTRY`` so that
``OptunaOptimizer(strategy_key='sss')`` resolves the SSS search space
without requiring ``SSSStrategy`` to extend the ``Strategy`` ABC.

Import this module (or ensure it is imported before running optimization)
to activate the registration::

    import src.strategy.strategies.sss.optuna_params  # noqa: F401

The adapter is registered under the key ``'sss'``.

Hard-coded params NOT in the search space
-----------------------------------------
The following SSS parameters are intentionally fixed and excluded from
Optuna's search space to prevent over-fitting on low-information axes:

- ``cbc_candles``       : always 3 (canonical CBC pattern)
- ``fifty_tap_level``   : always 0.5 (50% retracement by definition)
- ``spread_multiplier`` : cost model constant, not a strategy param
- ``min_stop_pips``     : safety floor, not a performance lever
- sequence state-machine transition logic
"""

from __future__ import annotations

from src.strategy.base import STRATEGY_REGISTRY


class SSSOptunaAdapter:
    """Thin adapter exposing the SSS Optuna parameter search space."""

    name = "sss"

    def suggest_params(self, trial) -> dict:
        """Sample one point from the SSS parameter search space.

        The bounds are intentionally widened enough to cover the current
        futures profile defaults, so optimization can compare the live
        profile against both stricter and looser SSS variants.
        """
        return {
            "active_strategies": ["sss"],
            "strategies": {
                "sss": {
                    "swing_lookback_n": trial.suggest_int(
                        "sss_swing_lookback_n", 2, 5
                    ),
                    "min_swing_pips": trial.suggest_float(
                        "sss_min_swing_pips", 0.3, 5.0, step=0.1
                    ),
                    "ss_candle_min": trial.suggest_int(
                        "sss_ss_candle_min", 6, 15
                    ),
                    "iss_candle_min": trial.suggest_int(
                        "sss_iss_candle_min", 2, 5
                    ),
                    "iss_candle_max": trial.suggest_int(
                        "sss_iss_candle_max", 5, 8
                    ),
                    "entry_mode": trial.suggest_categorical(
                        "sss_entry_mode", ["cbc_only", "fifty_tap", "combined"]
                    ),
                    "min_confluence_score": trial.suggest_int(
                        "sss_min_confluence_score", 0, 4
                    ),
                }
            }
        }


if "sss" not in STRATEGY_REGISTRY:
    STRATEGY_REGISTRY["sss"] = SSSOptunaAdapter
