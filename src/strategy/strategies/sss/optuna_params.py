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
    """Thin adapter exposing the SSS Optuna parameter search space.

    This class is not a full ``Strategy`` subclass — it exists solely to
    provide ``suggest_params`` for ``PropFirmObjective`` without coupling
    the SSS standalone implementation to the ABC hierarchy.

    The returned dict uses the nested ``strategies.sss.*`` path so it can
    be passed directly as the ``config`` argument to ``SSSBacktester`` or
    merged into a composite config dict.
    """

    name = "sss"

    def suggest_params(self, trial) -> dict:
        """Sample one point from the SSS parameter search space.

        Parameters
        ----------
        trial:
            An ``optuna.Trial`` instance used to call ``trial.suggest_*``.

        Returns
        -------
        dict
            Nested config dict: ``{"strategies": {"sss": {...}}}``.
            Ready to pass as *config* to the SSS backtester.

        Search space (7 parameters, ~2 000 combinations)
        -------------------------------------------------
        swing_lookback_n    : int [2, 5]   — bars back for swing detection
        min_swing_pips      : float [0.5, 5.0, step=0.5] — minimum swing size
        ss_candle_min       : int [8, 15]  — min candles in SS leg
        iss_candle_min      : int [3, 5]   — min candles in ISS leg
        iss_candle_max      : int [6, 8]   — max candles in ISS leg
        entry_mode          : categorical  — CBC-only, 50%-tap, or combined
        min_confluence_score: int [2, 6]   — minimum confluence points to trade
        """
        return {
            "active_strategies": ["sss"],
            "strategies": {
                "sss": {
                    "swing_lookback_n": trial.suggest_int(
                        "sss_swing_lookback_n", 2, 5
                    ),
                    "min_swing_pips": trial.suggest_float(
                        "sss_min_swing_pips", 0.5, 5.0, step=0.5
                    ),
                    "ss_candle_min": trial.suggest_int(
                        "sss_ss_candle_min", 8, 15
                    ),
                    "iss_candle_min": trial.suggest_int(
                        "sss_iss_candle_min", 3, 5
                    ),
                    "iss_candle_max": trial.suggest_int(
                        "sss_iss_candle_max", 6, 8
                    ),
                    "entry_mode": trial.suggest_categorical(
                        "sss_entry_mode", ["cbc_only"]
                    ),
                    "min_confluence_score": trial.suggest_int(
                        "sss_min_confluence_score", 1, 4
                    ),
                }
            }
        }


# Register on import — idempotent guard prevents duplicate-key errors when
# the module is re-imported (e.g. during test collection).
if "sss" not in STRATEGY_REGISTRY:
    STRATEGY_REGISTRY["sss"] = SSSOptunaAdapter
