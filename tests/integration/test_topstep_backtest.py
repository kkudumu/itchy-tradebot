"""End-to-end TopstepX backtest integration test (plan Task 15).

Runs the multi-strategy backtest engine on a small slice of the MGC
futures data under TopstepX rules and verifies the engine completes
without crashing, returns the expected structure, and persists the
strategy telemetry parquet. Does NOT assert that trades happen — the
telemetry and post-run summary tell us why if they don't.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


DATA_FILE = Path("data/projectx_mgc_1m_20260101_20260409.parquet")

_SKIP_IF_NO_DATA = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason=f"MGC data file missing: {DATA_FILE}",
)


def _load_slice(n_bars: int = 5_000) -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    return df.head(n_bars).copy()


def _topstep_config() -> dict:
    return {
        "active_strategies": ["ichimoku", "asian_breakout", "ema_pullback", "sss"],
        "active_strategy": "ichimoku",  # backward compat for loader
        "strategies": {
            "ichimoku": {
                "ichimoku": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52},
                "adx": {"period": 14, "threshold": 20},
                "atr": {"period": 14, "stop_multiplier": 2.5},
                "signal": {
                    "min_confluence_score": 1,
                    "tier_a_plus": 7,
                    "tier_b": 5,
                    "tier_c": 1,
                    "timeframes": ["15M", "5M"],
                },
            },
            "asian_breakout": {
                "enabled": True,
                "min_range_pips": 3,
                "max_range_pips": 80,
                "rr_ratio": 2.0,
                "pip_value": 0.1,
            },
            "ema_pullback": {
                "enabled": True,
                "fast_ema": 8,
                "mid_ema": 18,
                "slow_ema": 50,
                "rr_ratio": 2.0,
            },
            "sss": {
                "enabled": True,
                "swing_lookback_n": 2,
                "min_swing_pips": 0.5,
                "ss_candle_min": 8,
                "iss_candle_min": 3,
                "iss_candle_max": 8,
                "max_bars_in_state": 100,
                "require_cbc_context": False,
                "entry_mode": "cbc_only",
                "warmup_bars": 50,
            },
        },
        "prop_firm": {
            "style": "topstep_combine_dollar",
            "account_size": 50_000.0,
            "profit_target_usd": 3_000.0,
            "max_loss_limit_usd_trailing": 2_000.0,
            "daily_loss_limit_usd": 1_000.0,
            "consistency_pct": 50.0,
            "daily_reset_tz": "America/Chicago",
            "daily_reset_hour": 17,
        },
        "instrument": {
            "class": "futures",
            "tick_size": 0.10,
            "tick_value_usd": 1.0,
            "commission_per_contract_round_trip": 1.40,
            "slippage_ticks": 1,
        },
        "max_concurrent_positions": 3,
        "initial_risk_pct": 0.5,
        "reduced_risk_pct": 0.75,
        "phase_threshold_pct": 4.0,
    }


@_SKIP_IF_NO_DATA
class TestTopstepBacktestEndToEnd:
    def test_engine_instantiates_with_topstep_config(self) -> None:
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        bt = IchimokuBacktester(config=_topstep_config(), initial_balance=50_000.0)
        assert bt.active_prop_firm_tracker is not None
        assert type(bt.active_prop_firm_tracker).__name__ == "TopstepCombineTracker"
        assert bt._session_clock is not None
        names = [n for n, _ in bt._active_strategies]
        assert set(names) == {"asian_breakout", "ema_pullback", "sss"}

    def test_small_slice_backtest_runs_without_crashing(self, tmp_path) -> None:
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        cfg = _topstep_config()
        cfg["run_id"] = "test_topstep_small"
        cfg["telemetry_output_dir"] = str(tmp_path / "telemetry")

        df = _load_slice(n_bars=5_000)
        bt = IchimokuBacktester(config=cfg, initial_balance=50_000.0)
        result = bt.run(df, instrument="MGC", log_trades=False, enable_learning=False)

        # Structural assertions — don't care about trade count yet
        assert result is not None
        assert isinstance(result.prop_firm, dict)
        assert result.prop_firm.get("style") == "topstep_combine_dollar"
        assert "active_tracker" in result.prop_firm
        assert result.prop_firm["active_tracker"]["initial_balance"] == 50_000.0

        # Telemetry parquet was written
        telem_parquet = tmp_path / "telemetry" / "strategy_telemetry.parquet"
        telem_summary = tmp_path / "telemetry" / "strategy_telemetry_summary.json"
        assert telem_parquet.exists(), f"telemetry parquet missing: {telem_parquet}"
        assert telem_summary.exists(), f"telemetry summary missing: {telem_summary}"

        # Summary JSON should be well-formed
        summary = json.loads(telem_summary.read_text(encoding="utf-8"))
        assert "per_strategy" in summary
        assert "per_session" in summary
        assert "top_rejection_stages" in summary

    def test_result_prop_firm_has_topstep_fields(self, tmp_path) -> None:
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        cfg = _topstep_config()
        cfg["run_id"] = "test_topstep_fields"
        cfg["telemetry_output_dir"] = str(tmp_path / "telemetry")

        df = _load_slice(n_bars=3_000)
        bt = IchimokuBacktester(config=cfg, initial_balance=50_000.0)
        result = bt.run(df, instrument="MGC", log_trades=False, enable_learning=False)

        at = result.prop_firm.get("active_tracker") or {}
        for key in (
            "style",
            "status",
            "initial_balance",
            "current_balance",
            "mll",
            "mll_locked",
            "total_profit",
            "best_day_profit",
            "distance_to_mll",
            "distance_to_target",
        ):
            assert key in at, f"missing active_tracker field: {key}"
