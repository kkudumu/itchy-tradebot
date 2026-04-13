from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtesting.vectorbt_engine import IchimokuBacktester
from src.risk.circuit_breaker import DailyCircuitBreaker
from src.risk.instrument_sizer import FuturesContractSizer
from src.risk.exit_manager import HybridExitManager
from src.risk.position_sizer import AdaptivePositionSizer
from src.risk.trade_manager import TradeManager
from src.edges.base import EdgeContext


def _topstep_futures_config(initial_risk_pct: float = 0.5) -> dict:
    return {
        "risk": {
            "initial_risk_pct": initial_risk_pct,
            "reduced_risk_pct": 0.25,
            "phase_threshold_pct": 4.0,
            "daily_circuit_breaker_pct": 4.5,
            "max_concurrent_positions": 3,
        },
        "exit": {
            "tp_r_multiple": 1.5,
            "kijun_trail_start_r": 1.5,
            "higher_tf_kijun_start_r": 3.0,
        },
        "prop_firm": {
            "style": "topstep_combine_dollar",
            "name": "topstep_50k_combine",
            "account_size": 50_000.0,
            "profit_target_usd": 3_000.0,
            "max_loss_limit_usd_trailing": 2_000.0,
            "daily_loss_limit_usd": 1_000.0,
            "consistency_pct": 50.0,
            "max_micro_contracts": 50,
            "max_full_contracts": 5,
            "daily_reset_tz": "America/Chicago",
            "daily_reset_hour": 17,
        },
        "instrument_class": "futures",
        "instrument": {
            "class": "futures",
            "tick_size": 0.10,
            "tick_value_usd": 1.0,
            "commission_per_contract_round_trip": 1.02,
        },
        "strategies": {},
        "active_strategies": [],
    }


class TestTopstepFuturesBacktester:
    def test_nested_risk_config_wires_futures_sizer_and_point_value(self) -> None:
        backtester = IchimokuBacktester(
            config=_topstep_futures_config(initial_risk_pct=0.5),
            initial_balance=50_000.0,
        )

        assert backtester._point_value == pytest.approx(10.0)

        pos = backtester.trade_manager._sizer.calculate_position_size(
            account_equity=50_000.0,
            atr=5.0,
            atr_multiplier=1.0,
            point_value=1.0,
        )

        assert pos.risk_pct == pytest.approx(0.5)
        assert pos.risk_amount == pytest.approx(250.0)
        assert pos.lot_size == pytest.approx(5.0)

    def test_topstep_headroom_cap_keeps_three_losses_of_runway(self) -> None:
        backtester = IchimokuBacktester(
            config=_topstep_futures_config(initial_risk_pct=2.0),
            initial_balance=50_000.0,
        )
        ts = datetime(2026, 2, 9, 1, 0, tzinfo=timezone.utc)

        backtester.topstep_tracker.initialise(50_000.0, ts)
        backtester.trade_manager._breaker.start_day(50_000.0, ts=ts)
        backtester.topstep_tracker.update(ts, 50_000.0)

        capped_equity = backtester._cap_sizing_equity_for_open(
            current_balance=50_000.0,
            desired_sizing_equity=50_000.0,
        )

        assert capped_equity == pytest.approx((1_000.0 / 3.0) * 100.0 / 2.0)

    def test_mega_vision_shadow_mode_is_wired_from_config(self) -> None:
        cfg = _topstep_futures_config()
        cfg["mega_vision"] = {
            "mode": "shadow",
            "subscription_mode": True,
        }

        backtester = IchimokuBacktester(
            config=cfg,
            initial_balance=50_000.0,
        )

        assert backtester._mega_agent is not None
        assert backtester._mega_arbitrator is not None
        assert backtester._mega_shadow_recorder is not None


class TestDailyCircuitBreakerBudget:
    def test_remaining_budget_uses_dollar_limit_when_present(self) -> None:
        breaker = DailyCircuitBreaker(max_daily_loss_usd=1_000.0)
        ts = datetime(2026, 2, 9, 1, 0, tzinfo=timezone.utc)

        breaker.start_day(50_000.0, ts=ts)

        assert breaker.remaining_risk_budget(49_700.0) == pytest.approx(700.0)


class TestFuturesContractSizerRobustness:
    def test_non_finite_input_returns_zero_contracts(self) -> None:
        sizer = FuturesContractSizer(tick_size=0.10, tick_value_usd=1.0, max_contracts=50)

        assert sizer.size_for_risk(250.0, float("nan")) == 0


class TestTradeManagerEntryGuards:
    def test_non_finite_signal_price_is_rejected(self) -> None:
        trade_manager = TradeManager(
            position_sizer=AdaptivePositionSizer(initial_balance=50_000.0),
            circuit_breaker=DailyCircuitBreaker(),
            exit_manager=HybridExitManager(),
        )

        with pytest.raises(RuntimeError, match="not finite"):
            trade_manager.open_trade(
                entry_price=5000.0,
                stop_loss=float("nan"),
                take_profit=5010.0,
                direction="long",
                atr=5.0,
                point_value=10.0,
                account_equity=50_000.0,
            )

    def test_nan_atr_is_allowed_when_prices_are_valid(self) -> None:
        trade_manager = TradeManager(
            position_sizer=AdaptivePositionSizer(initial_balance=50_000.0),
            circuit_breaker=DailyCircuitBreaker(),
            exit_manager=HybridExitManager(),
        )

        trade_id, trade, pos = trade_manager.open_trade(
            entry_price=5000.0,
            stop_loss=4990.0,
            take_profit=5020.0,
            direction="long",
            atr=float("nan"),
            point_value=10.0,
            account_equity=50_000.0,
        )

        assert trade_id == 1
        assert trade.entry_price == 5000.0
        assert pos.lot_size > 0


class TestFridayCutoffEntryGuard:
    def test_friday_close_edge_blocks_new_entries_after_cutoff(self) -> None:
        backtester = IchimokuBacktester(
            config={
                "edges": {
                    "friday_close": {
                        "enabled": True,
                        "params": {"close_time_utc": "20:00", "day": 4},
                    }
                },
                "strategies": {},
                "active_strategies": [],
            },
            initial_balance=50_000.0,
        )
        context = EdgeContext(
            timestamp=datetime(2026, 2, 13, 21, 0, tzinfo=timezone.utc),
            day_of_week=4,
            close_price=5100.0,
            high_price=5105.0,
            low_price=5095.0,
            spread=0.0,
            session="new_york",
            adx=25.0,
            atr=10.0,
            indicator_values={},
        )

        allowed, results = backtester._check_entry_edges(context)

        assert allowed is False
        assert results[-1].edge_name == "friday_close"
