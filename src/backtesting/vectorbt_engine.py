"""
Main Ichimoku backtesting engine using Vectorbt.

Uses Portfolio.from_orders() (not from_signals()) because the strategy
requires partial position exits at 2R and Kijun-based trailing — features
that Portfolio.from_signals() cannot model.

Architecture
------------
The outer loop iterates through every 5M bar in chronological order.
At each bar the engine:
  1. Extracts multi-TF indicator values (all already shifted, no lookahead).
  2. Builds an EdgeContext from the bar data.
  3. Runs the full signal + edge pipeline when no trade is open.
  4. On a new signal, opens a trade via TradeManager.
  5. On an open trade, checks exits via HybridExitManager (partial at 2R,
     Kijun trail, edge exits, stop-loss).
  6. Logs every completed trade with market context and pgvector embedding.
  7. Updates PropFirmTracker and DailyCircuitBreaker each bar.

The resulting trade list + equity curve are wrapped in BacktestResult for
downstream analysis and export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.multi_tf import BacktestDataPreparer
from src.backtesting.metrics import PerformanceMetrics, PropFirmTracker, MultiPhasePropFirmTracker
from src.backtesting.trade_logger import TradeLogger
from src.edges.base import EdgeContext
from src.edges.manager import EdgeManager
from src.learning.adaptive_engine import AdaptiveLearningEngine
from src.learning.embeddings import EmbeddingEngine
from src.learning.memory_store import InMemorySimilarityStore, InMemoryStatsAnalyzer
from src.risk.circuit_breaker import DailyCircuitBreaker
from src.risk.exit_manager import ActiveTrade, HybridExitManager
from src.risk.position_sizer import AdaptivePositionSizer
from src.risk.trade_manager import TradeManager
from src.monitoring.health_monitor import StrategyHealthMonitor
from src.strategy.signal_engine import Signal, SignalEngine

logger = logging.getLogger(__name__)

# XAUUSD: $1 per 0.01 lot per $1 price move → point_value = 100 (per full lot)
_XAUUSD_POINT_VALUE: float = 100.0

# Minimum bars of warm-up data before the first signal scan
_WARMUP_BARS: int = 60


def _safe_float(val, default: float = 0.0) -> float:
    """Convert to float, returning default if NaN or None."""
    try:
        f = float(val) if val is not None else default
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


# =============================================================================
# BacktestResult
# =============================================================================

@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    trades: List[dict]
    """Full trade log — one dict per closed trade with context + embedding."""

    metrics: dict
    """Performance metrics from PerformanceMetrics.calculate()."""

    equity_curve: pd.Series
    """Portfolio equity at every 5M bar (DatetimeIndex)."""

    prop_firm: Dict[str, Any]
    """Prop firm tracking: status, profit_pct, max daily/total DD, etc."""

    daily_pnl: pd.Series
    """Daily P&L as a fraction of the start-of-day balance (DatetimeIndex)."""

    skipped_signals: int = 0
    """Number of valid signals that were blocked by edge filters."""

    total_signals: int = 0
    """Total raw signals generated before edge filtering."""

    challenge_simulation: Optional[Any] = None
    """ChallengeSimulationResult from post-backtest challenge simulation."""


# =============================================================================
# IchimokuBacktester
# =============================================================================

class IchimokuBacktester:
    """Event-driven Ichimoku backtester with full risk management.

    Parameters
    ----------
    config:
        Optional strategy configuration dict.  Missing keys fall back to
        SignalEngine / EdgeManager defaults.
    initial_balance:
        Starting account equity.  Default: 10 000.0.
    point_value:
        Monetary value of one price unit per full lot for the instrument.
        XAUUSD default: 100.0 (1 lot, $1 price move = $100 P&L).
    max_daily_loss_pct:
        DailyCircuitBreaker threshold.  Default: 2.0.
    prop_firm_profit_target_pct:
        Challenge profit target percentage.  Default: 8.0.
    prop_firm_max_daily_dd_pct:
        Challenge maximum daily drawdown percentage.  Default: 5.0.
    prop_firm_max_total_dd_pct:
        Challenge maximum total drawdown percentage.  Default: 10.0.
    prop_firm_time_limit_days:
        Challenge time limit in calendar days.  Default: 30.
    db_pool:
        Optional database connection pool for TradeLogger.  When None,
        trades are formatted but not persisted.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        initial_balance: float = 10_000.0,
        point_value: float = _XAUUSD_POINT_VALUE,
        max_daily_loss_pct: float = 2.0,
        prop_firm_profit_target_pct: float = 8.0,
        prop_firm_max_daily_dd_pct: float = 5.0,
        prop_firm_max_total_dd_pct: float = 10.0,
        prop_firm_time_limit_days: int = 30,
        db_pool=None,
    ) -> None:
        cfg = config or {}
        self._config = cfg
        self._initial_balance = initial_balance
        self._point_value = point_value

        # Trading costs — forex uses per-lot + spread points; futures
        # uses per-contract round-trip commission + tick slippage. The
        # engine reads the instrument config from cfg["instrument"]
        # (injected by the caller via run_demo_challenge.py) to pick
        # the right model.
        costs = cfg.get("edges", {}).get("trading_costs", {})
        self._commission_per_lot = float(costs.get("commission_per_lot", 0.0))
        self._spread_points = float(costs.get("spread_points", 0.0))

        # Futures cost path — opt-in via instrument.class: futures or
        # by explicit cfg["commission_per_contract_round_trip"].
        instrument_cfg = cfg.get("instrument") or {}
        self._instrument_class = str(
            instrument_cfg.get("class") or cfg.get("instrument_class") or "forex"
        ).lower()
        self._tick_size = float(
            instrument_cfg.get("tick_size") or cfg.get("tick_size") or 0.0
        )
        self._tick_value_usd = float(
            instrument_cfg.get("tick_value_usd")
            or cfg.get("tick_value_usd")
            or 0.0
        )
        self._commission_per_contract_rt = float(
            instrument_cfg.get("commission_per_contract_round_trip")
            or cfg.get("commission_per_contract_round_trip")
            or 0.0
        )
        self._slippage_ticks = int(
            instrument_cfg.get("slippage_ticks")
            or cfg.get("slippage_ticks")
            or (1 if self._instrument_class == "futures" else 0)
        )
        if self._instrument_class == "futures" and self._tick_size > 0:
            # Override point_value so the existing pnl_points * lot * pv
            # formula yields dollars for futures:
            #   $ per price-unit per contract = tick_value_usd / tick_size
            point_value = self._tick_value_usd / self._tick_size

        # Strategy components
        self.signal_engine = SignalEngine(config=cfg)
        self.edge_manager = EdgeManager(edge_configs=cfg.get("edges", {}))

        # Risk components
        sizer = AdaptivePositionSizer(
            initial_balance=initial_balance,
            initial_risk_pct=cfg.get("initial_risk_pct", 1.5),
            reduced_risk_pct=cfg.get("reduced_risk_pct", 0.75),
            phase_threshold_pct=cfg.get("phase_threshold_pct", 4.0),
        )
        breaker = DailyCircuitBreaker(
            max_daily_loss_pct=max_daily_loss_pct,
        )
        exit_mgr = HybridExitManager(
            tp_r_multiple=cfg.get("tp_r_multiple", 2.0),
            kijun_trail_start_r=cfg.get("kijun_trail_start_r", 1.5),
            higher_tf_kijun_start_r=cfg.get("higher_tf_kijun_start_r", 3.0),
        )
        self.trade_manager = TradeManager(
            position_sizer=sizer,
            circuit_breaker=breaker,
            exit_manager=exit_mgr,
            max_concurrent=cfg.get("max_concurrent_positions", 3),
        )

        # Learning + logging components
        self.embedding_engine = EmbeddingEngine()
        self.data_preparer = BacktestDataPreparer(
            ichi_tenkan_period=cfg.get("tenkan_period", 9),
            ichi_kijun_period=cfg.get("kijun_period", 26),
            ichi_senkou_b_period=cfg.get("senkou_b_period", 52),
            adx_period=cfg.get("adx_period", 14),
            atr_period=cfg.get("atr_period", 14),
        )
        self.trade_logger = TradeLogger(
            db_pool=db_pool,
            embedding_engine=self.embedding_engine,
        )

        # In-memory learning components for always-on backtesting
        self._memory_store = InMemorySimilarityStore(vector_dim=64)
        self._memory_stats = InMemoryStatsAnalyzer()
        self.learning_engine = AdaptiveLearningEngine(
            similarity_search=self._memory_store,
            embedding_engine=self.embedding_engine,
            stats_analyzer=self._memory_stats,
        )

        # Prop firm tracker selection — dispatches on prop_firm.style.
        # The legacy PropFirmTracker (single-phase pct) remains as a
        # fallback for callers that pass explicit pct targets via
        # __init__ kwargs, because existing test fixtures rely on it.
        self.prop_firm_tracker = PropFirmTracker(
            profit_target_pct=prop_firm_profit_target_pct,
            max_daily_dd_pct=prop_firm_max_daily_dd_pct,
            max_total_dd_pct=prop_firm_max_total_dd_pct,
            time_limit_days=prop_firm_time_limit_days,
        )

        prop_firm_cfg = cfg.get("prop_firm", {}) or {}
        prop_firm_style = prop_firm_cfg.get("style", "the5ers_pct_phased")
        self._prop_firm_style = prop_firm_style

        # Active tracker — the one the engine's main loop calls update() on.
        # Starts as the legacy PropFirmTracker for backward compat; replaced
        # below when a style discriminator is present.
        self.active_prop_firm_tracker = self.prop_firm_tracker

        # TopstepX Combine path — dollar-based trailing MLL
        if prop_firm_style == "topstep_combine_dollar":
            from src.risk.session_clock import SessionClock
            from src.risk.topstep_tracker import TopstepCombineTracker
            from src.config.models import TopstepCombineConfig

            tx_kwargs = {
                k: v for k, v in prop_firm_cfg.items() if k != "style"
            }
            tx_config = TopstepCombineConfig(**tx_kwargs)
            self.topstep_tracker = TopstepCombineTracker(config=tx_config)
            self.active_prop_firm_tracker = self.topstep_tracker
            # The5ers tracker is unused in topstep mode but we keep a
            # safe default instance for backward compat on any legacy
            # call sites that still reference it.
            self.multi_phase_tracker = None
            # SessionClock for 5pm CT day rollover in the circuit breaker
            self._session_clock = SessionClock(
                reset_hour_local=int(tx_config.daily_reset_hour),
                reset_tz=tx_config.daily_reset_tz,
            )
            # Rebuild the circuit breaker with dollar-based daily loss
            # limit + session clock
            self.trade_manager._breaker = DailyCircuitBreaker(
                max_daily_loss_usd=float(tx_config.daily_loss_limit_usd),
                session_clock=self._session_clock,
            )
        else:
            # Legacy the5ers 2-step / single-phase pct path — unchanged
            self.topstep_tracker = None
            self._session_clock = None
            p1 = prop_firm_cfg.get("phase_1", {}) or {}
            p2 = prop_firm_cfg.get("phase_2", {}) or {}
            funded = prop_firm_cfg.get("funded", {}) or {}
            self.multi_phase_tracker = MultiPhasePropFirmTracker(
                account_size=float(prop_firm_cfg.get("account_size", initial_balance)),
                phase_1_profit_target_pct=float(p1.get("profit_target_pct", 8.0)),
                phase_1_max_loss_pct=float(p1.get("max_loss_pct", 10.0)),
                phase_1_daily_loss_pct=float(p1.get("daily_loss_pct", 5.0)),
                phase_2_profit_target_pct=float(p2.get("profit_target_pct", 5.0)),
                phase_2_max_loss_pct=float(p2.get("max_loss_pct", 10.0)),
                phase_2_daily_loss_pct=float(p2.get("daily_loss_pct", 5.0)),
                funded_monthly_target_pct=float(funded.get("monthly_target_pct", 10.0)),
                funded_max_loss_pct=float(funded.get("max_loss_pct", 10.0)),
                funded_daily_loss_pct=float(funded.get("daily_loss_pct", 5.0)),
            )

        self._metrics_calc = PerformanceMetrics()

        # Health monitor — autonomous self-awareness layer
        self.health_monitor = StrategyHealthMonitor(
            signal_engine=self.signal_engine,
            edge_manager=self.edge_manager,
            config=cfg.get("health_monitor"),
            mode="backtest",
            drought_window=cfg.get("health_monitor_drought_window", 500),
            baseline_win_rate=cfg.get("baseline_win_rate", 0.55),
        )

        # Multi-strategy support
        from src.strategy.strategies.asian_breakout import AsianBreakoutStrategy
        from src.strategy.strategies.ema_pullback import EMAPullbackStrategy
        from src.strategy.strategies.sss.strategy import SSSStrategy
        from src.strategy.signal_blender import SignalBlender

        active_strategy_names = cfg.get("active_strategies", ["ichimoku"])
        self._active_strategies: List[tuple] = []
        strategy_configs = cfg.get("strategies", {}) if isinstance(cfg.get("strategies"), dict) else {}

        for name in active_strategy_names:
            if name == "ichimoku":
                pass  # Existing signal_engine handles ichimoku via _scan_for_signal
            elif name == "asian_breakout":
                ab_config = strategy_configs.get("asian_breakout", {})
                self._active_strategies.append(("asian_breakout", AsianBreakoutStrategy(config=ab_config)))
            elif name == "ema_pullback":
                ep_config = strategy_configs.get("ema_pullback", {})
                self._active_strategies.append(("ema_pullback", EMAPullbackStrategy(config=ep_config)))
            elif name == "sss":
                # Wire SSS into the multi-strategy dispatch loop. The
                # strategy is stateful (bar-by-bar swing history +
                # sequence tracker) so it must be instantiated once and
                # fed every bar, same as asian_breakout / ema_pullback.
                sss_config = strategy_configs.get("sss", {})
                self._active_strategies.append(("sss", SSSStrategy(config=sss_config)))

        self._signal_blender = SignalBlender(multi_agree_bonus=2)

        # Strategy telemetry collector — captures every signal event,
        # filter rejection, and trade exit for downstream analysis and
        # mega-vision training data.
        from src.backtesting.strategy_telemetry import StrategyTelemetryCollector

        self.telemetry = StrategyTelemetryCollector(
            run_id=cfg.get("run_id", "backtest")
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        candles_1m: pd.DataFrame,
        instrument: str = "XAUUSD",
        log_trades: bool = False,
        enable_learning: bool = True,
        enable_screenshots: bool = False,
        screenshot_dir: str = "backtest_screenshots",
        live_dashboard=None,
    ) -> BacktestResult:
        """Execute a full backtest on 1-minute candle data.

        Parameters
        ----------
        candles_1m:
            1-minute OHLCV DataFrame with UTC DatetimeIndex.
        instrument:
            Instrument symbol label.  Default: 'XAUUSD'.
        log_trades:
            When True, persist completed trades to the database via
            TradeLogger.  Default: False.
        enable_learning:
            When True, the adaptive learning engine is active during
            backtest.  Trades are recorded to the in-memory similarity
            store and stats analyzer, and pre_trade_analysis() is
            consulted before each entry.  Default: True.
        enable_screenshots:
            When True, generate mplfinance candlestick charts at trade
            entry and exit.  Default: False (slow for large datasets).
        screenshot_dir:
            Root directory for backtest screenshot storage.
        live_dashboard:
            Optional LiveDashboardServer instance.  When provided, the
            backtester pushes state updates every 5 bars for real-time
            monitoring in a browser.

        Returns
        -------
        BacktestResult containing trades, metrics, equity curve, prop
        firm status, and daily P&L series.
        """
        logger.info("Starting backtest on %d 1M bars for %s", len(candles_1m), instrument)

        # 1. Prepare multi-TF data
        tf_data = self.data_preparer.prepare(candles_1m)
        self._candles_1m = candles_1m  # Keep reference for signal scanning
        df_5m = tf_data["5M"]

        if len(df_5m) < _WARMUP_BARS:
            logger.warning(
                "Only %d 5M bars available; at least %d needed for warm-up",
                len(df_5m), _WARMUP_BARS,
            )

        # Reset in-memory learning state for this run
        if enable_learning:
            self._memory_store.clear()
            self._memory_stats.clear()
            self.learning_engine.set_total_trades(0)
            logger.info("Adaptive learning enabled — phases: mechanical → statistical → similarity")

        # Screenshot helper (lazy import to avoid hard mplfinance dependency)
        _screenshot_fn = None
        if enable_screenshots:
            _screenshot_fn = self._make_screenshot_fn(screenshot_dir, instrument)

        # 2. Initialise state
        balance = self._initial_balance
        equity_records: List[tuple] = []  # (timestamp, equity)
        closed_trades: List[dict] = []
        equity_by_r: List[float] = []  # closed R-multiples for equity curve modifier
        learning_skipped: int = 0  # signals skipped by adaptive learning

        # Active trade state — supports multiple concurrent trades
        # Each entry: {trade_id: {"trade": ActiveTrade, "signal": Signal,
        #              "context": dict, "candles": int, "strategy": str}}
        active_trades: Dict[int, dict] = {}
        # Backwards-compat aliases (used by dashboard / end-of-data cleanup)
        active_trade_id: Optional[int] = None
        active_trade: Optional[ActiveTrade] = None
        active_signal: Optional[Signal] = None
        active_context: Optional[dict] = None
        candles_since_entry: int = 0

        # Prop firm tracker init — always initialise the legacy tracker
        # (it's read by the final BacktestResult for backward compat)
        # AND the active tracker when they differ (e.g. TopstepX mode).
        first_ts = pd.Timestamp(df_5m.index[0]).to_pydatetime()
        self.prop_firm_tracker.initialise(self._initial_balance, first_ts)
        if self.active_prop_firm_tracker is not self.prop_firm_tracker:
            self.active_prop_firm_tracker.initialise(self._initial_balance, first_ts)

        # Circuit breaker — set up first day
        prev_date = None

        total_signals = 0
        skipped_signals = 0

        # Pipeline diagnostics — tracks signal attrition at each stage
        _pipeline_counts = {
            "signals_generated": 0,
            "signals_filtered_in_trade": 0,
            "signals_filtered_no_open": 0,
            "signals_filtered_edge": 0,
            "signals_filtered_learning": 0,
            "signals_filtered_open_rejected": 0,
            "signals_entered": 0,
        }

        # Live dashboard setup
        import time as _time_mod
        _bt_start_time = _time_mod.monotonic()
        _total_5m_bars = len(df_5m)
        _equity_sample: List[float] = []  # Sampled equity for live chart (max ~500 points)
        _sample_interval = max(1, _total_5m_bars // 500)
        _n_wins = 0
        _n_losses = 0

        if live_dashboard is not None:
            live_dashboard.update({
                "total_bars": _total_5m_bars,
                "initial_balance": self._initial_balance,
                "instrument": instrument,
                "data_start_date": str(df_5m.index[0]) if len(df_5m) > 0 else "",
                "data_end_date": str(df_5m.index[-1]) if len(df_5m) > 0 else "",
            })

        # 2.5. Pre-flight health check
        # Pre-flight only scans Ichimoku signals. When other strategies are active,
        # don't let a failed pre-flight halt the entire backtest.
        has_non_ichimoku = len(self._active_strategies) > 0
        if has_non_ichimoku:
            logger.info("Skipping Ichimoku-only pre-flight — non-Ichimoku strategies are active.")
        else:
            pre_flight_result = self.health_monitor.pre_flight(candles_1m)
            if pre_flight_result.aborted:
                logger.error("Pre-flight ABORTED: %s", pre_flight_result.message)

        # 3. Main event loop over 5M bars
        for bar_idx, (ts_raw, row_5m) in enumerate(df_5m.iterrows()):
            ts = pd.Timestamp(ts_raw).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            bar_signal: Optional[Signal] = None  # Track signal for health monitor
            current_date = ts.date()

            # Day rollover
            if current_date != prev_date:
                self.trade_manager._breaker.start_day(balance, current_date)
                prev_date = current_date

            close = float(row_5m.get("close", np.nan))
            high = float(row_5m.get("high", np.nan))
            low = float(row_5m.get("low", np.nan))

            if np.isnan(close):
                equity_records.append((ts, balance))
                self.prop_firm_tracker.update(ts, balance)
                if self.active_prop_firm_tracker is not self.prop_firm_tracker:
                    self.active_prop_firm_tracker.update(ts, balance)
                continue

            # Retrieve higher-TF values aligned to this 5M bar
            htf_vals = self._get_htf_values(tf_data, ts)

            # ------------------------------------------------------------------
            # Always update strategy state and collect signals.
            # Strategies like Asian Breakout need every bar for range tracking.
            # Signals are collected here; the entry section below consumes them.
            # ------------------------------------------------------------------
            open_price = float(row_5m.get("open", close))
            _bar_strategy_signals: List[Signal] = []
            for _sn, _sobj in self._active_strategies:
                try:
                    if _sn == "asian_breakout":
                        _sig = _sobj.on_bar(ts, high=high, low=low, close=close)
                    elif _sn == "ema_pullback":
                        _sig = _sobj.on_bar(
                            ts, open=open_price, high=high, low=low, close=close,
                            ema_fast=float(row_5m.get("ema_fast", 0.0)),
                            ema_mid=float(row_5m.get("ema_mid", 0.0)),
                            ema_slow=float(row_5m.get("ema_slow", 0.0)),
                            atr=float(row_5m.get("atr", 0.0) or 0.0),
                        )
                    elif _sn == "sss":
                        # SSSStrategy.on_bar signature:
                        # (timestamp, *, open, high, low, close, atr, spread=0.0)
                        _sig = _sobj.on_bar(
                            ts,
                            open=open_price,
                            high=high,
                            low=low,
                            close=close,
                            atr=float(row_5m.get("atr", 0.0) or 0.0),
                            spread=float(row_5m.get("spread", 0.0) or 0.0),
                        )
                    else:
                        _sig = None
                    if _sig is not None:
                        _sig.strategy_name = _sn
                        _bar_strategy_signals.append(_sig)
                        _pipeline_counts["signals_generated"] += 1
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # Manage ALL open trades (no longer blocks new entries)
            # ------------------------------------------------------------------
            kijun_5m = float(row_5m.get("kijun", np.nan))
            kijun_1h = htf_vals.get("kijun_1h", np.nan)
            _closed_this_bar: List[int] = []

            for _tid, _tstate in list(active_trades.items()):
                _tobj = _tstate["trade"]
                _tsig = _tstate["signal"]
                _tctx = _tstate["context"]
                _tcandles = _tstate["candles"] + 1
                _tstate["candles"] = _tcandles

                # Build edge context for exit checks
                edge_ctx = self._build_edge_context(
                    ts=ts,
                    row_5m=row_5m,
                    htf_vals=htf_vals,
                    active_trade=_tobj,
                    candles_since_entry=_tcandles,
                    equity_by_r=equity_by_r,
                    confluence_score=_tsig.confluence_score if _tsig else 0,
                )

                # Check edge-based exit conditions (friday close, time stop)
                edge_exit_triggered, edge_exit_results = self.edge_manager.check_exit(edge_ctx)

                if edge_exit_triggered:
                    exit_reason = next(
                        (r.reason for r in edge_exit_results if not r.allowed),
                        "edge_exit",
                    )
                    trade_summary = self.trade_manager.close_trade(
                        trade_id=_tid,
                        exit_price=close,
                        reason=exit_reason,
                    )
                    trade_summary = self._enrich_trade_summary(
                        trade_summary, _tsig, _tctx, instrument, balance
                    )
                    closed_trades.append(trade_summary)
                    _r = float(trade_summary.get("r_multiple") or 0.0)
                    equity_by_r.append(_r)
                    if _r > 0: _n_wins += 1
                    else: _n_losses += 1
                    balance = self._update_balance_from_trade(balance, trade_summary)
                    self._record_learning(trade_summary, _tctx, enable_learning)
                    self.health_monitor.on_trade_closed(won=_r > 0)
                    if live_dashboard is not None:
                        self._push_trade_to_dashboard(live_dashboard, trade_summary, ts, len(closed_trades))
                    if _screenshot_fn:
                        _screenshot_fn(df_5m, bar_idx, "exit", _tid)
                    _closed_this_bar.append(_tid)
                    continue

                # Check stop, partial exit, and trail via HybridExitManager
                decision = self.trade_manager.update_trade(
                    trade_id=_tid,
                    current_price=close,
                    kijun_value=kijun_5m if not np.isnan(kijun_5m) else close,
                    higher_tf_kijun=kijun_1h if not np.isnan(kijun_1h) else None,
                    bar_high=high,
                    bar_low=low,
                )

                if decision.action == "full_exit":
                    trade_summary = self.trade_manager.closed_trades[-1]
                    trade_summary = self._enrich_trade_summary(
                        dict(trade_summary), _tsig, _tctx, instrument, balance
                    )
                    closed_trades.append(trade_summary)
                    _r = float(trade_summary.get("r_multiple") or 0.0)
                    equity_by_r.append(_r)
                    if _r > 0: _n_wins += 1
                    else: _n_losses += 1
                    balance = self._update_balance_from_trade(balance, trade_summary)
                    self._record_learning(trade_summary, _tctx, enable_learning)
                    self.health_monitor.on_trade_closed(won=_r > 0)
                    if live_dashboard is not None:
                        self._push_trade_to_dashboard(live_dashboard, trade_summary, ts, len(closed_trades))
                    if _screenshot_fn:
                        _screenshot_fn(df_5m, bar_idx, "exit", _tid)
                    _closed_this_bar.append(_tid)

                elif decision.action == "partial_exit":
                    partial_pnl = self._partial_pnl(
                        trade=_tobj,
                        exit_price=close,
                        close_pct=0.5,
                    )
                    balance += partial_pnl
                    logger.debug(
                        "Partial exit at bar %d: 50%% closed @ %.2f (R=%.2f), pnl=%.2f",
                        bar_idx, close, decision.r_multiple, partial_pnl,
                    )

            # Remove closed trades from active dict
            for _tid in _closed_this_bar:
                active_trades.pop(_tid, None)

            # Keep legacy aliases in sync (for dashboard / end-of-data)
            if active_trades:
                _last_tid = max(active_trades)
                active_trade_id = _last_tid
                active_trade = active_trades[_last_tid]["trade"]
                active_signal = active_trades[_last_tid]["signal"]
                active_context = active_trades[_last_tid]["context"]
                candles_since_entry = active_trades[_last_tid]["candles"]
            else:
                active_trade_id = None
                active_trade = None
                active_signal = None
                active_context = None
                candles_since_entry = 0

            # ------------------------------------------------------------------
            # Look for new trade entries (runs even when trades are open)
            # ------------------------------------------------------------------
            if bar_idx >= _WARMUP_BARS:
                # Track which strategies already have an open trade
                _strategies_in_trade = {
                    _ts["strategy"] for _ts in active_trades.values()
                }

                can_open, reason = self.trade_manager.can_open_trade(balance, instrument)
                if can_open:
                    # Collect all signals
                    all_signals: List[Signal] = []

                    # Add strategy signals not already in a trade
                    for _sig in _bar_strategy_signals:
                        _sig_strat = getattr(_sig, "strategy_name", "unknown")
                        if _sig_strat not in _strategies_in_trade:
                            all_signals.append(_sig)
                        else:
                            _pipeline_counts["signals_filtered_in_trade"] += 1

                    cfg = self._config

                    # Also try Ichimoku scanner (if active and not already in trade)
                    active_names = [n for n, _ in self._active_strategies]
                    if ("ichimoku" not in _strategies_in_trade
                            and ("ichimoku" in active_names
                                 or "ichimoku" in cfg.get("active_strategies", ["ichimoku"]))):
                        ichi_signal = self._scan_for_signal(tf_data, bar_idx, instrument)
                        if ichi_signal is not None:
                            all_signals.append(ichi_signal)
                            _pipeline_counts["signals_generated"] += 1

                    # Pick best signal
                    signal = self._signal_blender.select(all_signals) if all_signals else None

                    if signal is not None:
                        total_signals += 1
                        bar_signal = signal

                        # Build edge context for entry check
                        edge_ctx_entry = self._build_edge_context(
                            ts=ts,
                            row_5m=row_5m,
                            htf_vals=htf_vals,
                            active_trade=None,
                            candles_since_entry=None,
                            equity_by_r=equity_by_r,
                            confluence_score=signal.confluence_score,
                        )
                        edge_ctx_entry.signal = signal

                        entry_allowed, entry_results = self.edge_manager.check_entry(edge_ctx_entry)

                        if not entry_allowed:
                            skipped_signals += 1
                            _pipeline_counts["signals_filtered_edge"] += 1
                            logger.debug(
                                "Signal at %s blocked by edge: %s",
                                ts.isoformat(),
                                next((r.reason for r in entry_results if not r.allowed), "unknown"),
                            )
                        else:
                            # Adaptive learning pre-trade gate
                            learning_size_mult = 1.0
                            learning_blocked = False
                            if enable_learning:
                                entry_context_pre = self._build_entry_context(
                                    signal=signal, row_5m=row_5m,
                                    htf_vals=htf_vals, ts=ts, risk_pct=0.0,
                                )
                                insight = self.learning_engine.pre_trade_analysis(entry_context_pre)

                                if insight.recommendation == "skip":
                                    skipped_signals += 1
                                    learning_skipped += 1
                                    learning_blocked = True
                                    _pipeline_counts["signals_filtered_learning"] += 1
                                    logger.debug(
                                        "Signal at %s blocked by learning (%s): %s",
                                        ts.isoformat(),
                                        self.learning_engine.get_phase(),
                                        insight.reasoning,
                                    )
                                elif insight.recommendation == "caution":
                                    learning_size_mult = 0.75

                            if not learning_blocked:
                                # Position size modifier from modifier edges
                                size_multiplier = self.edge_manager.get_combined_size_multiplier(edge_ctx_entry)
                                size_multiplier *= learning_size_mult

                                # Open the trade
                                try:
                                    atr_multiplier = self._config.get("atr_stop_multiplier", 1.5)
                                    trade_id, trade_obj, pos_size = self.trade_manager.open_trade(
                                        entry_price=signal.entry_price,
                                        stop_loss=signal.stop_loss,
                                        take_profit=signal.take_profit,
                                        direction=signal.direction,
                                        atr=signal.atr,
                                        point_value=self._point_value,
                                        account_equity=balance * size_multiplier,
                                        atr_multiplier=atr_multiplier,
                                        instrument=instrument,
                                        entry_time=ts,
                                    )

                                    _strat_name = getattr(signal, "strategy_name", "ichimoku")
                                    active_trades[trade_id] = {
                                        "trade": trade_obj,
                                        "signal": signal,
                                        "context": self._build_entry_context(
                                            signal=signal,
                                            row_5m=row_5m,
                                            htf_vals=htf_vals,
                                            ts=ts,
                                            risk_pct=pos_size.risk_pct,
                                        ),
                                        "candles": 0,
                                        "strategy": _strat_name,
                                    }
                                    # Sync legacy aliases
                                    active_trade_id = trade_id
                                    active_trade = trade_obj
                                    active_signal = signal
                                    active_context = active_trades[trade_id]["context"]
                                    candles_since_entry = 0
                                    _pipeline_counts["signals_entered"] += 1

                                    # Entry screenshot
                                    if _screenshot_fn:
                                        _screenshot_fn(df_5m, bar_idx, "entry", trade_id)

                                    logger.debug(
                                        "Trade opened: %s %s @ %.2f SL=%.2f TP=%.2f "
                                        "lots=%.2f risk=%.2f%% strategy=%s",
                                        signal.direction, instrument, signal.entry_price,
                                        signal.stop_loss, signal.take_profit,
                                        pos_size.lot_size, pos_size.risk_pct, _strat_name,
                                    )
                                except RuntimeError as exc:
                                    logger.debug("Trade open rejected: %s", exc)
                                    skipped_signals += 1
                                    _pipeline_counts["signals_filtered_open_rejected"] += 1
                else:
                    # can_open_trade returned False (circuit breaker / max concurrent)
                    if _bar_strategy_signals:
                        _pipeline_counts["signals_filtered_no_open"] += len(_bar_strategy_signals)

            # Record equity at this bar
            equity_records.append((ts, balance))
            self.prop_firm_tracker.update(ts, balance)
            if self.active_prop_firm_tracker is not self.prop_firm_tracker:
                self.active_prop_firm_tracker.update(ts, balance)

            # ── Checkpoint: early abort if strategy is clearly broken ──
            _CHECKPOINT_INTERVAL = 25_000  # Check every ~25K 5M bars (~1 month)
            _MIN_TRADES_PER_CHECKPOINT = 2  # Expect at least 2 trades per month
            if (bar_idx > 0
                and bar_idx % _CHECKPOINT_INTERVAL == 0
                and bar_idx < _total_5m_bars - _CHECKPOINT_INTERVAL):
                _cp_trades = len(closed_trades)
                _cp_expected = (bar_idx // _CHECKPOINT_INTERVAL) * _MIN_TRADES_PER_CHECKPOINT
                if _cp_trades < _cp_expected:
                    logger.warning(
                        "CHECKPOINT bar %d/%d: only %d trades (expected >= %d). "
                        "Aborting early — strategy is not generating enough signals.",
                        bar_idx, _total_5m_bars, _cp_trades, _cp_expected,
                    )
                    break

            # Health monitor per-bar update
            self.health_monitor.on_bar(bar_idx, signal=bar_signal)

            # HALTED enforcement disabled — health monitor is not strategy-aware
            # for multi-strategy systems. Strategies manage their own risk.
            if False and self.health_monitor.is_halted and active_trade_id is not None:
                trade_summary = self.trade_manager.close_trade(
                    trade_id=active_trade_id,
                    exit_price=close,
                    reason="health_monitor_halted",
                )
                trade_summary = self._enrich_trade_summary(
                    trade_summary, active_signal, active_context, instrument, balance
                )
                closed_trades.append(trade_summary)
                _r = float(trade_summary.get("r_multiple") or 0.0)
                equity_by_r.append(_r)
                if _r > 0: _n_wins += 1
                else: _n_losses += 1
                balance = self._update_balance_from_trade(balance, trade_summary)
                self._record_learning(trade_summary, active_context, enable_learning)
                self.health_monitor.on_trade_closed(won=_r > 0)
                if live_dashboard is not None:
                    self._push_trade_to_dashboard(live_dashboard, trade_summary, ts, len(closed_trades))
                logger.warning("HALTED: force-closed trade %s at bar %d", active_trade_id, bar_idx)
                active_trade_id = None
                active_trade = None
                active_signal = None
                active_context = None
                candles_since_entry = 0

            # Push candle data to dashboard
            if live_dashboard is not None:
                _time_val = int(ts.timestamp())
                live_dashboard.append_candle("5m", [
                    _time_val,
                    _safe_float(row_5m.get("open")),
                    _safe_float(row_5m.get("high")),
                    _safe_float(row_5m.get("low")),
                    _safe_float(row_5m.get("close")),
                    _safe_float(row_5m.get("volume", row_5m.get("tick_volume"))),
                    _safe_float(row_5m.get("tenkan")),
                    _safe_float(row_5m.get("kijun")),
                    _safe_float(row_5m.get("senkou_a")),
                    _safe_float(row_5m.get("senkou_b")),
                    _safe_float(row_5m.get("chikou")),
                ])
                # 1M candle push — push all 5 constituent 1M bars for this 5M bar
                if "1M" in tf_data:
                    tf_df_1m = tf_data["1M"]
                    _start_1m = pd.Timestamp(ts_raw)
                    _end_1m = _start_1m + pd.Timedelta(minutes=5)
                    _m_slice = tf_df_1m.loc[(_start_1m <= tf_df_1m.index) & (tf_df_1m.index < _end_1m)]
                    for _m_idx_ts, _m_row in _m_slice.iterrows():
                        _m_unix = int(_m_idx_ts.timestamp())
                        live_dashboard.append_candle("1m", [
                            _m_unix,
                            _safe_float(_m_row.get("open")),
                            _safe_float(_m_row.get("high")),
                            _safe_float(_m_row.get("low")),
                            _safe_float(_m_row.get("close")),
                            _safe_float(_m_row.get("volume", _m_row.get("tick_volume"))),
                            _safe_float(_m_row.get("tenkan")),
                            _safe_float(_m_row.get("kijun")),
                            _safe_float(_m_row.get("senkou_a")),
                            _safe_float(_m_row.get("senkou_b")),
                            _safe_float(_m_row.get("chikou")),
                        ])

                # Higher timeframe candle push
                for tf_key, tf_seconds in [("15M", 900), ("30M", 1800), ("1H", 3600), ("4H", 14400), ("1D", 86400)]:
                    if _time_val % tf_seconds == 0 and tf_key in tf_data:
                        tf_df = tf_data[tf_key]
                        mask = tf_df.index <= ts_raw
                        if mask.any():
                            tf_row = tf_df.loc[mask].iloc[-1]
                            live_dashboard.append_candle(tf_key.lower(), [
                                _time_val,
                                _safe_float(tf_row.get("open")),
                                _safe_float(tf_row.get("high")),
                                _safe_float(tf_row.get("low")),
                                _safe_float(tf_row.get("close")),
                                _safe_float(tf_row.get("volume", tf_row.get("tick_volume"))),
                                _safe_float(tf_row.get("tenkan")),
                                _safe_float(tf_row.get("kijun")),
                                _safe_float(tf_row.get("senkou_a")),
                                _safe_float(tf_row.get("senkou_b")),
                                _safe_float(tf_row.get("chikou")),
                            ])

            # Live dashboard update (every 5 bars to avoid overhead)
            if live_dashboard is not None and bar_idx % 5 == 0:
                if bar_idx % _sample_interval == 0:
                    _equity_sample.append(balance)

                elapsed = _time_mod.monotonic() - _bt_start_time
                pct = (bar_idx + 1) / _total_5m_bars * 100 if _total_5m_bars > 0 else 0

                # Build recent trades list for trade log
                recent = []
                for t in closed_trades[-10:]:
                    recent.append({
                        "id": len(closed_trades) - closed_trades[::-1].index(t),
                        "direction": t.get("direction", "?"),
                        "entry_price": float(t.get("entry_price", 0)),
                        "exit_price": float(t.get("exit_price", 0)),
                        "r_multiple": float(t.get("r_multiple", 0)),
                        "pnl_points": float(t.get("pnl_points", 0)),
                        "reason": t.get("reason", ""),
                        "entry_time": str(t.get("entry_time", "")),
                        "exit_time": str(t.get("exit_time", "")),
                    })

                # Running metrics
                n_t = len(closed_trades)
                wr = _n_wins / n_t if n_t > 0 else 0
                ret_pct = (balance / self._initial_balance - 1) * 100
                running_max = max(e[1] for e in equity_records) if equity_records else self._initial_balance
                max_dd = (balance - running_max) / running_max * 100 if running_max > 0 else 0

                live_dashboard.update({
                    "bar_index": bar_idx + 1,
                    "pct_complete": round(pct, 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "equity": round(balance, 2),
                    "equity_history": _equity_sample.copy(),
                    "balance_pct": round(ret_pct, 2),
                    "n_trades": n_t,
                    "n_wins": _n_wins,
                    "n_losses": _n_losses,
                    "win_rate": round(wr, 4),
                    "total_return_pct": round(ret_pct, 2),
                    "max_dd_pct": round(abs(max_dd), 2),
                    "worst_daily_dd_pct": 0.0,
                    "sharpe": 0.0,
                    "expectancy": round(sum(float(t.get("r_multiple", 0)) for t in closed_trades) / n_t, 3) if n_t > 0 else 0.0,
                    "learning_phase": self.learning_engine.get_phase() if enable_learning else "disabled",
                    "learning_trades": self._memory_store.trade_count if enable_learning else 0,
                    "learning_skipped": learning_skipped,
                    "prop_status": "active",
                    "prop_profit_pct": round(ret_pct, 2),
                    "total_signals": total_signals,
                    "skipped_signals": skipped_signals,
                    "recent_trades": recent,
                    "current_timestamp": ts.isoformat(),
                })
                # Health monitor status overlay
                _hs = self.health_monitor.get_status()
                live_dashboard.update({
                    "health_state": _hs.state.value,
                    "health_drought": _hs.is_drought,
                    "health_relaxation_tier": _hs.relaxation_tier,
                    "health_regime": _hs.regime,
                    "health_bottleneck": _hs.bottleneck_filter,
                    "health_message": _hs.message,
                })

        # ------------------------------------------------------------------
        # Pipeline diagnostics summary
        # ------------------------------------------------------------------
        logger.info(
            "Pipeline: generated=%d | filtered_in_trade=%d | filtered_no_open=%d | "
            "filtered_edge=%d | filtered_learning=%d | filtered_open_rejected=%d | entered=%d",
            _pipeline_counts["signals_generated"],
            _pipeline_counts["signals_filtered_in_trade"],
            _pipeline_counts["signals_filtered_no_open"],
            _pipeline_counts["signals_filtered_edge"],
            _pipeline_counts["signals_filtered_learning"],
            _pipeline_counts["signals_filtered_open_rejected"],
            _pipeline_counts["signals_entered"],
        )

        # ------------------------------------------------------------------
        # Force-close ALL trades still open at end of data
        # ------------------------------------------------------------------
        if active_trades and not df_5m.empty:
            final_row = df_5m.iloc[-1]
            final_close = float(final_row.get("close", 0.0))
            final_ts = pd.Timestamp(df_5m.index[-1]).to_pydatetime()
            if final_ts.tzinfo is None:
                final_ts = final_ts.replace(tzinfo=timezone.utc)

            for _tid, _tstate in list(active_trades.items()):
                trade_summary = self.trade_manager.close_trade(
                    trade_id=_tid,
                    exit_price=final_close,
                    reason="end_of_data",
                )
                trade_summary = self._enrich_trade_summary(
                    trade_summary, _tstate["signal"], _tstate["context"], instrument, balance
                )
                closed_trades.append(trade_summary)
                _r = float(trade_summary.get("r_multiple") or 0.0)
                equity_by_r.append(_r)
                if _r > 0: _n_wins += 1
                else: _n_losses += 1
                balance = self._update_balance_from_trade(balance, trade_summary)
                self._record_learning(trade_summary, _tstate["context"], enable_learning)
                self.health_monitor.on_trade_closed(won=_r > 0)
                if live_dashboard is not None:
                    self._push_trade_to_dashboard(live_dashboard, trade_summary, final_ts, len(closed_trades))
            active_trades.clear()

        # Signal live dashboard completion
        if live_dashboard is not None:
            live_dashboard.finish({
                "n_trades": len(closed_trades),
                "n_wins": _n_wins,
                "n_losses": _n_losses,
                "equity": round(balance, 2),
                "equity_history": _equity_sample,
                "total_return_pct": round((balance / self._initial_balance - 1) * 100, 2),
                "total_signals": total_signals,
                "skipped_signals": skipped_signals,
                "learning_skipped": learning_skipped,
            })

        # ------------------------------------------------------------------
        # Persist trades to DB if requested
        # ------------------------------------------------------------------
        if log_trades and closed_trades:
            try:
                self.trade_logger.log_batch(closed_trades, source="backtest")
                logger.info("Logged %d trades to database", len(closed_trades))
            except Exception as exc:
                logger.warning("Trade logging failed: %s", exc)

        # ------------------------------------------------------------------
        # Build result objects
        # ------------------------------------------------------------------
        equity_curve = self._build_equity_curve(equity_records, self._initial_balance)
        daily_pnl = self._build_daily_pnl(equity_curve)

        metrics = self._metrics_calc.calculate(
            trades=closed_trades,
            equity_curve=equity_curve,
            initial_balance=self._initial_balance,
        )

        metrics["pipeline_counts"] = _pipeline_counts

        prop_status = self.prop_firm_tracker.check_pass()

        # When running under a non-legacy tracker (e.g. TopstepX), the
        # active tracker is authoritative for the final verdict. Expose
        # its dict snapshot alongside the legacy pct-status dict.
        active_tracker_dict: Optional[dict] = None
        if self.active_prop_firm_tracker is not self.prop_firm_tracker:
            try:
                active_tracker_dict = self.active_prop_firm_tracker.to_dict()
            except Exception as exc:
                logger.warning("active tracker to_dict failed: %s", exc)

        # ------------------------------------------------------------------
        # Challenge simulation (Monte Carlo + rolling window)
        # ------------------------------------------------------------------
        challenge_sim = None
        if closed_trades:
            try:
                from src.backtesting.challenge_simulator import ChallengeSimulator

                sim_trades = []
                for t in closed_trades:
                    entry_time = t.get("entry_time", first_ts)
                    if hasattr(entry_time, "__sub__") and hasattr(first_ts, "__sub__"):
                        try:
                            day_index = (entry_time - first_ts).days
                        except (TypeError, AttributeError):
                            day_index = 0
                    else:
                        day_index = 0
                    sim_trades.append({
                        "r_multiple": float(t.get("r_multiple", 0.0)),
                        "risk_pct": float(t.get("risk_pct", 1.5)),
                        "day_index": day_index,
                    })

                simulator = ChallengeSimulator()
                total_days = (df_5m.index[-1] - df_5m.index[0]).days if len(df_5m) > 1 else 0
                challenge_sim = simulator.run(
                    sim_trades,
                    total_trading_days=max(total_days, 1),
                    n_mc_simulations=1000,
                )
            except Exception as exc:
                logger.warning("Challenge simulation failed: %s", exc)

        learning_phase = self.learning_engine.get_phase() if enable_learning else "disabled"
        logger.info(
            "Backtest complete: %d trades, win_rate=%.1f%%, return=%.2f%%, "
            "prop_firm=%s, learning_phase=%s, learning_skipped=%d",
            len(closed_trades),
            metrics.get("win_rate", 0) * 100,
            metrics.get("total_return_pct", 0),
            prop_status.status,
            learning_phase,
            learning_skipped,
        )

        prop_firm_dict: Dict[str, Any] = {
            "status": prop_status.status,
            "profit_pct": prop_status.profit_pct,
            "max_daily_dd_pct": prop_status.max_daily_dd_pct,
            "max_total_dd_pct": prop_status.max_total_dd_pct,
            "days_elapsed": prop_status.days_elapsed,
            "details": prop_status.details,
        }
        if active_tracker_dict is not None:
            # Merge in the active tracker's snapshot under a separate
            # key so both the legacy pct view and the active tracker
            # (e.g. TopstepX dollar view) are available to consumers.
            prop_firm_dict["style"] = active_tracker_dict.get("style", "unknown")
            prop_firm_dict["active_tracker"] = active_tracker_dict
            # Override top-level status with the active tracker's status
            # when it's further along (failed beats pending, etc.)
            active_status = active_tracker_dict.get("status")
            if active_status and active_status != "pending":
                prop_firm_dict["status"] = active_status

        # Flush telemetry to disk — caller can control the path via
        # cfg["telemetry_output_dir"] or we fall back to reports/<run_id>/.
        try:
            telem_dir = Path(
                self._config.get("telemetry_output_dir")
                or f"reports/{self.telemetry.run_id}"
            )
            telem_dir.mkdir(parents=True, exist_ok=True)
            self.telemetry.to_parquet(telem_dir / "strategy_telemetry.parquet")
            self.telemetry.to_summary_json(telem_dir / "strategy_telemetry_summary.json")
            logger.info(
                "Strategy telemetry: %d events written to %s",
                self.telemetry.event_count,
                telem_dir,
            )
            self.telemetry.log_console_summary()
        except Exception as exc:
            logger.warning("Telemetry flush failed: %s", exc)

        return BacktestResult(
            trades=closed_trades,
            metrics=metrics,
            equity_curve=equity_curve,
            prop_firm=prop_firm_dict,
            daily_pnl=daily_pnl,
            skipped_signals=skipped_signals,
            total_signals=total_signals,
            challenge_simulation=challenge_sim,
        )

    # ------------------------------------------------------------------
    # Dashboard trade push
    # ------------------------------------------------------------------

    @staticmethod
    def _push_trade_to_dashboard(live_dashboard, trade_summary: dict, ts: datetime, trade_id: int) -> None:
        """Push a completed trade to the live dashboard."""
        entry_time = trade_summary.get("entry_time", ts)
        entry_iso = entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)
        exit_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        live_dashboard.append_trade({
            "id": trade_id,
            "direction": trade_summary.get("direction", ""),
            "entry_time": entry_iso,
            "exit_time": exit_iso,
            "entry_price": float(trade_summary.get("entry_price", 0)),
            "exit_price": float(trade_summary.get("exit_price", 0)),
            "r_multiple": float(trade_summary.get("r_multiple", 0)),
            "pnl_points": float(trade_summary.get("pnl_points", 0)),
            "original_stop": float(trade_summary.get("original_stop", trade_summary.get("stop_loss", 0))),
            "stop_loss": float(trade_summary.get("original_stop", trade_summary.get("stop_loss", 0))),
            "take_profit": float(trade_summary.get("take_profit", 0)),
            "exit_reason": trade_summary.get("exit_reason", trade_summary.get("reason", "")),
            "confluence_score": int(trade_summary.get("confluence_score", 0)),
            "signal_tier": trade_summary.get("signal_tier", ""),
            "session": trade_summary.get("session", ""),
            "lot_size": float(trade_summary.get("lot_size", 0)),
            "risk_pct": float(trade_summary.get("risk_pct", 0)),
            "instrument": trade_summary.get("instrument", "XAUUSD"),
            "entry_reason": trade_summary.get("entry_reason", ""),
            "strategy_name": trade_summary.get("strategy_name", "ichimoku"),
        })

    # ------------------------------------------------------------------
    # Signal scanning
    # ------------------------------------------------------------------

    # Maximum 1M bars needed for signal scan: 52 periods on 4H = 52*240 = 12480,
    # plus Chikou lookback = 26*240 = 6240.  Round up to 20000 for safety.
    _SCAN_LOOKBACK_1M: int = 80_000

    def _scan_for_signal(
        self,
        tf_data: Dict[str, pd.DataFrame],
        bar_idx: int,
        instrument: str,
    ) -> Optional[Signal]:
        """Attempt to generate a signal at the given 5M bar index.

        Passes a trailing window of 1M data (sliced to the current 5M
        bar's timestamp) to SignalEngine.scan(), which resamples
        internally to all required timeframes.
        """
        df_5m = tf_data["5M"]

        if bar_idx < _WARMUP_BARS:
            return None

        # Get the timestamp of the current 5M bar to slice 1M data
        current_ts = df_5m.index[bar_idx]
        # Use a trailing window instead of the full history for performance
        end_loc = self._candles_1m.index.searchsorted(current_ts, side="right")
        start_loc = max(0, end_loc - self._SCAN_LOOKBACK_1M)
        data_1m_slice = self._candles_1m.iloc[start_loc:end_loc]

        if len(data_1m_slice) < _WARMUP_BARS * 5:
            return None

        try:
            signal = self.signal_engine.scan(data_1m=data_1m_slice, current_bar=-1)
            if signal is not None:
                signal.instrument = instrument
            return signal
        except Exception as exc:
            logger.debug("Signal scan error at bar %d: %s", bar_idx, exc)
            return None

    # ------------------------------------------------------------------
    # Edge context building
    # ------------------------------------------------------------------

    def _build_edge_context(
        self,
        ts: datetime,
        row_5m: Any,
        htf_vals: dict,
        active_trade: Optional[ActiveTrade],
        candles_since_entry: Optional[int],
        equity_by_r: List[float],
        confluence_score: int = 0,
    ) -> EdgeContext:
        """Build an EdgeContext from current bar data and trade state."""
        atr = float(row_5m.get("atr") or htf_vals.get("atr_15m") or 1.0)
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        kijun_5m = float(row_5m.get("kijun") or 0.0)
        if np.isnan(kijun_5m):
            kijun_5m = float(row_5m.get("close") or 0.0)

        cloud_thickness = _nan_to_zero(
            htf_vals.get("cloud_thickness_4h") or
            abs(float(row_5m.get("senkou_a") or 0.0) - float(row_5m.get("senkou_b") or 0.0))
        )

        current_r: Optional[float] = None
        if active_trade is not None:
            current_r = active_trade.current_r

        return EdgeContext(
            timestamp=ts,
            day_of_week=ts.weekday(),
            close_price=float(row_5m.get("close") or 0.0),
            high_price=float(row_5m.get("high") or 0.0),
            low_price=float(row_5m.get("low") or 0.0),
            spread=0.0,  # not available in backtesting data
            session=_session_from_hour(ts.hour),
            adx=_nan_to_zero(float(row_5m.get("adx") or htf_vals.get("adx_15m") or 0.0)),
            atr=atr,
            indicator_values={'kijun': kijun_5m, 'cloud_thickness': cloud_thickness},
            bb_squeeze=False,  # BB squeeze not computed at engine level
            confluence_score=confluence_score,
            current_r=current_r,
            candles_since_entry=candles_since_entry,
            equity_curve=list(equity_by_r),
        )

    # ------------------------------------------------------------------
    # Entry context for embedding
    # ------------------------------------------------------------------

    def _build_entry_context(
        self,
        signal: Signal,
        row_5m: Any,
        htf_vals: dict,
        ts: datetime,
        risk_pct: float,
    ) -> dict:
        """Build a market context dict for the embedding engine at trade entry."""
        mtf = signal.mtf_state

        return {
            # Ichimoku state
            "cloud_direction_4h": int(mtf.state_4h.cloud_direction) if mtf else 0,
            "cloud_direction_1h": int(mtf.state_1h.cloud_direction) if mtf else 0,
            "tk_cross_15m":       int(mtf.state_15m.tk_cross) if mtf else 0,
            "chikou_confirmation": bool(mtf.state_15m.chikou_confirmed) if mtf else False,
            "cloud_thickness_4h": float(mtf.state_4h.cloud_thickness) if (mtf and mtf.state_4h.cloud_thickness == mtf.state_4h.cloud_thickness) else 0.0,
            "cloud_position_15m": int(mtf.state_15m.cloud_position) if mtf else 0,
            "cloud_position_5m":  int(mtf.state_5m.cloud_position) if mtf else 0,
            # Trend / momentum
            "adx_value":          float(mtf.adx_15m) if mtf else 0.0,
            "atr_value":          float(signal.atr),
            "atr":                float(signal.atr),
            # Session / time
            "session":            mtf.session if mtf else _session_from_hour(ts.hour),
            "hour":               ts.hour,
            "day_of_week":        ts.weekday(),
            # Signal quality
            "confluence_score":   signal.confluence_score,
            "signal_tier":        signal.quality_tier,
            "direction":          signal.direction,
            "risk_pct":           risk_pct,
            "atr_stop_distance":  abs(signal.entry_price - signal.stop_loss),
            # Zone context
            "zone_confluence_count": signal.zone_context.get("nearby_zone_count", 0),
            # Kijun distance
            "kijun_distance_5m":  signal.entry_price - (mtf.kijun_5m if mtf else signal.entry_price),
        }

    # ------------------------------------------------------------------
    # HTF value extraction helpers
    # ------------------------------------------------------------------

    def _get_htf_values(
        self,
        tf_data: Dict[str, pd.DataFrame],
        ts: datetime,
    ) -> dict:
        """Extract higher-TF indicator values aligned to a 5M timestamp.

        Each higher-TF DataFrame is indexed at its own bar open times.
        We find the most recent bar that is <= ts (forward-fill semantics).
        """
        result: dict = {}
        for tf, prefix_key in (("15M", "15m"), ("1H", "1h"), ("4H", "4h")):
            df = tf_data.get(tf)
            if df is None or df.empty:
                continue
            # Find last HTF bar that has already closed by ts
            df_before = df[df.index <= ts]
            if df_before.empty:
                continue
            row = df_before.iloc[-1]
            for col in df.columns:
                result[f"{col}_{prefix_key}"] = float(row[col]) if not pd.isna(row[col]) else np.nan

        return result

    # ------------------------------------------------------------------
    # Trade lifecycle helpers
    # ------------------------------------------------------------------

    def _update_balance_from_trade(
        self, balance: float, trade_summary: dict
    ) -> float:
        """Apply trade P&L (minus trading costs) to the running balance."""
        pnl_points = float(trade_summary.get("pnl_points") or 0.0)
        lot_size = float(trade_summary.get("lot_size") or 0.0)
        remaining_pct = float(trade_summary.get("remaining_pct") or 1.0)

        # Gross P&L = points * lot_size * point_value * remaining fraction.
        # For futures, point_value was set in __init__ to tick_value_usd /
        # tick_size so this formula returns dollars for N contracts.
        monetary_pnl = pnl_points * lot_size * self._point_value * remaining_pct

        # Trading costs — model depends on instrument class.
        if self._instrument_class == "futures":
            # Futures: full round-trip commission on the final close,
            # scaled proportionally to the fraction closed. Slippage
            # is N ticks × 2 (entry + exit) × tick_value_usd.
            commission_cost = (
                self._commission_per_contract_rt * lot_size * remaining_pct
            )
            slippage_cost = (
                self._slippage_ticks
                * 2.0
                * self._tick_value_usd
                * lot_size
                * remaining_pct
            )
            spread_cost = 0.0
            commission_cost += slippage_cost
        else:
            spread_cost = self._spread_points * lot_size * self._point_value * remaining_pct
            commission_cost = self._commission_per_lot * lot_size * remaining_pct

        new_balance = balance + monetary_pnl - spread_cost - commission_cost

        # Prevent balance from going negative (margin call equivalent)
        new_balance = max(new_balance, 0.01)

        # Update the sizer so phase transitions are tracked correctly
        self.trade_manager._sizer.update_balance(new_balance)
        return new_balance

    def _partial_pnl(
        self, trade: ActiveTrade, exit_price: float, close_pct: float
    ) -> float:
        """Calculate monetary P&L for a partial exit, net of trading costs."""
        if trade.direction == "long":
            pnl_points = exit_price - trade.entry_price
        else:
            pnl_points = trade.entry_price - exit_price
        gross_pnl = pnl_points * trade.lot_size * self._point_value * close_pct

        if self._instrument_class == "futures":
            commission_cost = (
                self._commission_per_contract_rt * trade.lot_size * close_pct
            )
            slippage_cost = (
                self._slippage_ticks
                * self._tick_value_usd
                * trade.lot_size
                * close_pct
            )
            return gross_pnl - commission_cost - slippage_cost

        spread_cost = self._spread_points * trade.lot_size * self._point_value * close_pct
        commission_cost = self._commission_per_lot * trade.lot_size * close_pct
        return gross_pnl - spread_cost - commission_cost

    def _enrich_trade_summary(
        self,
        summary: dict,
        signal: Optional[Signal],
        context: Optional[dict],
        instrument: str,
        balance: float,
    ) -> dict:
        """Add signal metadata and embedding context to a closed trade summary."""
        summary["instrument"] = instrument
        summary["context"] = context or {}

        if signal is not None:
            summary["confluence_score"] = signal.confluence_score
            summary["signal_tier"] = signal.quality_tier
            summary["risk_pct"] = self.trade_manager._sizer.get_risk_pct()
            summary["take_profit"] = signal.take_profit
            summary["strategy_name"] = getattr(signal, "strategy_name", "ichimoku")
            # Build concise entry reason from signal reasoning trace
            reasoning = signal.reasoning or {}
            reason_parts = []
            for key in ("4h_filter", "1h_confirmation", "15m_signal", "5m_entry"):
                step = reasoning.get(key)
                if step and step.get("pass"):
                    reason_parts.append(step.get("reason", key.replace("_", " ")))
            if reason_parts:
                summary["entry_reason"] = " | ".join(reason_parts)
            else:
                summary["entry_reason"] = f"{signal.quality_tier} signal (score {signal.confluence_score})"

        # Generate and attach embedding
        trade_result = {"r_multiple": summary.get("r_multiple")}
        embed_data = self.embedding_engine.embed_trade(context or {}, trade_result)
        summary["context_embedding"] = embed_data["context_embedding"]

        return summary

    # ------------------------------------------------------------------
    # Adaptive learning helpers
    # ------------------------------------------------------------------

    def _record_learning(
        self,
        trade_summary: dict,
        context: Optional[dict],
        enable_learning: bool,
    ) -> None:
        """Record a closed trade to the in-memory learning components."""
        if not enable_learning:
            return

        r_multiple = float(trade_summary.get("r_multiple") or 0.0)

        # Record embedding to memory store for similarity search
        embedding_list = trade_summary.get("context_embedding")
        if embedding_list is not None:
            embedding = np.array(embedding_list, dtype=np.float64)
            self._memory_store.record_trade(
                embedding=embedding,
                r_multiple=r_multiple,
                context=context,
            )

        # Record trade stats for statistical filtering
        stats_record = {
            "r_multiple": r_multiple,
            "session": (context or {}).get("session", "unknown"),
            "adx_value": (context or {}).get("adx_value", 0.0),
            "confluence_score": trade_summary.get("confluence_score", 0),
            "day_of_week": (context or {}).get("day_of_week", 0),
            "direction": (context or {}).get("direction", "long"),
            "signal_tier": trade_summary.get("signal_tier", "C"),
        }
        self._memory_stats.record_trade(stats_record)

        # Update adaptive engine trade counter + post-trade analysis
        self.learning_engine.post_trade_analysis(stats_record)

        phase = self.learning_engine.get_phase()
        store_n = self._memory_store.trade_count
        if store_n % 50 == 0 and store_n > 0:
            logger.info(
                "Learning progress: %d trades recorded, phase=%s, "
                "memory_store=%d embeddings",
                store_n, phase, store_n,
            )

    # ------------------------------------------------------------------
    # Backtest screenshot helper
    # ------------------------------------------------------------------

    def _make_screenshot_fn(self, screenshot_dir: str, instrument: str):
        """Return a callable that generates mplfinance charts, or None on import failure."""
        try:
            import mplfinance as mpf
        except ImportError:
            logger.warning(
                "mplfinance not installed — backtest screenshots disabled. "
                "Install with: pip install mplfinance"
            )
            return None

        from pathlib import Path

        base_dir = Path(screenshot_dir) / instrument

        def _capture(df_5m: pd.DataFrame, bar_idx: int, phase: str, trade_id: Optional[int]) -> str:
            """Generate a candlestick chart around the trade event."""
            # Show 50 bars before and 10 after (or end of data)
            start = max(0, bar_idx - 50)
            end = min(len(df_5m), bar_idx + 10)
            chunk = df_5m.iloc[start:end].copy()

            if chunk.empty:
                return ""

            # mplfinance needs DatetimeIndex
            if not isinstance(chunk.index, pd.DatetimeIndex):
                chunk.index = pd.DatetimeIndex(chunk.index)

            # Rename tick_volume → volume if present
            if "tick_volume" in chunk.columns and "volume" not in chunk.columns:
                chunk = chunk.rename(columns={"tick_volume": "volume"})

            ts = chunk.index[min(bar_idx - start, len(chunk) - 1)]
            date_str = str(ts.date()).replace("-", "")
            tid_part = f"_{trade_id}" if trade_id is not None else ""
            filename = f"{phase}{tid_part}_{date_str}.png"
            file_path = base_dir / date_str / filename

            file_path.parent.mkdir(parents=True, exist_ok=True)

            title = f"{instrument} 5M — {phase}"
            if trade_id is not None:
                title += f" (trade #{trade_id})"

            try:
                mpf.plot(
                    chunk,
                    type="candle",
                    style="charles",
                    title=title,
                    volume=("volume" in chunk.columns),
                    savefig=str(file_path),
                    figsize=(12.8, 7.2),
                )
                return str(file_path)
            except Exception as exc:
                logger.debug("Screenshot generation failed: %s", exc)
                return ""

        return _capture

    # ------------------------------------------------------------------
    # Equity curve and daily P&L construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_equity_curve(
        records: List[tuple],
        initial_balance: float,
    ) -> pd.Series:
        """Build a DatetimeIndex equity Series from (timestamp, equity) pairs."""
        if not records:
            return pd.Series(dtype=float, name="equity")
        timestamps, equities = zip(*records)
        series = pd.Series(
            list(equities),
            index=pd.DatetimeIndex(list(timestamps), tz=timezone.utc),
            name="equity",
        )
        return series

    @staticmethod
    def _build_daily_pnl(equity_curve: pd.Series) -> pd.Series:
        """Compute daily P&L as a fraction of the start-of-day equity."""
        if equity_curve.empty:
            return pd.Series(dtype=float, name="daily_pnl")
        daily_open = equity_curve.resample("1D").first().dropna()
        daily_close = equity_curve.resample("1D").last().dropna()
        open_aligned, close_aligned = daily_open.align(daily_close, join="inner")
        daily_pnl = (close_aligned - open_aligned) / open_aligned
        daily_pnl.name = "daily_pnl"
        return daily_pnl


# =============================================================================
# StrategyBacktester
# =============================================================================

class StrategyBacktester:
    """Strategy-agnostic backtester that delegates to the active strategy.

    This is a transition wrapper: it loads the requested strategy from the
    STRATEGY_REGISTRY, builds an EvaluatorCoordinator from the strategy's
    declared evaluators, and delegates the actual bar-by-bar simulation to
    IchimokuBacktester (which handles the full risk / edge pipeline).

    Future strategies will have their own dedicated bar loop; for now this
    ensures the StrategyBacktester API is stable and importable.

    Parameters
    ----------
    strategy_key:
        Key used to look up the strategy in STRATEGY_REGISTRY.
        Default: 'ichimoku'.
    config:
        Optional configuration dict forwarded to both the strategy and the
        underlying IchimokuBacktester.
    initial_balance:
        Starting account equity.  Default: 10 000.0.
    data:
        Optional pre-loaded 1-minute OHLCV DataFrame.  Can also be supplied
        at ``run()`` time via the ``data_1m`` parameter.
    """

    def __init__(
        self,
        strategy_key: str = 'ichimoku',
        config: dict = None,
        initial_balance: float = 10_000.0,
        data: pd.DataFrame = None,
    ) -> None:
        from src.strategy.base import STRATEGY_REGISTRY
        from src.strategy.coordinator import EvaluatorCoordinator
        import src.strategy.evaluators   # noqa: F401 — populates EVALUATOR_REGISTRY
        import src.strategy.strategies   # noqa: F401 — populates STRATEGY_REGISTRY

        if strategy_key not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Strategy '{strategy_key}' not found in registry. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )

        strategy_cls = STRATEGY_REGISTRY[strategy_key]
        self._strategy = strategy_cls(config=config) if config else strategy_cls()
        self._coordinator = EvaluatorCoordinator(
            self._strategy.required_evaluators,
            warmup_bars=self._strategy.warmup_bars,
        )

        # Delegate to IchimokuBacktester for now (it handles the bar loop).
        # This is a transition — future strategies will have their own loop.
        self._backtester = IchimokuBacktester(
            config=config,
            initial_balance=initial_balance,
        )

        self._data = data

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data_1m: pd.DataFrame = None, **kwargs) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        data_1m:
            1-minute OHLCV DataFrame.  If omitted the DataFrame supplied at
            construction time (``data=``) is used.
        **kwargs:
            Forwarded verbatim to ``IchimokuBacktester.run()``.

        Returns
        -------
        BacktestResult
        """
        candles = data_1m if data_1m is not None else self._data
        if candles is None:
            raise ValueError(
                "No data provided. Pass data_1m to run() or supply data= at construction."
            )
        return self._backtester.run(candles_1m=candles, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self):
        """The instantiated Strategy object."""
        return self._strategy

    @property
    def coordinator(self):
        """The EvaluatorCoordinator built from strategy.required_evaluators."""
        return self._coordinator


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _nan_to_zero(value: Any) -> float:
    """Return 0.0 when value is NaN or None, else return float(value)."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        return 0.0 if (f != f) else f  # NaN comparison
    except (TypeError, ValueError):
        return 0.0


def _session_from_hour(hour: int) -> str:
    """Derive a rough session label from UTC hour."""
    if 8 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "new_york"
    if 0 <= hour < 8:
        return "asian"
    return "off_hours"
