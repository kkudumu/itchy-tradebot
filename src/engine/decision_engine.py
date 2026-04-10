"""Central orchestrator for the XAU/USD Ichimoku trading agent.

This module is intentionally thin — all domain logic lives in specialised
modules.  The engine connects the pipeline:

    scan → signal → edge filters → similarity → confidence → size → execute → log

For open trades:

    check exit edges → update trailing stops → log position updates

Supports two operating modes:
- live:     reads candles from a market-data provider, executes via an execution provider
- backtest: iterates a pre-loaded data feed, simulates execution
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many closed trades or minutes between edge_stats refreshes
_EDGE_STATS_TRADE_INTERVAL = 50
_EDGE_STATS_TIME_INTERVAL_SECONDS = 3600  # 1 hour


@dataclass
class Decision:
    """Full record of one scan cycle's reasoning and outcome."""

    timestamp: datetime
    instrument: str
    action: str             # 'enter', 'skip', 'exit', 'partial_exit', 'modify'
    signal: Optional[object]
    edge_results: Dict[str, Any]
    similarity_data: Dict[str, Any]
    confluence_score: int
    reasoning: str
    trade_id: Optional[int] = None
    executed: bool = False
    execution_detail: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """Central orchestrator for the trading agent.

    This is the THIN coordinator — all logic lives in specialised modules.

    Parameters
    ----------
    config:
        Engine configuration dict.  Relevant keys:
        - ``instrument``  (str, default 'XAUUSD')
        - ``scan_interval_minutes`` (int, default 5)
        - ``point_value``  (float, default 1.0)
        - ``atr_multiplier`` (float, default 1.5)
        - ``min_confidence_score`` (int, default 4)
        - ``similarity_k`` (int, default 10)
        - ``similarity_min_score`` (float, default 0.7)
    signal_engine:
        :class:`~src.strategy.signal_engine.SignalEngine` instance.
    edge_manager:
        :class:`~src.edges.manager.EdgeManager` instance.
    trade_manager:
        :class:`~src.risk.trade_manager.TradeManager` instance.
    similarity_search:
        :class:`~src.learning.similarity.SimilaritySearch` instance.
    embedding_engine:
        :class:`~src.learning.embeddings.EmbeddingEngine` instance.
    zone_manager:
        :class:`~src.zones.manager.ZoneManager` instance.
    market_data_provider:
        Live market-data provider. When omitted, MT5Bridge can still be passed
        and will be wrapped for backward compatibility.
    execution_provider:
        Live execution provider. When omitted, OrderManager can still be passed
        and will be wrapped for backward compatibility.
    account_provider:
        Live account provider. When omitted, AccountMonitor can still be passed
        and will be wrapped for backward compatibility.
    screenshot_capture:
        Optional callable ``(phase: str, trade_id: int) -> str | None``.
        Expected signature: ``capture(phase, trade_id) -> filepath_or_None``.
    db_pool:
        Database connection pool used by the trade logger.
    """

    def __init__(
        self,
        config: dict,
        signal_engine=None,
        edge_manager=None,
        trade_manager=None,
        similarity_search=None,
        embedding_engine=None,
        zone_manager=None,
        market_data_provider=None,
        execution_provider=None,
        account_provider=None,
        mt5_bridge=None,
        order_manager=None,
        account_monitor=None,
        screenshot_capture=None,
        db_pool=None,
        strategy_config=None,
    ) -> None:
        self._cfg = config or {}
        self._instrument = self._cfg.get("instrument", "XAUUSD")
        self._scan_interval = self._cfg.get("scan_interval_minutes", 5)
        self._point_value = self._cfg.get("point_value", 1.0)
        self._atr_multiplier = self._cfg.get("atr_multiplier", 1.5)
        self._min_confluence = self._cfg.get("min_confidence_score", 4)
        self._similarity_k = self._cfg.get("similarity_k", 10)
        self._similarity_min = self._cfg.get("similarity_min_score", 0.7)

        # Injected dependencies
        self.signal_engine = signal_engine
        self.edge_manager = edge_manager
        self.trade_manager = trade_manager
        self.similarity_search = similarity_search
        self.embedding_engine = embedding_engine
        self.zone_manager = zone_manager
        self.mt5_bridge = mt5_bridge
        self.order_manager = order_manager
        self.account_monitor = account_monitor
        self.screenshot_capture = screenshot_capture

        if market_data_provider is None and mt5_bridge is not None:
            from src.providers import MT5MarketDataProvider

            market_data_provider = MT5MarketDataProvider(mt5_bridge)
        if execution_provider is None and order_manager is not None:
            from src.providers import MT5ExecutionProvider

            execution_provider = MT5ExecutionProvider(order_manager)
        if account_provider is None and account_monitor is not None:
            from src.providers import MT5AccountProvider

            account_provider = MT5AccountProvider(account_monitor)

        self.market_data_provider = market_data_provider
        self.execution_provider = execution_provider
        self.account_provider = account_provider

        # Strategy interface (alternative to signal_engine)
        if signal_engine is not None:
            self._signal_engine = signal_engine
            self._strategy = None
            self._coordinator = None
        else:
            self._signal_engine = None
            if strategy_config is not None:
                from ..strategy.loader import StrategyLoader
                from ..strategy.coordinator import EvaluatorCoordinator
                loader = StrategyLoader(strategy_config)
                self._strategy = loader.load()
                self._coordinator = EvaluatorCoordinator(
                    self._strategy.required_evaluators,
                    warmup_bars=self._strategy.warmup_bars,
                )
            else:
                self._strategy = None
                self._coordinator = None

        # Lazy-import to avoid circular dependency at module level
        from .trade_logger import EngineTradeLogger
        self.trade_logger = EngineTradeLogger(
            db_pool=db_pool,
            embedding_engine=embedding_engine,
        )
        from .scheduler import ScanScheduler
        self._scheduler = ScanScheduler(interval_minutes=self._scan_interval)

        # Operating mode
        self._mode = "live" if self.market_data_provider is not None else "backtest"

        # Runtime state
        self._running = False
        self._stop_event = threading.Event()
        self._backtest_data: Optional[Any] = None  # set by caller for backtest mode

        # Edge stats refresh tracking
        self._closed_trade_count = 0
        self._last_edge_stats_refresh: Optional[datetime] = None

        # Health monitor — autonomous self-awareness layer
        self.health_monitor = None
        if signal_engine is not None and edge_manager is not None:
            try:
                from src.monitoring.health_monitor import StrategyHealthMonitor
                self.health_monitor = StrategyHealthMonitor(
                    signal_engine=signal_engine,
                    edge_manager=edge_manager,
                    config=self._cfg.get("health_monitor"),
                    mode=self._mode,
                    drought_window=self._cfg.get("health_monitor_drought_window", 500),
                    baseline_win_rate=self._cfg.get("baseline_win_rate", 0.55),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("StrategyHealthMonitor init failed: %s", exc)

        # Bar counter for health monitor
        self._bar_idx = 0

        # Last EvalMatrix produced by the coordinator (used in edge context and exit checks)
        self._last_eval_matrix = None

        logger.info(
            "DecisionEngine initialised — instrument=%s mode=%s interval=%dM",
            self._instrument,
            self._mode,
            self._scan_interval,
        )

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the main trading loop.

        Live mode: the scheduler waits for each 5M candle close.
        Backtest mode: caller should iterate bars and call ``scan()`` directly.
        """
        if self._running:
            logger.warning("DecisionEngine.start() called while already running")
            return

        self._running = True
        self._stop_event.clear()

        if self._mode == "live":
            logger.info("DecisionEngine starting live loop (interval=%dM)", self._scan_interval)
            try:
                self._scheduler.run_loop(self._live_scan_callback, self._stop_event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled exception in live loop: %s", exc)
            finally:
                self._running = False
        else:
            # Backtest: the engine is ready; caller drives iteration via scan()
            logger.info("DecisionEngine ready in backtest mode")

    def stop(self) -> None:
        """Graceful shutdown: signal the loop to exit, flush logs."""
        logger.info("DecisionEngine stop requested")
        self._stop_event.set()
        self._running = False

        # Flush pending log entries
        try:
            self.trade_logger.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error flushing trade logger on shutdown: %s", exc)

        logger.info("DecisionEngine stopped")

    # ------------------------------------------------------------------
    # Core scan cycle
    # ------------------------------------------------------------------

    def scan(self, data: Optional[dict] = None) -> Decision:
        """Run one complete scan cycle.

        Steps
        -----
        1. Retrieve current multi-timeframe data.
        2. Run zone maintenance on the latest candle.
        3. Manage any open trades (exit checks, trailing stops).
        4. Check for a new entry signal.
        5. Apply entry edge filters to the signal.
        6. Query similarity search for historical context.
        7. Calculate confidence-adjusted score.
        8. Check trade manager approval (circuit breaker, position limits).
        9. Size and execute the trade, or skip with a reason.
        10. Log the decision with full trace.

        Parameters
        ----------
        data:
            Optional pre-loaded data dict (backtest mode).  In live mode,
            data is fetched from MT5Bridge.

        Returns
        -------
        :class:`Decision` describing the outcome of the scan.
        """
        ts = datetime.now(timezone.utc)

        # Step 1: get market data
        if data is None:
            data = self._get_data()

        if not data:
            decision = Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=None,
                edge_results={},
                similarity_data={},
                confluence_score=0,
                reasoning="No market data available",
            )
            self._log_decision(decision)
            return decision

        # Step 2: zone maintenance
        self._zone_maintenance(data)

        # Step 3: manage open trades first
        open_trade_decisions = self._manage_open_trades(data)
        for td in open_trade_decisions:
            self._log_decision(td)

        # Step 4: check for a new entry signal (skip when health monitor is HALTED)
        signal = None
        self._last_eval_matrix = None  # reset per scan cycle
        if self.health_monitor is not None and self.health_monitor.is_halted:
            decision = Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=None,
                edge_results={},
                similarity_data={},
                confluence_score=0,
                reasoning="Health monitor HALTED — scanning disabled",
            )
            self._log_decision(decision)

            # Health monitor per-scan update
            self.health_monitor.on_bar(self._bar_idx, signal=None)
            self._bar_idx += 1

            return decision

        if self._strategy is not None and self._coordinator is not None:
            # Strategy interface path
            data_1m = data.get("1M")
            if data_1m is None:
                data_1m = data.get("M1")
            if data_1m is not None and not data_1m.empty:
                try:
                    eval_matrix = self._coordinator.evaluate(data_1m, self._bar_idx)
                    if eval_matrix is not None:
                        self._last_eval_matrix = eval_matrix
                        signal = self._strategy.decide(eval_matrix)
                except Exception as exc:  # noqa: BLE001
                    logger.error("strategy.decide failed: %s", exc)
        elif self.signal_engine is not None:
            data_1m = data.get("1M")
            if data_1m is None:
                data_1m = data.get("M1")
            if data_1m is not None and not data_1m.empty:
                try:
                    signal = self.signal_engine.scan(data_1m)
                except Exception as exc:  # noqa: BLE001
                    logger.error("signal_engine.scan failed: %s", exc)

        if signal is None:
            decision = Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=None,
                edge_results={},
                similarity_data={},
                confluence_score=0,
                reasoning="No signal detected",
            )
            self._log_decision(decision)
            return decision

        # Signal detected — run full pipeline
        decision = self._process_signal(signal, data, ts)
        self._log_decision(decision)

        # Health monitor per-scan update
        if self.health_monitor is not None:
            self.health_monitor.on_bar(self._bar_idx, signal=signal)
            self._bar_idx += 1

        # Periodically refresh edge stats materialized view
        self._maybe_refresh_edge_stats()

        return decision

    # ------------------------------------------------------------------
    # Signal pipeline
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal: Any,
        data: dict,
        ts: datetime,
    ) -> Decision:
        """Run a detected signal through the full decision pipeline.

        Returns a Decision with action='enter' or 'skip'.
        """
        edge_results: Dict[str, Any] = {}
        similarity_data: Dict[str, Any] = {}

        # Step 5: edge filter check
        edge_context = self._build_edge_context(signal, data)
        entry_allowed = True
        entry_reason = "All edge filters passed"

        if self.edge_manager is not None and edge_context is not None:
            try:
                entry_allowed, results = self.edge_manager.check_entry(edge_context)
                for r in results:
                    edge_results[r.edge_name] = {
                        "allowed": r.allowed,
                        "reason": r.reason,
                        "modifier": r.modifier,
                    }
                if not entry_allowed:
                    failed = [r for r in results if not r.allowed]
                    entry_reason = f"Blocked by edge: {failed[0].edge_name} — {failed[0].reason}"
            except Exception as exc:  # noqa: BLE001
                logger.error("edge_manager.check_entry failed: %s", exc)
                entry_allowed = False
                entry_reason = f"Edge manager error: {exc}"

        if not entry_allowed:
            return Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=signal,
                edge_results=edge_results,
                similarity_data=similarity_data,
                confluence_score=getattr(signal, "confluence_score", 0),
                reasoning=entry_reason,
            )

        # Step 6: modifier edges (position size multiplier)
        size_multiplier = 1.0
        if self.edge_manager is not None and edge_context is not None:
            try:
                size_multiplier = self.edge_manager.get_combined_size_multiplier(edge_context)
                modifiers = self.edge_manager.get_modifiers(edge_context)
                for name, val in modifiers.items():
                    edge_results[f"modifier_{name}"] = val
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge modifier calculation failed: %s", exc)

        # Step 7: similarity search
        adjusted_score = getattr(signal, "confluence_score", 0)
        if self.embedding_engine is not None and self.similarity_search is not None:
            try:
                context_dict = self._build_context_dict(signal, data)
                embedding = self.embedding_engine.create_embedding(context_dict)
                similar_trades = self.similarity_search.find_similar_trades(
                    context_embedding=embedding,
                    k=self._similarity_k,
                    min_similarity=self._similarity_min,
                )
                stats = self.similarity_search.get_performance_stats(similar_trades)
                similarity_data = {
                    "n_similar": stats.n_trades,
                    "win_rate": stats.win_rate,
                    "avg_r": stats.avg_r,
                    "expectancy": stats.expectancy,
                    "confidence": stats.confidence,
                }
                # Step 8: adjust confidence based on similarity results
                adjusted_score = self._adjust_confidence(signal, stats)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Similarity search failed: %s", exc)

        # Minimum score gate
        if adjusted_score < self._min_confluence:
            return Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=signal,
                edge_results=edge_results,
                similarity_data=similarity_data,
                confluence_score=adjusted_score,
                reasoning=f"Confidence score {adjusted_score} below minimum {self._min_confluence}",
            )

        # Step 9: trade manager pre-trade gate
        account_equity = self._get_account_equity()
        can_open = True
        gate_reason = "ok"

        if self.trade_manager is not None:
            try:
                can_open, gate_reason = self.trade_manager.can_open_trade(
                    current_balance=account_equity,
                    instrument=self._instrument,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("trade_manager.can_open_trade failed: %s", exc)
                can_open = False
                gate_reason = str(exc)

        if not can_open:
            return Decision(
                timestamp=ts,
                instrument=self._instrument,
                action="skip",
                signal=signal,
                edge_results=edge_results,
                similarity_data=similarity_data,
                confluence_score=adjusted_score,
                reasoning=f"Trade manager blocked entry: {gate_reason}",
            )

        # Step 10: position sizing
        lot_size = 0.01
        point_value = self._point_value
        if self._mode == "live" and self.market_data_provider is not None:
            try:
                spec = self.market_data_provider.get_contract_spec(self._instrument)
                point_value = float(spec.point_value or self._point_value)
                if spec.default_quantity is not None and spec.default_quantity > 0:
                    lot_size = float(spec.default_quantity)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not resolve contract spec for %s: %s", self._instrument, exc)
        if self.trade_manager is not None:
            try:
                _, active_trade, pos = self.trade_manager.open_trade(
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    direction=signal.direction,
                    atr=signal.atr,
                    point_value=point_value,
                    account_equity=account_equity,
                    atr_multiplier=self._atr_multiplier,
                    instrument=self._instrument,
                    entry_time=ts,
                )
                lot_size = pos.lot_size * size_multiplier
                # Internal trade_id from trade_manager
                internal_trade_id = self.trade_manager.active_trade_ids[-1] if self.trade_manager.active_trade_ids else None
            except Exception as exc:  # noqa: BLE001
                logger.error("trade_manager.open_trade failed: %s", exc)
                return Decision(
                    timestamp=ts,
                    instrument=self._instrument,
                    action="skip",
                    signal=signal,
                    edge_results=edge_results,
                    similarity_data=similarity_data,
                    confluence_score=adjusted_score,
                    reasoning=f"Position sizing failed: {exc}",
                )
        else:
            internal_trade_id = None

        # Step 11: execute
        execution_detail = self._execute_trade(signal, lot_size)
        executed = execution_detail.get("success", False)
        broker_trade_id = execution_detail.get("ticket")
        if executed and internal_trade_id is not None and self.trade_manager is not None:
            active = self.trade_manager._active_trades.get(internal_trade_id)  # noqa: SLF001
            if active is not None:
                setattr(active, "external_id", broker_trade_id)

        # Take pre-entry screenshot
        if self.screenshot_capture is not None and broker_trade_id:
            try:
                filepath = self.screenshot_capture("pre_entry", broker_trade_id)
                if filepath:
                    self.trade_logger.log_screenshot(broker_trade_id, "pre_entry", filepath)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Screenshot capture failed (pre_entry): %s", exc)

        # Log trade entry to DB with full context
        if executed:
            context = self._build_context_dict(signal, data)
            trade_record = {
                "instrument": self._instrument,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "lot_size": lot_size,
                "confluence_score": adjusted_score,
                "signal_tier": getattr(signal, "quality_tier", ""),
                "atr": signal.atr,
                "timestamp": ts.isoformat(),
                "ticket": broker_trade_id,
                "internal_trade_id": internal_trade_id,
                "similarity_data": similarity_data,
                "edge_results": edge_results,
                "zone_context": getattr(signal, "zone_context", {}),
                "signal_reasoning": getattr(signal, "reasoning", {}),
            }
            self.trade_logger.log_trade_entry(trade_record, context)

            # Take entry screenshot
            if self.screenshot_capture is not None and broker_trade_id:
                try:
                    filepath = self.screenshot_capture("entry", broker_trade_id)
                    if filepath:
                        self.trade_logger.log_screenshot(broker_trade_id, "entry", filepath)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Screenshot capture failed (entry): %s", exc)

        return Decision(
            timestamp=ts,
            instrument=self._instrument,
            action="enter" if executed else "skip",
            signal=signal,
            edge_results=edge_results,
            similarity_data=similarity_data,
            confluence_score=adjusted_score,
            reasoning="Trade executed" if executed else f"Execution failed: {execution_detail.get('error', 'unknown')}",
            trade_id=broker_trade_id or internal_trade_id,
            executed=executed,
            execution_detail=execution_detail,
        )

    # ------------------------------------------------------------------
    # Open trade management
    # ------------------------------------------------------------------

    def _manage_open_trades(self, data: dict) -> List[Decision]:
        """Check exit conditions and update trailing stops for all open trades."""
        decisions: List[Decision] = []

        if self.trade_manager is None:
            return decisions

        current_price = self._extract_current_price(data)
        kijun_value = self._extract_kijun(data, "5M") or self._extract_kijun(data, "15M") or current_price
        higher_kijun = self._extract_kijun(data, "1H")
        ts = datetime.now(timezone.utc)

        # Build edge context for exit filters
        edge_context = self._build_exit_edge_context(data, current_price)

        for trade_id in list(self.trade_manager.active_trade_ids):
            try:
                # Check edge-level exit conditions (friday_close, time_stop)
                force_close = False
                exit_reason = ""
                edge_exit_results: Dict[str, Any] = {}

                if self.edge_manager is not None and edge_context is not None:
                    exit_triggered, exit_results = self.edge_manager.check_exit(edge_context)
                    for r in exit_results:
                        edge_exit_results[r.edge_name] = {
                            "allowed": r.allowed,
                            "reason": r.reason,
                        }
                    if exit_triggered:
                        force_close = True
                        triggered = [r for r in exit_results if not r.allowed]
                        exit_reason = triggered[0].reason if triggered else "edge exit triggered"

                if force_close:
                    # Force close via order manager if live
                    exec_detail = self._close_trade_live(trade_id, current_price, exit_reason)
                    close_data = self.trade_manager.close_trade(
                        trade_id, current_price, reason=exit_reason
                    )

                    # Screenshot at exit
                    ticket = close_data.get("ticket") or trade_id
                    if self.screenshot_capture is not None:
                        try:
                            fp = self.screenshot_capture("exit", ticket)
                            if fp:
                                self.trade_logger.log_screenshot(ticket, "exit", fp)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Screenshot capture failed (exit): %s", exc)

                    self.trade_logger.log_trade_exit(trade_id, {**close_data, "current_price": current_price})
                    self._closed_trade_count += 1
                    if self.health_monitor is not None:
                        _r = float(close_data.get("r_multiple", 0))
                        self.health_monitor.on_trade_closed(won=_r > 0)

                    decisions.append(Decision(
                        timestamp=ts,
                        instrument=self._instrument,
                        action="exit",
                        signal=None,
                        edge_results=edge_exit_results,
                        similarity_data={},
                        confluence_score=0,
                        reasoning=exit_reason,
                        trade_id=trade_id,
                        executed=True,
                        execution_detail=exec_detail,
                    ))
                    continue

                # Strategy trading mode exit check (alternative to trade_manager)
                eval_matrix = getattr(self, "_last_eval_matrix", None)
                if self._strategy is not None and self._strategy.trading_mode is not None and eval_matrix is not None:
                    trade_obj = self.trade_manager._active_trades.get(trade_id) if self.trade_manager is not None else None  # noqa: SLF001
                    current_bar_data = {
                        "close": current_price,
                        "atr": 1.0,
                    }
                    try:
                        strategy_exit = self._strategy.trading_mode.check_exit(
                            trade_obj, current_bar_data, eval_matrix
                        )
                        if strategy_exit is not None and getattr(strategy_exit, "action", "hold") != "hold":
                            exit_decision = strategy_exit
                        else:
                            exit_decision = self.trade_manager.update_trade(
                                trade_id=trade_id,
                                current_price=current_price,
                                kijun_value=kijun_value,
                                higher_tf_kijun=higher_kijun,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("strategy.trading_mode.check_exit failed: %s", exc)
                        exit_decision = self.trade_manager.update_trade(
                            trade_id=trade_id,
                            current_price=current_price,
                            kijun_value=kijun_value,
                            higher_tf_kijun=higher_kijun,
                        )
                else:
                    # Normal exit manager check (take-profit, trailing)
                    exit_decision = self.trade_manager.update_trade(
                        trade_id=trade_id,
                        current_price=current_price,
                        kijun_value=kijun_value,
                        higher_tf_kijun=higher_kijun,
                    )

                if exit_decision.action == "full_exit":
                    self._close_trade_live(trade_id, current_price, exit_decision.reason)

                    # Screenshot at exit
                    if self.screenshot_capture is not None:
                        try:
                            fp = self.screenshot_capture("exit", trade_id)
                            if fp:
                                self.trade_logger.log_screenshot(trade_id, "exit", fp)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Screenshot capture failed (exit): %s", exc)

                    self.trade_logger.log_trade_exit(trade_id, {
                        "current_price": current_price,
                        "action": exit_decision.action,
                        "reason": exit_decision.reason,
                        "r_multiple": exit_decision.r_multiple,
                    })
                    self._closed_trade_count += 1
                    if self.health_monitor is not None:
                        _r = float(getattr(exit_decision, "r_multiple", 0) or 0)
                        self.health_monitor.on_trade_closed(won=_r > 0)

                    decisions.append(Decision(
                        timestamp=ts,
                        instrument=self._instrument,
                        action="exit",
                        signal=None,
                        edge_results={},
                        similarity_data={},
                        confluence_score=0,
                        reasoning=exit_decision.reason,
                        trade_id=trade_id,
                        executed=True,
                    ))

                elif exit_decision.action == "partial_exit":
                    self._partial_close_live(trade_id, exit_decision.close_pct, current_price)

                    decisions.append(Decision(
                        timestamp=ts,
                        instrument=self._instrument,
                        action="partial_exit",
                        signal=None,
                        edge_results={},
                        similarity_data={},
                        confluence_score=0,
                        reasoning=exit_decision.reason,
                        trade_id=trade_id,
                        executed=True,
                        execution_detail={
                            "close_pct": exit_decision.close_pct,
                            "r_multiple": exit_decision.r_multiple,
                        },
                    ))

                elif exit_decision.action == "trail_update" and exit_decision.new_stop is not None:
                    self._modify_stop_live(trade_id, exit_decision.new_stop)
                    decisions.append(Decision(
                        timestamp=ts,
                        instrument=self._instrument,
                        action="modify",
                        signal=None,
                        edge_results={},
                        similarity_data={},
                        confluence_score=0,
                        reasoning=f"Trailing stop updated to {exit_decision.new_stop}",
                        trade_id=trade_id,
                        executed=True,
                        execution_detail={"new_stop": exit_decision.new_stop},
                    ))

            except Exception as exc:  # noqa: BLE001
                logger.error("Error managing trade %s: %s", trade_id, exc)

        return decisions

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _get_data(self) -> dict:
        """Retrieve current multi-timeframe market data.

        Live mode:  from the configured market-data provider.
        Backtest:   from the backtest data feed set on the engine.
        """
        if self._mode == "live" and self.market_data_provider is not None:
            try:
                return self.market_data_provider.get_multi_tf_data(
                    instrument=self._instrument,
                    count=500,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to retrieve live data: %s", exc)
                return {}
        # Backtest: data is passed directly to scan(); return empty here
        return {}

    def _get_account_equity(self) -> float:
        """Return current account equity.

        Live: from the configured account provider.
        Backtest: from TradeManager equity summary.
        """
        if self._mode == "live" and self.account_provider is not None:
            try:
                info = self.account_provider.get_account_info()
                if info is not None:
                    return info.equity or 10_000.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("account_provider.get_account_info failed: %s", exc)

        if self.trade_manager is not None:
            try:
                summary = self.trade_manager.get_equity_summary()
                return summary.get("total_equity", 10_000.0)
            except Exception:  # noqa: BLE001
                pass

        return 10_000.0

    # ------------------------------------------------------------------
    # Edge context builders
    # ------------------------------------------------------------------

    def _build_edge_context(self, signal: Any, data: dict):
        """Construct an EdgeContext from the current signal and market state."""
        try:
            from src.edges.base import EdgeContext

            ts = getattr(signal, "timestamp", datetime.now(timezone.utc))
            day_of_week = ts.weekday() if hasattr(ts, "weekday") else 0
            mtf = getattr(signal, "mtf_state", None)

            close_price = getattr(signal, "entry_price", 0.0)
            high_price = close_price
            low_price = close_price
            spread = self._get_current_spread()
            session = getattr(mtf, "session", "unknown") if mtf else "unknown"
            adx = float(getattr(getattr(mtf, "state_15m", None), "adx", 0.0) if mtf else 0.0)
            atr = getattr(signal, "atr", 1.0)
            cloud_thickness = 0.0
            kijun_value = float(getattr(mtf, "kijun_5m", close_price) if mtf else close_price)
            bb_squeeze = False
            confluence_score = getattr(signal, "confluence_score", 0)

            # Try to extract cloud thickness from mtf state
            if mtf is not None:
                state_15m = getattr(mtf, "state_15m", None)
                if state_15m is not None:
                    sa = getattr(state_15m, "senkou_a", None)
                    sb = getattr(state_15m, "senkou_b", None)
                    if sa is not None and sb is not None:
                        try:
                            cloud_thickness = abs(float(sa) - float(sb))
                        except (TypeError, ValueError):
                            cloud_thickness = 0.0

            # Historical equity curve for equity_curve filter
            equity_curve: list = []
            if self.trade_manager is not None:
                equity_curve = [
                    t.get("r_multiple", 0.0)
                    for t in self.trade_manager.closed_trades
                ]

            # Use strategy's populate_edge_context if available, otherwise fall back
            eval_matrix = getattr(self, "_last_eval_matrix", None)
            if self._strategy is not None and eval_matrix is not None:
                indicator_values = self._strategy.populate_edge_context(eval_matrix)
            else:
                indicator_values = {'kijun': kijun_value, 'cloud_thickness': cloud_thickness}

            return EdgeContext(
                timestamp=ts,
                day_of_week=day_of_week,
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                spread=spread,
                session=session,
                adx=adx,
                atr=atr,
                indicator_values=indicator_values,
                bb_squeeze=bb_squeeze,
                confluence_score=confluence_score,
                equity_curve=equity_curve,
                signal=signal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build EdgeContext: %s", exc)
            return None

    def _build_exit_edge_context(self, data: dict, current_price: float):
        """Construct an EdgeContext for exit edge evaluation."""
        try:
            from src.edges.base import EdgeContext

            ts = datetime.now(timezone.utc)
            atr = 1.0
            spread = self._get_current_spread()

            equity_curve: list = []
            if self.trade_manager is not None:
                equity_curve = [
                    t.get("r_multiple", 0.0)
                    for t in self.trade_manager.closed_trades
                ]
                active_ids = self.trade_manager.active_trade_ids
                current_r = 0.0
                candles_since_entry = None
                if active_ids:
                    # Use first active trade for context
                    first_id = active_ids[0]
                    active = self.trade_manager._active_trades.get(first_id)  # noqa: SLF001
                    if active is not None:
                        current_r = getattr(active, "current_r", 0.0)
                        entry_time = getattr(active, "entry_time", None)
                        if entry_time:
                            delta = ts - entry_time
                            candles_since_entry = int(delta.total_seconds() / (self._scan_interval * 60))
            else:
                current_r = 0.0
                candles_since_entry = None

            return EdgeContext(
                timestamp=ts,
                day_of_week=ts.weekday(),
                close_price=current_price,
                high_price=current_price,
                low_price=current_price,
                spread=spread,
                session="unknown",
                adx=0.0,
                atr=atr,
                indicator_values={},
                bb_squeeze=False,
                confluence_score=0,
                current_r=current_r,
                candles_since_entry=candles_since_entry,
                equity_curve=equity_curve,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build exit EdgeContext: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def _adjust_confidence(self, signal: Any, similarity_stats: Any) -> int:
        """Adjust signal confidence score based on historical similarity results.

        Rules
        -----
        - If data confidence < 0.3 (fewer than 6 similar trades), no adjustment.
        - If expectancy > 0.5 and win_rate > 0.6, award +1 point.
        - If expectancy < -0.3 or win_rate < 0.4, deduct -1 point.
        - Score is clamped to [0, 8].

        Parameters
        ----------
        signal:
            The detected signal (carries ``confluence_score``).
        similarity_stats:
            :class:`~src.learning.similarity.PerformanceStats` from the search.
        """
        base_score = getattr(signal, "confluence_score", 0)

        if similarity_stats is None:
            return base_score

        data_confidence = getattr(similarity_stats, "confidence", 0.0)
        if data_confidence < 0.3:
            # Too few samples — trust the signal score as-is
            return base_score

        expectancy = getattr(similarity_stats, "expectancy", 0.0)
        win_rate = getattr(similarity_stats, "win_rate", 0.0)

        adjustment = 0
        if expectancy > 0.5 and win_rate > 0.6:
            adjustment = 1
        elif expectancy < -0.3 or win_rate < 0.4:
            adjustment = -1

        return max(0, min(8, base_score + adjustment))

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_trade(self, signal: Any, lot_size: float) -> dict:
        """Execute the trade order.

        Live mode: sends a market order via the execution provider.
        Backtest mode: simulates a fill at the signal entry price.
        """
        if self._mode == "live" and self.execution_provider is not None:
            try:
                result = self.execution_provider.place_market_order(
                    instrument=self._instrument,
                    direction=signal.direction,
                    quantity=lot_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    comment=f"Ichimoku {signal.quality_tier if hasattr(signal, 'quality_tier') else ''}",
                )
                return {
                    "success": result.success,
                    "ticket": result.order_id,
                    "price": result.fill_price,
                    "volume": result.quantity,
                    "retcode": result.error_code,
                    "slippage": result.raw.get("slippage", 0.0),
                    "error": result.error_message,
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("Order execution failed: %s", exc)
                return {"success": False, "error": str(exc)}

        # Backtest: simulate fill at signal entry price
        import random
        simulated_ticket = random.randint(100000, 999999)
        logger.debug(
            "Backtest fill: %s %s %.2f lots @ %.5f",
            signal.direction,
            self._instrument,
            lot_size,
            signal.entry_price,
        )
        return {
            "success": True,
            "ticket": simulated_ticket,
            "price": signal.entry_price,
            "volume": lot_size,
            "retcode": 10009,
            "slippage": 0.0,
            "error": "",
        }

    def _close_trade_live(self, trade_id: int, price: float, reason: str) -> dict:
        """Send a close order via the execution provider in live mode."""
        if self._mode == "live" and self.execution_provider is not None:
            try:
                self.execution_provider.close_position(self._instrument)
                logger.info("Closing trade %s at %.5f - %s", trade_id, price, reason)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error closing trade %s: %s", trade_id, exc)
        return {"success": True, "price": price, "reason": reason}

    def _partial_close_live(self, trade_id: int, close_pct: float, price: float) -> None:
        """Partial close in live mode."""
        if self._mode == "live" and self.execution_provider is not None:
            try:
                quantity = close_pct
                if self.trade_manager is not None:
                    active = self.trade_manager._active_trades.get(trade_id)  # noqa: SLF001
                    if active is not None:
                        quantity = float(getattr(active, "lot_size", 0.0)) * close_pct
                self.execution_provider.partial_close_position(self._instrument, quantity)
                logger.info("Partial close trade %s %.0f%% at %.5f", trade_id, close_pct * 100, price)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error partially closing trade %s: %s", trade_id, exc)

    def _modify_stop_live(self, trade_id: int, new_stop: float) -> None:
        """Modify trailing stop in live mode when the provider exposes it."""
        if self._mode == "live" and self.execution_provider is not None:
            logger.debug("Trail stop trade %s -> %.5f", trade_id, new_stop)

    # ------------------------------------------------------------------
    # Zone maintenance
    # ------------------------------------------------------------------

    def _zone_maintenance(self, data: dict) -> None:
        """Run ZoneManager maintenance on every candle close."""
        if self.zone_manager is None:
            return

        # Use the 5M candle (most recent close) for zone updates
        candle = self._extract_latest_candle(data, tf="5M") or self._extract_latest_candle(data, tf="M5")
        if candle is None:
            return

        try:
            self.zone_manager.maintenance(candle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ZoneManager maintenance failed: %s", exc)

    # ------------------------------------------------------------------
    # Edge stats refresh
    # ------------------------------------------------------------------

    def _maybe_refresh_edge_stats(self) -> None:
        """Refresh edge_stats materialized view periodically.

        Triggers when either:
        - ``_EDGE_STATS_TRADE_INTERVAL`` trades have closed since last refresh, or
        - ``_EDGE_STATS_TIME_INTERVAL_SECONDS`` seconds have elapsed.
        """
        now = datetime.now(timezone.utc)
        trade_threshold_met = (self._closed_trade_count % _EDGE_STATS_TRADE_INTERVAL == 0
                               and self._closed_trade_count > 0)
        time_threshold_met = (
            self._last_edge_stats_refresh is None
            or (now - self._last_edge_stats_refresh).total_seconds() >= _EDGE_STATS_TIME_INTERVAL_SECONDS
        )

        if trade_threshold_met or time_threshold_met:
            self._refresh_edge_stats()
            self._last_edge_stats_refresh = now

    def _refresh_edge_stats(self) -> None:
        """Refresh the edge_stats materialized view via the trade logger."""
        try:
            self.trade_logger.refresh_edge_stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge_stats refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def _log_decision(self, decision: Decision) -> None:
        """Persist the decision record with full reasoning trace."""
        try:
            self.trade_logger.log_decision(decision)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to log decision: %s", exc)

    # ------------------------------------------------------------------
    # Live loop callback
    # ------------------------------------------------------------------

    def _live_scan_callback(self) -> None:
        """Called by the scheduler on each scan interval."""
        try:
            self.scan()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Exception during live scan: %s", exc)

    # ------------------------------------------------------------------
    # Context builder for embeddings
    # ------------------------------------------------------------------

    def _build_context_dict(self, signal: Any, data: dict) -> dict:
        """Build a flat context dict for embedding generation."""
        mtf = getattr(signal, "mtf_state", None)

        def _safe(obj, attr, default=0.0):
            try:
                return float(getattr(obj, attr, default))
            except (TypeError, ValueError):
                return default

        state_4h = getattr(mtf, "state_4h", None) if mtf else None
        state_1h = getattr(mtf, "state_1h", None) if mtf else None
        state_15m = getattr(mtf, "state_15m", None) if mtf else None

        return {
            "cloud_direction_4h": _safe(state_4h, "cloud_direction"),
            "cloud_direction_1h": _safe(state_1h, "cloud_direction"),
            "tk_cross_15m": _safe(state_15m, "tk_cross"),
            "session": getattr(mtf, "session", "unknown") if mtf else "unknown",
            "adx_value": _safe(state_15m, "adx"),
            "atr_value": getattr(signal, "atr", 1.0),
            "rsi_value": 50.0,  # placeholder — RSI not yet in signal
            "nearest_sr_distance": 0.0,
            "zone_confluence_score": len(getattr(signal, "zone_context", {}).get("zones", [])),
            "confluence_score": getattr(signal, "confluence_score", 0),
            "direction": getattr(signal, "direction", "long"),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_current_spread(self) -> float:
        """Return current bid/ask spread in price units."""
        if self._mode == "live" and self.market_data_provider is not None:
            try:
                tick = self.market_data_provider.get_tick(self._instrument)
                return float(tick.get("spread", 0.3))
            except Exception:  # noqa: BLE001
                pass
        return 0.3  # default spread for XAUUSD in backtest

    def _extract_current_price(self, data: dict) -> float:
        """Extract the latest close price from the 5M data frame."""
        for tf in ("5M", "M5", "15M", "M15"):
            df = data.get(tf)
            if df is not None and not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
        return 0.0

    def _extract_kijun(self, data: dict, tf: str):
        """Extract the Kijun-Sen value from a timeframe DataFrame if present."""
        df = data.get(tf)
        if df is None or df.empty:
            return None
        if "kijun" in df.columns:
            v = df["kijun"].iloc[-1]
            try:
                import math
                if not math.isnan(float(v)):
                    return float(v)
            except (TypeError, ValueError):
                pass
        return None

    def _extract_latest_candle(self, data: dict, tf: str = "5M"):
        """Return the latest candle as a dict from a DataFrame."""
        df = data.get(tf)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        return {
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
        }
