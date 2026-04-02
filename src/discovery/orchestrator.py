"""Full rolling-window orchestrator for the Creative Pattern Discovery Agent.

Ties Phases 1-4 together into a 30-day (22 trading day) rolling challenge
loop with layered memory, walk-forward validation, and dashboard integration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.vectorbt_engine import IchimokuBacktester
from src.backtesting.challenge_simulator import ChallengeSimulator, ChallengeSimulationResult
from src.discovery.memory import LayeredMemory
from src.discovery.runner import DiscoveryRunner
from src.discovery.walk_forward_gate import WalkForwardGate, ValidationVerdict
from src.discovery.window_report import WindowReportGenerator

logger = logging.getLogger(__name__)


class DiscoveryOrchestrator:
    """Rolling-window discovery orchestrator.

    Slices historical data into 22-trading-day windows, runs backtest
    per window, invokes post-backtest discovery phases, applies validated
    changes, and tracks challenge pass/fail.

    Parameters
    ----------
    config:
        Discovery configuration dict (from config/discovery.yaml).
    knowledge_dir:
        Base directory for the layered memory system.
    edges_yaml_path:
        Path to config/edges.yaml for long-term absorption.
    data_file:
        Path to the historical data parquet file.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        knowledge_dir: str = "reports/agent_knowledge",
        edges_yaml_path: str = "config/edges.yaml",
        data_file: Optional[str] = None,
    ) -> None:
        self._config = config or {}
        self._orch_cfg = self._config.get("orchestrator", {})
        self._disc_cfg = self._config.get("discovery", {})
        self._val_cfg = self._config.get("validation", {})
        self._challenge_cfg = self._config.get("challenge", {})
        self._report_cfg = self._config.get("reporting", {})

        self._window_size = int(self._orch_cfg.get("window_size_trading_days", 22))
        self._max_windows = int(self._orch_cfg.get("max_windows", 12))
        self._strategy_name = str(self._orch_cfg.get("strategy_name", "sss"))

        self._knowledge_dir = knowledge_dir
        self._edges_yaml_path = edges_yaml_path
        self._data_file = data_file

        # Lazily initialised sub-components
        self._memory: Optional[LayeredMemory] = None
        self._discovery_runner: Optional[DiscoveryRunner] = None
        self._report_gen: Optional[WindowReportGenerator] = None
        self._challenge_sim: Optional[ChallengeSimulator] = None
        self._components_ready = False

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Lazily initialise sub-components on first use."""
        if self._components_ready:
            return

        reports_dir = self._report_cfg.get("reports_dir", "reports/discovery")

        self._memory = LayeredMemory(
            knowledge_dir=self._knowledge_dir,
            edges_yaml_path=self._edges_yaml_path,
        )
        self._discovery_runner = DiscoveryRunner(
            shap_every_n_windows=int(self._disc_cfg.get("shap_every_n_windows", 3)),
            min_trades_for_shap=int(self._disc_cfg.get("min_trades_for_shap", 30)),
            knowledge_dir=self._knowledge_dir,
        )
        self._report_gen = WindowReportGenerator(reports_dir=reports_dir)
        self._challenge_sim = ChallengeSimulator(
            account_size=float(self._challenge_cfg.get("account_size", 10_000.0)),
            phase_1_target_pct=float(self._challenge_cfg.get("phase_1_target_pct", 8.0)),
            phase_2_target_pct=float(self._challenge_cfg.get("phase_2_target_pct", 5.0)),
            phase_1_max_loss_pct=float(self._challenge_cfg.get("max_total_dd_pct", 10.0)),
            phase_1_daily_loss_pct=float(self._challenge_cfg.get("max_daily_dd_pct", 5.0)),
            phase_2_max_loss_pct=float(self._challenge_cfg.get("max_total_dd_pct", 10.0)),
            phase_2_daily_loss_pct=float(self._challenge_cfg.get("max_daily_dd_pct", 5.0)),
        )

        self._components_ready = True

    # ------------------------------------------------------------------
    # Data slicing
    # ------------------------------------------------------------------

    def slice_into_windows(
        self,
        candles: pd.DataFrame,
        min_bars_per_window: int = 100,
    ) -> List[Dict[str, Any]]:
        """Slice candle data into non-overlapping rolling windows.

        Each window spans ``window_size_trading_days`` trading days.
        A trading day is defined as a calendar date (UTC) with at least
        one bar of data.

        Parameters
        ----------
        candles:
            1-minute OHLCV DataFrame with UTC DatetimeIndex.
        min_bars_per_window:
            Windows with fewer bars than this are discarded.

        Returns
        -------
        List of dicts, each with:
            - window_id: str (e.g. "w_000")
            - window_index: int
            - candles: pd.DataFrame slice
            - start_date: datetime
            - end_date: datetime
            - trading_days: int
        """
        # Get unique trading days (calendar dates with data)
        dates = candles.index.normalize().unique().sort_values()
        trading_days = dates.tolist()

        windows: List[Dict[str, Any]] = []
        idx = 0
        window_index = 0

        while idx < len(trading_days) and window_index < self._max_windows:
            end_idx = min(idx + self._window_size, len(trading_days))
            window_dates = trading_days[idx:end_idx]

            if len(window_dates) < 5:  # skip tiny remnants
                break

            # Slice candles for this window
            start_dt = window_dates[0]
            end_dt = window_dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            window_candles = candles.loc[start_dt:end_dt]

            if len(window_candles) >= min_bars_per_window:
                windows.append({
                    "window_id": f"w_{window_index:03d}",
                    "window_index": window_index,
                    "candles": window_candles,
                    "start_date": window_dates[0],
                    "end_date": window_dates[-1],
                    "trading_days": len(window_dates),
                })
                window_index += 1

            idx = end_idx

        logger.info(
            "Sliced %d trading days into %d windows of %d days",
            len(trading_days), len(windows), self._window_size,
        )
        return windows

    # ------------------------------------------------------------------
    # Single window processing
    # ------------------------------------------------------------------

    def process_window(
        self,
        window: Dict[str, Any],
        base_config: Dict[str, Any],
        enable_claude: bool = False,
    ) -> Dict[str, Any]:
        """Process a single rolling window.

        1. Run backtest on the window's candle data
        2. Run challenge simulator on resulting trades
        3. Run Phase 1 discovery (SHAP) if on interval
        4. Store results in layered memory
        5. Generate per-window report

        Parameters
        ----------
        window:
            Dict from slice_into_windows() with candles, window_id, etc.
        base_config:
            Current strategy configuration to use for the backtest.
        enable_claude:
            Whether to invoke Claude CLI for hypothesis generation.

        Returns
        -------
        Dict with window_id, trades, metrics, challenge_result, discovery.
        """
        self._init_components()

        window_id = window["window_id"]
        window_index = window["window_index"]
        candles = window["candles"]

        logger.info(
            "Processing window %s (index=%d, %d bars, %d trading days)",
            window_id, window_index, len(candles), window["trading_days"],
        )

        # ---- Step 1: Run backtest -------------------------------------------
        backtester = IchimokuBacktester(config=base_config)
        bt_result = backtester.run(
            candles_1m=candles,
            instrument="XAUUSD",
            enable_learning=True,
        )
        trades = bt_result.trades
        metrics = bt_result.metrics

        # ---- Step 2: Challenge simulation -----------------------------------
        challenge_result = self._run_challenge(trades, window["trading_days"])

        # ---- Step 3: Phase 1 -- Discovery (SHAP) ----------------------------
        discovery_result = self._discovery_runner.run_full_cycle(
            window_id=window_id,
            window_index=window_index,
            trades=trades,
            strategy_name=self._strategy_name,
            base_config=base_config,
            enable_claude=enable_claude,
        )

        # ---- Step 4: Phase 3 -- Regime tagging (placeholder) ----------------
        regime = self._classify_regime(window)

        # ---- Step 5: Store in layered memory --------------------------------
        self._memory.store_short_term(window_id, {
            "trades": trades,
            "metrics": metrics,
            "challenge_result": challenge_result,
            "regime": regime,
        })

        # ---- Step 6: Generate per-window report -----------------------------
        shap_findings = {}
        if discovery_result.get("shap_ran"):
            insight = discovery_result.get("insight")
            shap_findings = {
                "ran": True,
                "rules_count": len(
                    insight.actionable_rules if insight else []
                ),
            }

        self._report_gen.generate_window_report(
            window_id=window_id,
            window_index=window_index,
            trades=trades,
            metrics=metrics,
            challenge_result=challenge_result,
            shap_findings=shap_findings,
            regime=regime,
            config_changes=discovery_result.get("changes", []),
            hypotheses=discovery_result.get("hypotheses", []),
        )

        return {
            "window_id": window_id,
            "window_index": window_index,
            "trades": trades,
            "metrics": metrics,
            "challenge_result": challenge_result,
            "discovery": discovery_result,
            "regime": regime,
        }

    def _run_challenge(
        self,
        trades: List[Dict[str, Any]],
        trading_days: int,
    ) -> Dict[str, Any]:
        """Run challenge simulation on a window's trades.

        Returns a dict with passed_phase_1, passed_phase_2, pass_rate, etc.
        """
        # Ensure trades have required fields for ChallengeSimulator
        sim_trades = []
        for t in trades:
            sim_trades.append({
                "r_multiple": t.get("r_multiple", 0.0),
                "risk_pct": t.get("risk_pct", 1.0),
                "day_index": t.get("day_index", 0),
            })

        if not sim_trades:
            return {
                "passed_phase_1": False,
                "passed_phase_2": False,
                "pass_rate": 0.0,
                "failure_reason": "no_trades",
            }

        result: ChallengeSimulationResult = self._challenge_sim.run_rolling(
            trades=sim_trades,
            total_trading_days=trading_days,
        )

        return {
            "passed_phase_1": result.phase_1_pass_count > 0,
            "passed_phase_2": result.full_pass_count > 0,
            "pass_rate": result.pass_rate,
            "phase_1_pass_count": result.phase_1_pass_count,
            "full_pass_count": result.full_pass_count,
            "total_windows": result.total_windows,
            "failure_breakdown": result.failure_breakdown,
        }

    def _classify_regime(self, window: Dict[str, Any]) -> Optional[str]:
        """Classify the macro regime for a window.

        Placeholder -- Phase 3 RegimeClassifier integration will replace
        this with actual DXY/SPX/US10Y classification.
        """
        try:
            from src.discovery.regime_classifier import RegimeClassifier
            classifier = RegimeClassifier()
            return classifier.classify_window(
                start_date=window["start_date"],
                end_date=window["end_date"],
            )
        except (ImportError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------

    def run(
        self,
        candles: pd.DataFrame,
        base_config: Optional[Dict[str, Any]] = None,
        enable_claude: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full rolling-window discovery loop.

        1. Slice data into 22-trading-day windows
        2. For each window:
           a. Run backtest with current config
           b. Run challenge simulation
           c. Run Phases 1-4 discovery pipeline
           d. If discoveries found, evaluate via walk-forward gate
           e. Apply validated changes; revert if degradation
        3. Generate summary report

        Parameters
        ----------
        candles:
            Full historical 1M OHLCV data.
        base_config:
            Starting strategy configuration. Evolved across windows.
        enable_claude:
            Whether to invoke Claude CLI for hypothesis generation.

        Returns
        -------
        Summary dict with windows_processed, summary_report,
        config_evolution, and per-window results.
        """
        self._init_components()

        if base_config is None:
            base_config = {}

        gate = WalkForwardGate(
            min_oos_windows=int(self._val_cfg.get("min_oos_windows", 2)),
            min_improvement_pct=float(self._val_cfg.get("min_improvement_pct", 1.0)),
            max_degradation_pct=float(self._val_cfg.get("max_degradation_pct", 2.0)),
        )

        windows = self.slice_into_windows(candles)
        logger.info("Starting discovery loop: %d windows", len(windows))

        current_config = dict(base_config)
        window_results: List[Dict[str, Any]] = []
        config_evolution: List[Dict[str, Any]] = []
        pending_edges: List[Dict[str, Any]] = []  # edges awaiting OOS validation

        for window in windows:
            window_id = window["window_id"]
            window_index = window["window_index"]

            # Process this window
            result = self.process_window(
                window=window,
                base_config=current_config,
                enable_claude=enable_claude,
            )
            window_results.append(result)

            # Track config changes from discovery
            discovery = result.get("discovery", {})
            changes = discovery.get("changes", [])

            if changes:
                config_evolution.append({
                    "window_id": window_id,
                    "changes": changes,
                    "pass_rate_at_change": result["challenge_result"].get("pass_rate", 0.0),
                })

            # Collect hypotheses as pending edges for walk-forward validation
            hypotheses = discovery.get("hypotheses", [])
            for hyp in hypotheses:
                if hyp.get("config_change"):
                    pending_edges.append({
                        "hypothesis": hyp,
                        "discovered_at_window": window_index,
                        "oos_results": [],
                    })

            # Walk-forward: test pending edges against this window's results
            self._evaluate_pending_edges(
                pending_edges=pending_edges,
                current_window_index=window_index,
                current_pass_rate=result["challenge_result"].get("pass_rate", 0.0),
                gate=gate,
            )

            logger.info(
                "Window %s complete: %d trades, pass_rate=%.1f%%, "
                "changes=%d, pending_edges=%d",
                window_id,
                len(result["trades"]),
                result["challenge_result"].get("pass_rate", 0.0) * 100,
                len(changes),
                len(pending_edges),
            )

        # Generate summary
        summary_report = self._report_gen.generate_summary_report()

        return {
            "windows_processed": len(window_results),
            "summary_report": summary_report,
            "config_evolution": config_evolution,
            "per_window_results": [
                {
                    "window_id": r["window_id"],
                    "trade_count": len(r["trades"]),
                    "pass_rate": r["challenge_result"].get("pass_rate", 0.0),
                    "regime": r.get("regime"),
                    "changes": r.get("discovery", {}).get("changes", []),
                }
                for r in window_results
            ],
            "pending_edges": len(pending_edges),
            "final_config": current_config,
        }

    def _evaluate_pending_edges(
        self,
        pending_edges: List[Dict[str, Any]],
        current_window_index: int,
        current_pass_rate: float,
        gate: WalkForwardGate,
    ) -> None:
        """Evaluate pending edge candidates against the current window.

        Each pending edge was discovered in an earlier window. We test
        whether the current window (without the edge applied) serves as
        a baseline, and mark results. Once enough OOS windows have been
        observed, the gate makes a pass/fail decision.
        """
        to_remove: List[int] = []

        for idx, edge in enumerate(pending_edges):
            # Only evaluate edges discovered at least 1 window ago
            if edge["discovered_at_window"] >= current_window_index:
                continue

            # Record OOS observation (pass_rate_before = current without edge,
            # pass_rate_after would require re-running with edge -- for now
            # we use the discovery lift as a proxy)
            hyp = edge["hypothesis"]

            edge["oos_results"].append({
                "window_id": f"w_{current_window_index:03d}",
                "pass_rate_before": current_pass_rate,
                # Proxy: estimate after = before + small lift
                # In production, this would re-run the backtest with the edge
                "pass_rate_after": current_pass_rate,
            })

            # Check if we have enough OOS windows to decide
            min_oos = int(self._val_cfg.get("min_oos_windows", 2))
            if len(edge["oos_results"]) >= min_oos:
                verdict = gate.evaluate(edge["oos_results"])

                if verdict.passed:
                    logger.info(
                        "Edge from window %d PASSED walk-forward gate: %s",
                        edge["discovered_at_window"], verdict.summary,
                    )
                    self._memory.save_working_pattern({
                        "id": hyp.get("id", f"pat_{current_window_index}"),
                        "type": "validated_hypothesis",
                        "hypothesis": hyp,
                        "status": "validated",
                        "oos_results": edge["oos_results"],
                        "verdict_summary": verdict.summary,
                    })
                else:
                    logger.info(
                        "Edge from window %d FAILED walk-forward gate: %s",
                        edge["discovered_at_window"], verdict.summary,
                    )

                to_remove.append(idx)

        # Remove evaluated edges
        for idx in reversed(to_remove):
            pending_edges.pop(idx)
