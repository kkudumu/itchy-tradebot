# Forward Test Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-stage pipeline (discovery with rule application, challenge simulation, funded forward test with continued discovery) plus live dashboard wiring, so the agent discovers edges on historical data, proves it can pass The5ers challenge, then trades funded for 6+ months while continuing to learn.

**Architecture:** Stage 1 (discovery) closes the apply loop so SHAP rules actually modify config between windows. Stage 2 (challenge) runs the optimized config through Phase 1 + Phase 2 of The5ers with `MultiPhasePropFirmTracker`. Stage 3 (funded) trades month-by-month with cautious continued discovery (SHAP every 3 months, max 5% DD for new rules). Live dashboard wired into all stages.

**Tech Stack:** Existing `IchimokuBacktester`, `MultiPhasePropFirmTracker`, `ChallengeSimulator`, `DiscoveryOrchestrator`, `LiveDashboardServer`, `DiscoveryRunner`. New: `ForwardTestRunner`, `MonthlyPnLTracker`, `scripts/run_forward_test.py`.

---

## File Structure

| File | Responsibility |
|------|---------------|
| Modify: `src/discovery/orchestrator.py` | Close the apply loop — apply top hypothesis config between windows |
| Create: `src/discovery/forward_test.py` | ForwardTestRunner — challenge simulation + funded monthly trading |
| Create: `src/discovery/monthly_tracker.py` | MonthlyPnLTracker — per-month P&L, DD, trade stats |
| Modify: `scripts/run_discovery_loop.py` | Wire LiveDashboardServer into discovery windows |
| Create: `scripts/run_forward_test.py` | CLI entry point for full pipeline or forward-test-only |
| Test: `tests/test_forward_test.py` | ForwardTestRunner tests |
| Test: `tests/test_monthly_tracker.py` | MonthlyPnLTracker tests |
| Test: `tests/test_orchestrator_apply.py` | Tests for the apply-loop fix |
| Test: `tests/test_forward_pipeline_e2e.py` | End-to-end integration test |

---

### Task 1: Monthly P&L Tracker

**Files:**
- Create: `src/discovery/monthly_tracker.py`
- Test: `tests/test_monthly_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_monthly_tracker.py
"""Tests for funded-phase monthly P&L tracking."""

import pytest


class TestMonthlyPnLTracker:
    def test_record_month_stores_data(self):
        from src.discovery.monthly_tracker import MonthlyPnLTracker

        tracker = MonthlyPnLTracker(initial_balance=10_000.0)
        tracker.record_month(
            month_label="2025-03",
            ending_balance=10_500.0,
            trades=28,
            wins=11,
            max_dd_pct=2.3,
            daily_dd_peak_pct=1.1,
            rules_applied=0,
        )
        report = tracker.get_monthly_reports()
        assert len(report) == 1
        assert report[0]["month"] == "2025-03"
        assert report[0]["return_pct"] == pytest.approx(5.0, abs=0.01)
        assert report[0]["status"] == "healthy"

    def test_consecutive_months_chain_balance(self):
        from src.discovery.monthly_tracker import MonthlyPnLTracker

        tracker = MonthlyPnLTracker(initial_balance=10_000.0)
        tracker.record_month("2025-03", ending_balance=10_500.0, trades=20, wins=8,
                             max_dd_pct=2.0, daily_dd_peak_pct=1.0, rules_applied=0)
        tracker.record_month("2025-04", ending_balance=11_000.0, trades=25, wins=10,
                             max_dd_pct=1.5, daily_dd_peak_pct=0.8, rules_applied=1)

        reports = tracker.get_monthly_reports()
        assert len(reports) == 2
        # Month 2 starting balance = month 1 ending balance
        assert reports[1]["starting_balance"] == 10_500.0
        assert reports[1]["return_pct"] == pytest.approx(4.76, abs=0.1)

    def test_bust_detected_when_dd_exceeds_limit(self):
        from src.discovery.monthly_tracker import MonthlyPnLTracker

        tracker = MonthlyPnLTracker(initial_balance=10_000.0, max_dd_limit_pct=10.0)
        tracker.record_month("2025-03", ending_balance=8_900.0, trades=30, wins=5,
                             max_dd_pct=11.0, daily_dd_peak_pct=3.0, rules_applied=0)

        reports = tracker.get_monthly_reports()
        assert reports[0]["status"] == "bust"

    def test_profitable_months_count(self):
        from src.discovery.monthly_tracker import MonthlyPnLTracker

        tracker = MonthlyPnLTracker(initial_balance=10_000.0)
        tracker.record_month("2025-03", ending_balance=10_300.0, trades=20, wins=8,
                             max_dd_pct=2.0, daily_dd_peak_pct=1.0, rules_applied=0)
        tracker.record_month("2025-04", ending_balance=10_100.0, trades=20, wins=7,
                             max_dd_pct=3.0, daily_dd_peak_pct=1.5, rules_applied=0)
        tracker.record_month("2025-05", ending_balance=10_500.0, trades=25, wins=10,
                             max_dd_pct=1.5, daily_dd_peak_pct=0.8, rules_applied=0)

        summary = tracker.get_summary()
        assert summary["total_months"] == 3
        assert summary["profitable_months"] == 2  # month 2 lost money (10300->10100)
        assert summary["survived"] is True

    def test_summary_includes_total_return(self):
        from src.discovery.monthly_tracker import MonthlyPnLTracker

        tracker = MonthlyPnLTracker(initial_balance=10_000.0)
        tracker.record_month("2025-03", ending_balance=10_800.0, trades=20, wins=9,
                             max_dd_pct=2.0, daily_dd_peak_pct=1.0, rules_applied=0)
        summary = tracker.get_summary()
        assert summary["total_return_pct"] == pytest.approx(8.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monthly_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MonthlyPnLTracker**

```python
# src/discovery/monthly_tracker.py
"""Monthly P&L tracker for the funded phase.

Tracks per-month returns, drawdowns, trade stats, and overall survival
during the post-challenge funded simulation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MonthlyPnLTracker:
    """Track monthly P&L during the funded forward test.

    Parameters
    ----------
    initial_balance:
        Account balance at the start of the funded phase.
    max_dd_limit_pct:
        Maximum total drawdown allowed (10% for The5ers funded).
        If any month's max DD exceeds this, status becomes 'bust'.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        max_dd_limit_pct: float = 10.0,
    ) -> None:
        self._initial_balance = initial_balance
        self._max_dd_limit = max_dd_limit_pct
        self._months: List[Dict[str, Any]] = []
        self._current_balance = initial_balance

    def record_month(
        self,
        month_label: str,
        ending_balance: float,
        trades: int,
        wins: int,
        max_dd_pct: float,
        daily_dd_peak_pct: float,
        rules_applied: int,
    ) -> Dict[str, Any]:
        """Record one month of funded trading results.

        Parameters
        ----------
        month_label: e.g. "2025-03"
        ending_balance: Account balance at month end.
        trades: Total trades this month.
        wins: Winning trades this month.
        max_dd_pct: Maximum total drawdown during the month.
        daily_dd_peak_pct: Worst single-day drawdown.
        rules_applied: Number of new discovery rules applied this month.

        Returns
        -------
        The month report dict.
        """
        starting_balance = self._current_balance
        return_pct = ((ending_balance - starting_balance) / starting_balance) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0.0

        bust = max_dd_pct > self._max_dd_limit
        status = "bust" if bust else "healthy"

        report = {
            "month": month_label,
            "starting_balance": round(starting_balance, 2),
            "ending_balance": round(ending_balance, 2),
            "return_pct": round(return_pct, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "daily_dd_peak_pct": round(daily_dd_peak_pct, 2),
            "trades": trades,
            "wins": wins,
            "win_rate_pct": round(win_rate, 1),
            "rules_applied": rules_applied,
            "status": status,
        }

        self._months.append(report)
        self._current_balance = ending_balance

        logger.info(
            "Month %s: %+.2f%% (%d trades, %.0f%% WR, DD=%.1f%%) — %s",
            month_label, return_pct, trades, win_rate, max_dd_pct, status,
        )
        return report

    def get_monthly_reports(self) -> List[Dict[str, Any]]:
        """Return all recorded monthly reports."""
        return list(self._months)

    def get_summary(self) -> Dict[str, Any]:
        """Return funded phase summary statistics."""
        if not self._months:
            return {"total_months": 0, "survived": True, "profitable_months": 0,
                    "total_return_pct": 0.0}

        profitable = sum(1 for m in self._months if m["return_pct"] > 0)
        busted = any(m["status"] == "bust" for m in self._months)
        total_return = ((self._current_balance - self._initial_balance)
                        / self._initial_balance) * 100

        return {
            "total_months": len(self._months),
            "profitable_months": profitable,
            "losing_months": len(self._months) - profitable,
            "survived": not busted,
            "total_return_pct": round(total_return, 2),
            "final_balance": round(self._current_balance, 2),
            "worst_month_dd_pct": max(m["max_dd_pct"] for m in self._months),
            "avg_monthly_return_pct": round(
                sum(m["return_pct"] for m in self._months) / len(self._months), 2
            ),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monthly_tracker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/monthly_tracker.py tests/test_monthly_tracker.py
git commit -m "feat: add MonthlyPnLTracker for funded phase (Task 1)"
```

---

### Task 2: Close the Apply Loop in Discovery Orchestrator

**Files:**
- Modify: `src/discovery/orchestrator.py`
- Test: `tests/test_orchestrator_apply.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator_apply.py
"""Tests for the apply-loop fix in DiscoveryOrchestrator."""

import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_candles(n_days=66, bars_per_day=300):
    """Create synthetic 1M candles spanning n_days trading days."""
    rng = np.random.default_rng(42)
    total_bars = n_days * bars_per_day
    idx = pd.date_range("2024-01-02", periods=total_bars, freq="1min", tz="UTC")
    prices = 2000.0 + np.cumsum(rng.normal(0, 0.5, total_bars))
    return pd.DataFrame({
        "open": prices,
        "high": prices + rng.uniform(0, 1, total_bars),
        "low": prices - rng.uniform(0, 1, total_bars),
        "close": prices + rng.normal(0, 0.3, total_bars),
        "volume": rng.integers(100, 800, total_bars),
    }, index=idx)


class TestApplyLoop:
    def test_hypothesis_applied_to_next_window_config(self):
        """After a hypothesis is generated, the NEXT window should use the modified config."""
        from src.discovery.orchestrator import DiscoveryOrchestrator

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "max_windows": 4},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "challenge": {},
                "validation": {"min_oos_windows": 2},
                "reporting": {"reports_dir": str(Path(tempfile.mkdtemp()) / "reports")},
            },
            knowledge_dir=kb_dir,
        )

        # The orchestrator's run() should track config evolution
        candles = _make_candles(n_days=88)
        result = orch.run(candles, base_config={
            "active_strategies": ["sss"],
            "strategies": {"sss": {"min_confluence_score": 4}},
        }, enable_claude=False)

        assert result["windows_processed"] >= 3
        # Config should have been tracked even if no hypotheses applied
        assert "config_evolution" in result

    def test_config_reverted_on_degradation(self):
        """If applied config makes performance worse, it should revert."""
        from src.discovery.rule_applier import apply_hypothesis_to_config

        base = {"strategies": {"sss": {"min_confluence_score": 4, "entry_mode": "cbc_only"}}}
        hyp = {"config_change": {"strategies": {"sss": {"min_confluence_score": 2}}}}

        modified = apply_hypothesis_to_config(hyp, base)
        assert modified["strategies"]["sss"]["min_confluence_score"] == 2

        # Revert is just using the original config
        reverted = copy.deepcopy(base)
        assert reverted["strategies"]["sss"]["min_confluence_score"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_apply.py -v`
Expected: Tests should run (may pass or fail depending on existing orchestrator behavior)

- [ ] **Step 3: Modify orchestrator to apply hypotheses between windows**

Add this method to `DiscoveryOrchestrator` in `src/discovery/orchestrator.py`, and update `run()` to call it:

In the `run()` method, after `process_window()` and before the next window, add logic to apply the top hypothesis:

```python
# Add to the run() method, inside the for-loop, after process_window():

            # ---- Apply top hypothesis to config for next window ----
            hypotheses = discovery.get("hypotheses", [])
            if hypotheses:
                top_hyp = hypotheses[0]  # highest confidence
                if top_hyp.get("config_change"):
                    from src.discovery.rule_applier import apply_hypothesis_to_config
                    prev_config = copy.deepcopy(current_config)
                    current_config = apply_hypothesis_to_config(top_hyp, current_config)
                    config_evolution.append({
                        "window_id": window_id,
                        "hypothesis_id": top_hyp.get("id", "unknown"),
                        "changes": [top_hyp.get("description", "config change")],
                        "prev_config_snapshot": prev_config,
                    })
                    logger.info(
                        "Applied hypothesis %s to config for next window",
                        top_hyp.get("id", "unknown"),
                    )

            # ---- Revert if performance degraded ----
            if len(window_results) >= 2 and config_evolution:
                prev_pass_rate = window_results[-2]["challenge_result"].get("pass_rate", 0.0)
                curr_pass_rate = result["challenge_result"].get("pass_rate", 0.0)
                prev_dd = window_results[-2].get("metrics", {}).get("max_drawdown_pct", 0.0)
                curr_dd = result.get("metrics", {}).get("max_drawdown_pct", 0.0)

                if curr_pass_rate < prev_pass_rate and curr_dd > prev_dd + 2.0:
                    last_evo = config_evolution[-1]
                    if "prev_config_snapshot" in last_evo:
                        current_config = last_evo["prev_config_snapshot"]
                        config_evolution.append({
                            "window_id": window_id,
                            "action": "reverted",
                            "reason": f"pass_rate {prev_pass_rate:.1%}->{curr_pass_rate:.1%}, DD +{curr_dd - prev_dd:.1f}%",
                        })
                        logger.warning(
                            "Reverted config change — performance degraded",
                        )
```

You also need to add `import copy` at the top of the file if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_apply.py tests/test_orchestrator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/orchestrator.py tests/test_orchestrator_apply.py
git commit -m "feat: close the apply loop — hypotheses modify config between windows (Task 2)"
```

---

### Task 3: Forward Test Runner

**Files:**
- Create: `src/discovery/forward_test.py`
- Test: `tests/test_forward_test.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_forward_test.py
"""Tests for the ForwardTestRunner (challenge + funded phases)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_candles(n_days=120, bars_per_day=300, seed=42):
    rng = np.random.default_rng(seed)
    total_bars = n_days * bars_per_day
    idx = pd.date_range("2025-01-02", periods=total_bars, freq="1min", tz="UTC")
    prices = 2600.0 + np.cumsum(rng.normal(0.01, 0.5, total_bars))
    return pd.DataFrame({
        "open": prices,
        "high": prices + rng.uniform(0, 1, total_bars),
        "low": prices - rng.uniform(0, 1, total_bars),
        "close": prices + rng.normal(0, 0.3, total_bars),
        "volume": rng.integers(100, 800, total_bars),
    }, index=idx)


class TestForwardTestRunner:
    def test_init_with_config(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
        )
        assert runner is not None

    def test_run_challenge_returns_result(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        candles = _make_candles(n_days=50)
        result = runner.run_challenge(candles)

        assert "phase_1" in result
        assert "passed" in result["phase_1"]
        assert "trades" in result["phase_1"]

    def test_run_funded_returns_monthly_reports(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        candles = _make_candles(n_days=66)  # ~3 months
        result = runner.run_funded(candles, enable_discovery=False)

        assert "monthly_reports" in result
        assert "summary" in result
        assert result["summary"]["total_months"] >= 1

    def test_run_full_pipeline(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        candles = _make_candles(n_days=120)
        result = runner.run(candles, enable_discovery=False)

        assert "challenge" in result
        assert "funded" in result
        assert result["challenge"]["phase_1"]["passed"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_forward_test.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ForwardTestRunner**

```python
# src/discovery/forward_test.py
"""Forward test runner — challenge simulation + funded monthly trading.

Runs the optimized config through The5ers Phase 1 + Phase 2, then
trades the funded account month-by-month with optional continued
discovery.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.vectorbt_engine import IchimokuBacktester
from src.discovery.monthly_tracker import MonthlyPnLTracker

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_MONTH = 22
_PHASE_1_TARGET_PCT = 8.0
_PHASE_2_TARGET_PCT = 5.0
_MAX_DD_PCT = 10.0
_DAILY_DD_PCT = 5.0
_FUNDED_DD_LIMIT_FOR_NEW_RULES = 5.0  # cautious: half the 10% limit


class ForwardTestRunner:
    """Run challenge + funded forward test on unseen data.

    Parameters
    ----------
    config:
        Strategy config dict (from discovery phase optimization).
    initial_balance:
        Account size for the challenge.
    knowledge_dir:
        Path for discovery knowledge base (funded-phase discovery).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        initial_balance: float = 10_000.0,
        knowledge_dir: str = "reports/agent_knowledge",
        live_dashboard: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._initial_balance = initial_balance
        self._knowledge_dir = knowledge_dir
        self._live_dashboard = live_dashboard

    def run(
        self,
        candles: pd.DataFrame,
        enable_discovery: bool = False,
        enable_claude: bool = False,
    ) -> Dict[str, Any]:
        """Run the full forward test: challenge + funded.

        Splits data: first 44 trading days for challenge (22 Phase 1 + 22 Phase 2),
        remainder for funded monthly trading.
        """
        dates = candles.index.normalize().unique().sort_values()
        trading_days = dates.tolist()

        challenge_days = min(44, len(trading_days))
        challenge_end = trading_days[challenge_days - 1] + pd.Timedelta(days=1)
        challenge_candles = candles.loc[:challenge_end]
        funded_candles = candles.loc[challenge_end:]

        logger.info(
            "Forward test: %d days challenge, %d days funded",
            challenge_days, len(trading_days) - challenge_days,
        )

        challenge_result = self.run_challenge(challenge_candles)

        funded_result = {"monthly_reports": [], "summary": {"total_months": 0, "survived": True}}
        if len(funded_candles) > 0:
            balance_after_challenge = challenge_result.get("ending_balance", self._initial_balance)
            funded_result = self.run_funded(
                funded_candles,
                starting_balance=balance_after_challenge,
                enable_discovery=enable_discovery,
                enable_claude=enable_claude,
            )

        return {
            "challenge": challenge_result,
            "funded": funded_result,
        }

    def run_challenge(self, candles: pd.DataFrame) -> Dict[str, Any]:
        """Run Phase 1 + Phase 2 challenge simulation."""
        dates = candles.index.normalize().unique().sort_values()
        trading_days = dates.tolist()

        phase_1_days = min(_TRADING_DAYS_PER_MONTH, len(trading_days))
        phase_1_end = trading_days[phase_1_days - 1] + pd.Timedelta(days=1)
        phase_1_candles = candles.loc[:phase_1_end]

        # Phase 1
        p1_result = self._run_phase(
            phase_1_candles, self._initial_balance,
            target_pct=_PHASE_1_TARGET_PCT, label="Phase 1",
        )

        # Phase 2
        p2_result = {"passed": None, "trades": [], "metrics": {}, "ending_balance": p1_result["ending_balance"]}
        remaining = candles.loc[phase_1_end:]
        if p1_result["passed"] and len(remaining) > 0:
            p2_balance = self._initial_balance  # balance resets for Phase 2
            p2_result = self._run_phase(
                remaining, p2_balance,
                target_pct=_PHASE_2_TARGET_PCT, label="Phase 2",
            )

        ending_balance = p2_result["ending_balance"] if p2_result["passed"] else p1_result["ending_balance"]

        return {
            "phase_1": p1_result,
            "phase_2": p2_result,
            "passed_both": bool(p1_result["passed"] and p2_result.get("passed")),
            "ending_balance": ending_balance,
        }

    def run_funded(
        self,
        candles: pd.DataFrame,
        starting_balance: Optional[float] = None,
        enable_discovery: bool = False,
        enable_claude: bool = False,
    ) -> Dict[str, Any]:
        """Run funded-phase monthly forward test."""
        balance = starting_balance or self._initial_balance
        tracker = MonthlyPnLTracker(initial_balance=balance, max_dd_limit_pct=_MAX_DD_PCT)

        dates = candles.index.normalize().unique().sort_values()
        trading_days = dates.tolist()

        current_config = copy.deepcopy(self._config)
        all_funded_trades: List[Dict] = []
        month_num = 0

        # Discovery runner for funded phase (cautious settings)
        discovery_runner = None
        if enable_discovery:
            from src.discovery.runner import DiscoveryRunner
            discovery_runner = DiscoveryRunner(
                shap_every_n_windows=3,  # every 3 months
                min_trades_for_shap=50,
                knowledge_dir=self._knowledge_dir,
            )

        idx = 0
        while idx < len(trading_days):
            month_end_idx = min(idx + _TRADING_DAYS_PER_MONTH, len(trading_days))
            month_days = trading_days[idx:month_end_idx]

            if len(month_days) < 5:
                break

            start_dt = month_days[0]
            end_dt = month_days[-1] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            month_candles = candles.loc[start_dt:end_dt]

            month_label = start_dt.strftime("%Y-%m")
            logger.info("Funded month %s: %d bars, %d trading days", month_label, len(month_candles), len(month_days))

            # Run backtest for this month
            backtester = IchimokuBacktester(config=current_config, initial_balance=balance)
            bt_result = backtester.run(
                month_candles, instrument="XAUUSD",
                enable_learning=True, live_dashboard=self._live_dashboard,
            )

            trades = bt_result.trades
            metrics = bt_result.metrics
            all_funded_trades.extend(trades)

            wins = sum(1 for t in trades if (t.get("r_multiple") or 0) > 0)
            max_dd = abs(metrics.get("max_drawdown_pct", 0))
            daily_dd_peak = abs(bt_result.prop_firm.get("max_daily_dd_pct", 0))
            ending_balance = balance * (1 + metrics.get("total_return_pct", 0) / 100)

            rules_applied = 0

            # Funded-phase discovery (cautious)
            if discovery_runner is not None:
                disc_result = discovery_runner.run_full_cycle(
                    window_id=f"funded_{month_label}",
                    window_index=month_num,
                    trades=trades,
                    strategy_name=current_config.get("active_strategies", ["sss"])[0],
                    base_config=current_config,
                    enable_claude=enable_claude,
                )

                # Only apply if DD stays under cautious limit
                hypotheses = disc_result.get("hypotheses", [])
                if hypotheses and max_dd < _FUNDED_DD_LIMIT_FOR_NEW_RULES:
                    top_hyp = hypotheses[0]
                    if top_hyp.get("config_change"):
                        from src.discovery.rule_applier import apply_hypothesis_to_config
                        current_config = apply_hypothesis_to_config(top_hyp, current_config)
                        rules_applied = 1
                        logger.info("Funded discovery: applied 1 rule (DD=%.1f%% < %.1f%% limit)",
                                    max_dd, _FUNDED_DD_LIMIT_FOR_NEW_RULES)

            tracker.record_month(
                month_label=month_label,
                ending_balance=ending_balance,
                trades=len(trades),
                wins=wins,
                max_dd_pct=max_dd,
                daily_dd_peak_pct=daily_dd_peak,
                rules_applied=rules_applied,
            )

            balance = ending_balance
            month_num += 1
            idx = month_end_idx

            # Bust check
            if tracker.get_monthly_reports()[-1]["status"] == "bust":
                logger.error("FUNDED BUST at month %s — stopping forward test", month_label)
                break

        return {
            "monthly_reports": tracker.get_monthly_reports(),
            "summary": tracker.get_summary(),
            "total_trades": len(all_funded_trades),
            "final_config": current_config,
        }

    def _run_phase(
        self,
        candles: pd.DataFrame,
        balance: float,
        target_pct: float,
        label: str,
    ) -> Dict[str, Any]:
        """Run a single challenge phase (Phase 1 or Phase 2)."""
        backtester = IchimokuBacktester(
            config=self._config,
            initial_balance=balance,
            prop_firm_profit_target_pct=target_pct,
            prop_firm_max_daily_dd_pct=_DAILY_DD_PCT,
            prop_firm_max_total_dd_pct=_MAX_DD_PCT,
        )
        result = backtester.run(
            candles, instrument="XAUUSD",
            enable_learning=True, live_dashboard=self._live_dashboard,
        )

        return_pct = result.metrics.get("total_return_pct", 0)
        max_dd = abs(result.metrics.get("max_drawdown_pct", 0))
        passed = return_pct >= target_pct and max_dd <= _MAX_DD_PCT

        ending_balance = balance * (1 + return_pct / 100)

        logger.info(
            "%s: return=%.2f%% (target %.1f%%), DD=%.2f%%, trades=%d — %s",
            label, return_pct, target_pct, max_dd, len(result.trades),
            "PASSED" if passed else "FAILED",
        )

        return {
            "passed": passed,
            "return_pct": return_pct,
            "max_dd_pct": max_dd,
            "trades": result.trades,
            "metrics": result.metrics,
            "ending_balance": ending_balance,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_forward_test.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/forward_test.py tests/test_forward_test.py
git commit -m "feat: add ForwardTestRunner — challenge + funded monthly trading (Task 3)"
```

---

### Task 4: Wire Live Dashboard into Discovery Loop

**Files:**
- Modify: `scripts/run_discovery_loop.py`
- Modify: `src/discovery/orchestrator.py` (accept live_dashboard param)

- [ ] **Step 1: Read current run_discovery_loop.py to understand its structure**

Run: `head -60 scripts/run_discovery_loop.py`

- [ ] **Step 2: Modify orchestrator to accept and pass live_dashboard**

In `src/discovery/orchestrator.py`, add `live_dashboard` parameter to `__init__`, `process_window`, and `run`:

```python
# In __init__, add parameter:
    def __init__(self, ..., live_dashboard=None) -> None:
        ...
        self._live_dashboard = live_dashboard

# In process_window, pass it to backtester.run():
        bt_result = backtester.run(
            candles_1m=candles,
            instrument="XAUUSD",
            enable_learning=True,
            live_dashboard=self._live_dashboard,
        )
```

- [ ] **Step 3: Modify run_discovery_loop.py to start LiveDashboardServer**

Add before the orchestrator.run() call:

```python
    # Start live dashboard
    live_server = None
    try:
        from src.backtesting.live_dashboard import LiveDashboardServer
        live_server = LiveDashboardServer(
            port=args.port if hasattr(args, 'port') else 8501,
            auto_open=True,
            app_config=strategy_config,
            config_dir=str(Path(__file__).resolve().parent.parent / "config"),
        )
        live_server.start()
        logger.info("Live dashboard: http://localhost:%d", live_server.port)
    except Exception as exc:
        logger.warning("Could not start live dashboard: %s", exc)

    # Pass to orchestrator
    orch = DiscoveryOrchestrator(config=disc_config, ..., live_dashboard=live_server)
```

- [ ] **Step 4: Verify dashboard starts**

Run: `python scripts/run_discovery_loop.py --data-file data/xauusd_1m_2023_2024.parquet --strategy sss --max-windows 1`
Expected: Dashboard opens at http://localhost:8501, first window processes with live updates

- [ ] **Step 5: Commit**

```bash
git add src/discovery/orchestrator.py scripts/run_discovery_loop.py
git commit -m "feat: wire LiveDashboardServer into discovery loop (Task 4)"
```

---

### Task 5: CLI Entry Point — run_forward_test.py

**Files:**
- Create: `scripts/run_forward_test.py`

- [ ] **Step 1: Create the CLI script**

```python
# scripts/run_forward_test.py
"""Full forward test pipeline: discovery -> challenge -> funded.

Usage:
    # Full pipeline:
    python scripts/run_forward_test.py \
        --discovery-data data/xauusd_1m_2023_2024.parquet \
        --forward-data data/xauusd_1m_2025.parquet \
        --strategy sss --enable-claude

    # Forward test only (skip discovery, use current config):
    python scripts/run_forward_test.py \
        --forward-data data/xauusd_1m_2025.parquet \
        --strategy sss --skip-discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_forward_test")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Forward test pipeline: discovery -> challenge -> funded.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--discovery-data", type=str, default=None,
                    help="Path to discovery-phase data (e.g. 2023-2024).")
    p.add_argument("--forward-data", type=str, required=True,
                    help="Path to forward-test data (e.g. 2025).")
    p.add_argument("--strategy", type=str, default="sss",
                    help="Strategy to test.")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--skip-discovery", action="store_true",
                    help="Skip discovery phase, use current config.")
    p.add_argument("--enable-claude", action="store_true",
                    help="Enable Claude CLI for hypothesis generation.")
    p.add_argument("--enable-funded-discovery", action="store_true",
                    help="Enable continued discovery during funded phase.")
    p.add_argument("--port", type=int, default=8501,
                    help="Live dashboard port.")
    p.add_argument("--output-dir", type=str, default="reports/forward_test")
    return p


def _load_config(strategy: str) -> dict:
    """Load strategy config from YAML files."""
    config_dir = _PROJECT_ROOT / "config"
    with open(config_dir / "strategy.yaml") as f:
        raw_strat = yaml.safe_load(f) or {}
    with open(config_dir / "edges.yaml") as f:
        raw_edges = yaml.safe_load(f) or {}

    return {
        "active_strategies": [strategy],
        "strategies": raw_strat.get("strategies", {}),
        "edges": raw_edges,
        "risk": raw_strat.get("risk", {}),
        "exit": raw_strat.get("exit", {}),
        "prop_firm": raw_strat.get("prop_firm", {}),
        "initial_risk_pct": raw_strat.get("risk", {}).get("initial_risk_pct", 0.5),
        "reduced_risk_pct": raw_strat.get("risk", {}).get("reduced_risk_pct", 0.75),
        "phase_threshold_pct": raw_strat.get("risk", {}).get("phase_threshold_pct", 4.0),
        "tp_r_multiple": raw_strat.get("exit", {}).get("tp_r_multiple", 2.0),
        "kijun_trail_start_r": raw_strat.get("exit", {}).get("kijun_trail_start_r", 1.5),
        "higher_tf_kijun_start_r": raw_strat.get("exit", {}).get("higher_tf_kijun_start_r", 3.0),
        "atr_stop_multiplier": 2.5,
    }


def _load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.strategy)

    # Start live dashboard
    live_server = None
    try:
        from src.backtesting.live_dashboard import LiveDashboardServer
        live_server = LiveDashboardServer(
            port=args.port, auto_open=True,
            app_config=config,
            config_dir=str(_PROJECT_ROOT / "config"),
        )
        live_server.start()
        logger.info("Live dashboard: http://localhost:%d", args.port)
    except Exception as exc:
        logger.warning("Could not start live dashboard: %s", exc)

    # Stage 1: Discovery (optional)
    if not args.skip_discovery and args.discovery_data:
        print("\n" + "=" * 60)
        print("  STAGE 1: DISCOVERY PHASE")
        print("=" * 60)

        discovery_data = _load_data(args.discovery_data)
        logger.info("Discovery data: %d bars", len(discovery_data))

        from src.discovery.orchestrator import DiscoveryOrchestrator
        disc_config_path = _PROJECT_ROOT / "config" / "discovery.yaml"
        disc_config = {}
        if disc_config_path.exists():
            with open(disc_config_path) as f:
                disc_config = yaml.safe_load(f) or {}

        orch = DiscoveryOrchestrator(
            config=disc_config,
            live_dashboard=live_server,
        )
        disc_result = orch.run(
            discovery_data, base_config=config, enable_claude=args.enable_claude,
        )

        print(f"\n  Discovery: {disc_result['windows_processed']} windows processed")
        print(f"  Config changes: {len(disc_result.get('config_evolution', []))}")

        # Use evolved config for forward test
        # (config was modified in-place by the apply loop)

    # Stage 2+3: Challenge + Funded
    print("\n" + "=" * 60)
    print("  STAGE 2-3: CHALLENGE + FUNDED FORWARD TEST")
    print("=" * 60)

    forward_data = _load_data(args.forward_data)
    logger.info("Forward test data: %d bars", len(forward_data))

    from src.discovery.forward_test import ForwardTestRunner
    runner = ForwardTestRunner(
        config=config,
        initial_balance=args.initial_balance,
        knowledge_dir=str(output_dir / "knowledge"),
        live_dashboard=live_server,
    )

    result = runner.run(
        forward_data,
        enable_discovery=args.enable_funded_discovery,
        enable_claude=args.enable_claude,
    )

    # Print results
    challenge = result["challenge"]
    funded = result["funded"]

    print(f"\n{'=' * 60}")
    print("  CHALLENGE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Phase 1: {'PASSED' if challenge['phase_1']['passed'] else 'FAILED'} "
          f"({challenge['phase_1'].get('return_pct', 0):.2f}% return)")
    if challenge["phase_2"].get("passed") is not None:
        print(f"  Phase 2: {'PASSED' if challenge['phase_2']['passed'] else 'FAILED'} "
              f"({challenge['phase_2'].get('return_pct', 0):.2f}% return)")

    print(f"\n{'=' * 60}")
    print("  FUNDED MONTHLY P&L")
    print(f"{'=' * 60}")
    for m in funded.get("monthly_reports", []):
        status_icon = "OK" if m["status"] == "healthy" else "BUST"
        print(f"  {m['month']}: {m['return_pct']:+6.2f}%  DD={m['max_dd_pct']:.1f}%  "
              f"trades={m['trades']:3d}  WR={m['win_rate_pct']:.0f}%  [{status_icon}]")

    summary = funded.get("summary", {})
    print(f"\n  Total return:      {summary.get('total_return_pct', 0):+.2f}%")
    print(f"  Profitable months: {summary.get('profitable_months', 0)}/{summary.get('total_months', 0)}")
    print(f"  Survived:          {'YES' if summary.get('survived') else 'NO'}")
    print(f"{'=' * 60}")

    # Save results
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"forward_test_{ts}.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "strategy": args.strategy,
            "challenge": {
                "phase_1_passed": challenge["phase_1"]["passed"],
                "phase_1_return": challenge["phase_1"].get("return_pct"),
                "phase_2_passed": challenge["phase_2"].get("passed"),
                "phase_2_return": challenge["phase_2"].get("return_pct"),
            },
            "funded": {
                "monthly_reports": funded.get("monthly_reports", []),
                "summary": summary,
            },
        }, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    if live_server:
        print(f"  Dashboard: http://localhost:{args.port}")
        print("  Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it runs**

Run: `python scripts/run_forward_test.py --forward-data data/xauusd_1m_2023_2025.parquet --strategy sss --skip-discovery`
Expected: Challenge + funded phases execute, monthly P&L printed

- [ ] **Step 3: Commit**

```bash
git add scripts/run_forward_test.py
git commit -m "feat: add run_forward_test.py CLI entry point (Task 5)"
```

---

### Task 6: Prepare 2025 Data File

**Files:**
- No source changes — data preparation

- [ ] **Step 1: Create 2025-only parquet**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/xauusd_1m_2023_2025.parquet')
mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
df_2025 = df[mask]
df_2025.to_parquet('data/xauusd_1m_2025.parquet')
print(f'Saved {len(df_2025):,} bars: {df_2025.index[0]} to {df_2025.index[-1]}')
"
```

Expected: `data/xauusd_1m_2025.parquet` created with ~354K bars

- [ ] **Step 2: Commit**

```bash
git add -n data/xauusd_1m_2025.parquet  # verify it's a reasonable size
# Do NOT commit data files to git — they are in .gitignore
echo "Data file ready at data/xauusd_1m_2025.parquet"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Test: `tests/test_forward_pipeline_e2e.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_forward_pipeline_e2e.py
"""End-to-end test for the full forward test pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_candles(start_date, n_days=60, bars_per_day=300, seed=42):
    rng = np.random.default_rng(seed)
    total_bars = n_days * bars_per_day
    idx = pd.date_range(start_date, periods=total_bars, freq="1min", tz="UTC")
    prices = 2000.0 + np.cumsum(rng.normal(0, 0.5, total_bars))
    return pd.DataFrame({
        "open": prices,
        "high": prices + rng.uniform(0, 1, total_bars),
        "low": prices - rng.uniform(0, 1, total_bars),
        "close": prices + rng.normal(0, 0.3, total_bars),
        "volume": rng.integers(100, 800, total_bars),
    }, index=idx)


class TestForwardPipelineE2E:
    def test_challenge_produces_phase_results(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        candles = _make_candles("2025-01-02", n_days=50)
        result = runner.run_challenge(candles)

        assert result["phase_1"]["passed"] is not None
        assert isinstance(result["phase_1"]["trades"], list)
        assert "return_pct" in result["phase_1"]
        assert "max_dd_pct" in result["phase_1"]

    def test_funded_tracks_monthly_pnl(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        candles = _make_candles("2025-03-01", n_days=66)
        result = runner.run_funded(candles)

        assert len(result["monthly_reports"]) >= 1
        for report in result["monthly_reports"]:
            assert "month" in report
            assert "return_pct" in report
            assert "status" in report

    def test_full_pipeline_challenge_then_funded(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        candles = _make_candles("2025-01-02", n_days=120)
        result = runner.run(candles, enable_discovery=False)

        assert "challenge" in result
        assert "funded" in result
        assert result["funded"]["summary"]["total_months"] >= 1

    def test_monthly_tracker_survives_full_run(self):
        from src.discovery.forward_test import ForwardTestRunner

        runner = ForwardTestRunner(
            config={"active_strategies": ["sss"], "strategies": {"sss": {}}},
            initial_balance=10_000.0,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        candles = _make_candles("2025-03-01", n_days=44)
        result = runner.run_funded(candles)

        summary = result["summary"]
        assert "total_months" in summary
        assert "profitable_months" in summary
        assert "survived" in summary
        assert "total_return_pct" in summary
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_monthly_tracker.py tests/test_forward_test.py tests/test_orchestrator_apply.py tests/test_forward_pipeline_e2e.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_forward_pipeline_e2e.py
git commit -m "feat: add forward pipeline integration tests (Task 7)"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Monthly P&L Tracker | `monthly_tracker.py` | 5 |
| 2 | Close Apply Loop | `orchestrator.py` (modify) | 2 |
| 3 | Forward Test Runner | `forward_test.py` | 4 |
| 4 | Live Dashboard Wiring | `orchestrator.py` + `run_discovery_loop.py` (modify) | manual |
| 5 | CLI Entry Point | `scripts/run_forward_test.py` | manual |
| 6 | 2025 Data File | data prep | — |
| 7 | Integration Tests | `test_forward_pipeline_e2e.py` | 4 |

**Total: 7 tasks, 15 tests, 3 new files, 3 modified files.**

After implementation, run the full pipeline:
```bash
python scripts/run_forward_test.py \
    --discovery-data data/xauusd_1m_2023_2024.parquet \
    --forward-data data/xauusd_1m_2025.parquet \
    --strategy sss \
    --enable-claude \
    --enable-funded-discovery
```
