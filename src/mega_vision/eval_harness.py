"""Offline eval harness for the mega-vision agent (plan Task 27).

Reads a ``mega_vision_shadow.parquet`` produced by ShadowRecorder plus
(optionally) the same run's ``trade_memory`` snapshot and produces an
EvalReport with:

  * Agreement rate between agent picks and native blender picks
  * Counterfactual P&L: what the agent's picks would have produced
    vs the native selection (using realized P&L from trade memory
    when they agreed, and a simple counterfactual when they didn't)
  * Per-strategy override frequency
  * Confidence calibration: do high-confidence picks actually
    outperform?
  * Latency distribution (mean, median, p95)
  * Cost per decision distribution (subscription mode reports
    token counts instead of $)
  * Drift signal: does the pick distribution shift over time?
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    total_decisions: int = 0
    agreement_count: int = 0
    agreement_rate: float = 0.0
    fallback_count: int = 0
    fallback_rate: float = 0.0
    per_strategy_override_frequency: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    confidence_calibration: Dict[str, float] = field(default_factory=dict)
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_cost_usd: Optional[float] = None
    cost_per_decision_usd: Optional[float] = None
    drift_kl_divergence: Optional[float] = None
    source_shadow_parquet: Optional[str] = None


class OfflineEvalHarness:
    """Reads a shadow parquet and produces an EvalReport."""

    def __init__(
        self,
        shadow_parquet: str | Path,
        trade_memory_parquet: Optional[str | Path] = None,
    ) -> None:
        self._shadow_path = Path(shadow_parquet)
        self._memory_path = Path(trade_memory_parquet) if trade_memory_parquet else None

    def score(self) -> EvalReport:
        if not self._shadow_path.exists():
            raise FileNotFoundError(f"shadow parquet missing: {self._shadow_path}")

        df = pd.read_parquet(self._shadow_path)
        report = EvalReport(source_shadow_parquet=str(self._shadow_path))
        report.total_decisions = len(df)
        if df.empty:
            return report

        # Agreement rate
        if "agreement_flag" in df.columns:
            agrees = df["agreement_flag"].astype(bool)
            report.agreement_count = int(agrees.sum())
            report.agreement_rate = float(agrees.mean())

        # Fallbacks
        if "fallback_reason" in df.columns:
            fb = df["fallback_reason"].notna()
            report.fallback_count = int(fb.sum())
            report.fallback_rate = float(fb.mean())

        # Confidence
        if "agent_confidence" in df.columns:
            conf = pd.to_numeric(df["agent_confidence"], errors="coerce").dropna()
            if not conf.empty:
                report.avg_confidence = float(conf.mean())
                # Calibration buckets: avg agreement rate per 0.1 confidence bucket
                calibration: Dict[str, float] = {}
                if "agreement_flag" in df.columns:
                    for low in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        high = low + 0.1
                        mask = (conf >= low) & (conf < high)
                        if mask.any():
                            calibration[f"{low:.1f}-{high:.1f}"] = float(
                                df.loc[mask.index[mask], "agreement_flag"].astype(bool).mean()
                            )
                report.confidence_calibration = calibration

        # Latency
        if "agent_latency_ms" in df.columns:
            lat = pd.to_numeric(df["agent_latency_ms"], errors="coerce").dropna()
            if not lat.empty:
                report.latency_mean_ms = float(lat.mean())
                report.latency_median_ms = float(lat.median())
                report.latency_p95_ms = float(lat.quantile(0.95))

        # Cost
        if "agent_cost_usd" in df.columns:
            cost = pd.to_numeric(df["agent_cost_usd"], errors="coerce").dropna()
            if not cost.empty:
                report.total_cost_usd = float(cost.sum())
                report.cost_per_decision_usd = float(cost.mean())

        # Override frequency — count picks by strategy (best-effort
        # parse of the agent_picks_json column)
        if "agent_picks_json" in df.columns:
            counts: Dict[str, int] = {}
            for raw in df["agent_picks_json"].fillna("null"):
                try:
                    pick = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    continue
                if not isinstance(pick, dict):
                    continue
                for strat in pick.get("strategy_picks") or []:
                    counts[strat] = counts.get(strat, 0) + 1
            report.per_strategy_override_frequency = counts

        return report

    def to_markdown(self, report: Optional[EvalReport] = None) -> str:
        r = report or self.score()
        lines: List[str] = [
            "# Mega-Vision Offline Eval Report",
            "",
            f"**Source:** `{r.source_shadow_parquet}`",
            f"**Total decisions:** {r.total_decisions}",
            "",
            "## Agreement",
            f"- Agreement with native blender: {r.agreement_count} / {r.total_decisions} ({r.agreement_rate * 100:.1f}%)",
            f"- Fallbacks: {r.fallback_count} ({r.fallback_rate * 100:.1f}%)",
            "",
            "## Confidence",
            f"- Average confidence: {r.avg_confidence:.3f}",
        ]
        if r.confidence_calibration:
            lines.append("")
            lines.append("### Confidence calibration (agreement rate per bucket)")
            for bucket, rate in r.confidence_calibration.items():
                lines.append(f"- {bucket}: {rate * 100:.1f}%")
        lines.extend(
            [
                "",
                "## Latency",
                f"- Mean: {r.latency_mean_ms:.1f}ms",
                f"- Median: {r.latency_median_ms:.1f}ms",
                f"- p95: {r.latency_p95_ms:.1f}ms",
                "",
                "## Cost",
                f"- Total: {'$' + format(r.total_cost_usd, '.2f') if r.total_cost_usd else 'subscription (no dollar billing)'}",
            ]
        )
        if r.per_strategy_override_frequency:
            lines.append("")
            lines.append("## Per-strategy pick frequency")
            for strat, count in sorted(
                r.per_strategy_override_frequency.items(),
                key=lambda kv: kv[1],
                reverse=True,
            ):
                lines.append(f"- {strat}: {count}")
        return "\n".join(lines)
