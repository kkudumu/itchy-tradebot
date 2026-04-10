"""Training data pipeline — shadow logs → labeled (state, outcome) examples.

Reads a mega_vision_shadow.parquet + the same run's trade_memory
parquet + (optionally) the screenshot directory. Joins on timestamp
to produce one training example per decision that has a downstream
trade outcome. Each example is:

  {
    "ts": "2026-04-09T14:30:00Z",
    "context": { candidate_signals, agent_picks, reasoning, ... },
    "outcome": { pnl_usd, r_multiple, agreed_with_native },
    "screenshot_path": "screenshots/run_id/sss_...png" | None
  }

This is the shape a future fine-tuning run would consume. We don't
actually fine-tune here — we just make sure the data exists and is
well-formed.
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
class TrainingExample:
    ts: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    screenshot_path: Optional[str] = None


class TrainingDataPipeline:
    """Joins shadow decisions with trade outcomes to produce training examples."""

    def build_dataset(
        self,
        shadow_parquet: str | Path,
        trade_memory_parquet: Optional[str | Path],
        output_dir: str | Path,
        max_join_window_minutes: int = 180,
    ) -> int:
        """Build the dataset and write it to *output_dir*.

        Returns the number of examples produced.
        """
        shadow_path = Path(shadow_parquet)
        if not shadow_path.exists():
            logger.warning("shadow parquet missing: %s", shadow_path)
            return 0
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        shadow_df = pd.read_parquet(shadow_path)
        if shadow_df.empty:
            # Empty training set — still write a valid parquet
            pd.DataFrame(columns=["ts", "context", "outcome", "screenshot_path"]).to_parquet(
                out / "examples.parquet"
            )
            return 0

        trades_df: Optional[pd.DataFrame] = None
        if trade_memory_parquet is not None:
            tmpath = Path(trade_memory_parquet)
            if tmpath.exists():
                trades_df = pd.read_parquet(tmpath)

        examples: List[TrainingExample] = []
        for _, row in shadow_df.iterrows():
            ts = str(row.get("ts_utc") or "")
            context = {
                "candidate_signals": self._parse_json(row.get("candidate_signals_json")),
                "agent_picks": self._parse_json(row.get("agent_picks_json")),
                "native_picks": self._parse_json(row.get("native_picks_json")),
                "reasoning": str(row.get("agent_reasoning") or ""),
                "confidence": float(row.get("agent_confidence") or 0.0),
            }

            outcome = self._find_outcome(trades_df, ts, max_join_window_minutes)
            example = TrainingExample(
                ts=ts,
                context=context,
                outcome=outcome,
                screenshot_path=None,
            )
            examples.append(example)

        # Write parquet — each example becomes a row with JSON-serialized
        # context + outcome so pyarrow doesn't try to infer nested
        # struct schemas.
        rows = [
            {
                "ts": e.ts,
                "context_json": json.dumps(e.context, default=str),
                "outcome_json": json.dumps(e.outcome, default=str),
                "screenshot_path": e.screenshot_path,
            }
            for e in examples
        ]
        pd.DataFrame(rows).to_parquet(out / "examples.parquet", index=False)
        return len(examples)

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return None

    @staticmethod
    def _find_outcome(
        trades_df: Optional[pd.DataFrame],
        decision_ts: str,
        window_minutes: int,
    ) -> Dict[str, Any]:
        """Find the trade that opened closest after *decision_ts*.

        Returns a shallow dict (trade_id, pnl_usd, r_multiple) or an
        empty dict when no matching trade is found within the window.
        """
        if trades_df is None or trades_df.empty:
            return {}
        if "opened_at" not in trades_df.columns:
            return {}
        try:
            dec = pd.Timestamp(decision_ts)
        except Exception:
            return {}

        # Compute time deltas and find the closest opened_at > decision_ts
        opened = pd.to_datetime(trades_df["opened_at"], errors="coerce")
        delta_minutes = (opened - dec).dt.total_seconds() / 60.0
        valid = (delta_minutes >= 0) & (delta_minutes <= window_minutes)
        if not valid.any():
            return {}
        idx = delta_minutes[valid].idxmin()
        trade = trades_df.loc[idx]
        return {
            "trade_id": str(trade.get("trade_id") or ""),
            "pnl_usd": float(trade.get("pnl_usd") or 0.0),
            "r_multiple": float(trade.get("r_multiple") or 0.0),
        }
