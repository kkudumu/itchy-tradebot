"""Shadow mode recorder — persists every mega-vision decision to parquet.

Called by the engine (or live runner) on every agent decision when
``mega_vision.mode`` is ``shadow`` or ``authority``. The records
feed the offline eval harness (Task 27) and the training data
pipeline (Task 27's second deliverable).

Schema (one row per decision):
  ts_utc
  candidate_signals_json
  agent_picks_json
  native_picks_json
  agreement_flag
  agent_confidence
  agent_reasoning
  agent_latency_ms
  agent_cost_usd
  fallback_reason
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShadowRecord:
    ts_utc: str = ""
    candidate_signals_json: str = "[]"
    agent_picks_json: str = "null"
    native_picks_json: str = "[]"
    agreement_flag: bool = False
    agent_confidence: float = 0.0
    agent_reasoning: str = ""
    agent_latency_ms: float = 0.0
    agent_cost_usd: Optional[float] = None
    fallback_reason: Optional[str] = None


class ShadowRecorder:
    """Buffered shadow-mode decision log."""

    def __init__(self, run_id: str, out_dir: Path | str) -> None:
        self.run_id = run_id
        self._out_dir = Path(out_dir)
        self._records: List[ShadowRecord] = []

    # ------------------------------------------------------------------

    def record(
        self,
        ts: datetime,
        candidate_signals: List[Any],
        agent_pick: Optional[Dict[str, Any]],
        native_picks: List[Any],
        latency_ms: float = 0.0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Append a decision to the buffer."""
        candidate_payload = [
            _signal_summary(s) for s in candidate_signals
        ]
        native_payload = [_signal_summary(s) for s in native_picks]

        picks = (agent_pick or {}).get("strategy_picks") or []
        native_names = [p.get("strategy_name") for p in native_payload]
        agreement = set(picks) == set(n for n in native_names if n)

        record = ShadowRecord(
            ts_utc=_to_iso(ts),
            candidate_signals_json=json.dumps(candidate_payload, default=str),
            agent_picks_json=json.dumps(agent_pick, default=str),
            native_picks_json=json.dumps(native_payload, default=str),
            agreement_flag=agreement,
            agent_confidence=float((agent_pick or {}).get("confidence") or 0.0),
            agent_reasoning=str((agent_pick or {}).get("reasoning") or ""),
            agent_latency_ms=float(latency_ms or 0.0),
            agent_cost_usd=float(cost_usd) if cost_usd is not None else None,
            fallback_reason=(agent_pick or {}).get("reasoning")
            if (agent_pick or {}).get("fallback")
            else None,
        )
        self._records.append(record)

    # ------------------------------------------------------------------

    @property
    def record_count(self) -> int:
        return len(self._records)

    def flush_to_parquet(self, path: Path | str | None = None) -> Path:
        """Write the buffered records to parquet. Returns the path."""
        out = Path(path) if path else self._out_dir / "mega_vision_shadow.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            pd.DataFrame(columns=[f.name for f in ShadowRecord.__dataclass_fields__.values()]).to_parquet(out)
            return out
        df = pd.DataFrame([asdict(r) for r in self._records])
        df.to_parquet(out, index=False)
        return out

    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_summary(signal: Any) -> Dict[str, Any]:
    return {
        "strategy_name": getattr(signal, "strategy_name", None),
        "direction": getattr(signal, "direction", None),
        "entry_price": getattr(signal, "entry_price", None),
        "stop_loss": getattr(signal, "stop_loss", None),
        "take_profit": getattr(signal, "take_profit", None),
        "confluence_score": getattr(signal, "confluence_score", None),
    }


def _to_iso(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)
