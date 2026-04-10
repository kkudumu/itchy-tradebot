"""Chart screenshot capture for mega-vision training data (plan Task 21).

Renders a matplotlib candlestick chart + metadata JSON sidecar for
every signal generation event when screenshot capture is enabled.
The paired PNG + JSON is the training-data shape the mega-vision
pipeline consumes (Task 27).

Configuration (from run config):
  screenshot_capture:
    enabled: false        # default off — heavy
    every_signal: true
    every_n_bars: 0       # when non-zero, also capture every N bars
    out_dir: "screenshots/<run_id>"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotConfig:
    enabled: bool = False
    every_signal: bool = True
    every_n_bars: int = 0
    out_dir: str = "screenshots"

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ScreenshotConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            every_signal=bool(data.get("every_signal", True)),
            every_n_bars=int(data.get("every_n_bars", 0) or 0),
            out_dir=str(data.get("out_dir") or "screenshots"),
        )


class ScreenshotCapture:
    """Renders OHLCV windows to PNG + JSON sidecar.

    Uses ``mplfinance`` for the chart render. The class is a no-op
    when ``config.enabled`` is ``False`` so callers can construct it
    unconditionally without paying the matplotlib import cost.
    """

    def __init__(self, config: ScreenshotConfig, run_id: str = "run") -> None:
        self._cfg = config
        self._run_id = run_id
        self._out_dir: Optional[Path] = None
        self._bars_since_last_capture: int = 0

    def capture(
        self,
        candles: Any,  # pd.DataFrame with OHLCV columns
        timestamp: datetime,
        strategy_name: str,
        event_type: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Path | None:
        """Render a chart + sidecar JSON for the given candles window.

        Returns the PNG path on success, or ``None`` when capture is
        disabled / the event doesn't match the capture policy. Errors
        are logged at WARNING level and swallowed — a screenshot
        failure should never crash the backtest.
        """
        if not self._cfg.enabled:
            return None

        if event_type == "signal_generated" and not self._cfg.every_signal:
            return None

        try:
            out_dir = self._ensure_out_dir()
            filename = self._filename(timestamp, strategy_name, event_type)
            png_path = out_dir / f"{filename}.png"
            json_path = out_dir / f"{filename}.json"

            # mplfinance import is deferred because it's heavy and
            # most runs won't have capture enabled.
            import mplfinance as mpf  # type: ignore[import-untyped]

            # Tail the candles to a reasonable window
            window = candles.tail(100) if hasattr(candles, "tail") else candles

            mpf.plot(
                window,
                type="candle",
                style="nightclouds",
                savefig=dict(fname=str(png_path), dpi=80),
            )

            sidecar = {
                "run_id": self._run_id,
                "timestamp_utc": timestamp.isoformat()
                if hasattr(timestamp, "isoformat")
                else str(timestamp),
                "strategy_name": strategy_name,
                "event_type": event_type,
                "window_bars": len(window) if hasattr(window, "__len__") else 0,
                **(metadata or {}),
            }
            json_path.write_text(
                json.dumps(sidecar, indent=2, default=str),
                encoding="utf-8",
            )
            return png_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screenshot capture failed: %s", exc)
            return None

    def on_bar_end(self) -> bool:
        """Increment the per-bar counter; return True when the every_n_bars
        cadence should trigger a capture."""
        if not self._cfg.enabled or self._cfg.every_n_bars <= 0:
            return False
        self._bars_since_last_capture += 1
        if self._bars_since_last_capture >= self._cfg.every_n_bars:
            self._bars_since_last_capture = 0
            return True
        return False

    # ------------------------------------------------------------------

    def _ensure_out_dir(self) -> Path:
        if self._out_dir is None:
            # Always scope under run_id so concurrent runs don't
            # clobber each other's screenshots.
            self._out_dir = Path(self._cfg.out_dir) / self._run_id
            self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir

    def _filename(
        self,
        timestamp: datetime,
        strategy_name: str,
        event_type: str,
    ) -> str:
        ts = (
            timestamp.strftime("%Y%m%dT%H%M%S")
            if hasattr(timestamp, "strftime")
            else str(timestamp).replace(":", "").replace(" ", "_")
        )
        return f"{strategy_name}_{ts}_{event_type}"
