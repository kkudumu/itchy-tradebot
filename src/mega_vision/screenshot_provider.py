"""Bridges the Task 21 ScreenshotCapture into agent inference.

The mega-vision agent wants to pass a chart image as vision content
in its user message. This module handles two things:

  1. ``render_for_decision(ts, candles_window)`` — calls the underlying
     :class:`ScreenshotCapture` with the current bar context.
  2. ``load_image_as_content_block(path)`` — reads a PNG file from
     disk and returns the dict the Agent SDK expects for vision
     input: ``{"type": "image", "source": {"type": "base64",
     "media_type": "image/png", "data": "<base64>"}}``.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ScreenshotProvider:
    """Thin wrapper around :class:`ScreenshotCapture` for agent inference."""

    def __init__(self, screenshot_capture: Any | None = None, run_id: str = "megavision") -> None:
        self._capture = screenshot_capture
        self._run_id = run_id

    def render_for_decision(
        self,
        ts: datetime,
        candles_window: Any,
    ) -> Optional[Path]:
        """Render a chart for the given bar context and return its path.

        Returns ``None`` when no :class:`ScreenshotCapture` was
        provided or when capture is disabled at the config level.
        """
        if self._capture is None:
            return None
        try:
            return self._capture.capture(
                candles=candles_window,
                timestamp=ts,
                strategy_name="megavision",
                event_type="decision",
                metadata={"run_id": self._run_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ScreenshotProvider.render_for_decision failed: %s", exc)
            return None


def load_image_as_content_block(path: str | Path) -> Dict[str, Any]:
    """Read a PNG file and return an Agent SDK vision content block.

    The block shape matches what ``claude-agent-sdk`` expects for
    image input on a user message. Callers embed this dict in the
    content list alongside any text content.
    """
    p = Path(path)
    data = p.read_bytes()
    b64 = base64.standard_b64encode(data).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        },
    }
