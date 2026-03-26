"""Chart screenshot capture at key trade lifecycle events.

Attempts to use mt5.chart_screenshot() when available.  Falls back to
generating a candlestick chart via mplfinance when MT5 screenshot API is
unavailable (common in Docker environments without a running MT5 terminal
with an active GUI).

Screenshots are saved under:
    <save_dir>/<instrument>/<YYYYMMDD>/<phase>_<trade_id>_<timestamp>.png
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Capture MT5 chart screenshots at trade lifecycle events.

    Parameters
    ----------
    bridge:
        Connected :class:`~src.execution.mt5_bridge.MT5Bridge` instance.
    save_dir:
        Root directory for screenshot storage. Created if it does not exist.
    width:
        Screenshot width in pixels (passed to mt5.chart_screenshot). Default: 1280.
    height:
        Screenshot height in pixels. Default: 720.
    """

    VALID_PHASES = frozenset({"pre_entry", "entry", "during", "exit"})

    def __init__(
        self,
        bridge,
        save_dir: str = "screenshots",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.bridge = bridge
        self.save_dir = Path(save_dir)
        self.width = width
        self.height = height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        instrument: str,
        timeframe: str,
        phase: str,
        trade_id: Optional[int] = None,
    ) -> str:
        """Capture a chart screenshot and save it to disk.

        Parameters
        ----------
        instrument:
            Symbol string (e.g. 'XAUUSD').
        timeframe:
            Timeframe label (e.g. '1H', '15M').
        phase:
            One of 'pre_entry', 'entry', 'during', 'exit'.
        trade_id:
            Optional trade identifier included in the filename for traceability.

        Returns
        -------
        Absolute path of the saved screenshot file, or an empty string on failure.
        """
        if phase not in self.VALID_PHASES:
            logger.warning("Unknown screenshot phase '%s' — using as-is", phase)

        file_path = self._build_path(instrument, timeframe, phase, trade_id)
        self._ensure_dir(file_path.parent)

        # Attempt MT5 native screenshot first
        mt5_path = self._try_mt5_screenshot(instrument, timeframe, str(file_path))
        if mt5_path:
            logger.info("Screenshot saved via MT5: %s", mt5_path)
            return mt5_path

        # Fallback: matplotlib chart from live data
        return self._fallback_chart(instrument, timeframe, str(file_path), phase, trade_id)

    # ------------------------------------------------------------------
    # MT5 native screenshot
    # ------------------------------------------------------------------

    def _try_mt5_screenshot(
        self, instrument: str, timeframe: str, file_path: str
    ) -> str:
        """Attempt to capture via mt5.chart_screenshot().

        Returns the file path on success, empty string on failure.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return ""

        # chart_screenshot may not be available in all MT5 Python API versions
        if not hasattr(mt5, "chart_screenshot"):
            logger.debug("mt5.chart_screenshot not available — using matplotlib fallback")
            return ""

        try:
            tf_const = self.bridge.timeframe_constant(timeframe)
            if tf_const is None:
                logger.warning("Unknown timeframe '%s' for screenshot", timeframe)
                return ""

            # MT5 requires the chart to be open in the terminal
            ok = mt5.chart_screenshot(
                symbol=instrument,
                timeframe=tf_const,
                path=file_path,
                width=self.width,
                height=self.height,
            )

            if ok:
                return file_path

            error = mt5.last_error()
            logger.debug("chart_screenshot failed (%s) — falling back to matplotlib", error)
            return ""

        except Exception as exc:  # noqa: BLE001
            logger.debug("chart_screenshot raised %s — falling back to matplotlib", exc)
            return ""

    # ------------------------------------------------------------------
    # Matplotlib fallback
    # ------------------------------------------------------------------

    def _fallback_chart(
        self,
        instrument: str,
        timeframe: str,
        file_path: str,
        phase: str,
        trade_id: Optional[int],
    ) -> str:
        """Generate a candlestick chart using mplfinance and save it.

        Returns the file path on success, empty string on failure.
        """
        try:
            import mplfinance as mpf  # soft dependency
        except ImportError:
            logger.warning(
                "mplfinance is not installed — cannot generate fallback chart. "
                "Install with: pip install mplfinance"
            )
            return ""

        try:
            # Pull the most recent bars from the bridge
            tf_const = self.bridge.timeframe_constant(timeframe)
            if tf_const is None:
                logger.warning("Unknown timeframe '%s' — cannot generate chart", timeframe)
                return ""

            ohlcv = self.bridge.get_rates(instrument, tf_const, count=100)
            if ohlcv.empty:
                logger.warning("No OHLCV data returned for %s %s chart", instrument, timeframe)
                return ""

            chart_path = self._generate_chart(
                ohlcv=ohlcv,
                instrument=instrument,
                timeframe=timeframe,
                file_path=file_path,
                phase=phase,
                trade_id=trade_id,
                mpf=mpf,
            )
            return chart_path

        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallback chart generation failed: %s", exc)
            return ""

    def _generate_chart(
        self,
        ohlcv: pd.DataFrame,
        instrument: str,
        timeframe: str,
        file_path: str,
        phase: str,
        trade_id: Optional[int],
        markers: Optional[List[dict]] = None,
        mpf=None,
    ) -> str:
        """Render OHLCV data as a candlestick chart and save to disk.

        Parameters
        ----------
        ohlcv:
            DataFrame with columns: time, open, high, low, close, tick_volume.
        markers:
            Optional list of price-level markers to annotate on the chart.
            Each dict may contain keys: price, label, color.
        mpf:
            Pre-imported mplfinance module (avoids repeated imports in tests).

        Returns
        -------
        Absolute file path of the saved PNG.
        """
        if mpf is None:
            import mplfinance as mpf  # noqa: PLC0415

        # mplfinance expects a DatetimeIndex
        df = ohlcv.copy()
        if "time" in df.columns:
            df = df.set_index("time")
        df.index = pd.DatetimeIndex(df.index)

        # Rename tick_volume → volume if present
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})

        title = f"{instrument} {timeframe} — {phase}"
        if trade_id is not None:
            title += f" (trade #{trade_id})"

        addplots = []
        if markers:
            for marker in markers:
                price = marker.get("price")
                color = marker.get("color", "blue")
                if price is not None:
                    line = [price] * len(df)
                    addplots.append(mpf.make_addplot(line, color=color, linestyle="--"))

        kwargs = dict(
            type="candle",
            style="charles",
            title=title,
            volume=("volume" in df.columns),
            savefig=file_path,
            figsize=(self.width / 100, self.height / 100),
        )
        if addplots:
            kwargs["addplot"] = addplots

        mpf.plot(df, **kwargs)

        logger.info("Fallback chart saved: %s", file_path)
        return file_path

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _build_path(
        self,
        instrument: str,
        timeframe: str,
        phase: str,
        trade_id: Optional[int],
    ) -> Path:
        """Construct the output file path for a screenshot."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        ts_str = now.strftime("%H%M%S")

        tid_part = f"_{trade_id}" if trade_id is not None else ""
        filename = f"{phase}{tid_part}_{ts_str}.png"

        return self.save_dir / instrument / date_str / filename

    @staticmethod
    def _ensure_dir(directory: Path) -> None:
        """Create a directory (and parents) if it does not already exist."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create screenshot directory %s: %s", directory, exc)
