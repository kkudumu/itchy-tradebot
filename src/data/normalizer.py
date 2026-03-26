"""
Tick data normalisation and 1-minute OHLCV aggregation.

Responsibilities
----------------
- Aggregate raw tick DataFrames into 1-minute OHLCV bars
- Ensure all timestamps carry an explicit UTC timezone
- Detect gaps in the time series and classify their likely cause
- Validate data quality before database ingestion
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Gold price sanity bounds (USD per troy ounce).
# These are intentionally loose to accommodate multi-decade history.
_GOLD_PRICE_MIN = 200.0
_GOLD_PRICE_MAX = 10_000.0

# Minimum volume per 1-minute bar considered valid (in Dukascopy lots).
_MIN_VOLUME = 0.0


class DataNormalizer:
    """
    Stateless collection of normalisation and validation routines.

    All methods accept and return pandas DataFrames; no state is mutated
    between calls.
    """

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def ticks_to_1m_ohlcv(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate a tick DataFrame into 1-minute OHLCV bars.

        Mid prices ((bid + ask) / 2) are used for OHLCV construction so that
        the bars represent fair value rather than a single side of the spread.
        Volume is the sum of ask_vol and bid_vol across all ticks in the minute.

        Parameters
        ----------
        ticks : pd.DataFrame
            Columns: timestamp (UTC), bid, ask, bid_vol, ask_vol.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, open, high, low, close, volume.
            timestamp is the start (floor) of each 1-minute bucket, UTC.
            Empty DataFrame when *ticks* is empty.
        """
        if ticks.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        ticks = ticks.copy()
        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
        ticks = ticks.sort_values("timestamp").reset_index(drop=True)

        # Mid price for OHLCV; fall back to bid if ask is absent
        if "ask" in ticks.columns and "bid" in ticks.columns:
            ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2.0
        elif "bid" in ticks.columns:
            ticks["mid"] = ticks["bid"]
        else:
            raise ValueError("ticks DataFrame must contain at least a 'bid' column")

        # Volume
        vol_cols = [c for c in ("bid_vol", "ask_vol") if c in ticks.columns]
        ticks["total_vol"] = ticks[vol_cols].sum(axis=1) if vol_cols else 0.0

        # Floor timestamp to the minute bucket
        ticks["minute"] = ticks["timestamp"].dt.floor("min")

        grouped = ticks.groupby("minute", sort=True)
        bars = grouped["mid"].ohlc()
        bars["volume"] = grouped["total_vol"].sum()
        bars.index.name = "timestamp"
        bars = bars.reset_index()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)

        return bars[["timestamp", "open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Timestamp normalisation
    # ------------------------------------------------------------------

    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the *timestamp* column is timezone-aware UTC.

        Timezone-naive timestamps are assumed to already be UTC and are
        localised (not converted).  Any other timezone is converted to UTC.

        Returns a copy — the original DataFrame is not modified.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])

        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")

        df["timestamp"] = ts
        return df

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_gaps(
        self,
        df: pd.DataFrame,
        expected_freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Identify gaps in the 1-minute OHLCV time series.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a *timestamp* column (UTC, sorted ascending).
        expected_freq : str
            Pandas offset alias for the expected bar frequency (default "1min").

        Returns
        -------
        pd.DataFrame
            One row per gap with columns:
            - gap_start   : last timestamp before the gap
            - gap_end     : first timestamp after the gap
            - duration    : timedelta length of the gap
            - missing_bars: number of missing 1-minute bars
            - reason      : "weekend" | "daily_break" | "holiday" | "unknown"
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame(
                columns=["gap_start", "gap_end", "duration", "missing_bars", "reason"]
            )

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        freq = pd.tseries.frequencies.to_offset(expected_freq)
        one_bar = pd.Timedelta(freq)

        gaps: list[dict] = []
        timestamps = df["timestamp"]

        for i in range(1, len(timestamps)):
            prev = timestamps.iloc[i - 1]
            curr = timestamps.iloc[i]
            delta = curr - prev

            if delta <= one_bar:
                continue  # consecutive bars — no gap

            missing = int(round(delta / one_bar)) - 1
            reason = _classify_gap(prev, curr, delta)
            gaps.append(
                {
                    "gap_start":    prev,
                    "gap_end":      curr,
                    "duration":     delta,
                    "missing_bars": missing,
                    "reason":       reason,
                }
            )
            if reason == "unknown":
                logger.warning(
                    "Unexpected gap: %s → %s  (%d missing bars)",
                    prev.isoformat(),
                    curr.isoformat(),
                    missing,
                )
            else:
                logger.debug(
                    "Gap [%s]: %s → %s  (%d missing bars)",
                    reason,
                    prev.isoformat(),
                    curr.isoformat(),
                    missing,
                )

        if not gaps:
            return pd.DataFrame(
                columns=["gap_start", "gap_end", "duration", "missing_bars", "reason"]
            )

        result = pd.DataFrame(gaps)
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> list[str]:
        """
        Validate data quality of a 1-minute OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Columns: timestamp, open, high, low, close, volume.

        Returns
        -------
        list[str]
            List of human-readable error/warning strings.  Empty list means
            the data passed all checks.
        """
        errors: list[str] = []
        if df.empty:
            errors.append("DataFrame is empty.")
            return errors

        now_utc = datetime.now(tz=timezone.utc)

        # --- Timestamp checks ---
        ts = pd.to_datetime(df["timestamp"], utc=True)

        future_mask = ts > now_utc
        if future_mask.any():
            count = int(future_mask.sum())
            errors.append(f"{count} future timestamp(s) detected (max: {ts[future_mask].max()}).")

        if ts.duplicated().any():
            count = int(ts.duplicated().sum())
            errors.append(f"{count} duplicate timestamp(s) found.")

        # --- Price checks ---
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'.")
                continue
            series = df[col].astype(float)
            neg = (series <= 0).sum()
            if neg:
                errors.append(f"{neg} non-positive value(s) in '{col}'.")
            out_of_range = ((series < _GOLD_PRICE_MIN) | (series > _GOLD_PRICE_MAX)).sum()
            if out_of_range:
                errors.append(
                    f"{out_of_range} price(s) in '{col}' outside expected gold range "
                    f"[{_GOLD_PRICE_MIN}, {_GOLD_PRICE_MAX}]."
                )

        # --- OHLC consistency ---
        if all(c in df.columns for c in ("open", "high", "low", "close")):
            o = df["open"].astype(float)
            h = df["high"].astype(float)
            lo = df["low"].astype(float)
            c = df["close"].astype(float)

            high_violated = (h < o.combine(c, max)).sum()
            if high_violated:
                errors.append(
                    f"{high_violated} bar(s) where high < max(open, close)."
                )

            low_violated = (lo > o.combine(c, min)).sum()
            if low_violated:
                errors.append(
                    f"{low_violated} bar(s) where low > min(open, close)."
                )

        # --- Volume checks ---
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            zero_vol = (vol <= _MIN_VOLUME).sum()
            if zero_vol:
                errors.append(f"{zero_vol} bar(s) with zero or negative volume.")

        return errors


# =============================================================================
# Private helpers
# =============================================================================

def _classify_gap(
    prev: pd.Timestamp,
    curr: pd.Timestamp,
    delta: pd.Timedelta,
) -> str:
    """
    Heuristically classify a time-series gap by its cause.

    Rules
    -----
    weekend
        Gap spans Saturday UTC.  Gold closes Friday ~21:00 UTC and reopens
        Sunday ~21:00 UTC — total pause ~48 hours.

    daily_break
        Gap is < 4 hours and falls around the 21:00–22:00 UTC daily server
        maintenance window.

    holiday
        Gap is 20–30 hours and occurs on a known holiday date or is preceded by
        late December / early January.

    unknown
        Does not match any of the above patterns.
    """
    hours = delta.total_seconds() / 3600

    # Weekend: gap spans Saturday (weekday 5)
    # The end of the gap lands on Saturday or Sunday, or the gap is ~48 h
    if prev.weekday() == 4 and hours >= 44:  # Friday → Sunday
        return "weekend"
    if 44 <= hours <= 52:
        return "weekend"

    # Daily break: short gap around 21:00–22:00 UTC
    if hours < 4 and 20 <= prev.hour <= 22:
        return "daily_break"

    # Holiday heuristic: gap of 20–30 hours not on a weekend
    if 20 <= hours <= 30:
        # Christmas / New Year region
        month = prev.month
        day = prev.day
        if month == 12 and day >= 24:
            return "holiday"
        if month == 1 and day <= 2:
            return "holiday"
        return "holiday"  # other single-day closures (e.g. Good Friday)

    return "unknown"
