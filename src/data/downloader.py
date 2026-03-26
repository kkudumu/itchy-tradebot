"""
Dukascopy .bi5 tick data downloader for historical OHLCV construction.

Binary format
-------------
Each .bi5 file is LZMA-compressed and contains a sequence of 20-byte tick records:

    Offset  Size  Type    Field
    ------  ----  ------  -------------------------------------------
       0       4  uint32  ms offset from the start of the hour (big-endian)
       4       4  uint32  ask price in pippettes (big-endian)
       8       4  uint32  bid price in pippettes (big-endian)
      12       4  float   ask volume (big-endian, IEEE 754)
      16       4  float   bid volume (big-endian, IEEE 754)

URL pattern
-----------
https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5

Note: months in the URL are 0-indexed (January = 00, December = 11).

Price conversion
----------------
Dukascopy stores prices as integer "pippettes".  For gold (XAUUSD) divide by
1000.0 to obtain USD/oz.  For standard forex pairs divide by 100000.0.
"""

from __future__ import annotations

import io
import logging
import lzma
import struct
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Struct pattern for one 20-byte tick record (big-endian):
#   I = uint32 (ms offset), I = uint32 (ask), I = uint32 (bid),
#   f = float32 (ask vol), f = float32 (bid vol)
_TICK_STRUCT = struct.Struct(">IIIff")
_TICK_SIZE = _TICK_STRUCT.size  # 20 bytes


def _build_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    timeout: int = 30,
) -> requests.Session:
    """Return a requests Session with automatic retry and exponential back-off."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; history-pipeline/1.0)"})
    return session


class DukascopyDownloader:
    """
    Download and decode historical tick data from Dukascopy's public data feed.

    Parameters
    ----------
    instrument : str
        Dukascopy symbol string, e.g. "XAUUSD".
    point_divisor : float
        Divisor applied to raw integer prices.  Use 1000.0 for gold, 100000.0
        for standard forex pairs.
    rate_limit_secs : float
        Minimum pause between consecutive HTTP requests (default: 0.05 s).
    request_timeout : int
        HTTP request timeout in seconds (default: 30).
    max_retries : int
        Maximum number of retry attempts per request (default: 5).
    """

    BASE_URL = "https://datafeed.dukascopy.com/datafeed"

    def __init__(
        self,
        instrument: str = "XAUUSD",
        point_divisor: float = 1000.0,
        rate_limit_secs: float = 0.05,
        request_timeout: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.instrument = instrument.upper()
        self.point_divisor = point_divisor
        self.rate_limit_secs = rate_limit_secs
        self.request_timeout = request_timeout
        self._session = _build_session(retries=max_retries)
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_hour(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
    ) -> pd.DataFrame:
        """
        Download and decode one hour of tick data.

        Parameters
        ----------
        year, month, day, hour : int
            Calendar values in UTC.  month is 1-indexed (January = 1).

        Returns
        -------
        pd.DataFrame
            Columns: timestamp (DatetimeTZDtype UTC), bid, ask, bid_vol, ask_vol.
            Empty DataFrame when the hour has no data (market closed, holiday).
        """
        url = self._build_url(year, month, day, hour)
        raw = self._fetch(url)
        if raw is None:
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])

        base_ts = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
        ticks = self._decode_bi5(raw, base_ts)
        if not ticks:
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])

        df = pd.DataFrame(ticks)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def download_range(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Download tick data for a date range.

        Weekends (Saturday UTC and most of Sunday UTC) are skipped
        automatically because the gold market is closed.

        Parameters
        ----------
        start_date : datetime
            Inclusive start (UTC).  Timezone-naive is treated as UTC.
        end_date : datetime
            Exclusive end (UTC).  Timezone-naive is treated as UTC.
        progress_callback : callable, optional
            Called as ``progress_callback(hours_done, total_hours)`` after
            each hour is processed.

        Returns
        -------
        pd.DataFrame
            Combined tick data sorted by timestamp.
        """
        start_date = _ensure_utc(start_date)
        end_date = _ensure_utc(end_date)

        # Build list of all UTC hours in range
        hours = _generate_market_hours(start_date, end_date)
        total = len(hours)
        logger.info(
            "Downloading %s %s → %s (%d market hours)",
            self.instrument,
            start_date.date(),
            end_date.date(),
            total,
        )

        frames: list[pd.DataFrame] = []
        for idx, (y, mo, d, h) in enumerate(hours, start=1):
            df = self.download_hour(y, mo, d, h)
            if not df.empty:
                frames.append(df)
            if progress_callback:
                progress_callback(idx, total)

        if not frames:
            logger.warning("No tick data retrieved for the requested range.")
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])

        result = pd.concat(frames, ignore_index=True)
        result.sort_values("timestamp", inplace=True)
        result.reset_index(drop=True, inplace=True)
        logger.info("Downloaded %d ticks total.", len(result))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, year: int, month: int, day: int, hour: int) -> str:
        """Construct the Dukascopy URL for a single hour file.

        Dukascopy months are 0-indexed in the URL path, so January = 00.
        """
        month_zero = month - 1  # convert to 0-indexed
        return (
            f"{self.BASE_URL}/{self.instrument}"
            f"/{year}/{month_zero:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        )

    def _fetch(self, url: str) -> Optional[bytes]:
        """
        Fetch raw bytes from ``url``.

        Returns None for 404 (hour has no data), raises on other errors after
        the retry budget is exhausted.
        """
        # Honour rate limit
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.rate_limit_secs:
            time.sleep(self.rate_limit_secs - elapsed)

        try:
            resp = self._session.get(url, timeout=self.request_timeout)
            self._last_request_ts = time.monotonic()

            if resp.status_code == 404:
                logger.debug("No data: %s", url)
                return None

            resp.raise_for_status()
            return resp.content

        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            raise

    def _decode_bi5(self, data: bytes, base_timestamp: datetime) -> list[dict]:
        """
        Decode LZMA-compressed .bi5 binary data into a list of tick dicts.

        Each record in the decompressed stream is _TICK_SIZE bytes:
          - uint32 big-endian: millisecond offset from base_timestamp
          - uint32 big-endian: ask in pippettes
          - uint32 big-endian: bid in pippettes
          - float32 big-endian: ask volume
          - float32 big-endian: bid volume

        Prices are divided by self.point_divisor to produce decimal values.
        """
        try:
            decompressed = lzma.decompress(data)
        except lzma.LZMAError as exc:
            logger.error("LZMA decompression failed: %s", exc)
            return []

        n_ticks = len(decompressed) // _TICK_SIZE
        if n_ticks == 0:
            return []

        ticks: list[dict] = []
        base_ms = int(base_timestamp.timestamp() * 1000)

        for i in range(n_ticks):
            offset = i * _TICK_SIZE
            chunk = decompressed[offset : offset + _TICK_SIZE]
            if len(chunk) < _TICK_SIZE:
                break

            ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = _TICK_STRUCT.unpack(chunk)
            abs_ms = base_ms + ms_offset
            ticks.append(
                {
                    "timestamp": datetime.fromtimestamp(abs_ms / 1000.0, tz=timezone.utc),
                    "ask":       ask_raw / self.point_divisor,
                    "bid":       bid_raw / self.point_divisor,
                    "ask_vol":   float(ask_vol),
                    "bid_vol":   float(bid_vol),
                }
            )

        return ticks


# =============================================================================
# Module-level helpers
# =============================================================================

def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone.  Attaches UTC if timezone-naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _generate_market_hours(start: datetime, end: datetime) -> list[tuple[int, int, int, int]]:
    """
    Produce a list of (year, month, day, hour) tuples for all expected market
    hours between *start* (inclusive) and *end* (exclusive).

    Gold (XAUUSD) trades Sunday ~22:00 UTC through Friday ~21:00 UTC.
    Saturday UTC and most of Sunday UTC are excluded.

    The daily maintenance break (typically ~21:00–22:00 UTC) and holidays are
    not pre-filtered here; missing hours are handled by a 404 from the server.
    """
    hours: list[tuple[int, int, int, int]] = []
    current = start.replace(minute=0, second=0, microsecond=0)

    while current < end:
        # weekday(): 5 = Saturday, 6 = Sunday
        wd = current.weekday()
        if wd == 5:
            # Skip all of Saturday
            current += timedelta(hours=1)
            continue
        if wd == 6 and current.hour < 21:
            # Gold reopens Sunday ~21:00 UTC — skip earlier hours
            current += timedelta(hours=1)
            continue

        hours.append((current.year, current.month, current.day, current.hour))
        current += timedelta(hours=1)

    return hours
