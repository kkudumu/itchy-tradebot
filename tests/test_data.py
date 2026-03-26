"""
Unit tests for the historical data pipeline.

All tests run without network access or a live database — mocks and in-memory
fixtures are used exclusively.

Test groups
-----------
TestBi5Decoding         — _decode_bi5 with a hand-crafted binary fixture
TestDownloaderHour      — download_hour mocking HTTP layer
TestTickTo1mAggregation — DataNormalizer.ticks_to_1m_ohlcv
TestGapDetection        — DataNormalizer.detect_gaps
TestValidation          — DataNormalizer.validate
TestTimestampNorm       — DataNormalizer.normalize_timestamps
TestBulkInsertSQL       — TimescaleLoader.bulk_insert (mocked psycopg2)
"""

from __future__ import annotations

import io
import lzma
import struct
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers to build .bi5 fixtures
# ---------------------------------------------------------------------------

_TICK_STRUCT = struct.Struct(">IIIff")  # ms_offset, ask, bid, ask_vol, bid_vol


def _build_bi5(ticks: list[tuple]) -> bytes:
    """
    Build a synthetic .bi5 byte string from a list of raw tick tuples.

    Each tuple: (ms_offset: int, ask_pippette: int, bid_pippette: int,
                 ask_vol: float, bid_vol: float)
    """
    raw = b"".join(_TICK_STRUCT.pack(*t) for t in ticks)
    return lzma.compress(raw)


def _utc(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. .bi5 Decoding
# ---------------------------------------------------------------------------

class TestBi5Decoding(unittest.TestCase):
    """Verify that _decode_bi5 correctly unpacks the binary format."""

    def setUp(self):
        from src.data.downloader import DukascopyDownloader
        self.dl = DukascopyDownloader(instrument="XAUUSD", point_divisor=1000.0)
        self.base = _utc(2023, 6, 15, 10)  # 2023-06-15 10:00 UTC

    def test_single_tick_price_conversion(self):
        """Ask/bid pippette values should be divided by point_divisor."""
        # 1 950 000 / 1000.0 = 1950.0 USD/oz
        bi5 = _build_bi5([(500, 1_950_000, 1_949_900, 0.5, 0.5)])
        ticks = self.dl._decode_bi5(bi5, self.base)

        self.assertEqual(len(ticks), 1)
        self.assertAlmostEqual(ticks[0]["ask"], 1950.0, places=6)
        self.assertAlmostEqual(ticks[0]["bid"], 1949.9, places=6)

    def test_timestamp_calculation(self):
        """Tick timestamp = base_timestamp + ms_offset milliseconds."""
        ms_offset = 2_500  # 2.5 seconds
        bi5 = _build_bi5([(ms_offset, 1_900_000, 1_899_000, 1.0, 1.0)])
        ticks = self.dl._decode_bi5(bi5, self.base)

        expected = self.base + timedelta(milliseconds=ms_offset)
        self.assertEqual(ticks[0]["timestamp"], expected)

    def test_multiple_ticks_decoded(self):
        """All tick records in the binary blob are decoded."""
        raw_ticks = [(i * 1000, 1_900_000 + i, 1_899_000 + i, 0.1, 0.1) for i in range(10)]
        bi5 = _build_bi5(raw_ticks)
        ticks = self.dl._decode_bi5(bi5, self.base)

        self.assertEqual(len(ticks), 10)

    def test_volume_fields_preserved(self):
        """ask_vol and bid_vol are parsed as floats."""
        bi5 = _build_bi5([(0, 1_800_000, 1_799_000, 2.5, 3.75)])
        ticks = self.dl._decode_bi5(bi5, self.base)

        self.assertAlmostEqual(ticks[0]["ask_vol"], 2.5, places=4)
        self.assertAlmostEqual(ticks[0]["bid_vol"], 3.75, places=4)

    def test_empty_data_returns_empty_list(self):
        """LZMA-compressed empty payload yields no tick records."""
        bi5 = lzma.compress(b"")
        ticks = self.dl._decode_bi5(bi5, self.base)
        self.assertEqual(ticks, [])

    def test_invalid_lzma_returns_empty_list(self):
        """Corrupt binary data must not raise — returns empty list."""
        ticks = self.dl._decode_bi5(b"\x00\x01\x02\x03", self.base)
        self.assertEqual(ticks, [])

    def test_timestamp_is_utc_aware(self):
        """Every decoded tick carries a UTC-aware timestamp."""
        bi5 = _build_bi5([(100, 2_000_000, 1_999_000, 1.0, 1.0)])
        ticks = self.dl._decode_bi5(bi5, self.base)
        self.assertIsNotNone(ticks[0]["timestamp"].tzinfo)
        self.assertEqual(ticks[0]["timestamp"].tzinfo, timezone.utc)


# ---------------------------------------------------------------------------
# 2. download_hour (HTTP mocked)
# ---------------------------------------------------------------------------

class TestDownloaderHour(unittest.TestCase):
    """Test download_hour behaviour against a mocked requests session."""

    def _make_downloader(self):
        from src.data.downloader import DukascopyDownloader
        dl = DukascopyDownloader()
        return dl

    def test_successful_hour_returns_dataframe(self):
        """A valid bi5 response is decoded into a non-empty DataFrame."""
        raw_ticks = [(1000 * i, 1_900_000, 1_899_500, 0.5, 0.5) for i in range(5)]
        bi5 = _build_bi5(raw_ticks)

        dl = self._make_downloader()
        resp = MagicMock()
        resp.status_code = 200
        resp.content = bi5
        resp.raise_for_status = MagicMock()
        dl._session.get = MagicMock(return_value=resp)

        result = dl.download_hour(2023, 6, 15, 10)
        self.assertFalse(result.empty)
        self.assertIn("timestamp", result.columns)
        self.assertEqual(len(result), 5)

    def test_404_returns_empty_dataframe(self):
        """A 404 response (hour has no data) produces an empty DataFrame."""
        dl = self._make_downloader()
        resp = MagicMock()
        resp.status_code = 404
        dl._session.get = MagicMock(return_value=resp)

        result = dl.download_hour(2023, 1, 1, 3)
        self.assertTrue(result.empty)

    def test_url_uses_zero_indexed_month(self):
        """Dukascopy URL month is 0-indexed (January = 00)."""
        dl = self._make_downloader()
        url = dl._build_url(2023, 1, 15, 10)  # January = month 1
        self.assertIn("/00/", url)  # 0-indexed → 00

        url_dec = dl._build_url(2023, 12, 25, 0)  # December
        self.assertIn("/11/", url_dec)  # 11 in URL

    def test_timestamps_are_utc(self):
        """Every tick in the returned DataFrame has UTC timezone."""
        bi5 = _build_bi5([(500, 1_950_000, 1_949_000, 1.0, 1.0)])
        dl = self._make_downloader()
        resp = MagicMock(status_code=200, content=bi5)
        resp.raise_for_status = MagicMock()
        dl._session.get = MagicMock(return_value=resp)

        df = dl.download_hour(2023, 6, 15, 14)
        self.assertEqual(df["timestamp"].dt.tz, timezone.utc)


# ---------------------------------------------------------------------------
# 3. Tick → 1-minute OHLCV
# ---------------------------------------------------------------------------

class TestTickTo1mAggregation(unittest.TestCase):
    """DataNormalizer.ticks_to_1m_ohlcv correctness."""

    def setUp(self):
        from src.data.normalizer import DataNormalizer
        self.n = DataNormalizer()

    def _make_ticks(self, minute_prices: dict[int, list[float]]) -> pd.DataFrame:
        """
        Build a synthetic tick DataFrame.

        minute_prices : {minute_offset: [price1, price2, ...]}
        All timestamps are anchored to 2023-01-02 10:00 UTC.
        """
        base = _utc(2023, 1, 2, 10)
        rows = []
        for m, prices in sorted(minute_prices.items()):
            for i, p in enumerate(prices):
                rows.append(
                    {
                        "timestamp": base + timedelta(minutes=m, seconds=i),
                        "bid": p,
                        "ask": p + 0.1,
                        "bid_vol": 1.0,
                        "ask_vol": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_ohlc_within_single_minute(self):
        """OHLCV values are correctly computed for one bar."""
        ticks = self._make_ticks({0: [100.0, 110.0, 90.0, 105.0]})
        bars = self.n.ticks_to_1m_ohlcv(ticks)

        self.assertEqual(len(bars), 1)
        row = bars.iloc[0]
        mid = lambda p: p + 0.05  # mid = (bid + ask) / 2

        self.assertAlmostEqual(row["open"],  mid(100.0), places=4)
        self.assertAlmostEqual(row["high"],  mid(110.0), places=4)
        self.assertAlmostEqual(row["low"],   mid(90.0),  places=4)
        self.assertAlmostEqual(row["close"], mid(105.0), places=4)

    def test_volume_sums_bid_and_ask(self):
        """Volume = sum of bid_vol + ask_vol across all ticks in the minute."""
        ticks = self._make_ticks({0: [1900.0, 1901.0]})
        bars = self.n.ticks_to_1m_ohlcv(ticks)
        # 2 ticks × (1.0 bid_vol + 1.0 ask_vol) = 4.0
        self.assertAlmostEqual(bars.iloc[0]["volume"], 4.0, places=6)

    def test_multiple_minutes_produce_correct_bar_count(self):
        """Each minute with ticks produces exactly one bar."""
        ticks = self._make_ticks({0: [1900.0], 1: [1901.0], 5: [1902.0]})
        bars = self.n.ticks_to_1m_ohlcv(ticks)
        self.assertEqual(len(bars), 3)

    def test_empty_ticks_returns_empty_df(self):
        """Empty input produces an empty DataFrame with the correct columns."""
        bars = self.n.ticks_to_1m_ohlcv(pd.DataFrame())
        self.assertTrue(bars.empty)
        self.assertIn("open", bars.columns)

    def test_output_timestamps_are_utc(self):
        """Output timestamp column is UTC-aware."""
        ticks = self._make_ticks({0: [1950.0]})
        bars = self.n.ticks_to_1m_ohlcv(ticks)
        self.assertEqual(bars["timestamp"].dt.tz, timezone.utc)

    def test_bar_timestamp_is_minute_floor(self):
        """Bar timestamp is floored to the minute boundary."""
        ticks = self._make_ticks({0: [1900.0]})
        bars = self.n.ticks_to_1m_ohlcv(ticks)
        ts = bars.iloc[0]["timestamp"]
        self.assertEqual(ts.second, 0)
        self.assertEqual(ts.microsecond, 0)


# ---------------------------------------------------------------------------
# 4. Gap detection
# ---------------------------------------------------------------------------

class TestGapDetection(unittest.TestCase):
    """DataNormalizer.detect_gaps classification and counting."""

    def setUp(self):
        from src.data.normalizer import DataNormalizer
        self.n = DataNormalizer()

    def _bars(self, timestamps: list[datetime]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open":  [1900.0] * len(timestamps),
                "high":  [1910.0] * len(timestamps),
                "low":   [1890.0] * len(timestamps),
                "close": [1905.0] * len(timestamps),
                "volume":[1.0]   * len(timestamps),
            }
        )

    def test_no_gap_in_consecutive_bars(self):
        """Perfectly consecutive minutes have no gaps."""
        base = _utc(2023, 6, 15, 10)
        ts = [base + timedelta(minutes=i) for i in range(60)]
        gaps = self.n.detect_gaps(self._bars(ts))
        self.assertTrue(gaps.empty)

    def test_weekend_gap_detected(self):
        """~48-hour gap spanning Saturday is classified as 'weekend'."""
        # Friday 21:00 UTC → Sunday 22:00 UTC  (~49 h gap)
        friday = _utc(2023, 6, 16, 21, 0)   # Friday
        sunday = _utc(2023, 6, 18, 22, 0)   # Sunday
        gaps = self.n.detect_gaps(self._bars([friday, sunday]))

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0]["reason"], "weekend")

    def test_daily_break_detected(self):
        """Short gap around 21:00 UTC classified as 'daily_break'."""
        bar1 = _utc(2023, 6, 15, 21, 0)
        bar2 = _utc(2023, 6, 15, 22, 30)  # 90-minute gap
        gaps = self.n.detect_gaps(self._bars([bar1, bar2]))

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0]["reason"], "daily_break")

    def test_holiday_gap_detected(self):
        """~24-hour gap on Christmas Eve is classified as 'holiday'."""
        bar1 = _utc(2023, 12, 24, 20, 0)
        bar2 = _utc(2023, 12, 25, 22, 0)  # ~26-hour gap
        gaps = self.n.detect_gaps(self._bars([bar1, bar2]))

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0]["reason"], "holiday")

    def test_missing_bars_count(self):
        """The missing_bars field correctly counts absent 1-minute bars."""
        base = _utc(2023, 6, 15, 12)
        # 10-minute gap → 9 missing bars
        ts = [base, base + timedelta(minutes=10)]
        gaps = self.n.detect_gaps(self._bars(ts))

        self.assertEqual(gaps.iloc[0]["missing_bars"], 9)

    def test_empty_df_returns_empty_gaps(self):
        """Empty input produces empty gap DataFrame."""
        gaps = self.n.detect_gaps(pd.DataFrame(columns=["timestamp"]))
        self.assertTrue(gaps.empty)

    def test_single_bar_returns_empty_gaps(self):
        """A single bar cannot form a gap."""
        gaps = self.n.detect_gaps(self._bars([_utc(2023, 1, 1, 10)]))
        self.assertTrue(gaps.empty)


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    """DataNormalizer.validate catches known bad data patterns."""

    def setUp(self):
        from src.data.normalizer import DataNormalizer
        self.n = DataNormalizer()

    def _good_bar(self, ts: datetime | None = None) -> dict:
        return {
            "timestamp": ts or _utc(2023, 6, 15, 10, 0),
            "open":  1900.0,
            "high":  1910.0,
            "low":   1890.0,
            "close": 1905.0,
            "volume": 2.0,
        }

    def test_valid_data_passes(self):
        """Well-formed data produces zero errors."""
        df = pd.DataFrame([self._good_bar()])
        errors = self.n.validate(df)
        self.assertEqual(errors, [])

    def test_future_timestamp_flagged(self):
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([self._good_bar(future)])
        errors = self.n.validate(df)
        self.assertTrue(any("future" in e.lower() for e in errors))

    def test_negative_price_flagged(self):
        row = self._good_bar()
        row["close"] = -1.0
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("close" in e.lower() for e in errors))

    def test_zero_price_flagged(self):
        row = self._good_bar()
        row["high"] = 0.0
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("high" in e.lower() for e in errors))

    def test_price_out_of_range_flagged(self):
        """A price outside [200, 10000] is flagged as out of gold range."""
        row = self._good_bar()
        row["open"] = 50.0  # absurdly low
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("range" in e.lower() for e in errors))

    def test_high_below_open_flagged(self):
        """High must be >= max(open, close)."""
        row = self._good_bar()
        row["high"] = 1800.0  # below open=1900
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("high" in e.lower() for e in errors))

    def test_low_above_close_flagged(self):
        """Low must be <= min(open, close)."""
        row = self._good_bar()
        row["low"] = 2000.0  # above close=1905
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("low" in e.lower() for e in errors))

    def test_zero_volume_flagged(self):
        row = self._good_bar()
        row["volume"] = 0.0
        df = pd.DataFrame([row])
        errors = self.n.validate(df)
        self.assertTrue(any("volume" in e.lower() for e in errors))

    def test_duplicate_timestamp_flagged(self):
        ts = _utc(2023, 6, 15, 10, 0)
        df = pd.DataFrame([self._good_bar(ts), self._good_bar(ts)])
        errors = self.n.validate(df)
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_empty_df_flagged(self):
        errors = self.n.validate(pd.DataFrame())
        self.assertTrue(any("empty" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# 6. Timestamp normalisation
# ---------------------------------------------------------------------------

class TestTimestampNorm(unittest.TestCase):
    """DataNormalizer.normalize_timestamps handles naive and aware inputs."""

    def setUp(self):
        from src.data.normalizer import DataNormalizer
        self.n = DataNormalizer()

    def test_naive_timestamps_get_utc(self):
        """Timezone-naive timestamps are localised to UTC."""
        df = pd.DataFrame({"timestamp": [datetime(2023, 6, 15, 10, 0)]})
        result = self.n.normalize_timestamps(df)
        self.assertEqual(result["timestamp"].dt.tz, timezone.utc)

    def test_utc_timestamps_unchanged(self):
        """Already-UTC timestamps pass through without modification."""
        ts = datetime(2023, 6, 15, 10, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({"timestamp": [ts]})
        result = self.n.normalize_timestamps(df)
        self.assertEqual(result["timestamp"].dt.tz, timezone.utc)

    def test_non_utc_timezone_converted(self):
        """Non-UTC aware timestamps are converted to UTC."""
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo("America/New_York")
        # 06:00 EDT (UTC-4 in summer) = 10:00 UTC
        ts = datetime(2023, 6, 15, 6, 0, tzinfo=eastern)
        # Build a Series manually so pandas infers tz from the dtype cleanly
        ts_series = pd.Series([ts]).dt.tz_convert("UTC")
        df = pd.DataFrame({"timestamp": ts_series})
        result = self.n.normalize_timestamps(df)
        self.assertEqual(result["timestamp"].dt.tz, timezone.utc)
        self.assertEqual(result["timestamp"].iloc[0].hour, 10)

    def test_empty_df_passes_through(self):
        """Empty DataFrame is returned without error."""
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]")})
        result = self.n.normalize_timestamps(df)
        self.assertTrue(result.empty)


# ---------------------------------------------------------------------------
# 7. Bulk insert (mocked psycopg2)
# ---------------------------------------------------------------------------

class TestBulkInsertSQL(unittest.TestCase):
    """
    TimescaleLoader.bulk_insert calls execute_values with the correct SQL
    and tuple structure without touching a real database.
    """

    def _make_loader(self):
        from src.data.loader import TimescaleLoader
        return TimescaleLoader(
            {"host": "localhost", "port": 5432, "dbname": "test", "user": "u", "password": ""}
        )

    def _make_bars(self, n: int = 5) -> pd.DataFrame:
        base = _utc(2023, 6, 15, 10)
        return pd.DataFrame(
            {
                "timestamp": [base + timedelta(minutes=i) for i in range(n)],
                "open":  [1900.0 + i for i in range(n)],
                "high":  [1910.0 + i for i in range(n)],
                "low":   [1890.0 + i for i in range(n)],
                "close": [1905.0 + i for i in range(n)],
                "volume":[1.0]   * n,
            }
        )

    def test_execute_values_called(self):
        """execute_values is invoked at least once for a non-empty DataFrame."""
        loader = self._make_loader()

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 5
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.data.loader.psycopg2.connect", return_value=mock_conn), \
             patch("src.data.loader.execute_values") as mock_ev:
            mock_ev.return_value = None
            loader.bulk_insert(self._make_bars(), instrument="XAUUSD")
            self.assertTrue(mock_ev.called)

    def test_sql_contains_on_conflict(self):
        """The INSERT statement must include ON CONFLICT DO NOTHING."""
        loader = self._make_loader()
        captured_sql: list[str] = []

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 3
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        def capture_ev(cur, sql, rows, **kwargs):
            captured_sql.append(sql)

        with patch("src.data.loader.psycopg2.connect", return_value=mock_conn), \
             patch("src.data.loader.execute_values", side_effect=capture_ev):
            loader.bulk_insert(self._make_bars(), instrument="XAUUSD")

        self.assertTrue(any("ON CONFLICT" in sql for sql in captured_sql))

    def test_empty_df_skipped(self):
        """Empty DataFrame returns 0 without calling the database."""
        loader = self._make_loader()
        with patch("src.data.loader.psycopg2.connect") as mock_connect:
            result = loader.bulk_insert(pd.DataFrame(), instrument="XAUUSD")
            mock_connect.assert_not_called()
        self.assertEqual(result, 0)

    def test_tuples_include_instrument(self):
        """Each row tuple passed to execute_values includes the instrument string."""
        loader = self._make_loader()
        captured_rows: list = []

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 2
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        def capture_ev(cur, sql, rows, **kwargs):
            captured_rows.extend(rows)

        with patch("src.data.loader.psycopg2.connect", return_value=mock_conn), \
             patch("src.data.loader.execute_values", side_effect=capture_ev):
            loader.bulk_insert(self._make_bars(2), instrument="XAUUSD")

        for row in captured_rows:
            # tuple order: (timestamp, instrument, open, high, low, close, volume)
            self.assertEqual(row[1], "XAUUSD")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
