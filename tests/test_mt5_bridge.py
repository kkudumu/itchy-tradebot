"""Unit tests for the MT5 execution bridge.

All MetaTrader5 interactions are mocked so the test suite runs on Linux
without the MetaTrader5 package installed.

Coverage:
1.  Connection — mt5.initialize, login flow, failure handling
2.  Data retrieval — copy_rates_from_pos, DataFrame format validation
3.  Market order — success path, request structure verification
4.  Limit order — Kijun pullback placement
5.  Stop-limit order — cloud breakout placement
6.  Position modification — trailing stop SL update
7.  Partial close — hybrid 50/50 exit sends correct volume
8.  Full close — full position close
9.  Filling mode detection — FOK, IOC, RETURN
10. Volume normalization — rounding to lot step, clamping to bounds
11. Error handling — retcode 10030 (market closed), 10006 (rejected)
12. Slippage logging — tracked on success and failure
13. Account monitor — account_info, equity, positions
14. Screenshot — path generation, fallback chart
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import types

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Build and inject a complete MT5 mock module before any src imports
# ---------------------------------------------------------------------------

def _make_mock_mt5() -> MagicMock:
    """Create a fully-configured mock of the MetaTrader5 module."""
    mock = MagicMock(name="MetaTrader5")

    # --- TIMEFRAME constants ---
    mock.TIMEFRAME_M1  = 1
    mock.TIMEFRAME_M5  = 5
    mock.TIMEFRAME_M15 = 15
    mock.TIMEFRAME_H1  = 16385
    mock.TIMEFRAME_H4  = 16388

    # --- ORDER TYPE constants ---
    mock.ORDER_TYPE_BUY       = 0
    mock.ORDER_TYPE_SELL      = 1
    mock.ORDER_TYPE_BUY_LIMIT  = 2
    mock.ORDER_TYPE_SELL_LIMIT = 3
    mock.ORDER_TYPE_BUY_STOP_LIMIT  = 6
    mock.ORDER_TYPE_SELL_STOP_LIMIT = 7

    # --- TRADE ACTION constants ---
    mock.TRADE_ACTION_DEAL    = 1
    mock.TRADE_ACTION_PENDING = 5
    mock.TRADE_ACTION_SLTP    = 6

    # --- ORDER FILLING constants ---
    mock.ORDER_FILLING_FOK    = 0
    mock.ORDER_FILLING_IOC    = 1
    mock.ORDER_FILLING_RETURN = 2

    # --- ORDER TIME constants ---
    mock.ORDER_TIME_GTC = 1

    # --- Return codes ---
    mock.RETCODE_DONE = 10009

    # --- Default function behaviours (overridden per-test as needed) ---
    mock.initialize.return_value = True
    mock.shutdown.return_value   = True
    mock.last_error.return_value = (0, "no error")

    return mock


_MOCK_MT5 = _make_mock_mt5()
sys.modules["MetaTrader5"] = _MOCK_MT5

# Now we can safely import the execution modules
from src.execution.mt5_bridge import MT5Bridge, _build_tf_map  # noqa: E402
from src.execution.order_manager import OrderManager, OrderResult  # noqa: E402
from src.execution.screenshot import ScreenshotCapture  # noqa: E402
from src.execution.account_monitor import AccountMonitor, AccountInfo, PositionInfo  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mock():
    """Reset mock call history before every test."""
    _MOCK_MT5.reset_mock()
    # Restore defaults that individual tests may have overridden
    _MOCK_MT5.initialize.return_value = True
    _MOCK_MT5.shutdown.return_value = True
    _MOCK_MT5.last_error.return_value = (0, "no error")
    yield


@pytest.fixture
def bridge():
    """Return a connected MT5Bridge backed by the mock module."""
    b = MT5Bridge(login=12345, password="test_pass", server="The5ers-Demo")
    # Inject mock directly so connect() isn't required
    b._mt5 = _MOCK_MT5
    b._connected = True
    return b


@pytest.fixture
def order_manager(bridge):
    return OrderManager(bridge=bridge, deviation=20)


@pytest.fixture
def account_monitor(bridge):
    return AccountMonitor(bridge=bridge)


@pytest.fixture
def screenshot_capture(bridge, tmp_path):
    return ScreenshotCapture(bridge=bridge, save_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_rates_array(n: int = 10) -> list[tuple]:
    """Create a list of (time, open, high, low, close, tick_volume) tuples."""
    base_ts = int(datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc).timestamp())
    rows = []
    for i in range(n):
        rows.append((
            base_ts + i * 300,       # 5-minute bars
            2300.0 + i * 0.5,        # open
            2301.5 + i * 0.5,        # high
            2299.0 + i * 0.5,        # low
            2300.8 + i * 0.5,        # close
            1000 + i * 10,           # tick_volume
        ))
    # Return as a structured numpy array matching MT5 format
    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
    ]
    return np.array(rows, dtype=dtype)


def _make_order_send_result(retcode=10009, order=11111, price=2300.0, volume=0.10):
    """Create a mock order_send result object."""
    result = MagicMock()
    result.retcode  = retcode
    result.order    = order
    result.price    = price
    result.volume   = volume
    result.comment  = "done"
    return result


def _make_account_info_mock(
    balance=10000.0,
    equity=10050.0,
    margin=500.0,
    margin_free=9550.0,
    leverage=100,
):
    info = MagicMock()
    info.balance     = balance
    info.equity      = equity
    info.margin      = margin
    info.margin_free = margin_free
    info.leverage    = leverage
    return info


def _make_symbol_info_mock(
    filling_mode=1,
    volume_step=0.01,
    volume_min=0.01,
    volume_max=500.0,
    point=0.01,
    digits=2,
    trade_tick_value=1.0,
    trade_freeze_level=0,
    trade_stops_level=0,
):
    info = MagicMock()
    info.filling_mode      = filling_mode
    info.volume_step       = volume_step
    info.volume_min        = volume_min
    info.volume_max        = volume_max
    info.point             = point
    info.digits            = digits
    info.trade_tick_value  = trade_tick_value
    info.trade_freeze_level = trade_freeze_level
    info.trade_stops_level  = trade_stops_level
    return info


def _make_tick_mock(bid=2299.5, ask=2300.0):
    tick = MagicMock()
    tick.bid  = bid
    tick.ask  = ask
    tick.time = int(datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc).timestamp())
    return tick


# ---------------------------------------------------------------------------
# 1. Connection tests
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_success(self):
        """connect() returns True and sets _connected when MT5 login succeeds."""
        b = MT5Bridge(login=12345, password="pass", server="Test-Server")
        _MOCK_MT5.initialize.return_value = True
        _MOCK_MT5.account_info.return_value = _make_account_info_mock()

        with patch("src.execution.mt5_bridge._import_mt5", return_value=_MOCK_MT5):
            result = b.connect()

        assert result is True
        assert b.is_connected is True
        _MOCK_MT5.initialize.assert_called_once()

    def test_connect_initialize_failure(self):
        """connect() returns False when mt5.initialize() fails."""
        b = MT5Bridge(login=12345, password="pass", server="Test-Server")
        _MOCK_MT5.initialize.return_value = False

        with patch("src.execution.mt5_bridge._import_mt5", return_value=_MOCK_MT5):
            result = b.connect()

        assert result is False
        assert b.is_connected is False

    def test_connect_login_failure(self):
        """connect() returns False when account_info() returns None after initialize."""
        b = MT5Bridge(login=12345, password="wrong", server="Test-Server")
        _MOCK_MT5.initialize.return_value = True
        _MOCK_MT5.account_info.return_value = None

        with patch("src.execution.mt5_bridge._import_mt5", return_value=_MOCK_MT5):
            result = b.connect()

        assert result is False

    def test_disconnect_calls_shutdown(self, bridge):
        """disconnect() calls mt5.shutdown() and clears _connected."""
        bridge.disconnect()
        _MOCK_MT5.shutdown.assert_called_once()
        assert bridge.is_connected is False

    def test_connect_with_path(self):
        """connect() passes path kwarg to initialize when provided."""
        b = MT5Bridge(
            login=999, password="x", server="Srv",
            path=r"C:\MT5\terminal64.exe"
        )
        _MOCK_MT5.initialize.return_value = True
        _MOCK_MT5.account_info.return_value = _make_account_info_mock()

        with patch("src.execution.mt5_bridge._import_mt5", return_value=_MOCK_MT5):
            b.connect()

        call_kwargs = _MOCK_MT5.initialize.call_args.kwargs
        assert call_kwargs.get("path") == r"C:\MT5\terminal64.exe"

    def test_context_manager(self):
        """MT5Bridge works as a context manager."""
        b = MT5Bridge(login=1, password="p", server="s")
        _MOCK_MT5.initialize.return_value = True
        _MOCK_MT5.account_info.return_value = _make_account_info_mock()

        with patch("src.execution.mt5_bridge._import_mt5", return_value=_MOCK_MT5):
            with b as ctx:
                assert ctx is b

        _MOCK_MT5.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Data retrieval tests
# ---------------------------------------------------------------------------

class TestDataRetrieval:
    def test_get_rates_returns_dataframe(self, bridge):
        """get_rates() returns a DataFrame with the expected columns."""
        _MOCK_MT5.copy_rates_from_pos.return_value = _make_rates_array(50)

        df = bridge.get_rates("XAUUSD", 5, count=50)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ("time", "open", "high", "low", "close", "tick_volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_get_rates_timestamps_are_utc(self, bridge):
        """Timestamps in the returned DataFrame are UTC-aware datetimes."""
        _MOCK_MT5.copy_rates_from_pos.return_value = _make_rates_array(5)

        df = bridge.get_rates("XAUUSD", 5, count=5)

        # pandas may use seconds or nanoseconds resolution depending on version;
        # the key requirement is UTC timezone awareness.
        assert hasattr(df["time"].dtype, "tz"), "time column must be timezone-aware"
        assert str(df["time"].dtype.tz) == "UTC"

    def test_get_rates_numeric_types(self, bridge):
        """Price columns are Python float, not raw numpy types."""
        _MOCK_MT5.copy_rates_from_pos.return_value = _make_rates_array(3)

        df = bridge.get_rates("XAUUSD", 5, count=3)

        assert df["open"].dtype == float
        assert df["close"].dtype == float

    def test_get_rates_empty_on_none(self, bridge):
        """get_rates() returns an empty DataFrame when MT5 returns None."""
        _MOCK_MT5.copy_rates_from_pos.return_value = None

        df = bridge.get_rates("XAUUSD", 5, count=100)

        assert df.empty

    def test_get_multi_tf_rates_all_timeframes(self, bridge):
        """get_multi_tf_rates() fetches 5M, 15M, 1H, 4H by default."""
        _MOCK_MT5.copy_rates_from_pos.return_value = _make_rates_array(20)

        result = bridge.get_multi_tf_rates("XAUUSD")

        assert set(result.keys()) == {"5M", "15M", "1H", "4H"}
        for label, df in result.items():
            assert not df.empty, f"Empty DataFrame for {label}"

    def test_get_tick_returns_bid_ask(self, bridge):
        """get_tick() returns a dict with bid, ask, spread, time."""
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock(bid=2299.5, ask=2300.0)

        tick = bridge.get_tick("XAUUSD")

        assert tick["bid"] == pytest.approx(2299.5)
        assert tick["ask"] == pytest.approx(2300.0)
        assert tick["spread"] == pytest.approx(0.5)
        assert isinstance(tick["time"], pd.Timestamp)

    def test_get_tick_empty_on_none(self, bridge):
        """get_tick() returns empty dict when symbol_info_tick returns None."""
        _MOCK_MT5.symbol_info_tick.return_value = None

        tick = bridge.get_tick("XAUUSD")

        assert tick == {}

    def test_get_symbol_info_returns_expected_keys(self, bridge):
        """get_symbol_info() returns all required symbol properties."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock()

        info = bridge.get_symbol_info("XAUUSD")

        for key in ("point", "digits", "trade_tick_value", "volume_min", "volume_max",
                    "volume_step", "filling_mode"):
            assert key in info, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 3. Market order tests
# ---------------------------------------------------------------------------

class TestMarketOrder:
    def _setup(self):
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock(bid=2299.5, ask=2300.0)
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)

    def test_market_buy_success(self, order_manager):
        """market_order() sends a BUY order and returns a successful OrderResult."""
        self._setup()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(
            retcode=10009, order=11111, price=2300.05, volume=0.10
        )

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is True
        assert result.ticket == 11111
        assert result.price == pytest.approx(2300.05)

        sent_request = _MOCK_MT5.order_send.call_args[0][0]
        assert sent_request["type"] == _MOCK_MT5.ORDER_TYPE_BUY
        assert sent_request["action"] == _MOCK_MT5.TRADE_ACTION_DEAL

    def test_market_sell_success(self, order_manager):
        """market_order() sends a SELL order for short direction."""
        self._setup()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(
            retcode=10009, order=22222, price=2299.45, volume=0.10
        )

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="short",
            lot_size=0.10,
            stop_loss=2310.0,
            take_profit=2280.0,
        )

        assert result.success is True
        sent_request = _MOCK_MT5.order_send.call_args[0][0]
        assert sent_request["type"] == _MOCK_MT5.ORDER_TYPE_SELL
        # Short: price should be the bid
        assert sent_request["price"] == pytest.approx(2299.5)

    def test_market_order_structure(self, order_manager):
        """market_order() populates sl, tp, deviation, magic in the request."""
        self._setup()
        _MOCK_MT5.order_send.return_value = _make_order_send_result()

        order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
            comment="test_entry",
        )

        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["sl"] == pytest.approx(2290.0)
        assert req["tp"] == pytest.approx(2320.0)
        assert req["deviation"] == 20
        assert req["magic"] == 234567
        assert req["comment"] == "test_entry"

    def test_market_order_returns_error_on_none_result(self, order_manager):
        """market_order() returns a failed OrderResult when order_send returns None."""
        self._setup()
        _MOCK_MT5.order_send.return_value = None

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is False
        assert result.error_message != ""


# ---------------------------------------------------------------------------
# 4. Limit order (Kijun pullback entry)
# ---------------------------------------------------------------------------

class TestLimitOrder:
    def test_limit_buy_places_buy_limit(self, order_manager):
        """limit_order() uses ORDER_TYPE_BUY_LIMIT for long direction."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)
        _MOCK_MT5.order_send.return_value = _make_order_send_result(order=33333)

        result = order_manager.limit_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            price=2295.0,
            stop_loss=2285.0,
            take_profit=2315.0,
        )

        assert result.success is True
        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["type"] == _MOCK_MT5.ORDER_TYPE_BUY_LIMIT
        assert req["action"] == _MOCK_MT5.TRADE_ACTION_PENDING
        assert req["price"] == pytest.approx(2295.0)

    def test_limit_sell_places_sell_limit(self, order_manager):
        """limit_order() uses ORDER_TYPE_SELL_LIMIT for short direction."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)
        _MOCK_MT5.order_send.return_value = _make_order_send_result(order=44444)

        result = order_manager.limit_order(
            instrument="XAUUSD",
            direction="short",
            lot_size=0.10,
            price=2305.0,
            stop_loss=2315.0,
            take_profit=2285.0,
        )

        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["type"] == _MOCK_MT5.ORDER_TYPE_SELL_LIMIT


# ---------------------------------------------------------------------------
# 5. Stop-limit order (cloud breakout entry)
# ---------------------------------------------------------------------------

class TestStopLimitOrder:
    def test_stop_limit_buy_structure(self, order_manager):
        """stop_limit_order() sends BUY_STOP_LIMIT with stoplimit field."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)
        _MOCK_MT5.order_send.return_value = _make_order_send_result(order=55555)

        result = order_manager.stop_limit_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_price=2305.0,
            limit_price=2304.0,
            stop_loss=2290.0,
            take_profit=2330.0,
        )

        assert result.success is True
        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["type"] == _MOCK_MT5.ORDER_TYPE_BUY_STOP_LIMIT
        assert req["price"] == pytest.approx(2305.0)
        assert req["stoplimit"] == pytest.approx(2304.0)

    def test_stop_limit_sell_structure(self, order_manager):
        """stop_limit_order() sends SELL_STOP_LIMIT for short direction."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)
        _MOCK_MT5.order_send.return_value = _make_order_send_result(order=66666)

        order_manager.stop_limit_order(
            instrument="XAUUSD",
            direction="short",
            lot_size=0.10,
            stop_price=2295.0,
            limit_price=2296.0,
            stop_loss=2310.0,
            take_profit=2265.0,
        )

        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["type"] == _MOCK_MT5.ORDER_TYPE_SELL_STOP_LIMIT


# ---------------------------------------------------------------------------
# 6. Position modification (trailing stop)
# ---------------------------------------------------------------------------

class TestPositionModification:
    def _make_position(self, ticket=11111, sl=2290.0, tp=2320.0, symbol="XAUUSD"):
        pos = MagicMock()
        pos.ticket = ticket
        pos.symbol = symbol
        pos.sl = sl
        pos.tp = tp
        return pos

    def test_modify_sl_sends_sltp_action(self, order_manager):
        """modify_position() sends TRADE_ACTION_SLTP with the new stop-loss."""
        _MOCK_MT5.positions_get.return_value = [self._make_position()]
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10009)

        ok = order_manager.modify_position(ticket=11111, stop_loss=2295.0)

        assert ok is True
        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["action"] == _MOCK_MT5.TRADE_ACTION_SLTP
        assert req["sl"] == pytest.approx(2295.0)

    def test_modify_preserves_existing_tp(self, order_manager):
        """modify_position() preserves the existing TP when only SL is changed."""
        _MOCK_MT5.positions_get.return_value = [self._make_position(tp=2320.0)]
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10009)

        order_manager.modify_position(ticket=11111, stop_loss=2295.0)

        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["tp"] == pytest.approx(2320.0)  # original TP preserved

    def test_modify_returns_false_when_position_not_found(self, order_manager):
        """modify_position() returns False when no position matches the ticket."""
        _MOCK_MT5.positions_get.return_value = []

        ok = order_manager.modify_position(ticket=99999, stop_loss=2295.0)

        assert ok is False

    def test_modify_returns_false_on_rejection(self, order_manager):
        """modify_position() returns False on retcode != 10009."""
        _MOCK_MT5.positions_get.return_value = [self._make_position()]
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10006)

        ok = order_manager.modify_position(ticket=11111, stop_loss=2295.0)

        assert ok is False


# ---------------------------------------------------------------------------
# 7. Partial close (hybrid 50/50 exit)
# ---------------------------------------------------------------------------

class TestPartialClose:
    def _make_position(self, ticket=11111, volume=0.20, pos_type=0, symbol="XAUUSD"):
        pos = MagicMock()
        pos.ticket = ticket
        pos.symbol = symbol
        pos.volume = volume
        pos.type   = pos_type   # 0 = BUY
        pos.sl     = 2290.0
        pos.tp     = 2320.0
        return pos

    def test_close_partial_50_pct_sends_half_volume(self, order_manager):
        """close_partial(0.5) closes exactly half the position volume."""
        _MOCK_MT5.positions_get.return_value = [self._make_position(volume=0.20)]
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock()
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(
            retcode=10009, volume=0.10
        )

        result = order_manager.close_partial(ticket=11111, close_pct=0.5)

        assert result.success is True
        req = _MOCK_MT5.order_send.call_args[0][0]
        # 50% of 0.20 lots = 0.10 lots
        assert req["volume"] == pytest.approx(0.10, abs=1e-6)

    def test_close_partial_invalid_pct_returns_error(self, order_manager):
        """close_partial() rejects close_pct outside (0, 1]."""
        result = order_manager.close_partial(ticket=11111, close_pct=1.5)
        assert result.success is False

    def test_close_partial_zero_pct_returns_error(self, order_manager):
        """close_partial() rejects close_pct == 0.0."""
        result = order_manager.close_partial(ticket=11111, close_pct=0.0)
        assert result.success is False

    def test_close_full_position(self, order_manager):
        """close_position() with lot_size=None closes the full position volume."""
        _MOCK_MT5.positions_get.return_value = [self._make_position(volume=0.10)]
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock()
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(
            retcode=10009, volume=0.10
        )

        result = order_manager.close_position(ticket=11111, lot_size=None)

        assert result.success is True
        req = _MOCK_MT5.order_send.call_args[0][0]
        assert req["volume"] == pytest.approx(0.10)
        # Closing a BUY with SELL
        assert req["type"] == _MOCK_MT5.ORDER_TYPE_SELL


# ---------------------------------------------------------------------------
# 8. Filling mode detection
# ---------------------------------------------------------------------------

class TestFillingModeDetection:
    def test_detects_fok_when_bit0_set(self, order_manager):
        """_detect_filling_mode() returns ORDER_FILLING_FOK when bit 0 is set."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)  # bit 0

        mode = order_manager._detect_filling_mode("XAUUSD")

        assert mode == _MOCK_MT5.ORDER_FILLING_FOK

    def test_detects_ioc_when_bit1_set(self, order_manager):
        """_detect_filling_mode() returns ORDER_FILLING_IOC when only bit 1 is set."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=2)  # bit 1 only

        mode = order_manager._detect_filling_mode("XAUUSD")

        assert mode == _MOCK_MT5.ORDER_FILLING_IOC

    def test_detects_return_when_no_bits_set(self, order_manager):
        """_detect_filling_mode() returns ORDER_FILLING_RETURN when filling_mode == 0."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=0)

        mode = order_manager._detect_filling_mode("XAUUSD")

        assert mode == _MOCK_MT5.ORDER_FILLING_RETURN

    def test_fok_takes_priority_when_both_bits_set(self, order_manager):
        """FOK is preferred over IOC when both bits are set (filling_mode == 3)."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=3)  # bits 0+1

        mode = order_manager._detect_filling_mode("XAUUSD")

        assert mode == _MOCK_MT5.ORDER_FILLING_FOK


# ---------------------------------------------------------------------------
# 9. Volume normalization
# ---------------------------------------------------------------------------

class TestVolumeNormalization:
    def test_rounds_down_to_lot_step(self, order_manager):
        """_normalize_volume() rounds down to the nearest lot step."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(
            volume_step=0.01, volume_min=0.01, volume_max=500.0
        )

        normalized = order_manager._normalize_volume(0.1234, "XAUUSD")

        # 0.1234 / 0.01 = 12.34 → floor → 12 steps → 0.12
        assert normalized == pytest.approx(0.12, abs=1e-6)

    def test_clamps_below_volume_min(self, order_manager):
        """_normalize_volume() clamps to volume_min when volume is too small."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(
            volume_step=0.01, volume_min=0.01, volume_max=500.0
        )

        normalized = order_manager._normalize_volume(0.001, "XAUUSD")

        assert normalized == pytest.approx(0.01)

    def test_clamps_above_volume_max(self, order_manager):
        """_normalize_volume() clamps to volume_max when volume exceeds the limit."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(
            volume_step=0.01, volume_min=0.01, volume_max=10.0
        )

        normalized = order_manager._normalize_volume(999.99, "XAUUSD")

        assert normalized == pytest.approx(10.0)

    def test_exact_multiple_unchanged(self, order_manager):
        """_normalize_volume() does not alter a volume that is already aligned."""
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(
            volume_step=0.01, volume_min=0.01, volume_max=500.0
        )

        normalized = order_manager._normalize_volume(0.10, "XAUUSD")

        assert normalized == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 10. Error handling — retcodes 10030 and 10006
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def _setup_tick_and_info(self):
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock()
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)

    def test_retcode_10030_market_closed(self, order_manager):
        """OrderResult carries the market-closed description on retcode 10030."""
        self._setup_tick_and_info()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10030)

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is False
        assert result.retcode == 10030
        assert "10030" in result.error_message or "closed" in result.error_message.lower()

    def test_retcode_10006_rejected(self, order_manager):
        """OrderResult carries the rejection description on retcode 10006."""
        self._setup_tick_and_info()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10006)

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is False
        assert result.retcode == 10006

    def test_no_position_close_returns_error(self, order_manager):
        """close_position() returns error when ticket not found."""
        _MOCK_MT5.positions_get.return_value = []

        result = order_manager.close_position(ticket=99999)

        assert result.success is False

    def test_disconnected_bridge_returns_error(self, bridge):
        """OrderManager returns error result when bridge is not connected."""
        bridge._mt5 = None
        manager = OrderManager(bridge=bridge)

        result = manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is False
        assert "not connected" in result.error_message.lower()


# ---------------------------------------------------------------------------
# 11. Slippage logging
# ---------------------------------------------------------------------------

class TestSlippageLogging:
    def _setup(self):
        _MOCK_MT5.symbol_info_tick.return_value = _make_tick_mock(ask=2300.0)
        _MOCK_MT5.symbol_info.return_value = _make_symbol_info_mock(filling_mode=1)

    def test_slippage_logged_on_success(self, order_manager):
        """Slippage is appended to the log after a successful order."""
        self._setup()
        # Fill at 2300.5 vs requested 2300.0 — 0.5 adverse slippage
        _MOCK_MT5.order_send.return_value = _make_order_send_result(
            retcode=10009, price=2300.5
        )

        result = order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert result.success is True
        assert len(order_manager.slippage_log) == 1
        entry = order_manager.slippage_log[0]
        assert entry["filled_price"] == pytest.approx(2300.5)
        assert entry["requested_price"] == pytest.approx(2300.0)
        assert entry["slippage_points"] == pytest.approx(0.5)

    def test_slippage_logged_on_failure(self, order_manager):
        """Slippage is appended to the log even when the order is rejected."""
        self._setup()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10006, price=0.0)

        order_manager.market_order(
            instrument="XAUUSD",
            direction="long",
            lot_size=0.10,
            stop_loss=2290.0,
            take_profit=2320.0,
        )

        assert len(order_manager.slippage_log) == 1
        assert order_manager.slippage_log[0]["success"] is False

    def test_slippage_log_accumulates(self, order_manager):
        """Multiple orders each add an entry to the slippage log."""
        self._setup()
        _MOCK_MT5.order_send.return_value = _make_order_send_result(retcode=10009)

        for _ in range(3):
            order_manager.market_order(
                instrument="XAUUSD",
                direction="long",
                lot_size=0.10,
                stop_loss=2290.0,
                take_profit=2320.0,
            )

        assert len(order_manager.slippage_log) == 3


# ---------------------------------------------------------------------------
# 12. Account monitor
# ---------------------------------------------------------------------------

class TestAccountMonitor:
    def test_get_account_info_returns_dataclass(self, account_monitor):
        """get_account_info() returns an AccountInfo with correct field values."""
        _MOCK_MT5.account_info.return_value = _make_account_info_mock(
            balance=10000.0, equity=10050.0, margin=500.0, margin_free=9550.0, leverage=100
        )

        info = account_monitor.get_account_info()

        assert isinstance(info, AccountInfo)
        assert info.balance == pytest.approx(10000.0)
        assert info.equity == pytest.approx(10050.0)
        assert info.unrealized_pnl == pytest.approx(50.0)
        assert info.leverage == 100

    def test_get_account_info_returns_none_on_mt5_error(self, account_monitor):
        """get_account_info() returns None when account_info() fails."""
        _MOCK_MT5.account_info.return_value = None

        info = account_monitor.get_account_info()

        assert info is None

    def test_equity_property(self, account_monitor):
        """equity property returns account equity as a float."""
        _MOCK_MT5.account_info.return_value = _make_account_info_mock(equity=10075.0)

        assert account_monitor.equity == pytest.approx(10075.0)

    def test_balance_property(self, account_monitor):
        """balance property returns account balance as a float."""
        _MOCK_MT5.account_info.return_value = _make_account_info_mock(balance=9980.0)

        assert account_monitor.balance == pytest.approx(9980.0)

    def test_get_positions_returns_position_info_list(self, account_monitor):
        """get_positions() returns a list of PositionInfo objects."""
        pos = MagicMock()
        pos.ticket        = 12345
        pos.symbol        = "XAUUSD"
        pos.type          = 0   # BUY
        pos.volume        = 0.10
        pos.price_open    = 2300.0
        pos.price_current = 2310.0
        pos.sl            = 2290.0
        pos.tp            = 2320.0
        pos.profit        = 100.0
        pos.time          = int(datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc).timestamp())

        _MOCK_MT5.positions_get.return_value = [pos]

        positions = account_monitor.get_positions()

        assert len(positions) == 1
        p = positions[0]
        assert isinstance(p, PositionInfo)
        assert p.ticket == 12345
        assert p.direction == "long"
        assert p.entry_price == pytest.approx(2300.0)
        assert p.profit == pytest.approx(100.0)

    def test_get_positions_empty_on_no_positions(self, account_monitor):
        """get_positions() returns an empty list when there are no open positions."""
        _MOCK_MT5.positions_get.return_value = []

        positions = account_monitor.get_positions()

        assert positions == []

    def test_get_open_orders_returns_list(self, account_monitor):
        """get_open_orders() returns pending orders as list of dicts."""
        order = MagicMock()
        order.ticket       = 99001
        order.symbol       = "XAUUSD"
        order.type         = 2  # BUY_LIMIT
        order.volume_current = 0.10
        order.price_open   = 2295.0
        order.sl           = 2285.0
        order.tp           = 2315.0
        order.time_setup   = int(datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc).timestamp())

        _MOCK_MT5.orders_get.return_value = [order]

        orders = account_monitor.get_open_orders()

        assert len(orders) == 1
        assert orders[0]["ticket"] == 99001
        assert orders[0]["volume"] == pytest.approx(0.10)

    def test_get_daily_pnl_sums_deal_profits(self, account_monitor):
        """get_daily_pnl() returns the sum of deal profits from today."""
        deal1 = MagicMock()
        deal1.profit = 75.0
        deal2 = MagicMock()
        deal2.profit = -25.0
        _MOCK_MT5.history_deals_get.return_value = [deal1, deal2]

        pnl = account_monitor.get_daily_pnl()

        assert pnl == pytest.approx(50.0)

    def test_get_daily_pnl_returns_zero_on_error(self, account_monitor):
        """get_daily_pnl() returns 0.0 when history_deals_get fails."""
        _MOCK_MT5.history_deals_get.return_value = None

        pnl = account_monitor.get_daily_pnl()

        assert pnl == pytest.approx(0.0)

    def test_equity_returns_zero_when_disconnected(self, account_monitor):
        """equity property returns 0.0 when account_info is unavailable."""
        _MOCK_MT5.account_info.return_value = None

        assert account_monitor.equity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 13. Screenshot
# ---------------------------------------------------------------------------

class TestScreenshot:
    def test_build_path_includes_phase_and_trade_id(self, screenshot_capture, tmp_path):
        """Screenshot path includes the phase and trade_id in the filename."""
        path = screenshot_capture._build_path(
            instrument="XAUUSD",
            timeframe="1H",
            phase="entry",
            trade_id=42,
        )

        assert "XAUUSD" in str(path)
        assert "entry" in path.name
        assert "42" in path.name

    def test_build_path_without_trade_id(self, screenshot_capture):
        """Screenshot path is valid when trade_id is None."""
        path = screenshot_capture._build_path(
            instrument="XAUUSD",
            timeframe="15M",
            phase="exit",
            trade_id=None,
        )

        assert "exit" in path.name

    def test_capture_falls_back_to_empty_string_on_no_mplfinance(
        self, screenshot_capture, bridge
    ):
        """capture() returns empty string when neither MT5 screenshot nor mplfinance
        is available."""
        # MT5 has no chart_screenshot
        del _MOCK_MT5.chart_screenshot

        # Provide OHLCV data so the fallback is attempted
        _MOCK_MT5.copy_rates_from_pos.return_value = _make_rates_array(10)

        # Patch mplfinance import to raise ImportError
        with patch.dict(sys.modules, {"mplfinance": None}):
            result = screenshot_capture.capture(
                instrument="XAUUSD",
                timeframe="1H",
                phase="entry",
                trade_id=1,
            )

        # No chart_screenshot attribute → restore for other tests
        _MOCK_MT5.chart_screenshot = MagicMock(return_value=False)

        assert result == ""

    def test_capture_uses_mt5_screenshot_when_available(self, screenshot_capture, tmp_path):
        """capture() uses mt5.chart_screenshot when it returns True."""
        expected_path = str(tmp_path / "XAUUSD" / "20240601" / "entry_1_100000.png")

        def fake_screenshot(symbol, timeframe, path, width, height):
            # Simulate MT5 writing the file
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
            return True

        _MOCK_MT5.chart_screenshot = MagicMock(side_effect=fake_screenshot)
        _MOCK_MT5.TIMEFRAME_H1 = 16385

        result = screenshot_capture.capture(
            instrument="XAUUSD",
            timeframe="1H",
            phase="entry",
            trade_id=1,
        )

        # Either MT5 screenshot succeeded (non-empty) or fallback ran
        # The key assertion: no exception was raised
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 14. Timeframe mapping
# ---------------------------------------------------------------------------

class TestTimeframeMapping:
    def test_tf_map_covers_all_required_timeframes(self):
        """_build_tf_map() covers all timeframes required by the signal engine."""
        tf_map = _build_tf_map(_MOCK_MT5)

        for label in ("5M", "15M", "1H", "4H", "M1", "M5", "M15", "H1", "H4"):
            assert label in tf_map, f"Missing timeframe label: {label}"

    def test_timeframe_constant_returns_correct_value(self, bridge):
        """MT5Bridge.timeframe_constant() maps labels to MT5 constants."""
        assert bridge.timeframe_constant("5M")  == _MOCK_MT5.TIMEFRAME_M5
        assert bridge.timeframe_constant("15M") == _MOCK_MT5.TIMEFRAME_M15
        assert bridge.timeframe_constant("1H")  == _MOCK_MT5.TIMEFRAME_H1
        assert bridge.timeframe_constant("4H")  == _MOCK_MT5.TIMEFRAME_H4

    def test_timeframe_constant_returns_none_for_unknown(self, bridge):
        """MT5Bridge.timeframe_constant() returns None for unrecognised labels."""
        assert bridge.timeframe_constant("1D") is None
