"""
Unit tests for database models and connection configuration.

All tests run without a real PostgreSQL connection.  The connection tests
verify pool configuration, error handling, and state management using
unittest.mock to patch psycopg2.

Run with:
    pytest tests/test_database.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.database.models import (
    Candle,
    Decision,
    DecisionAction,
    MarketContext,
    PatternSignature,
    Screenshot,
    ScreenshotPhase,
    Trade,
    TradeDirection,
    TradeSource,
    TradeStatus,
    Zone,
    ZoneConfluence,
    ZoneStatus,
    ZoneType,
)
from src.database.connection import DBConfig, DatabasePool


# =============================================================================
# Fixtures
# =============================================================================

NOW = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


def _trade_row(**overrides):
    base = {
        "id":               1,
        "instrument":       "XAUUSD",
        "source":           "backtest",
        "direction":        "long",
        "entry_time":       NOW,
        "exit_time":        None,
        "entry_price":      2000.0,
        "exit_price":       None,
        "stop_loss":        1990.0,
        "take_profit":      2020.0,
        "lot_size":         0.1,
        "risk_pct":         1.0,
        "r_multiple":       None,
        "pnl":              None,
        "pnl_pct":          None,
        "status":           "open",
        "confluence_score": 75,
        "signal_tier":      "A",
        "created_at":       NOW,
    }
    base.update(overrides)
    return base


def _candle_row(**overrides):
    base = {
        "timestamp":  NOW,
        "instrument": "XAUUSD",
        "open":       1999.5,
        "high":       2001.0,
        "low":        1998.5,
        "close":      2000.0,
        "volume":     1234.5,
    }
    base.update(overrides)
    return base


def _context_row(**overrides):
    base = {
        "id":                    10,
        "trade_id":              1,
        "timestamp":             NOW,
        "instrument":            "XAUUSD",
        "cloud_direction_4h":    "bullish",
        "cloud_direction_1h":    "bullish",
        "tk_cross_15m":          "bullish",
        "chikou_confirmation":   True,
        "cloud_thickness_4h":    15.5,
        "adx_value":             28.0,
        "atr_value":             12.3,
        "rsi_value":             58.0,
        "bb_width_percentile":   65.0,
        "session":               "london",
        "nearest_sr_distance":   3.2,
        "zone_confluence_score": 4,
        "context_embedding":     [0.1] * 64,
        "created_at":            NOW,
    }
    base.update(overrides)
    return base


# =============================================================================
# Enum tests
# =============================================================================

class TestEnums:
    def test_trade_source_values(self):
        assert TradeSource.BACKTEST.value == "backtest"
        assert TradeSource.LIVE.value     == "live"
        assert TradeSource.PAPER.value    == "paper"

    def test_trade_direction_values(self):
        assert TradeDirection.LONG.value  == "long"
        assert TradeDirection.SHORT.value == "short"

    def test_trade_status_values(self):
        assert TradeStatus.OPEN.value      == "open"
        assert TradeStatus.PARTIAL.value   == "partial"
        assert TradeStatus.CLOSED.value    == "closed"
        assert TradeStatus.CANCELLED.value == "cancelled"

    def test_zone_type_values(self):
        assert ZoneType.SUPPORT.value    == "support"
        assert ZoneType.RESISTANCE.value == "resistance"
        assert ZoneType.SUPPLY.value     == "supply"
        assert ZoneType.DEMAND.value     == "demand"
        assert ZoneType.PIVOT.value      == "pivot"

    def test_zone_status_values(self):
        assert ZoneStatus.ACTIVE.value      == "active"
        assert ZoneStatus.TESTED.value      == "tested"
        assert ZoneStatus.INVALIDATED.value == "invalidated"

    def test_decision_action_values(self):
        assert DecisionAction.ENTER.value        == "enter"
        assert DecisionAction.SKIP.value         == "skip"
        assert DecisionAction.EXIT.value         == "exit"
        assert DecisionAction.PARTIAL_EXIT.value == "partial_exit"
        assert DecisionAction.MODIFY.value       == "modify"

    def test_screenshot_phase_values(self):
        assert ScreenshotPhase.PRE_ENTRY.value == "pre_entry"
        assert ScreenshotPhase.ENTRY.value     == "entry"
        assert ScreenshotPhase.DURING.value    == "during"
        assert ScreenshotPhase.EXIT.value      == "exit"

    def test_invalid_enum_raises(self):
        with pytest.raises(ValueError):
            TradeSource("invalid_source")

    def test_enum_str_comparison(self):
        # Enums inherit from str, so direct string comparison works
        assert TradeSource.LIVE == "live"
        assert DecisionAction.ENTER == "enter"


# =============================================================================
# Candle model
# =============================================================================

class TestCandle:
    def test_from_row_basic(self):
        row = _candle_row()
        c = Candle.from_row(row)
        assert c.instrument == "XAUUSD"
        assert c.open   == 1999.5
        assert c.high   == 2001.0
        assert c.low    == 1998.5
        assert c.close  == 2000.0
        assert c.volume == 1234.5
        assert c.timestamp == NOW

    def test_to_dict_has_iso_timestamp(self):
        c = Candle.from_row(_candle_row())
        d = c.to_dict()
        assert isinstance(d["timestamp"], str)
        assert "2024" in d["timestamp"]

    def test_to_dict_all_fields_present(self):
        c = Candle.from_row(_candle_row())
        d = c.to_dict()
        for key in ("timestamp", "instrument", "open", "high", "low", "close", "volume"):
            assert key in d

    def test_numeric_coercion(self):
        # Values coming in as strings (e.g. from some adapters) should be coerced
        row = _candle_row(open="2000.1", high="2002.5", low="1999.0", close="2001.0", volume="500")
        c = Candle.from_row(row)
        assert c.open  == 2000.1
        assert c.high  == 2002.5
        assert c.low   == 1999.0
        assert c.close == 2001.0


# =============================================================================
# Trade model
# =============================================================================

class TestTrade:
    def test_from_row_open_trade(self):
        t = Trade.from_row(_trade_row())
        assert t.id         == 1
        assert t.instrument == "XAUUSD"
        assert t.source     == TradeSource.BACKTEST
        assert t.direction  == TradeDirection.LONG
        assert t.status     == TradeStatus.OPEN
        assert t.stop_loss  == 1990.0
        assert t.lot_size   == 0.1
        assert t.signal_tier == "A"
        assert t.confluence_score == 75

    def test_from_row_closed_trade(self):
        t = Trade.from_row(_trade_row(
            exit_time=NOW, exit_price=2015.0,
            r_multiple=1.5, pnl=150.0, pnl_pct=0.75,
            status="closed",
        ))
        assert t.status       == TradeStatus.CLOSED
        assert t.exit_price   == 2015.0
        assert t.r_multiple   == 1.5
        assert t.pnl          == 150.0

    def test_to_dict_serialises_enums_as_strings(self):
        t = Trade.from_row(_trade_row())
        d = t.to_dict()
        assert d["source"]    == "backtest"
        assert d["direction"] == "long"
        assert d["status"]    == "open"

    def test_to_dict_iso_datetimes(self):
        t = Trade.from_row(_trade_row())
        d = t.to_dict()
        assert isinstance(d["entry_time"], str)
        assert d["exit_time"] is None

    def test_optional_fields_default_to_none(self):
        t = Trade.from_row(_trade_row(
            exit_time=None, exit_price=None,
            r_multiple=None, pnl=None, take_profit=None,
        ))
        assert t.exit_time   is None
        assert t.exit_price  is None
        assert t.r_multiple  is None
        assert t.pnl         is None
        assert t.take_profit is None

    def test_all_source_values_accepted(self):
        for src in ("backtest", "live", "paper"):
            t = Trade.from_row(_trade_row(source=src))
            assert t.source.value == src


# =============================================================================
# MarketContext model
# =============================================================================

class TestMarketContext:
    def test_from_row_full(self):
        mc = MarketContext.from_row(_context_row())
        assert mc.id                    == 10
        assert mc.instrument            == "XAUUSD"
        assert mc.cloud_direction_4h    == "bullish"
        assert mc.chikou_confirmation   is True
        assert mc.adx_value             == 28.0
        assert mc.session               == "london"
        assert mc.zone_confluence_score == 4
        assert len(mc.context_embedding) == 64

    def test_embedding_as_string_parsed(self):
        # Simulate receiving the vector as a bracketed string
        embedding_str = "[" + ",".join(["0.1"] * 64) + "]"
        mc = MarketContext.from_row(_context_row(context_embedding=embedding_str))
        assert len(mc.context_embedding) == 64
        assert abs(mc.context_embedding[0] - 0.1) < 1e-6

    def test_null_embedding_allowed(self):
        mc = MarketContext.from_row(_context_row(context_embedding=None))
        assert mc.context_embedding is None

    def test_to_dict_embedding_preserved(self):
        mc = MarketContext.from_row(_context_row())
        d = mc.to_dict()
        assert len(d["context_embedding"]) == 64

    def test_to_dict_timestamp_is_string(self):
        mc = MarketContext.from_row(_context_row())
        d = mc.to_dict()
        assert isinstance(d["timestamp"], str)


# =============================================================================
# PatternSignature model
# =============================================================================

class TestPatternSignature:
    def test_from_row_basic(self):
        row = {
            "id":            5,
            "context_id":    10,
            "trade_id":      1,
            "embedding":     [0.5] * 64,
            "outcome_r":     2.1,
            "win":           True,
            "cluster_label": 3,
            "created_at":    NOW,
        }
        ps = PatternSignature.from_row(row)
        assert ps.id            == 5
        assert ps.outcome_r     == 2.1
        assert ps.win           is True
        assert ps.cluster_label == 3
        assert len(ps.embedding) == 64

    def test_embedding_string_parsed(self):
        s = "[" + ",".join(["0.25"] * 64) + "]"
        ps = PatternSignature.from_row({
            "id": 1, "embedding": s,
            "context_id": None, "trade_id": None,
            "outcome_r": None, "win": None,
            "cluster_label": None, "created_at": None,
        })
        assert len(ps.embedding) == 64

    def test_to_dict(self):
        row = {
            "id": 1, "context_id": None, "trade_id": None,
            "embedding": [0.0] * 64,
            "outcome_r": 1.0, "win": False,
            "cluster_label": 0, "created_at": None,
        }
        d = PatternSignature.from_row(row).to_dict()
        assert d["outcome_r"] == 1.0
        assert d["win"] is False


# =============================================================================
# Screenshot model
# =============================================================================

class TestScreenshot:
    def test_from_row(self):
        row = {
            "id": 7, "trade_id": 1, "phase": "entry",
            "file_path": "/data/screenshots/trade_1_entry.png",
            "timeframe": "15m", "created_at": NOW,
        }
        s = Screenshot.from_row(row)
        assert s.phase     == ScreenshotPhase.ENTRY
        assert s.file_path == "/data/screenshots/trade_1_entry.png"
        assert s.timeframe == "15m"

    def test_null_phase_allowed(self):
        row = {
            "id": 1, "trade_id": None, "phase": None,
            "file_path": "/tmp/shot.png", "timeframe": None, "created_at": None,
        }
        s = Screenshot.from_row(row)
        assert s.phase is None

    def test_to_dict_phase_as_string(self):
        row = {
            "id": 1, "trade_id": 1, "phase": "pre_entry",
            "file_path": "/tmp/x.png", "timeframe": "4h", "created_at": NOW,
        }
        d = Screenshot.from_row(row).to_dict()
        assert d["phase"] == "pre_entry"


# =============================================================================
# Zone model
# =============================================================================

class TestZone:
    def test_from_row(self):
        row = {
            "id": 3, "instrument": "XAUUSD",
            "zone_type": "support", "price_high": 2005.0, "price_low": 2000.0,
            "timeframe": "1h", "strength": 0.85, "touch_count": 3,
            "status": "active", "first_seen": NOW,
            "last_tested": NOW, "created_at": NOW,
        }
        z = Zone.from_row(row)
        assert z.zone_type   == ZoneType.SUPPORT
        assert z.price_high  == 2005.0
        assert z.status      == ZoneStatus.ACTIVE
        assert z.touch_count == 3
        assert z.strength    == 0.85

    def test_defaults_applied_when_missing(self):
        row = {
            "id": None, "instrument": "XAUUSD",
            "zone_type": "demand", "price_high": 1980.0, "price_low": 1975.0,
            "timeframe": "4h", "first_seen": NOW,
            "strength": None, "touch_count": None,
            "status": None, "last_tested": None, "created_at": None,
        }
        z = Zone.from_row(row)
        assert z.strength    == 0.0
        assert z.touch_count == 0
        assert z.status      == ZoneStatus.ACTIVE

    def test_to_dict_enums_as_strings(self):
        row = {
            "id": 1, "instrument": "XAUUSD",
            "zone_type": "pivot", "price_high": 2010.0, "price_low": 2008.0,
            "timeframe": "1d", "strength": 0.5, "touch_count": 1,
            "status": "tested", "first_seen": NOW,
            "last_tested": None, "created_at": None,
        }
        d = Zone.from_row(row).to_dict()
        assert d["zone_type"] == "pivot"
        assert d["status"]    == "tested"


# =============================================================================
# ZoneConfluence model
# =============================================================================

class TestZoneConfluence:
    def test_from_row(self):
        row = {
            "id": 2, "zone_id": 3,
            "confluence_type": "fibonacci_618",
            "value": 2002.5, "timeframe": "1h", "created_at": NOW,
        }
        zc = ZoneConfluence.from_row(row)
        assert zc.confluence_type == "fibonacci_618"
        assert zc.value           == 2002.5

    def test_null_value_allowed(self):
        row = {
            "id": 1, "zone_id": None,
            "confluence_type": "round_number",
            "value": None, "timeframe": None, "created_at": None,
        }
        zc = ZoneConfluence.from_row(row)
        assert zc.value is None

    def test_to_dict(self):
        row = {
            "id": 1, "zone_id": 5,
            "confluence_type": "cloud_edge",
            "value": 2000.0, "timeframe": "4h", "created_at": NOW,
        }
        d = ZoneConfluence.from_row(row).to_dict()
        assert d["confluence_type"] == "cloud_edge"
        assert d["value"]           == 2000.0


# =============================================================================
# Decision model
# =============================================================================

class TestDecision:
    def test_from_row_basic(self):
        row = {
            "id": 20, "timestamp": NOW, "instrument": "XAUUSD",
            "action": "enter", "trade_id": 1,
            "signal_data":     {"tk_cross": True},
            "edge_results":    {"ichimoku_alignment": True, "adx_filter": False},
            "similarity_data": {"top_k": [{"id": 5, "score": 0.92}]},
            "confluence_score": 80,
            "reasoning":       "Strong bullish Ichimoku setup above cloud",
            "created_at":      NOW,
        }
        d = Decision.from_row(row)
        assert d.action           == DecisionAction.ENTER
        assert d.confluence_score == 80
        assert d.signal_data["tk_cross"] is True
        assert d.edge_results["ichimoku_alignment"] is True

    def test_json_string_fields_parsed(self):
        # Simulate receiving JSONB as a string (non-adapted connection)
        row = {
            "id": 1, "timestamp": NOW, "instrument": "XAUUSD",
            "action": "skip", "trade_id": None,
            "signal_data":     json.dumps({"reason": "no_setup"}),
            "edge_results":    json.dumps({"adx_filter": False}),
            "similarity_data": json.dumps({}),
            "confluence_score": 20,
            "reasoning":       "ADX below threshold",
            "created_at":      None,
        }
        d = Decision.from_row(row)
        assert d.signal_data["reason"] == "no_setup"

    def test_null_json_fields(self):
        row = {
            "id": 1, "timestamp": NOW, "instrument": "XAUUSD",
            "action": "skip", "trade_id": None,
            "signal_data": None, "edge_results": None,
            "similarity_data": None, "confluence_score": None,
            "reasoning": None, "created_at": None,
        }
        d = Decision.from_row(row)
        assert d.signal_data    is None
        assert d.edge_results   is None
        assert d.similarity_data is None

    def test_to_dict_action_as_string(self):
        row = {
            "id": 1, "timestamp": NOW, "instrument": "XAUUSD",
            "action": "partial_exit", "trade_id": 2,
            "signal_data": None, "edge_results": None,
            "similarity_data": None, "confluence_score": None,
            "reasoning": None, "created_at": NOW,
        }
        d = Decision.from_row(row).to_dict()
        assert d["action"] == "partial_exit"
        assert isinstance(d["timestamp"], str)

    def test_all_actions_accepted(self):
        for action in ("enter", "skip", "exit", "partial_exit", "modify"):
            row = {
                "id": 1, "timestamp": NOW, "instrument": "XAUUSD",
                "action": action, "trade_id": None,
                "signal_data": None, "edge_results": None,
                "similarity_data": None, "confluence_score": None,
                "reasoning": None, "created_at": None,
            }
            d = Decision.from_row(row)
            assert d.action.value == action


# =============================================================================
# DBConfig
# =============================================================================

class TestDBConfig:
    def test_defaults(self):
        cfg = DBConfig()
        assert cfg.host            == "localhost"
        assert cfg.port            == 5432
        assert cfg.dbname          == "trade_agent"
        assert cfg.user            == "postgres"
        assert cfg.min_connections == 2
        assert cfg.max_connections == 10

    def test_dsn_contains_all_fields(self):
        cfg = DBConfig(host="db.local", port=5433, dbname="mydb", user="admin", password="secret")
        dsn = cfg.dsn()
        assert "db.local"    in dsn
        assert "5433"        in dsn
        assert "mydb"        in dsn
        assert "admin"       in dsn
        assert "secret"      in dsn

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DB_HOST",     "envhost")
        monkeypatch.setenv("DB_PORT",     "5555")
        monkeypatch.setenv("DB_NAME",     "envdb")
        monkeypatch.setenv("DB_USER",     "envuser")
        monkeypatch.setenv("DB_PASSWORD", "envpass")
        monkeypatch.setenv("DB_MIN_CONN", "3")
        monkeypatch.setenv("DB_MAX_CONN", "20")
        cfg = DBConfig()
        assert cfg.host            == "envhost"
        assert cfg.port            == 5555
        assert cfg.dbname          == "envdb"
        assert cfg.user            == "envuser"
        assert cfg.password        == "envpass"
        assert cfg.min_connections == 3
        assert cfg.max_connections == 20


# =============================================================================
# DatabasePool (no real DB — psycopg2 is mocked)
# =============================================================================

class TestDatabasePool:
    def _make_pool(self):
        """Return a DatabasePool backed by a mock ThreadedConnectionPool."""
        cfg = DBConfig(
            host="localhost", port=5432, dbname="test_db",
            user="test", password="test",
            min_connections=1, max_connections=5,
        )
        return DatabasePool(cfg)

    def test_not_open_before_initialise(self):
        pool = self._make_pool()
        assert pool.is_open is False

    def test_is_open_after_initialise(self):
        pool = self._make_pool()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_cls.return_value = MagicMock()
            pool.initialise()
        assert pool.is_open is True

    def test_double_initialise_is_noop(self):
        pool = self._make_pool()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            pool.initialise()
            pool.initialise()  # second call should not create another pool
            assert mock_cls.call_count == 1

    def test_close_sets_is_open_false(self):
        pool = self._make_pool()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_cls.return_value = MagicMock()
            pool.initialise()
            pool.close()
        assert pool.is_open is False

    def test_get_connection_before_init_raises(self):
        pool = self._make_pool()
        with pytest.raises(RuntimeError, match="not been initialised"):
            with pool.get_connection():
                pass

    def test_get_connection_commits_on_success(self):
        pool = self._make_pool()
        mock_conn = MagicMock()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_pool = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            mock_cls.return_value = mock_pool
            pool.initialise()
            with pool.get_connection() as conn:
                assert conn is mock_conn
            mock_conn.commit.assert_called_once()
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_rolls_back_on_exception(self):
        pool = self._make_pool()
        mock_conn = MagicMock()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_pool = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            mock_cls.return_value = mock_pool
            pool.initialise()
            with pytest.raises(ValueError):
                with pool.get_connection():
                    raise ValueError("simulated failure")
            mock_conn.rollback.assert_called_once()
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_health_check_returns_true_when_db_responds(self):
        pool = self._make_pool()
        mock_conn  = MagicMock()
        mock_cur   = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__  = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = {"?column?": 1}
        mock_conn.cursor.return_value = mock_cur
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_pool = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            mock_cls.return_value = mock_pool
            pool.initialise()
            result = pool.health_check()
        assert result is True

    def test_health_check_returns_false_on_error(self):
        pool = self._make_pool()
        with patch("src.database.connection.ThreadedConnectionPool") as mock_cls:
            mock_pool = MagicMock()
            mock_pool.getconn.side_effect = Exception("connection refused")
            mock_cls.return_value = mock_pool
            pool.initialise()
            result = pool.health_check()
        assert result is False

    def test_config_accessible(self):
        pool = self._make_pool()
        assert pool.config.dbname == "test_db"


# =============================================================================
# Round-trip serialisation sanity checks
# =============================================================================

class TestRoundTrip:
    """Verify that from_row → to_dict preserves all critical field values."""

    def test_trade_round_trip(self):
        row = _trade_row(
            exit_time=NOW, exit_price=2018.0,
            r_multiple=1.8, pnl=180.0, pnl_pct=0.9,
            status="closed",
        )
        d = Trade.from_row(row).to_dict()
        assert d["entry_price"]  == 2000.0
        assert d["exit_price"]   == 2018.0
        assert d["r_multiple"]   == 1.8
        assert d["source"]       == "backtest"
        assert d["direction"]    == "long"
        assert d["status"]       == "closed"

    def test_candle_round_trip(self):
        row = _candle_row()
        d = Candle.from_row(row).to_dict()
        assert d["open"]  == 1999.5
        assert d["close"] == 2000.0

    def test_market_context_round_trip(self):
        row = _context_row()
        d = MarketContext.from_row(row).to_dict()
        assert d["cloud_direction_4h"]  == "bullish"
        assert d["session"]             == "london"
        assert len(d["context_embedding"]) == 64

    def test_decision_round_trip(self):
        row = {
            "id": 1, "timestamp": NOW, "instrument": "XAUUSD",
            "action": "enter", "trade_id": 1,
            "signal_data": {"tk_cross": True},
            "edge_results": {"ichimoku_alignment": True},
            "similarity_data": None,
            "confluence_score": 70,
            "reasoning": "Setup valid",
            "created_at": NOW,
        }
        d = Decision.from_row(row).to_dict()
        assert d["action"]           == "enter"
        assert d["confluence_score"] == 70
        assert d["edge_results"]["ichimoku_alignment"] is True
