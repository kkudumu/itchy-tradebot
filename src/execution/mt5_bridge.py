"""MetaTrader 5 terminal interface — connection, data retrieval, tick access.

All MT5 library calls are isolated in this module so that the rest of the
system can be tested on Linux by mocking the MetaTrader5 import.

Timeframe constants (mirrors MetaTrader5 values):
    TIMEFRAME_M1  = 1
    TIMEFRAME_M5  = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1  = 16385
    TIMEFRAME_H4  = 16388

In production, these are resolved at runtime from the real mt5 module.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MT5 import — replaced by mock in tests
# ---------------------------------------------------------------------------

def _import_mt5():
    """Return the MetaTrader5 module, raising ImportError on non-Windows systems."""
    try:
        import MetaTrader5 as mt5  # noqa: N811
        return mt5
    except ImportError as exc:
        raise ImportError(
            "MetaTrader5 package is not available on this platform. "
            "Use Docker (gmag11/MetaTrader5-Docker) or a Windows VPS for live trading."
        ) from exc


# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

# String labels → MT5 TIMEFRAME_* constants resolved at runtime
_TF_LABELS: tuple[str, ...] = ("5M", "15M", "1H", "4H")


def _build_tf_map(mt5) -> dict[str, int]:
    """Build the timeframe label → MT5 constant mapping from a live or mock mt5 object."""
    return {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        # Aliases used by the signal engine
        "5M":  mt5.TIMEFRAME_M5,
        "15M": mt5.TIMEFRAME_M15,
        "1H":  mt5.TIMEFRAME_H1,
        "4H":  mt5.TIMEFRAME_H4,
    }


# ---------------------------------------------------------------------------
# MT5Bridge
# ---------------------------------------------------------------------------

class MT5Bridge:
    """Interface to the MetaTrader 5 terminal.

    Parameters
    ----------
    login:
        MT5 account login number.
    password:
        MT5 account password.
    server:
        Broker server name (e.g. 'The5ers-Demo').
    path:
        Optional absolute path to the MT5 terminal executable.
        If None, MT5 searches for the terminal automatically.
    timeout:
        Connection timeout in milliseconds. Default: 60 000 ms.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: Optional[str] = None,
        timeout: int = 60_000,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._connected = False
        self._mt5 = None  # set on first connect

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the MT5 terminal and log in.

        Returns True on success, False on failure.  All errors are logged so
        the caller can decide whether to retry or abort.
        """
        try:
            mt5 = _import_mt5()
            self._mt5 = mt5

            kwargs: dict = {
                "login": self._login,
                "password": self._password,
                "server": self._server,
                "timeout": self._timeout,
            }
            if self._path:
                kwargs["path"] = self._path

            if not mt5.initialize(**kwargs):
                error = mt5.last_error()
                logger.error("MT5 initialize failed: %s", error)
                return False

            # Verify the login was successful by checking account info
            account = mt5.account_info()
            if account is None:
                logger.error("MT5 login failed for account %s on %s", self._login, self._server)
                mt5.shutdown()
                return False

            self._connected = True
            logger.info(
                "MT5 connected — account %s on %s (balance=%.2f)",
                self._login,
                self._server,
                float(account.balance),
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error connecting to MT5: %s", exc)
            return False

    def disconnect(self) -> None:
        """Shut down the MT5 connection gracefully."""
        try:
            if self._mt5 is not None:
                self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during MT5 disconnect: %s", exc)

    # ------------------------------------------------------------------
    # Market data retrieval
    # ------------------------------------------------------------------

    def get_rates(
        self,
        instrument: str,
        timeframe: int,
        count: int = 200,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars from MT5.

        Parameters
        ----------
        instrument:
            Symbol string, e.g. 'XAUUSD'.
        timeframe:
            MT5 TIMEFRAME_* constant (use the constants from the mt5 object
            or the helper method :meth:`timeframe_constant`).
        count:
            Number of bars to retrieve.

        Returns
        -------
        DataFrame with columns: time (UTC datetime), open, high, low, close,
        tick_volume.  Returns an empty DataFrame on error.
        """
        if not self._connected or self._mt5 is None:
            logger.warning("get_rates called while disconnected")
            return pd.DataFrame()

        try:
            rates = self._mt5.copy_rates_from_pos(instrument, timeframe, 0, count)

            if rates is None or len(rates) == 0:
                error = self._mt5.last_error()
                logger.error(
                    "copy_rates_from_pos returned None for %s tf=%s: %s",
                    instrument,
                    timeframe,
                    error,
                )
                return pd.DataFrame()

            df = pd.DataFrame(rates)

            # MT5 returns time as Unix timestamps — convert to UTC datetimes
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

            # Normalise MT5 numpy types to Python floats for downstream safety
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df[col] = df[col].astype(float)

            if "tick_volume" in df.columns:
                df["tick_volume"] = df["tick_volume"].astype(int)

            # Drop columns we don't use (real_volume, spread) if present
            keep = ["time", "open", "high", "low", "close", "tick_volume"]
            df = df[[c for c in keep if c in df.columns]]

            return df.reset_index(drop=True)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error retrieving rates for %s: %s", instrument, exc)
            return pd.DataFrame()

    def get_multi_tf_rates(
        self,
        instrument: str,
        timeframes: Optional[Dict[str, int]] = None,
        count: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve OHLCV data for all required timeframes in one call.

        Parameters
        ----------
        instrument:
            Symbol string.
        timeframes:
            Mapping of label → MT5 TIMEFRAME constant.  Defaults to the four
            timeframes used by the signal engine: 5M, 15M, 1H, 4H.
        count:
            Number of bars per timeframe.

        Returns
        -------
        Dict mapping each timeframe label to its DataFrame.
        """
        if timeframes is None:
            mt5 = self._mt5
            if mt5 is None:
                logger.error("get_multi_tf_rates called before connect()")
                return {}
            tf_map = _build_tf_map(mt5)
            timeframes = {label: tf_map[label] for label in _TF_LABELS}

        result: Dict[str, pd.DataFrame] = {}
        for label, tf_const in timeframes.items():
            df = self.get_rates(instrument, tf_const, count)
            result[label] = df
            if df.empty:
                logger.warning("Empty DataFrame returned for %s %s", instrument, label)

        return result

    def get_tick(self, instrument: str) -> dict:
        """Get the current best bid/ask tick for an instrument.

        Returns
        -------
        Dict with keys: bid, ask, spread (in points), time (UTC datetime).
        Returns an empty dict on error.
        """
        if not self._connected or self._mt5 is None:
            return {}

        try:
            tick = self._mt5.symbol_info_tick(instrument)
            if tick is None:
                logger.error("symbol_info_tick returned None for %s", instrument)
                return {}

            # Normalise numpy types to plain Python types
            return {
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "spread": float(tick.ask - tick.bid),
                "time": pd.Timestamp(tick.time, unit="s", tz="UTC"),
            }

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error getting tick for %s: %s", instrument, exc)
            return {}

    def get_symbol_info(self, instrument: str) -> dict:
        """Get static symbol properties needed for order sizing.

        Returns
        -------
        Dict with keys: point, digits, trade_tick_value, volume_min,
        volume_max, volume_step, filling_mode.
        Returns an empty dict if the symbol is not found.
        """
        if not self._connected or self._mt5 is None:
            return {}

        try:
            info = self._mt5.symbol_info(instrument)
            if info is None:
                logger.error("symbol_info returned None for %s", instrument)
                return {}

            return {
                "point": float(info.point),
                "digits": int(info.digits),
                "trade_tick_value": float(info.trade_tick_value),
                "volume_min": float(info.volume_min),
                "volume_max": float(info.volume_max),
                "volume_step": float(info.volume_step),
                "filling_mode": int(info.filling_mode),
                # freeze_level — minimum distance from current price for SL/TP
                "trade_freeze_level": int(getattr(info, "trade_freeze_level", 0)),
                "trade_stops_level": int(getattr(info, "trade_stops_level", 0)),
            }

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error getting symbol info for %s: %s", instrument, exc)
            return {}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def timeframe_constant(self, label: str) -> Optional[int]:
        """Convert a string label to the corresponding MT5 TIMEFRAME constant.

        Parameters
        ----------
        label:
            One of '5M', '15M', '1H', '4H', 'M1', 'M5', 'M15', 'H1', 'H4'.

        Returns
        -------
        The MT5 integer constant, or None if the label is unknown.
        """
        if self._mt5 is None:
            return None
        tf_map = _build_tf_map(self._mt5)
        return tf_map.get(label.upper())

    @property
    def is_connected(self) -> bool:
        """True while an active MT5 session is established."""
        return self._connected

    @property
    def mt5(self):
        """Direct reference to the MetaTrader5 module (for order_manager use)."""
        return self._mt5

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MT5Bridge":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
