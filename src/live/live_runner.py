"""LiveRunner — orchestrates a live trading session.

Wires a market data provider + execution provider + strategies +
order router + telemetry collector into a single loop that can run
paper or real. Paper mode uses :class:`PaperExecutionProvider` which
accepts orders without hitting the real broker and maintains a local
position ledger.

SignalR WebSocket support is stubbed — the runner currently polls the
provider on a cadence for new bars. Full SignalR integration is a
follow-on step that upgrades the poll to a push stream without
changing the runner's interface.
"""

from __future__ import annotations

import logging
import signal as _signal_module
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.providers.base import (
    AccountProvider,
    ExecutionProvider,
    ExecutionResult,
    MarketDataProvider,
    PositionSnapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper execution provider — used when live_runner runs in paper mode
# ---------------------------------------------------------------------------


@dataclass
class _PaperPosition:
    instrument: str
    direction: str
    quantity: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    opened_at: datetime


class PaperExecutionProvider(ExecutionProvider):
    """Stand-in execution provider for paper-mode live runs.

    Accepts orders without hitting any real broker. Maintains a simple
    in-memory position ledger that the live runner can inspect. Fills
    happen instantly at the provided entry price — there's no
    slippage or latency simulation at this layer (the backtest engine
    handles those concerns offline).
    """

    def __init__(self) -> None:
        self._positions: Dict[str, _PaperPosition] = {}
        self._next_order_id: int = 1

    def place_market_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        order_id = self._next_order_id
        self._next_order_id += 1
        # Paper mode fills at the provided take_profit-stop midpoint
        # for lack of a real bid/ask. The live runner typically
        # provides signal.entry_price as the "entry" anyway.
        fill_price = (stop_loss or 0.0)
        if take_profit and stop_loss:
            fill_price = (float(stop_loss) + float(take_profit)) / 2.0
        self._positions[f"paper-{order_id}"] = _PaperPosition(
            instrument=instrument,
            direction=direction,
            quantity=quantity,
            entry_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.now(timezone.utc),
        )
        logger.info(
            "[PAPER] %s %s x%.0f @ %.2f SL=%.2f TP=%.2f (%s)",
            direction,
            instrument,
            quantity,
            fill_price,
            stop_loss or 0.0,
            take_profit or 0.0,
            comment,
        )
        return ExecutionResult(
            success=True,
            order_id=f"paper-{order_id}",
            fill_price=fill_price,
            quantity=quantity,
        )

    def place_limit_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        return self.place_market_order(
            instrument, direction, quantity, stop_loss, take_profit, comment
        )

    def modify_order(
        self,
        order_id: int | str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
    ) -> bool:
        return True

    def cancel_order(self, order_id: int | str) -> bool:
        return True

    def close_position(self, instrument: str, quantity: float | None = None) -> bool:
        # Remove all positions for this instrument from the ledger
        closed = [
            oid for oid, p in self._positions.items() if p.instrument == instrument
        ]
        for oid in closed:
            self._positions.pop(oid, None)
        return bool(closed)

    def partial_close_position(self, instrument: str, quantity: float) -> bool:
        return self.close_position(instrument)

    # --- introspection for tests ---

    def open_positions(self) -> List[_PaperPosition]:
        return list(self._positions.values())


# ---------------------------------------------------------------------------
# LiveRunner
# ---------------------------------------------------------------------------


@dataclass
class LiveRunnerConfig:
    instrument: str
    poll_interval_seconds: float = 5.0
    duration_seconds: float | None = None  # None = run until ctrl-c
    paper: bool = True
    initial_balance: float = 50_000.0


class LiveRunner:
    """Live trading orchestrator.

    The runner is intentionally provider-agnostic — it takes already-
    constructed :class:`MarketDataProvider`, :class:`ExecutionProvider`,
    and :class:`AccountProvider` instances plus a callable that
    produces the next signal given recent bars (the "strategy fn").

    The strategy fn abstracts away the backtest's multi-strategy
    dispatch: in live mode we can pass a lambda that delegates to
    :class:`IchimokuBacktester`'s per-bar logic, or a simpler
    function for smoke tests.
    """

    def __init__(
        self,
        config: LiveRunnerConfig,
        market_data: MarketDataProvider,
        execution: ExecutionProvider,
        account: AccountProvider | None,
        strategy_fn: Callable[[Any], Any] | None = None,
        order_router: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self._cfg = config
        self._md = market_data
        self._exec = execution
        self._account = account
        self._strategy_fn = strategy_fn
        self._router = order_router
        self._telemetry = telemetry
        self._stop_requested: bool = False
        self._last_bar_ts: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the runner to exit at the next loop boundary."""
        self._stop_requested = True

    def run(self) -> Dict[str, Any]:
        """Main loop. Returns a summary dict when the loop exits.

        Does NOT reset ``_stop_requested`` — a caller that set it before
        run() will see the loop exit immediately. Callers that want to
        restart after a stop must construct a new LiveRunner.
        """
        started_at = datetime.now(timezone.utc)
        deadline = None
        if self._cfg.duration_seconds is not None:
            deadline = time.monotonic() + float(self._cfg.duration_seconds)

        logger.info(
            "LiveRunner started: instrument=%s paper=%s poll=%.1fs duration=%s",
            self._cfg.instrument,
            self._cfg.paper,
            self._cfg.poll_interval_seconds,
            self._cfg.duration_seconds,
        )

        bars_seen = 0
        signals_processed = 0
        orders_placed = 0

        try:
            while not self._stop_requested:
                if deadline is not None and time.monotonic() >= deadline:
                    logger.info("LiveRunner duration reached — exiting cleanly")
                    break

                # Poll for the latest bar(s). Uses the provider's
                # fetch_bars with a tiny count — a real SignalR push
                # stream will replace this block without changing the
                # rest of the loop.
                try:
                    mtf = self._md.get_multi_tf_data(
                        instrument=self._cfg.instrument,
                        count=100,
                        include_partial_bar=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("market data poll failed: %s", exc)
                    time.sleep(self._cfg.poll_interval_seconds)
                    continue

                new_bar = self._detect_new_bar(mtf)
                if new_bar:
                    bars_seen += 1
                    signal = self._maybe_call_strategy(mtf)
                    if signal is not None:
                        signals_processed += 1
                        placed = self._maybe_route_signal(signal)
                        if placed:
                            orders_placed += 1

                time.sleep(self._cfg.poll_interval_seconds)

        finally:
            # On exit: log summary. Real-money mode would flatten
            # positions here; paper mode leaves them in the ledger
            # for inspection.
            summary = {
                "instrument": self._cfg.instrument,
                "paper": self._cfg.paper,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "bars_seen": bars_seen,
                "signals_processed": signals_processed,
                "orders_placed": orders_placed,
            }
            logger.info("LiveRunner summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_new_bar(self, mtf: Dict[str, Any]) -> bool:
        """Return True when the latest 5M bar timestamp is newer than the last seen."""
        df_5m = mtf.get("5M") if isinstance(mtf, dict) else None
        if df_5m is None or len(df_5m) == 0:
            return False
        try:
            latest_ts = df_5m.index[-1]
            if hasattr(latest_ts, "to_pydatetime"):
                latest_ts = latest_ts.to_pydatetime()
        except Exception:
            return False

        if self._last_bar_ts is None or latest_ts > self._last_bar_ts:
            self._last_bar_ts = latest_ts
            return True
        return False

    def _maybe_call_strategy(self, mtf: Dict[str, Any]) -> Any | None:
        if self._strategy_fn is None:
            return None
        try:
            return self._strategy_fn(mtf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("strategy_fn raised: %s", exc)
            return None

    def _maybe_route_signal(self, signal: Any) -> bool:
        if self._router is None:
            logger.debug("No router configured — signal ignored: %s", signal)
            return False

        # Account equity lookup
        equity = float(self._cfg.initial_balance)
        if self._account is not None:
            try:
                info = self._account.get_account_info()
                if info is not None and info.equity:
                    equity = float(info.equity)
            except Exception as exc:  # noqa: BLE001
                logger.debug("account.get_account_info failed: %s", exc)

        result = self._router.route(
            signal=signal,
            instrument=self._cfg.instrument,
            account_equity=equity,
            risk_pct=0.5,  # TODO: pull from strategy config
        )
        from src.live.order_router import RouterRejection, RouterResult

        if isinstance(result, RouterRejection):
            logger.info("Signal rejected by router: %s", result.reason)
            return False
        return isinstance(result, RouterResult) and result.executed
