"""Instrument-class-aware sizing adapters.

The :class:`AdaptivePositionSizer` in ``position_sizer.py`` owns the
risk-percentage logic (phase switching, safety rails, risk amount per
trade). This module handles the *other* half of the sizing pipeline:
converting a risk amount (in dollars) and a stop distance (in price
units) into the broker-facing quantity — a float lot count for forex,
or an integer contract count for futures.

Separating the two lets ``AdaptivePositionSizer`` stay class-agnostic
while the conversion formulas live next to the instrument metadata
they depend on (tick_size, pip_size, pip_value_per_lot, contract caps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.config.profile import InstrumentClass


@runtime_checkable
class InstrumentSizer(Protocol):
    """Converts (risk_usd, stop_distance_price) to a broker quantity."""

    def size_for_risk(self, risk_usd: float, stop_distance_price: float) -> float | int:
        """Return the quantity to send to the broker.

        *stop_distance_price* is the absolute price delta between entry
        and stop in the instrument's native price units (e.g. 0.50 USD
        for a half-dollar stop on MGC). *risk_usd* is the total dollar
        amount the caller is willing to lose if the stop fires.
        """
        ...

    def min_size(self) -> float | int:
        """Broker-enforced minimum quantity (e.g. 0.01 forex lots, 1 contract)."""
        ...

    def max_size(self) -> float | int:
        """Broker-enforced maximum quantity (account risk cap, contract cap)."""
        ...


# ---------------------------------------------------------------------------
# Forex implementation: float lot sizing
# ---------------------------------------------------------------------------


@dataclass
class ForexLotSizer:
    """Forex lot sizer.

    ``pip_value_per_lot`` is the USD value of one pip per standard lot
    at the reference lot size. For XAU/USD with pip_size=0.01 and a
    $1/pip/lot convention, a 50-pip stop on $50 risk yields 1.0 lot.

    The minimum/maximum lot size clamp the output at broker limits.
    """

    pip_size: float
    pip_value_per_lot: float
    max_lot_size: float = 10.0
    min_lot_size: float = 0.01

    def size_for_risk(self, risk_usd: float, stop_distance_price: float) -> float:
        if risk_usd <= 0 or stop_distance_price <= 0:
            return self.min_lot_size
        stop_pips = stop_distance_price / self.pip_size
        dollars_per_lot_at_stop = stop_pips * self.pip_value_per_lot
        if dollars_per_lot_at_stop <= 0:
            return self.min_lot_size
        raw = risk_usd / dollars_per_lot_at_stop
        clamped = min(self.max_lot_size, max(self.min_lot_size, raw))
        return round(clamped, 2)

    def min_size(self) -> float:
        return self.min_lot_size

    def max_size(self) -> float:
        return self.max_lot_size


# ---------------------------------------------------------------------------
# Futures implementation: integer contract sizing
# ---------------------------------------------------------------------------


@dataclass
class FuturesContractSizer:
    """Futures contract sizer.

    Converts risk dollars to an integer contract count given a tick
    size and a USD value per tick. A $50 risk on a $5 stop for MGC
    ($1/tick) yields ``50 // (50 * 1) = 1`` contract. The sizer rounds
    DOWN — a risk amount that doesn't buy at least ``min_contracts``
    contracts returns 0, letting the engine skip the trade rather than
    take on more risk than requested.
    """

    tick_size: float
    tick_value_usd: float
    max_contracts: int = 50
    min_contracts: int = 1

    def size_for_risk(self, risk_usd: float, stop_distance_price: float) -> int:
        if risk_usd <= 0 or stop_distance_price <= 0:
            return 0
        stop_ticks = stop_distance_price / self.tick_size
        dollars_per_contract_at_stop = stop_ticks * self.tick_value_usd
        if dollars_per_contract_at_stop <= 0:
            return 0
        raw = risk_usd / dollars_per_contract_at_stop
        # Round down — never over-risk because of rounding
        contracts = int(raw)
        if contracts < self.min_contracts:
            # Risk amount doesn't cover a single contract at this stop;
            # return 0 so the caller can skip the trade with a clean
            # telemetry rejection rather than silently under-sizing.
            return 0
        return min(self.max_contracts, contracts)

    def min_size(self) -> int:
        return self.min_contracts

    def max_size(self) -> int:
        return self.max_contracts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def sizer_for_instrument(instrument_config: Any) -> InstrumentSizer:
    """Return the appropriate :class:`InstrumentSizer` for an instrument.

    Reads ``instrument_config.class_`` (or ``.instrument_class``) and
    returns a ``ForexLotSizer`` or ``FuturesContractSizer`` pre-populated
    from the instrument's tick/pip metadata. Caps default to broker-safe
    values when the instrument config doesn't specify them.
    """
    cls = getattr(instrument_config, "class_", None) or getattr(
        instrument_config, "instrument_class", None
    )
    if isinstance(cls, str):
        try:
            cls = InstrumentClass(cls)
        except ValueError:
            cls = None

    if cls == InstrumentClass.FUTURES:
        tick_size = float(getattr(instrument_config, "tick_size", 0.0) or 0.0)
        tick_value = float(
            getattr(instrument_config, "tick_value_usd", None)
            or getattr(instrument_config, "tick_value", 0.0)
            or 0.0
        )
        max_contracts = int(
            getattr(instrument_config, "max_micro_contracts", None)
            or getattr(instrument_config, "max_contracts", None)
            or 50
        )
        return FuturesContractSizer(
            tick_size=tick_size,
            tick_value_usd=tick_value,
            max_contracts=max_contracts,
        )

    # Default to forex
    pip_size = float(
        getattr(instrument_config, "pip_size", None)
        or getattr(instrument_config, "tick_size", 0.01)
        or 0.01
    )
    pip_value = float(
        getattr(instrument_config, "pip_value_per_lot", None)
        or getattr(instrument_config, "pip_value_usd", 1.0)
        or 1.0
    )
    max_lot = float(getattr(instrument_config, "max_lot_size", 10.0) or 10.0)
    return ForexLotSizer(
        pip_size=pip_size,
        pip_value_per_lot=pip_value,
        max_lot_size=max_lot,
    )
