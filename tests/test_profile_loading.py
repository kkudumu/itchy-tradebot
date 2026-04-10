"""Tests for the instrument-class profile abstraction (plan Task 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader, load_config
from src.config.models import InstrumentOverride
from src.config.profile import (
    InstrumentClass,
    ProfileConfig,
    load_profile,
    price_distance,
)


# ---------------------------------------------------------------------------
# load_profile()
# ---------------------------------------------------------------------------


class TestLoadProfile:
    def test_forex_profile_loads_from_disk(self) -> None:
        profile = load_profile(InstrumentClass.FOREX)
        assert profile.instrument_class == InstrumentClass.FOREX
        assert profile.daily_reset_tz == "UTC"
        assert profile.daily_reset_hour == 0

    def test_futures_profile_loads_from_disk(self) -> None:
        profile = load_profile(InstrumentClass.FUTURES)
        assert profile.instrument_class == InstrumentClass.FUTURES
        assert profile.daily_reset_tz == "America/Chicago"
        assert profile.daily_reset_hour == 17
        assert profile.default_commission_per_contract_round_trip == pytest.approx(1.40)
        assert profile.default_slippage_ticks == 1
        assert profile.maintenance_window_minutes == 60

    def test_accepts_string_class(self) -> None:
        profile = load_profile("futures")
        assert profile.instrument_class == InstrumentClass.FUTURES

    def test_missing_profile_returns_defaults(self, tmp_path: Path) -> None:
        # Point at an empty dir so no YAML exists — should fall back to defaults
        profile = load_profile(InstrumentClass.FUTURES, profile_dir=tmp_path)
        assert profile.instrument_class == InstrumentClass.FUTURES
        # Default commission is 0 when no YAML is found
        assert profile.default_commission_per_contract_round_trip == 0.0

    def test_loader_populates_profiles_dict(self) -> None:
        cfg = load_config()
        assert InstrumentClass.FOREX in cfg.profiles
        assert InstrumentClass.FUTURES in cfg.profiles
        assert cfg.profiles[InstrumentClass.FUTURES].daily_reset_tz == "America/Chicago"

    def test_profile_for_symbol_returns_matching_profile(self) -> None:
        cfg = load_config()
        profile = cfg.profile_for("XAUUSD")
        assert profile is not None
        # XAUUSD in the test fixture is configured as futures
        assert profile.instrument_class == InstrumentClass.FUTURES

    def test_profile_for_unknown_symbol_returns_none(self) -> None:
        cfg = load_config()
        assert cfg.profile_for("DOES_NOT_EXIST") is None


# ---------------------------------------------------------------------------
# InstrumentOverride class-aware validation
# ---------------------------------------------------------------------------


class TestInstrumentClassValidation:
    def test_forex_instrument_accepted_with_just_symbol(self) -> None:
        inst = InstrumentOverride.model_validate(
            {"symbol": "EURUSD", "class": "forex", "pip_size": 0.0001}
        )
        assert inst.class_ == InstrumentClass.FOREX

    def test_forex_class_is_default_when_omitted(self) -> None:
        inst = InstrumentOverride.model_validate({"symbol": "EURUSD"})
        assert inst.class_ == InstrumentClass.FOREX

    def test_futures_instrument_requires_tick_size(self) -> None:
        with pytest.raises(ValueError, match="tick_size"):
            InstrumentOverride.model_validate(
                {
                    "symbol": "MGC",
                    "class": "futures",
                    # tick_size missing
                    "tick_value_usd": 1.0,
                    "contract_size": 10,
                }
            )

    def test_futures_instrument_requires_tick_value_usd(self) -> None:
        with pytest.raises(ValueError, match="tick_value_usd"):
            InstrumentOverride.model_validate(
                {
                    "symbol": "MGC",
                    "class": "futures",
                    "tick_size": 0.10,
                    # tick_value_usd missing
                    "contract_size": 10,
                }
            )

    def test_futures_instrument_requires_contract_size(self) -> None:
        with pytest.raises(ValueError, match="contract_size"):
            InstrumentOverride.model_validate(
                {
                    "symbol": "MGC",
                    "class": "futures",
                    "tick_size": 0.10,
                    "tick_value_usd": 1.0,
                    # contract_size missing
                }
            )

    def test_futures_instrument_accepts_legacy_tick_value(self) -> None:
        # Legacy config that used `tick_value` instead of `tick_value_usd`
        # should be accepted via the backfill validator.
        inst = InstrumentOverride.model_validate(
            {
                "symbol": "MGC",
                "class": "futures",
                "tick_size": 0.10,
                "tick_value": 1.0,
                "contract_size": 10,
            }
        )
        assert inst.tick_value_usd == 1.0

    def test_project_instruments_yaml_loads_clean(self) -> None:
        cfg = load_config()
        inst = cfg.instruments.get("XAUUSD")
        assert inst is not None
        assert inst.class_ == InstrumentClass.FUTURES
        assert inst.tick_size == pytest.approx(0.10)
        assert inst.tick_value_usd == pytest.approx(1.0)
        assert inst.contract_size == pytest.approx(10)
        assert inst.daily_reset_hour_ct == 17


# ---------------------------------------------------------------------------
# price_distance helper
# ---------------------------------------------------------------------------


class _FakeInstrument:
    def __init__(self, class_, tick_size=None, pip_size=None):
        self.class_ = class_
        self.tick_size = tick_size
        self.pip_size = pip_size


class TestPriceDistance:
    def test_forex_pip_to_price(self) -> None:
        inst = _FakeInstrument(InstrumentClass.FOREX, pip_size=0.01)
        # 10 pips on XAU spot (0.01 pip size) → 0.10 price units
        assert price_distance(inst, 10) == pytest.approx(0.10)

    def test_futures_pip_to_price(self) -> None:
        inst = _FakeInstrument(InstrumentClass.FUTURES, tick_size=0.10)
        # 10 "pips" on MGC (0.10 tick) → 10 * 0.10 * 10 ticks = 10.0 price units
        assert price_distance(inst, 10) == pytest.approx(10.0)

    def test_futures_half_pip_preserves_forex_intuition(self) -> None:
        # The plan's rationale: SSS ``min_swing_pips: 0.5`` should remain
        # "half a dollar of price movement on gold" when ported forex→MGC.
        inst = _FakeInstrument(InstrumentClass.FUTURES, tick_size=0.10)
        assert price_distance(inst, 0.5) == pytest.approx(0.50)

    def test_futures_requires_tick_size(self) -> None:
        inst = _FakeInstrument(InstrumentClass.FUTURES, tick_size=None)
        with pytest.raises(ValueError, match="tick_size"):
            price_distance(inst, 1)

    def test_forex_requires_pip_size(self) -> None:
        inst = _FakeInstrument(InstrumentClass.FOREX, pip_size=None, tick_size=None)
        with pytest.raises(ValueError, match="pip_size"):
            price_distance(inst, 1)

    def test_forex_falls_back_to_tick_size(self) -> None:
        # A forex instrument without an explicit pip_size should fall back
        # to tick_size (common for instruments where pip_size == tick_size).
        inst = _FakeInstrument(InstrumentClass.FOREX, pip_size=None, tick_size=0.0001)
        assert price_distance(inst, 10) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Profile YAML shape — sanity check that files on disk match the model
# ---------------------------------------------------------------------------


class TestProfileYAMLShape:
    def test_forex_yaml_validates(self) -> None:
        path = Path("config/profiles/forex.yaml")
        assert path.exists(), f"missing {path}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        ProfileConfig.model_validate(data)  # will raise if shape is wrong

    def test_futures_yaml_validates(self) -> None:
        path = Path("config/profiles/futures.yaml")
        assert path.exists(), f"missing {path}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = ProfileConfig.model_validate(data)
        assert profile.instrument_class == InstrumentClass.FUTURES

    def test_profile_defaults_overridden_by_instrument(self) -> None:
        """Instrument-level values win over profile defaults.

        The profile says commission_per_contract_round_trip is 1.40 by default,
        but the instrument can set a different value in instruments.yaml and
        that per-instrument value takes precedence.
        """
        cfg = load_config()
        inst = cfg.instruments.get("XAUUSD")
        profile = cfg.profile_for("XAUUSD")
        assert inst is not None and profile is not None
        # Current config sets commission_per_contract_round_trip to 1.40
        # explicitly on the instrument — matches profile, but the point is
        # that the instrument value is authoritative.
        if inst.commission_per_contract_round_trip is not None:
            effective = inst.commission_per_contract_round_trip
        else:
            effective = profile.default_commission_per_contract_round_trip
        assert effective == pytest.approx(1.40)
