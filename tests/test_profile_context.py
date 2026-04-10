"""Tests for the learning engine profile context (plan Task 20)."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.profile import InstrumentClass
from src.learning.profile_context import (
    PROFILE_FEATURE_DIM,
    pad_embedding_with_profile,
    profile_embedding_features,
    profile_metadata,
)


class _FakeInst:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakePropFirm:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestProfileMetadata:
    def test_forex_instrument_returns_forex_class(self) -> None:
        inst = _FakeInst(class_=InstrumentClass.FOREX, pip_size=0.01, pip_value_usd=1.0)
        meta = profile_metadata(inst)
        assert meta["instrument_class"] == "forex"
        assert meta["pip_size"] == 0.01
        assert meta["pip_value_per_lot"] == 1.0
        assert meta["tick_size"] is None

    def test_futures_instrument_returns_futures_class(self) -> None:
        inst = _FakeInst(
            class_=InstrumentClass.FUTURES,
            tick_size=0.10,
            tick_value_usd=1.0,
        )
        meta = profile_metadata(inst)
        assert meta["instrument_class"] == "futures"
        assert meta["tick_size"] == 0.10
        assert meta["tick_value_usd"] == 1.0

    def test_topstep_prop_firm_populates_dollar_fields(self) -> None:
        inst = _FakeInst(class_=InstrumentClass.FUTURES, tick_size=0.10, tick_value_usd=1.0)
        pf = _FakePropFirm(
            style="topstep_combine_dollar",
            account_size=50_000.0,
            max_loss_limit_usd_trailing=2_000.0,
            daily_loss_limit_usd=1_000.0,
        )
        meta = profile_metadata(inst, prop_firm_config=pf)
        assert meta["prop_firm_style"] == "topstep_combine_dollar"
        assert meta["account_size_usd"] == 50_000.0
        assert meta["max_loss_limit_usd"] == 2_000.0
        assert meta["daily_loss_limit_usd"] == 1_000.0

    def test_the5ers_prop_firm_converts_pct_to_usd(self) -> None:
        inst = _FakeInst(class_=InstrumentClass.FOREX, pip_size=0.01)
        pf_dict = {
            "style": "the5ers_pct_phased",
            "account_size": 10_000.0,
            "phase_1": {"max_loss_pct": 10.0, "daily_loss_pct": 5.0},
        }
        meta = profile_metadata(inst, prop_firm_config=pf_dict)
        assert meta["prop_firm_style"] == "the5ers_pct_phased"
        assert meta["max_loss_limit_usd"] == 1_000.0
        assert meta["daily_loss_limit_usd"] == 500.0


class TestProfileEmbeddingFeatures:
    def test_output_length_is_constant(self) -> None:
        for meta in ({}, {"instrument_class": "futures"}, {"account_size_usd": 50_000}):
            vec = profile_embedding_features(meta)
            assert len(vec) == PROFILE_FEATURE_DIM

    def test_futures_flag_set(self) -> None:
        vec = profile_embedding_features({"instrument_class": "futures"})
        assert vec[0] == 1.0  # is_futures
        assert vec[1] == 0.0  # is_forex

    def test_forex_flag_set(self) -> None:
        vec = profile_embedding_features({"instrument_class": "forex"})
        assert vec[0] == 0.0
        assert vec[1] == 1.0

    def test_account_size_log_normalized(self) -> None:
        vec = profile_embedding_features({"account_size_usd": 50_000.0})
        # log10(50000) / 6 = 4.699 / 6 ≈ 0.783
        assert 0.75 <= vec[2] <= 0.82

    def test_topstep_style_flag(self) -> None:
        vec = profile_embedding_features({"prop_firm_style": "topstep_combine_dollar"})
        assert vec[6] == 1.0
        assert vec[7] == 0.0

    def test_the5ers_style_flag(self) -> None:
        vec = profile_embedding_features({"prop_firm_style": "the5ers_pct_phased"})
        assert vec[6] == 0.0
        assert vec[7] == 1.0

    def test_vector_is_deterministic(self) -> None:
        meta = {"instrument_class": "futures", "account_size_usd": 50_000.0}
        v1 = profile_embedding_features(meta)
        v2 = profile_embedding_features(meta)
        assert np.allclose(v1, v2)


class TestPadEmbeddingWithProfile:
    def test_appends_8_dims(self) -> None:
        base = np.zeros(64, dtype=np.float32)
        padded = pad_embedding_with_profile(base, {"instrument_class": "futures"})
        assert padded.shape == (72,)
        # First 64 dims are the base (zeros)
        assert np.all(padded[:64] == 0)
        # Next 8 are profile features
        assert padded[64] == 1.0  # is_futures
