# tests/test_feature_vector_macro.py
"""Tests for macro regime dimensions (59-63) in FeatureVectorBuilder."""

import numpy as np
import pytest


class TestMacroFeatureDims:
    def test_dxy_pct_change_dim59(self):
        """Dim 59 should encode DXY daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"dxy_pct_change": 0.5}  # 0.5% DXY move
        vec = builder.build(ctx)

        # Normalised to [0, 1] from range [-2%, 2%]
        assert 0.0 <= vec[59] <= 1.0
        assert vec[59] > 0.5  # Positive DXY change -> above midpoint

    def test_spx_pct_change_dim60(self):
        """Dim 60 should encode SPX daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"spx_pct_change": -1.0}  # 1% SPX drop
        vec = builder.build(ctx)

        assert 0.0 <= vec[60] <= 1.0
        assert vec[60] < 0.5  # Negative SPX change -> below midpoint

    def test_us10y_pct_change_dim61(self):
        """Dim 61 should encode US10Y daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"us10y_pct_change": 2.0}  # 2% yield spike
        vec = builder.build(ctx)

        assert 0.0 <= vec[61] <= 1.0
        assert vec[61] > 0.5  # Positive yield change -> above midpoint

    def test_regime_one_hot_dim62(self):
        """Dim 62 should encode regime as ordinal 0-1."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()

        # risk_on = 0.0, risk_off = 0.25, dollar_driven = 0.5,
        # inflation_fear = 0.75, mixed = 1.0
        for regime, expected_range in [
            ("risk_on", (0.0, 0.1)),
            ("risk_off", (0.2, 0.3)),
            ("dollar_driven", (0.4, 0.6)),
            ("inflation_fear", (0.7, 0.8)),
            ("mixed", (0.9, 1.0)),
        ]:
            vec = builder.build({"macro_regime": regime})
            assert expected_range[0] <= vec[62] <= expected_range[1], (
                f"regime={regime}, vec[62]={vec[62]}"
            )

    def test_event_proximity_dim63(self):
        """Dim 63 should encode event proximity (0=far, 1=imminent)."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()

        # Far from event
        vec_far = builder.build({"hours_to_event": 24.0})
        # Near event
        vec_near = builder.build({"hours_to_event": 1.0})

        assert vec_far[63] < vec_near[63]  # Nearer = higher value

    def test_no_macro_data_defaults_to_zero(self):
        """Missing macro fields should default to 0.0 (midpoint or zero)."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        vec = builder.build({})

        # Dims 59-63 should all be 0.0 when no macro data provided
        for dim in range(59, 64):
            assert vec[dim] == 0.0, f"dim {dim} = {vec[dim]}, expected 0.0"

    def test_overall_vector_still_64_dims(self):
        """Vector dimension must remain 64."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        vec = builder.build({
            "dxy_pct_change": 0.3,
            "spx_pct_change": -0.5,
            "us10y_pct_change": 1.2,
            "macro_regime": "risk_off",
            "hours_to_event": 3.0,
        })
        assert len(vec) == 64
        assert vec.min() >= 0.0
        assert vec.max() <= 1.0
