# tests/test_regime_classifier.py
"""Tests for daily macro regime classification."""

import numpy as np
import pandas as pd
import pytest


class TestRegimeLabel:
    def test_all_labels_are_strings(self):
        from src.macro.regime_classifier import RegimeLabel

        for label in RegimeLabel:
            assert isinstance(label.value, str)

    def test_five_regimes_exist(self):
        from src.macro.regime_classifier import RegimeLabel

        assert len(RegimeLabel) == 5


class TestClassifySingleDay:
    def test_risk_on(self):
        """SPX up, DXY down, US10Y stable -> risk_on."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=-0.5, spx_pct=1.2, us10y_pct=0.01
        )
        assert regime == RegimeLabel.RISK_ON

    def test_risk_off(self):
        """SPX down, DXY up, US10Y down (flight to safety) -> risk_off."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.8, spx_pct=-1.5, us10y_pct=-0.3
        )
        assert regime == RegimeLabel.RISK_OFF

    def test_dollar_driven(self):
        """DXY strong move, SPX mixed -> dollar_driven."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=1.2, spx_pct=0.1, us10y_pct=0.1
        )
        assert regime == RegimeLabel.DOLLAR_DRIVEN

    def test_inflation_fear(self):
        """US10Y spikes, SPX down -> inflation_fear."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.3, spx_pct=-0.8, us10y_pct=3.0
        )
        assert regime == RegimeLabel.INFLATION_FEAR

    def test_mixed(self):
        """Small moves in all -> mixed."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.05, spx_pct=-0.05, us10y_pct=0.02
        )
        assert regime == RegimeLabel.MIXED


class TestRegimeClassifier:
    def _make_panel(self, n=20, seed=42):
        """Generate a synthetic macro panel."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
        return pd.DataFrame({
            "dxy_close": 103.0 + np.cumsum(rng.normal(0, 0.3, n)),
            "dxy_pct_change": rng.normal(0, 0.5, n),
            "spx_close": 4800.0 + np.cumsum(rng.normal(0, 20, n)),
            "spx_pct_change": rng.normal(0, 0.8, n),
            "us10y_close": 4.25 + np.cumsum(rng.normal(0, 0.05, n)),
            "us10y_pct_change": rng.normal(0, 1.0, n),
        }, index=dates)

    def test_classify_panel_returns_series(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(panel)
        assert regimes.name == "regime"

    def test_all_values_are_valid_labels(self):
        from src.macro.regime_classifier import RegimeClassifier, RegimeLabel

        classifier = RegimeClassifier()
        panel = self._make_panel(n=50)
        regimes = classifier.classify(panel)

        valid_labels = {label.value for label in RegimeLabel}
        for regime in regimes:
            assert regime in valid_labels

    def test_get_regime_for_date(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        target_date = panel.index[5]
        regime = classifier.get_regime_for_date(regimes, target_date)
        assert isinstance(regime, str)

    def test_get_regime_for_missing_date_returns_mixed(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        missing_date = pd.Timestamp("2020-01-01", tz="UTC")
        regime = classifier.get_regime_for_date(regimes, missing_date)
        assert regime == "mixed"
