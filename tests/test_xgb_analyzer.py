"""Tests for XGBoost trade analysis pipeline."""

import numpy as np
import pandas as pd
import pytest


def _make_trade(r_multiple: float, context: dict) -> dict:
    """Helper: build a minimal trade dict."""
    return {"r_multiple": r_multiple, "context": context}


def _make_context(**overrides) -> dict:
    """Helper: build a context dict with sensible defaults."""
    base = {
        "cloud_direction_4h": 1.0,
        "cloud_direction_1h": 1.0,
        "tk_cross_15m": True,
        "chikou_confirmation": True,
        "adx_value": 30.0,
        "atr_value": 5.0,
        "session": "london",
        "confluence_score": 5,
        "signal_tier": "B",
        "direction": "long",
    }
    base.update(overrides)
    return base


class TestBuildTrainingData:
    def test_returns_correct_shapes(self):
        from src.discovery.xgb_analyzer import build_training_data

        trades = [
            _make_trade(1.5, _make_context()),
            _make_trade(-1.0, _make_context(adx_value=15.0)),
            _make_trade(2.0, _make_context(session="new_york")),
        ]
        X, y_binary, y_r = build_training_data(trades)

        assert isinstance(X, pd.DataFrame)
        assert X.shape == (3, 64)
        assert len(y_binary) == 3
        assert len(y_r) == 3

    def test_labels_match_r_multiple_sign(self):
        from src.discovery.xgb_analyzer import build_training_data

        trades = [
            _make_trade(1.5, _make_context()),
            _make_trade(-1.0, _make_context()),
            _make_trade(0.0, _make_context()),  # breakeven = loss
        ]
        _, y_binary, y_r = build_training_data(trades)

        assert y_binary.iloc[0] == 1  # win
        assert y_binary.iloc[1] == 0  # loss
        assert y_binary.iloc[2] == 0  # breakeven = loss
        assert y_r.iloc[0] == 1.5
        assert y_r.iloc[1] == -1.0

    def test_skips_trades_without_r_multiple(self):
        from src.discovery.xgb_analyzer import build_training_data

        trades = [
            _make_trade(1.5, _make_context()),
            {"context": _make_context()},  # no r_multiple
        ]
        X, y_binary, _ = build_training_data(trades)

        assert len(X) == 1

    def test_feature_values_in_zero_one_range(self):
        from src.discovery.xgb_analyzer import build_training_data

        trades = [_make_trade(1.0, _make_context()) for _ in range(5)]
        X, _, _ = build_training_data(trades)

        assert X.min().min() >= 0.0
        assert X.max().max() <= 1.0


class TestTrainClassifier:
    def _make_dataset(self, n=50, win_rate=0.37):
        """Generate a synthetic dataset with known class balance."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.random((n, 64)),
            columns=[f"f{i}" for i in range(64)],
        )
        n_wins = int(n * win_rate)
        y = pd.Series([1] * n_wins + [0] * (n - n_wins))
        y_r = pd.Series(
            list(rng.uniform(0.5, 3.0, n_wins)) + list(rng.uniform(-2.0, -0.1, n - n_wins))
        )
        return X, y, y_r

    def test_returns_fitted_model(self):
        from src.discovery.xgb_analyzer import train_classifier

        X, y, y_r = self._make_dataset()
        model = train_classifier(X, y, y_r)

        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_handles_small_dataset(self):
        from src.discovery.xgb_analyzer import train_classifier

        X, y, y_r = self._make_dataset(n=20)
        model = train_classifier(X, y, y_r)
        assert model is not None

    def test_uses_scale_pos_weight(self):
        from src.discovery.xgb_analyzer import train_classifier

        X, y, y_r = self._make_dataset(n=100, win_rate=0.30)
        model = train_classifier(X, y, y_r)

        # Model should be aware of class imbalance
        params = model.get_params()
        assert params["scale_pos_weight"] > 1.0
