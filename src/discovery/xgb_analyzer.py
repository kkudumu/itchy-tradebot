"""XGBoost + SHAP trade signal analyzer.

Trains on backtest trade logs (64-dim embeddings from FeatureVectorBuilder)
to discover which feature combinations predict wins vs losses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.learning.feature_vector import FeatureVectorBuilder

logger = logging.getLogger(__name__)

VECTOR_DIM = FeatureVectorBuilder.VECTOR_DIM

# Feature names mirror FeatureVectorBuilder layout (64 dims).
# Used as DataFrame column names for SHAP interpretability.
FEATURE_NAMES: List[str] = [
    # Ichimoku (0-14)
    "cloud_dir_4h", "cloud_dir_1h", "tk_cross_15m", "chikou_confirmed",
    "cloud_thick_4h", "cloud_thick_1h", "cloud_pos_15m", "cloud_pos_5m",
    "kijun_dist_5m", "tk_spread", "cloud_twist_15m", "cloud_breakout_15m",
    "ichi_r12", "ichi_r13", "ichi_r14",
    # Trend (15-24)
    "adx_value", "adx_trending", "rsi_value", "rsi_overbought", "rsi_oversold",
    "bb_width_pct", "bb_squeeze", "atr_norm", "trend_r23", "trend_r24",
    # Zone (25-34)
    "sr_distance", "zone_confluence", "in_supply", "in_demand", "at_pivot",
    "zone_strength", "zone_freshness", "zone_r32", "zone_r33", "zone_r34",
    # Session (35-44)
    "sess_london", "sess_ny", "sess_asian", "sess_overlap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "time_r43", "time_r44",
    # Signal (45-54)
    "confluence_score", "signal_tier", "direction", "risk_pct", "atr_stop_dist",
    "sig_r50", "sig_r51", "sig_r52", "sig_r53", "sig_r54",
    # Regime (55-63)
    "trend_strength", "vol_regime", "spread_norm", "daily_range_vs_atr",
    "regime_r59", "regime_r60", "regime_r61", "regime_r62", "regime_r63",
]

assert len(FEATURE_NAMES) == VECTOR_DIM


def build_training_data(
    trades: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert backtest trade logs into XGBoost-ready training data.

    Parameters
    ----------
    trades:
        List of trade dicts. Each must have "r_multiple" (float) and
        "context" (dict compatible with FeatureVectorBuilder.build()).

    Returns
    -------
    (X, y_binary, y_r):
        X: DataFrame (n_trades, 64) with named feature columns.
        y_binary: Series of 0/1 labels (loss/win).
        y_r: Series of raw R-multiples.
    """
    builder = FeatureVectorBuilder()

    rows: List[np.ndarray] = []
    labels: List[int] = []
    r_multiples: List[float] = []

    for trade in trades:
        r = trade.get("r_multiple")
        if r is None:
            continue
        r = float(r)
        context = trade.get("context") or {}

        vec = builder.build(context)
        rows.append(vec)
        labels.append(1 if r > 0 else 0)
        r_multiples.append(r)

    X = pd.DataFrame(np.array(rows) if rows else np.empty((0, VECTOR_DIM)),
                      columns=FEATURE_NAMES)
    y_binary = pd.Series(labels, name="win")
    y_r = pd.Series(r_multiples, name="r_multiple")

    logger.info(
        "Built training data: %d trades, %.1f%% wins, avg_r=%.3f",
        len(X), y_binary.mean() * 100 if len(y_binary) else 0, y_r.mean() if len(y_r) else 0,
    )
    return X, y_binary, y_r


# ---------------------------------------------------------------------------
# XGBoost classifier
# ---------------------------------------------------------------------------

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    y_r: pd.Series,
) -> xgb.XGBClassifier:
    """Train XGBoost on trade outcomes with class imbalance handling.

    Uses scale_pos_weight for 37% win rate imbalance, R-weighted samples
    to prioritize big winners, shallow trees to prevent overfitting on
    small datasets (~100 trades), and TimeSeriesSplit CV.

    Parameters
    ----------
    X: Feature matrix (n_trades, 64).
    y: Binary labels (0=loss, 1=win).
    y_r: R-multiples for sample weighting.

    Returns
    -------
    Fitted XGBClassifier.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Sample weights: up-weight big winners so model learns what makes
    # a GREAT trade, not just any win.
    sample_weight = np.ones(len(y))
    win_mask = y == 1
    sample_weight[win_mask] = np.clip(y_r[win_mask].values, 1.0, 5.0)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # TimeSeriesSplit CV: never peek at future trades.
    # Used only for early-stopping calibration; the final model is
    # trained on all data with a fresh instance to avoid class mismatch.
    best_n_estimators = 200
    n_splits = min(3, max(2, len(X) // 15))
    if len(X) >= 30:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in tscv.split(X):
            # Skip folds where train or val has only one class
            if len(set(y.iloc[train_idx])) < 2 or len(set(y.iloc[val_idx])) < 2:
                continue
            cv_model = xgb.XGBClassifier(**model.get_params())
            cv_model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                sample_weight=sample_weight[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            best_n_estimators = cv_model.best_iteration + 1 if hasattr(cv_model, "best_iteration") and cv_model.best_iteration > 0 else best_n_estimators

    # Final fit on all data with fresh model
    model = xgb.XGBClassifier(**{**model.get_params(), "n_estimators": best_n_estimators})
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model
