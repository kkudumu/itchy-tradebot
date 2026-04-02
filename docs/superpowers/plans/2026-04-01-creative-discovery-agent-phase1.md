# Creative Pattern Discovery Agent (Phase 1: Discovery Engine Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strategy-agnostic discovery engine that analyzes backtest trade logs via XGBoost/SHAP to find feature interactions that predict trade success, generates hypotheses, and persists validated findings to a JSON knowledge base.

**Architecture:** Post-backtest analysis loop: after each 30-day window backtest completes, the engine trains XGBoost on accumulated trades (64-dim embeddings), runs SHAP interaction analysis to discover feature pairs that predict wins/losses, Claude reasons about the findings to form hypotheses, and validated hypotheses are absorbed into config. Discoveries persist in `reports/agent_knowledge/` as JSON files. Strategy-scoped via existing `strategy_profiles` in edges.yaml.

**Tech Stack:** XGBoost, SHAP, scikit-learn (TimeSeriesSplit), existing FeatureVectorBuilder (64-dim), Claude Code CLI (`claude -p`) / Codex CLI fallback, JSON file storage.

**Phases overview (this is Phase 1 of 5):**
- **Phase 1 (this plan):** XGBoost/SHAP analysis + hypothesis loop + knowledge base
- Phase 2: PatternPy chart patterns + selective screenshots + Claude visual analysis
- Phase 3: Macro regime (DXY synthesis, SPX, US10Y, econ calendar)
- Phase 4: LLM-generated EdgeFilter code with AST/test/backtest safety
- Phase 5: Full orchestrator tying phases 1-4 into the 30-day rolling challenge loop

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/discovery/__init__.py` | Package init, public exports |
| `src/discovery/xgb_analyzer.py` | XGBoost training + SHAP analysis on trade embeddings |
| `src/discovery/hypothesis_engine.py` | Claude-powered hypothesis generation from SHAP insights |
| `src/discovery/knowledge_base.py` | JSON persistence for hypotheses, rules, validated edges |
| `src/discovery/rule_applier.py` | Converts SHAP rules into config overrides for the next backtest window |
| `src/discovery/runner.py` | Orchestrates: accumulate trades -> analyze -> hypothesize -> apply |
| `tests/test_xgb_analyzer.py` | Tests for training data builder, XGBoost, SHAP pipeline |
| `tests/test_hypothesis_engine.py` | Tests for hypothesis formatting, prompt construction |
| `tests/test_knowledge_base.py` | Tests for JSON read/write/query |
| `tests/test_rule_applier.py` | Tests for SHAP rules -> config override conversion |
| `tests/test_discovery_runner.py` | Integration test for the full loop |

---

### Task 1: Training Data Builder

**Files:**
- Create: `src/discovery/__init__.py`
- Create: `src/discovery/xgb_analyzer.py`
- Test: `tests/test_xgb_analyzer.py`

- [ ] **Step 1: Write failing test for build_training_data**

```python
# tests/test_xgb_analyzer.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xgb_analyzer.py::TestBuildTrainingData -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.discovery'`

- [ ] **Step 3: Create package init and build_training_data**

```python
# src/discovery/__init__.py
"""Creative Pattern Discovery Agent.

Strategy-agnostic edge discovery via XGBoost/SHAP analysis,
hypothesis generation, and validated rule absorption.
"""

from src.discovery.xgb_analyzer import build_training_data

__all__ = ["build_training_data"]
```

```python
# src/discovery/xgb_analyzer.py
"""XGBoost + SHAP trade signal analyzer.

Trains on backtest trade logs (64-dim embeddings from FeatureVectorBuilder)
to discover which feature combinations predict wins vs losses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.learning.feature_vector import FeatureVectorBuilder, VECTOR_DIM

logger = logging.getLogger(__name__)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xgb_analyzer.py::TestBuildTrainingData -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/__init__.py src/discovery/xgb_analyzer.py tests/test_xgb_analyzer.py
git commit -m "feat: add discovery engine training data builder (Task 1)"
```

---

### Task 2: XGBoost Classifier with Imbalance Handling

**Files:**
- Modify: `src/discovery/xgb_analyzer.py`
- Test: `tests/test_xgb_analyzer.py`

- [ ] **Step 1: Write failing test for train_classifier**

```python
# Append to tests/test_xgb_analyzer.py

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xgb_analyzer.py::TestTrainClassifier -v`
Expected: FAIL with `ImportError: cannot import name 'train_classifier'`

- [ ] **Step 3: Implement train_classifier**

Append to `src/discovery/xgb_analyzer.py`:

```python
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

    # TimeSeriesSplit: never peek at future trades
    n_splits = min(3, max(2, len(X) // 15))
    if len(X) >= 30:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in tscv.split(X):
            model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                sample_weight=sample_weight[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )

    # Final fit on all data
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install xgboost shap scikit-learn && pytest tests/test_xgb_analyzer.py::TestTrainClassifier -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/xgb_analyzer.py tests/test_xgb_analyzer.py
git commit -m "feat: add XGBoost classifier with imbalance handling (Task 2)"
```

---

### Task 3: SHAP Interaction Analysis

**Files:**
- Modify: `src/discovery/xgb_analyzer.py`
- Test: `tests/test_xgb_analyzer.py`

- [ ] **Step 1: Write failing test for run_shap_analysis**

```python
# Append to tests/test_xgb_analyzer.py
from dataclasses import fields as dc_fields


class TestSHAPAnalysis:
    def _train_model(self, n=60):
        from src.discovery.xgb_analyzer import train_classifier
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((n, 64)), columns=[f"f{i}" for i in range(64)])
        y = pd.Series(rng.choice([0, 1], n, p=[0.63, 0.37]))
        y_r = pd.Series(np.where(y == 1, rng.uniform(0.5, 3.0, n), rng.uniform(-2.0, -0.1, n)))
        model = train_classifier(X, y, y_r)
        return model, X, y, y_r

    def test_returns_shap_insight(self):
        from src.discovery.xgb_analyzer import run_shap_analysis, SHAPInsight

        model, X, y, y_r = self._train_model()
        insight = run_shap_analysis(model, X, y, y_r)

        assert isinstance(insight, SHAPInsight)
        assert len(insight.feature_importance) > 0
        assert isinstance(insight.top_interactions, list)
        assert isinstance(insight.actionable_rules, list)

    def test_feature_importance_sorted_descending(self):
        from src.discovery.xgb_analyzer import run_shap_analysis

        model, X, y, y_r = self._train_model()
        insight = run_shap_analysis(model, X, y, y_r)

        values = list(insight.feature_importance.values())
        assert values == sorted(values, reverse=True)

    def test_rules_have_required_keys(self):
        from src.discovery.xgb_analyzer import run_shap_analysis

        model, X, y, y_r = self._train_model(n=100)
        insight = run_shap_analysis(model, X, y, y_r)

        required_keys = {
            "feature_a", "feature_b", "condition", "quadrant_win_rate",
            "baseline_win_rate", "n_trades", "lift", "recommendation",
        }
        for rule in insight.actionable_rules:
            assert required_keys.issubset(rule.keys()), f"Missing keys: {required_keys - rule.keys()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xgb_analyzer.py::TestSHAPAnalysis -v`
Expected: FAIL with `ImportError: cannot import name 'run_shap_analysis'`

- [ ] **Step 3: Implement SHAPInsight dataclass and run_shap_analysis**

Append to `src/discovery/xgb_analyzer.py`:

```python
import shap
from dataclasses import dataclass, field


@dataclass
class SHAPInsight:
    """Results of SHAP analysis on the trade classifier."""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_interactions: List[Tuple[Tuple[str, str], float]] = field(default_factory=list)
    actionable_rules: List[Dict[str, Any]] = field(default_factory=list)


def run_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    y_r: pd.Series,
    top_k_features: int = 15,
    top_k_interactions: int = 10,
    min_trades_per_bucket: int = 10,
) -> SHAPInsight:
    """Run SHAP feature importance + interaction analysis.

    1. Computes per-feature mean |SHAP| (global importance).
    2. Computes SHAP interaction values to find feature PAIRS.
    3. Splits data into quadrants per interaction pair and extracts
       actionable rules where quadrant win rate deviates from baseline.

    Returns SHAPInsight with importance, interactions, and rules.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Feature importance (sorted descending)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(X.columns.tolist(), mean_abs.tolist()))
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k_features]
    )

    # Interaction values
    shap_interaction = explainer.shap_interaction_values(X)
    mean_inter = np.abs(shap_interaction).mean(axis=0)
    np.fill_diagonal(mean_inter, 0)

    cols = X.columns.tolist()
    pairs: List[Tuple[Tuple[str, str], float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            strength = float(mean_inter[i, j] + mean_inter[j, i])
            if strength > 0.001:
                pairs.append(((cols[i], cols[j]), strength))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_interactions = pairs[:top_k_interactions]

    # Extract rules from top interactions
    baseline_wr = float(y.mean())
    rules: List[Dict[str, Any]] = []

    for (feat_a, feat_b), strength in top_interactions:
        col_a, col_b = X[feat_a], X[feat_b]
        thresh_a = 0.5 if col_a.nunique() <= 3 else float(col_a.median())
        thresh_b = 0.5 if col_b.nunique() <= 3 else float(col_b.median())

        quadrants = {
            f"{feat_a}>={thresh_a:.2f} AND {feat_b}>={thresh_b:.2f}": (col_a >= thresh_a) & (col_b >= thresh_b),
            f"{feat_a}>={thresh_a:.2f} AND {feat_b}<{thresh_b:.2f}": (col_a >= thresh_a) & (col_b < thresh_b),
            f"{feat_a}<{thresh_a:.2f} AND {feat_b}>={thresh_b:.2f}": (col_a < thresh_a) & (col_b >= thresh_b),
            f"{feat_a}<{thresh_a:.2f} AND {feat_b}<{thresh_b:.2f}": (col_a < thresh_a) & (col_b < thresh_b),
        }

        for cond_str, mask in quadrants.items():
            n_in = int(mask.sum())
            if n_in < min_trades_per_bucket:
                continue
            quad_wr = float(y[mask].mean())
            quad_avg_r = float(y_r[mask].mean())
            if abs(quad_wr - baseline_wr) < 0.05:
                continue
            lift = quad_wr / baseline_wr if baseline_wr > 0 else 0.0

            if quad_wr >= baseline_wr * 1.3 and n_in >= 15:
                rec = "strong_filter"
            elif quad_wr >= baseline_wr * 1.15 or quad_wr <= baseline_wr * 0.7:
                rec = "weak_filter"
            else:
                rec = "informational"

            rules.append({
                "feature_a": feat_a,
                "feature_b": feat_b,
                "condition": cond_str,
                "quadrant_win_rate": round(quad_wr, 4),
                "quadrant_avg_r": round(quad_avg_r, 4),
                "baseline_win_rate": round(baseline_wr, 4),
                "n_trades": n_in,
                "lift": round(lift, 3),
                "interaction_strength": round(strength, 4),
                "recommendation": rec,
            })

    rules.sort(key=lambda r: r["lift"], reverse=True)

    return SHAPInsight(
        feature_importance=sorted_importance,
        top_interactions=top_interactions,
        actionable_rules=rules,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xgb_analyzer.py::TestSHAPAnalysis -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/xgb_analyzer.py tests/test_xgb_analyzer.py
git commit -m "feat: add SHAP interaction analysis with rule extraction (Task 3)"
```

---

### Task 4: JSON Knowledge Base

**Files:**
- Create: `src/discovery/knowledge_base.py`
- Test: `tests/test_knowledge_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge_base.py
"""Tests for the discovery agent JSON knowledge base."""

import json
import tempfile
from pathlib import Path

import pytest


class TestKnowledgeBase:
    def _make_kb(self, tmp_path):
        from src.discovery.knowledge_base import KnowledgeBase
        return KnowledgeBase(base_dir=str(tmp_path / "agent_knowledge"))

    def test_save_and_load_hypothesis(self, tmp_path):
        kb = self._make_kb(tmp_path)
        hyp = {
            "id": "hyp_001",
            "description": "London session + ADX > 30 improves win rate",
            "source": "shap_interaction",
            "strategy": "sss",
            "status": "proposed",
            "evidence": {"feature_a": "sess_london", "feature_b": "adx_trending", "lift": 1.35},
        }
        kb.save_hypothesis(hyp)
        loaded = kb.load_hypothesis("hyp_001")
        assert loaded["description"] == hyp["description"]
        assert loaded["status"] == "proposed"

    def test_update_hypothesis_status(self, tmp_path):
        kb = self._make_kb(tmp_path)
        hyp = {"id": "hyp_002", "description": "test", "status": "proposed"}
        kb.save_hypothesis(hyp)
        kb.update_hypothesis_status("hyp_002", "validated", metrics={"win_rate_delta": 0.08})
        loaded = kb.load_hypothesis("hyp_002")
        assert loaded["status"] == "validated"
        assert loaded["validation_metrics"]["win_rate_delta"] == 0.08

    def test_save_and_load_shap_rules(self, tmp_path):
        kb = self._make_kb(tmp_path)
        rules = [
            {"condition": "adx>0.5 AND sess_london>=0.5", "lift": 1.3, "recommendation": "strong_filter"},
        ]
        kb.save_shap_rules(rules, window_id="w_001")
        loaded = kb.load_shap_rules("w_001")
        assert len(loaded) == 1
        assert loaded[0]["lift"] == 1.3

    def test_list_hypotheses_by_status(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_hypothesis({"id": "h1", "status": "proposed"})
        kb.save_hypothesis({"id": "h2", "status": "validated"})
        kb.save_hypothesis({"id": "h3", "status": "proposed"})

        proposed = kb.list_hypotheses(status="proposed")
        assert len(proposed) == 2
        validated = kb.list_hypotheses(status="validated")
        assert len(validated) == 1

    def test_get_accumulated_trades(self, tmp_path):
        kb = self._make_kb(tmp_path)
        trades_w1 = [{"r_multiple": 1.0}, {"r_multiple": -0.5}]
        trades_w2 = [{"r_multiple": 2.0}]
        kb.save_window_trades(trades_w1, window_id="w_001")
        kb.save_window_trades(trades_w2, window_id="w_002")
        all_trades = kb.get_accumulated_trades()
        assert len(all_trades) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement KnowledgeBase**

```python
# src/discovery/knowledge_base.py
"""JSON-based knowledge base for the discovery agent.

Persists hypotheses, SHAP rules, and accumulated trade data to
reports/agent_knowledge/ as inspectable, git-trackable JSON files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """File-based knowledge store for discovery agent learnings.

    Directory layout:
        base_dir/
            hypotheses/
                hyp_001.json
                hyp_002.json
            shap_rules/
                w_001.json
            window_trades/
                w_001.json
            index.json
    """

    def __init__(self, base_dir: str = "reports/agent_knowledge") -> None:
        self._base = Path(base_dir)
        self._hyp_dir = self._base / "hypotheses"
        self._rules_dir = self._base / "shap_rules"
        self._trades_dir = self._base / "window_trades"

        for d in (self._hyp_dir, self._rules_dir, self._trades_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- Hypotheses --

    def save_hypothesis(self, hypothesis: Dict[str, Any]) -> Path:
        hyp_id = hypothesis["id"]
        hypothesis.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        path = self._hyp_dir / f"{hyp_id}.json"
        path.write_text(json.dumps(hypothesis, indent=2, default=str), encoding="utf-8")
        return path

    def load_hypothesis(self, hyp_id: str) -> Dict[str, Any]:
        path = self._hyp_dir / f"{hyp_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def update_hypothesis_status(
        self, hyp_id: str, status: str, metrics: Optional[Dict] = None
    ) -> None:
        hyp = self.load_hypothesis(hyp_id)
        hyp["status"] = status
        hyp["updated_at"] = datetime.now(timezone.utc).isoformat()
        if metrics:
            hyp["validation_metrics"] = metrics
        self.save_hypothesis(hyp)

    def list_hypotheses(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self._hyp_dir.glob("*.json")):
            hyp = json.loads(p.read_text(encoding="utf-8"))
            if status is None or hyp.get("status") == status:
                results.append(hyp)
        return results

    # -- SHAP Rules --

    def save_shap_rules(self, rules: List[Dict[str, Any]], window_id: str) -> Path:
        path = self._rules_dir / f"{window_id}.json"
        path.write_text(json.dumps(rules, indent=2, default=str), encoding="utf-8")
        return path

    def load_shap_rules(self, window_id: str) -> List[Dict[str, Any]]:
        path = self._rules_dir / f"{window_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # -- Window Trades (for accumulation) --

    def save_window_trades(self, trades: List[Dict[str, Any]], window_id: str) -> Path:
        path = self._trades_dir / f"{window_id}.json"
        path.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        return path

    def get_accumulated_trades(self) -> List[Dict[str, Any]]:
        all_trades: List[Dict[str, Any]] = []
        for p in sorted(self._trades_dir.glob("*.json")):
            trades = json.loads(p.read_text(encoding="utf-8"))
            all_trades.extend(trades)
        return all_trades
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_base.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/knowledge_base.py tests/test_knowledge_base.py
git commit -m "feat: add JSON knowledge base for discovery agent (Task 4)"
```

---

### Task 5: Hypothesis Engine (Claude-Powered Reasoning)

**Files:**
- Create: `src/discovery/hypothesis_engine.py`
- Test: `tests/test_hypothesis_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hypothesis_engine.py
"""Tests for hypothesis generation from SHAP insights."""

import pytest


class TestBuildPrompt:
    def test_prompt_contains_shap_rules(self):
        from src.discovery.hypothesis_engine import build_hypothesis_prompt
        from src.discovery.xgb_analyzer import SHAPInsight

        insight = SHAPInsight(
            feature_importance={"adx_value": 0.15, "sess_london": 0.12},
            top_interactions=[(("adx_value", "sess_london"), 0.08)],
            actionable_rules=[{
                "feature_a": "adx_value",
                "feature_b": "sess_london",
                "condition": "adx_value>=0.50 AND sess_london>=0.50",
                "quadrant_win_rate": 0.55,
                "baseline_win_rate": 0.37,
                "n_trades": 25,
                "lift": 1.486,
                "recommendation": "strong_filter",
            }],
        )
        prompt = build_hypothesis_prompt(insight, strategy_name="sss", window_id="w_003")

        assert "adx_value" in prompt
        assert "sess_london" in prompt
        assert "strong_filter" in prompt
        assert "55.0%" in prompt or "0.55" in prompt
        assert "sss" in prompt

    def test_prompt_includes_strategy_context(self):
        from src.discovery.hypothesis_engine import build_hypothesis_prompt
        from src.discovery.xgb_analyzer import SHAPInsight

        insight = SHAPInsight()
        prompt = build_hypothesis_prompt(insight, strategy_name="ichimoku", window_id="w_001")
        assert "ichimoku" in prompt


class TestParseHypotheses:
    def test_parses_json_block(self):
        from src.discovery.hypothesis_engine import parse_hypotheses_response

        response = '''Here's my analysis:

```json
[
  {
    "description": "ADX trending + London session improves SSS win rate",
    "config_change": {"strategies": {"sss": {"min_confluence_score": 3}}},
    "expected_improvement": "Win rate +8% based on SHAP lift 1.49",
    "confidence": "high"
  }
]
```

This is based on the interaction between ADX and session.'''

        hypotheses = parse_hypotheses_response(response, strategy_name="sss")
        assert len(hypotheses) == 1
        assert hypotheses[0]["description"] == "ADX trending + London session improves SSS win rate"
        assert hypotheses[0]["config_change"]["strategies"]["sss"]["min_confluence_score"] == 3
        assert hypotheses[0]["strategy"] == "sss"
        assert "id" in hypotheses[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement hypothesis_engine**

```python
# src/discovery/hypothesis_engine.py
"""Claude-powered hypothesis generation from SHAP insights.

Builds a structured prompt from SHAPInsight, sends it to Claude Code CLI
(or Codex CLI as fallback), and parses the response into hypothesis dicts.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from src.discovery.xgb_analyzer import SHAPInsight

logger = logging.getLogger(__name__)


def build_hypothesis_prompt(
    insight: SHAPInsight,
    strategy_name: str,
    window_id: str,
    extra_context: str = "",
) -> str:
    """Build a structured prompt for Claude to generate hypotheses.

    The prompt includes SHAP feature importance, top interactions,
    actionable rules with win rates, and asks Claude to propose
    concrete config changes as testable hypotheses.
    """
    # Format feature importance
    feat_lines = ""
    for name, imp in insight.feature_importance.items():
        feat_lines += f"  {name}: {imp:.4f}\n"

    # Format interactions
    inter_lines = ""
    for (f_a, f_b), strength in insight.top_interactions:
        inter_lines += f"  {f_a} x {f_b}: {strength:.4f}\n"

    # Format rules
    rule_lines = ""
    for r in insight.actionable_rules:
        wr_pct = r['quadrant_win_rate'] * 100
        base_pct = r['baseline_win_rate'] * 100
        rule_lines += (
            f"  [{r['recommendation'].upper()}] {r['condition']}\n"
            f"    Win rate: {wr_pct:.1f}% vs baseline {base_pct:.1f}% "
            f"(lift {r['lift']:.2f}x, n={r['n_trades']})\n"
        )

    prompt = f"""You are a quantitative trading analyst reviewing SHAP analysis results
for the {strategy_name} strategy on XAU/USD (gold). Window: {window_id}.

## Top Features by Importance (mean |SHAP|)
{feat_lines or '  (no significant features)'}

## Top Feature Interactions (pairwise SHAP)
{inter_lines or '  (no significant interactions)'}

## Actionable Rules Discovered
{rule_lines or '  (no actionable rules found)'}

{extra_context}

## Your Task

Based on the SHAP analysis above, propose 1-3 testable hypotheses as concrete
config changes for the {strategy_name} strategy. Each hypothesis should:

1. Explain WHY the feature interaction matters for gold trading
2. Propose a specific config change (edge filter threshold, strategy parameter, etc.)
3. Estimate the expected improvement based on the SHAP evidence

Respond with a JSON array of hypothesis objects:

```json
[
  {{
    "description": "Human-readable explanation of the hypothesis",
    "config_change": {{"strategies": {{"{strategy_name}": {{"param": "value"}}}}}},
    "expected_improvement": "What we expect to see and why",
    "confidence": "high|medium|low"
  }}
]
```

Be creative but grounded in the data. Only propose changes supported by the SHAP evidence."""

    return prompt


def parse_hypotheses_response(
    response: str,
    strategy_name: str,
) -> List[Dict[str, Any]]:
    """Parse Claude's response into structured hypothesis dicts.

    Extracts the JSON array from a ```json code block in the response.
    Adds id, strategy, and status fields to each hypothesis.
    """
    # Extract JSON block
    match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if not match:
        # Try raw JSON array
        match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
        if not match:
            logger.warning("Could not parse hypotheses from response")
            return []

    try:
        raw = match.group(1) if match.lastindex else match.group(0)
        hypotheses = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return []

    if not isinstance(hypotheses, list):
        hypotheses = [hypotheses]

    for hyp in hypotheses:
        hyp["id"] = f"hyp_{uuid.uuid4().hex[:8]}"
        hyp["strategy"] = strategy_name
        hyp["status"] = "proposed"

    return hypotheses


def generate_hypotheses(
    insight: SHAPInsight,
    strategy_name: str,
    window_id: str,
    extra_context: str = "",
    cli_command: Optional[List[str]] = None,
    timeout: int = 300,
) -> List[Dict[str, Any]]:
    """Generate hypotheses by invoking Claude Code CLI.

    Parameters
    ----------
    insight: SHAP analysis results.
    strategy_name: Which strategy to target.
    window_id: Current optimization window identifier.
    extra_context: Additional context (e.g., prior learnings).
    cli_command: CLI command list. Defaults to ["claude", "-p"].
    timeout: Subprocess timeout in seconds.

    Returns
    -------
    List of hypothesis dicts with id, description, config_change, etc.
    """
    prompt = build_hypothesis_prompt(insight, strategy_name, window_id, extra_context)

    cmd = cli_command or ["claude", "-p"]
    logger.info("Generating hypotheses via %s for %s window %s", cmd[0], strategy_name, window_id)

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        response = result.stdout
        if not response.strip():
            logger.warning("Empty response from CLI")
            return []
    except subprocess.TimeoutExpired:
        logger.error("CLI timed out after %ds", timeout)
        return []
    except FileNotFoundError:
        logger.error("CLI command not found: %s", cmd)
        return []

    hypotheses = parse_hypotheses_response(response, strategy_name)
    logger.info("Generated %d hypotheses", len(hypotheses))
    return hypotheses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hypothesis_engine.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/hypothesis_engine.py tests/test_hypothesis_engine.py
git commit -m "feat: add Claude-powered hypothesis engine (Task 5)"
```

---

### Task 6: Rule Applier (SHAP Rules -> Config Overrides)

**Files:**
- Create: `src/discovery/rule_applier.py`
- Test: `tests/test_rule_applier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rule_applier.py
"""Tests for converting SHAP rules into config overrides."""

import pytest


class TestRuleApplier:
    def test_strong_boost_rule_lowers_confluence(self):
        from src.discovery.rule_applier import apply_rules_to_config

        rules = [{
            "feature_a": "adx_value",
            "feature_b": "sess_london",
            "condition": "adx_value>=0.50 AND sess_london>=0.50",
            "quadrant_win_rate": 0.55,
            "baseline_win_rate": 0.37,
            "lift": 1.486,
            "recommendation": "strong_filter",
            "n_trades": 25,
        }]
        base_config = {"strategies": {"sss": {"min_confluence_score": 4}}}
        new_config, changes = apply_rules_to_config(rules, base_config, strategy="sss")

        assert len(changes) > 0

    def test_no_rules_returns_unchanged_config(self):
        from src.discovery.rule_applier import apply_rules_to_config

        base_config = {"strategies": {"sss": {"min_confluence_score": 4}}}
        new_config, changes = apply_rules_to_config([], base_config, strategy="sss")

        assert new_config == base_config
        assert len(changes) == 0

    def test_hypothesis_config_change_applied(self):
        from src.discovery.rule_applier import apply_hypothesis_to_config

        hypothesis = {
            "config_change": {"strategies": {"sss": {"min_confluence_score": 3}}},
        }
        base_config = {"strategies": {"sss": {"min_confluence_score": 4, "entry_mode": "cbc_only"}}}
        new_config = apply_hypothesis_to_config(hypothesis, base_config)

        assert new_config["strategies"]["sss"]["min_confluence_score"] == 3
        assert new_config["strategies"]["sss"]["entry_mode"] == "cbc_only"  # preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rule_applier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rule_applier**

```python
# src/discovery/rule_applier.py
"""Converts SHAP rules and hypotheses into config overrides.

Takes actionable rules from the SHAP analysis or hypotheses from Claude
and produces modified config dicts for the next backtest window.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def apply_rules_to_config(
    rules: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    strategy: str,
    max_changes: int = 2,
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply SHAP-derived rules as config modifications.

    Only strong_filter rules are applied automatically. Each rule
    generates a config change based on the feature interaction pattern.

    Parameters
    ----------
    rules: Actionable rules from SHAPInsight.
    base_config: Current strategy config dict.
    strategy: Strategy name (e.g., "sss").
    max_changes: Maximum config changes to apply per iteration.

    Returns
    -------
    (new_config, changes): Modified config and list of change descriptions.
    """
    new_config = copy.deepcopy(base_config)
    changes: List[str] = []

    strong_rules = [r for r in rules if r.get("recommendation") == "strong_filter"]
    if not strong_rules:
        return new_config, changes

    strat_cfg = new_config.get("strategies", {}).get(strategy, {})

    for rule in strong_rules[:max_changes]:
        change = _rule_to_config_change(rule, strat_cfg, strategy)
        if change:
            param, old_val, new_val, desc = change
            strat_cfg[param] = new_val
            changes.append(desc)
            logger.info("Applied rule: %s", desc)

    new_config.setdefault("strategies", {})[strategy] = strat_cfg
    return new_config, changes


def _rule_to_config_change(
    rule: Dict[str, Any],
    strat_cfg: Dict[str, Any],
    strategy: str,
) -> Tuple[str, Any, Any, str] | None:
    """Map a SHAP rule to a concrete config parameter change.

    Returns (param_name, old_value, new_value, description) or None.
    """
    feat_a = rule.get("feature_a", "")
    feat_b = rule.get("feature_b", "")
    lift = rule.get("lift", 1.0)

    # High-lift rules where confluence is involved -> adjust min_confluence_score
    if "confluence" in feat_a or "confluence" in feat_b:
        old = strat_cfg.get("min_confluence_score", 4)
        if lift > 1.2:
            new = max(1, old - 1)  # lower threshold since these conditions boost WR
        else:
            new = min(6, old + 1)  # raise threshold to filter low-quality
        if new != old:
            return (
                "min_confluence_score", old, new,
                f"Adjusted min_confluence_score {old}->{new} "
                f"(SHAP: {feat_a} x {feat_b}, lift={lift:.2f})"
            )

    # ADX interaction -> adjust swing parameters
    if "adx" in feat_a or "adx" in feat_b:
        old = strat_cfg.get("min_swing_pips", 1.0)
        if lift > 1.3:
            new = max(0.5, old - 0.5)
        else:
            new = min(5.0, old + 0.5)
        if new != old:
            return (
                "min_swing_pips", old, new,
                f"Adjusted min_swing_pips {old}->{new} "
                f"(SHAP: {feat_a} x {feat_b}, lift={lift:.2f})"
            )

    return None


def apply_hypothesis_to_config(
    hypothesis: Dict[str, Any],
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a hypothesis's config_change on top of base_config.

    Deep-merges at the strategy level so unmodified params are preserved.
    """
    new_config = copy.deepcopy(base_config)
    config_change = hypothesis.get("config_change", {})

    for key, val in config_change.items():
        if key == "strategies" and isinstance(val, dict):
            for strat_name, strat_params in val.items():
                if strat_name in new_config.get("strategies", {}) and isinstance(strat_params, dict):
                    new_config["strategies"][strat_name] = {
                        **new_config["strategies"][strat_name],
                        **strat_params,
                    }
                else:
                    new_config.setdefault("strategies", {})[strat_name] = strat_params
        else:
            new_config[key] = val

    return new_config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rule_applier.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/rule_applier.py tests/test_rule_applier.py
git commit -m "feat: add rule applier for SHAP rules -> config overrides (Task 6)"
```

---

### Task 7: Discovery Runner (Orchestrator)

**Files:**
- Create: `src/discovery/runner.py`
- Test: `tests/test_discovery_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery_runner.py
"""Tests for the discovery engine orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestDiscoveryRunner:
    def test_should_run_shap_respects_accumulation_window(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        assert runner.should_run_shap(window_index=0) is False
        assert runner.should_run_shap(window_index=1) is False
        assert runner.should_run_shap(window_index=2) is True
        assert runner.should_run_shap(window_index=3) is False
        assert runner.should_run_shap(window_index=5) is True

    def test_accumulate_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        trades_w1 = [{"r_multiple": 1.0, "context": {}}]
        trades_w2 = [{"r_multiple": -0.5, "context": {}}]

        runner.accumulate_window("w_001", trades_w1)
        runner.accumulate_window("w_002", trades_w2)

        all_trades = runner.get_accumulated_trades()
        assert len(all_trades) == 2

    def test_analyze_returns_insight_when_enough_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            min_trades_for_shap=5,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        # Add enough trades
        import numpy as np
        rng = np.random.default_rng(42)
        trades = []
        for _ in range(20):
            trades.append({
                "r_multiple": float(rng.choice([-1.0, 1.5])),
                "context": {"adx_value": float(rng.uniform(10, 40)), "session": "london"},
            })

        runner.accumulate_window("w_001", trades)
        insight = runner.analyze(strategy_name="sss")
        assert insight is not None

    def test_analyze_returns_none_when_too_few_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            min_trades_for_shap=50,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        runner.accumulate_window("w_001", [{"r_multiple": 1.0, "context": {}}])
        insight = runner.analyze(strategy_name="sss")
        assert insight is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DiscoveryRunner**

```python
# src/discovery/runner.py
"""Discovery engine orchestrator.

Coordinates the full loop: accumulate trades across windows, run
XGBoost/SHAP analysis at intervals, generate hypotheses via Claude,
and persist everything to the knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.discovery.knowledge_base import KnowledgeBase
from src.discovery.xgb_analyzer import (
    SHAPInsight,
    build_training_data,
    run_shap_analysis,
    train_classifier,
)

logger = logging.getLogger(__name__)


class DiscoveryRunner:
    """Orchestrates the discovery engine across optimization windows.

    Parameters
    ----------
    shap_every_n_windows:
        Run full SHAP analysis every N windows (default 3 = ~90 days).
    min_trades_for_shap:
        Minimum accumulated trades before SHAP runs (default 30).
    knowledge_dir:
        Path for the JSON knowledge base.
    """

    def __init__(
        self,
        shap_every_n_windows: int = 3,
        min_trades_for_shap: int = 30,
        knowledge_dir: str = "reports/agent_knowledge",
    ) -> None:
        self._shap_interval = shap_every_n_windows
        self._min_trades = min_trades_for_shap
        self._kb = KnowledgeBase(base_dir=knowledge_dir)

    def should_run_shap(self, window_index: int) -> bool:
        """Check if SHAP analysis should run for this window."""
        return window_index > 0 and (window_index + 1) % self._shap_interval == 0

    def accumulate_window(self, window_id: str, trades: List[Dict[str, Any]]) -> None:
        """Save a window's trades to the knowledge base for accumulation."""
        self._kb.save_window_trades(trades, window_id=window_id)
        logger.info("Accumulated %d trades for window %s", len(trades), window_id)

    def get_accumulated_trades(self) -> List[Dict[str, Any]]:
        """Get all accumulated trades across windows."""
        return self._kb.get_accumulated_trades()

    def analyze(self, strategy_name: str = "sss") -> Optional[SHAPInsight]:
        """Run XGBoost/SHAP analysis on accumulated trades.

        Returns SHAPInsight if enough trades, None otherwise.
        """
        all_trades = self.get_accumulated_trades()
        if len(all_trades) < self._min_trades:
            logger.info(
                "Only %d accumulated trades (need %d) — skipping SHAP",
                len(all_trades), self._min_trades,
            )
            return None

        logger.info("Running SHAP analysis on %d accumulated trades", len(all_trades))
        X, y_binary, y_r = build_training_data(all_trades)

        if len(X) < self._min_trades:
            return None

        model = train_classifier(X, y_binary, y_r)
        insight = run_shap_analysis(model, X, y_binary, y_r)

        logger.info(
            "SHAP complete: %d features, %d interactions, %d rules",
            len(insight.feature_importance),
            len(insight.top_interactions),
            len(insight.actionable_rules),
        )
        return insight

    def run_full_cycle(
        self,
        window_id: str,
        window_index: int,
        trades: List[Dict[str, Any]],
        strategy_name: str,
        base_config: Dict[str, Any],
        enable_claude: bool = True,
    ) -> Dict[str, Any]:
        """Run one full discovery cycle for a window.

        1. Accumulate this window's trades
        2. If SHAP interval reached, run analysis
        3. If analysis produced rules, optionally generate hypotheses
        4. Return results dict

        Returns
        -------
        Dict with keys: shap_ran, insight, hypotheses, rules_applied
        """
        self.accumulate_window(window_id, trades)

        result: Dict[str, Any] = {
            "window_id": window_id,
            "window_index": window_index,
            "shap_ran": False,
            "insight": None,
            "hypotheses": [],
            "changes": [],
        }

        if not self.should_run_shap(window_index):
            return result

        insight = self.analyze(strategy_name=strategy_name)
        if insight is None:
            return result

        result["shap_ran"] = True
        result["insight"] = insight

        # Save SHAP rules to knowledge base
        self._kb.save_shap_rules(insight.actionable_rules, window_id=window_id)

        # Generate hypotheses via Claude (if enabled and rules found)
        if enable_claude and insight.actionable_rules:
            try:
                from src.discovery.hypothesis_engine import generate_hypotheses
                hypotheses = generate_hypotheses(
                    insight, strategy_name, window_id,
                )
                for hyp in hypotheses:
                    self._kb.save_hypothesis(hyp)
                result["hypotheses"] = hypotheses
            except Exception as exc:
                logger.warning("Hypothesis generation failed: %s", exc)

        # Apply strong rules as config changes
        from src.discovery.rule_applier import apply_rules_to_config
        _, changes = apply_rules_to_config(
            insight.actionable_rules, base_config, strategy=strategy_name,
        )
        result["changes"] = changes

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_discovery_runner.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/runner.py tests/test_discovery_runner.py
git commit -m "feat: add discovery runner orchestrator (Task 7)"
```

---

### Task 8: Integration Test — Full Discovery Cycle

**Files:**
- Test: `tests/test_discovery_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_discovery_integration.py
"""Integration test: full discovery cycle from trades to rules."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestFullDiscoveryCycle:
    def _generate_trades(self, n=100, seed=42):
        """Generate realistic-ish trade data."""
        rng = np.random.default_rng(seed)
        trades = []
        for _ in range(n):
            win = rng.random() < 0.37
            r = float(rng.uniform(0.5, 3.0) if win else rng.uniform(-2.0, -0.1))
            trades.append({
                "r_multiple": r,
                "context": {
                    "cloud_direction_4h": float(rng.choice([0.0, 0.5, 1.0])),
                    "adx_value": float(rng.uniform(10, 50)),
                    "atr_value": float(rng.uniform(1, 10)),
                    "session": rng.choice(["london", "new_york", "asian"]),
                    "confluence_score": int(rng.integers(1, 8)),
                },
            })
        return trades

    def test_full_cycle_produces_insight_and_rules(self):
        from src.discovery.runner import DiscoveryRunner

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            min_trades_for_shap=20,
            knowledge_dir=kb_dir,
        )

        # Simulate 3 windows of trading
        for i in range(3):
            trades = self._generate_trades(n=40, seed=42 + i)
            result = runner.run_full_cycle(
                window_id=f"w_{i:03d}",
                window_index=i,
                trades=trades,
                strategy_name="sss",
                base_config={"strategies": {"sss": {"min_confluence_score": 4}}},
                enable_claude=False,  # skip actual CLI call in tests
            )

        # Window 2 (index=2) should trigger SHAP
        assert result["shap_ran"] is True
        assert result["insight"] is not None
        assert len(result["insight"].feature_importance) > 0

    def test_knowledge_base_persists_across_windows(self):
        from src.discovery.runner import DiscoveryRunner
        from src.discovery.knowledge_base import KnowledgeBase

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            min_trades_for_shap=20,
            knowledge_dir=kb_dir,
        )

        for i in range(3):
            trades = self._generate_trades(n=30, seed=i)
            runner.accumulate_window(f"w_{i:03d}", trades)

        kb = KnowledgeBase(base_dir=kb_dir)
        all_trades = kb.get_accumulated_trades()
        assert len(all_trades) == 90  # 3 windows * 30 trades
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_discovery_integration.py tests/test_xgb_analyzer.py tests/test_knowledge_base.py tests/test_hypothesis_engine.py tests/test_rule_applier.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_discovery_integration.py
git commit -m "feat: add discovery engine integration tests (Task 8)"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Training data builder | `xgb_analyzer.py` | 4 tests |
| 2 | XGBoost classifier | `xgb_analyzer.py` | 3 tests |
| 3 | SHAP interaction analysis | `xgb_analyzer.py` | 3 tests |
| 4 | JSON knowledge base | `knowledge_base.py` | 5 tests |
| 5 | Hypothesis engine | `hypothesis_engine.py` | 3 tests |
| 6 | Rule applier | `rule_applier.py` | 3 tests |
| 7 | Discovery runner | `runner.py` | 4 tests |
| 8 | Integration test | — | 2 tests |

**Total: 8 tasks, 27 tests, 6 new source files.**

Phase 2-5 plans will be written separately once Phase 1 is validated.
