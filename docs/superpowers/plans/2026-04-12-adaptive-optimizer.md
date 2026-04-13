# Adaptive Strategy Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a continuously running optimization loop that backtests the 5-strategy blender across MGC, MCL, MNQ, MYM futures, persists every trial to PostgreSQL + pgvector with 64-dim embeddings and signal logs, and uses embedding similarity to warm-start future runs.

**Architecture:** Each epoch cycles through instruments: download data → embed market context → query past experience → Optuna optimize with warm-starts → persist all trials → check guardrails → repeat forever. The DB accumulates a dense experience map. No regime labels — pure pattern similarity.

**Tech Stack:** Python 3.11, PostgreSQL 16 (port 5433) + pgvector + TimescaleDB, Optuna (TPE sampler), pandas, numpy, psycopg2, existing IchimokuBacktester engine.

---

## File Structure

```
NEW FILES:
  src/optimization/context_embedder.py    — Compute 64-dim (20+24+20) embeddings from market/params/outcome
  src/optimization/experience_store.py    — Persist and query optimization_runs via pgvector
  src/optimization/signal_persister.py    — Batch-insert signal_log rows to Postgres
  src/optimization/guardrails.py          — 3x combine pass + permutation p<0.05 checker
  src/optimization/adaptive_runner.py     — Core loop: data→embed→query→optimize→persist→check
  src/optimization/data_manager.py        — Download/update data per instrument from ProjectX
  scripts/run_adaptive_optimizer.py       — CLI entry point
  config/optimizer_instruments.yaml       — Instrument queue (MGC, MCL, MNQ, MYM)
  src/database/migrations/003_adaptive_optimizer.sql — New tables
  tests/test_context_embedder.py          — Unit tests for embedder
  tests/test_experience_store.py          — Unit tests for DB layer
  tests/test_guardrails.py                — Unit tests for guardrail checker
  tests/test_adaptive_runner.py           — Integration test for one epoch

MODIFIED FILES:
  config/instruments.yaml                 — Add MNQ and MYM instrument entries
  src/backtesting/strategy_telemetry.py   — Add hook for signal persister callback
```

---

### Task 1: Database Migration — New Tables

**Files:**
- Create: `src/database/migrations/003_adaptive_optimizer.sql`

- [ ] **Step 1: Write the migration SQL**

```sql
-- 003_adaptive_optimizer.sql
-- Adaptive optimizer tables for experience-driven optimization

CREATE EXTENSION IF NOT EXISTS vector;

-- Every Optuna trial result
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument      TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_start      TIMESTAMPTZ NOT NULL,
    data_end        TIMESTAMPTZ NOT NULL,
    market_embedding  VECTOR(20),
    params_embedding  VECTOR(24),
    outcome_embedding VECTOR(20),
    full_params     JSONB NOT NULL,
    active_strategies TEXT[] NOT NULL DEFAULT '{}',
    outcome         JSONB NOT NULL DEFAULT '{}',
    passed_combine  BOOLEAN NOT NULL DEFAULT FALSE,
    passed_permutation BOOLEAN NOT NULL DEFAULT FALSE,
    proven          BOOLEAN NOT NULL DEFAULT FALSE,
    epoch           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_opt_runs_instrument_proven
    ON optimization_runs (instrument, proven, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_opt_runs_market_embed
    ON optimization_runs USING ivfflat (market_embedding vector_cosine_ops)
    WITH (lists = 20);

-- Signal-level log for every signal in every trial
CREATE TABLE IF NOT EXISTS signal_log (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ NOT NULL,
    strategy_name   TEXT NOT NULL,
    direction       TEXT NOT NULL,
    confluence_score INTEGER NOT NULL DEFAULT 0,
    entry_price     DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    filtered_by     TEXT,
    entered         BOOLEAN NOT NULL DEFAULT FALSE,
    trade_result_r  DOUBLE PRECISION,
    exit_reason     TEXT,
    pnl_usd         DOUBLE PRECISION,
    market_snapshot  JSONB
);

CREATE INDEX IF NOT EXISTS idx_signal_log_run_id
    ON signal_log (run_id, strategy_name);

-- Only configs that passed all guardrails
CREATE TABLE IF NOT EXISTS proven_configs (
    id              SERIAL PRIMARY KEY,
    instrument      TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id          UUID REFERENCES optimization_runs(run_id),
    params          JSONB NOT NULL,
    win_rate        DOUBLE PRECISION,
    total_return_pct DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    combine_passes  INTEGER NOT NULL DEFAULT 0,
    data_start      TIMESTAMPTZ,
    data_end        TIMESTAMPTZ,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_proven_configs_instrument
    ON proven_configs (instrument, active, timestamp DESC);
```

- [ ] **Step 2: Apply the migration**

Run:
```bash
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d tradebot -f src/database/migrations/003_adaptive_optimizer.sql
```

Expected: Tables created without errors. Verify:
```bash
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d tradebot -c "\dt optimization_runs; \dt signal_log; \dt proven_configs;"
```

- [ ] **Step 3: Commit**

```bash
git add src/database/migrations/003_adaptive_optimizer.sql
git commit -m "feat: add adaptive optimizer DB tables (optimization_runs, signal_log, proven_configs)"
```

---

### Task 2: Instrument Config — Add MNQ and MYM

**Files:**
- Modify: `config/instruments.yaml`
- Create: `config/optimizer_instruments.yaml`

- [ ] **Step 1: Discover MNQ and MYM contract specs from ProjectX API**

Run:
```bash
python scripts/list_projectx_contracts.py --query "MNQ" 2>/dev/null | head -15
python scripts/list_projectx_contracts.py --query "MYM" 2>/dev/null | head -15
```

Note the `symbol_id`, `tick_size`, and `tick_value` from the output.

- [ ] **Step 2: Add MNQ and MYM to instruments.yaml**

Append to `config/instruments.yaml` after the MCLOIL entry:

```yaml
  - symbol: "MNQ"
    class: futures
    provider: "projectx"
    default_quantity: 1
    contract_id: ""              # Auto-discovered at runtime
    symbol_id: "F.US.MNQE"
    tick_size: 0.25
    tick_value_usd: 0.50         # $0.50 per tick (Micro Nasdaq)
    contract_size: 2             # $2 per point
    commission_per_contract_round_trip: 1.04
    session_open_ct: "17:00"
    session_close_ct: "16:00"
    daily_reset_hour_ct: 17
    tick_value: 0.50
    adx_threshold: 25
    spread_max_points: 5
    atr_stop_multiplier: 2.0
    pip_value_usd: 0.50

  - symbol: "MYM"
    class: futures
    provider: "projectx"
    default_quantity: 1
    contract_id: ""              # Auto-discovered at runtime
    symbol_id: "F.US.MYME"
    tick_size: 1.0
    tick_value_usd: 0.50         # $0.50 per tick (Micro Dow)
    contract_size: 0.50          # $0.50 per point
    commission_per_contract_round_trip: 1.04
    session_open_ct: "17:00"
    session_close_ct: "16:00"
    daily_reset_hour_ct: 17
    tick_value: 0.50
    adx_threshold: 25
    spread_max_points: 20
    atr_stop_multiplier: 2.0
    pip_value_usd: 0.50
```

Note: Verify tick_size/tick_value from Step 1 output and update if different.

- [ ] **Step 3: Create optimizer_instruments.yaml**

Write `config/optimizer_instruments.yaml`:

```yaml
# Instruments for the adaptive optimization loop.
# Adding a new instrument = one entry here. System auto-discovers contract specs.
instruments:
  - symbol: MGC
    symbol_id: F.US.MGC
    name: Micro Gold
    data_file: data/projectx_mgc_1m_last30d_20260310_20260409.parquet

  - symbol: MCL
    symbol_id: F.US.MCLE
    name: Micro Crude Oil
    data_file: data/projectx_mcl_1m_20260101_20260411.parquet

  - symbol: MNQ
    symbol_id: F.US.MNQE
    name: Micro E-mini Nasdaq
    data_file: null  # Will be downloaded

  - symbol: MYM
    symbol_id: F.US.MYME
    name: Micro E-mini Dow
    data_file: null  # Will be downloaded
```

- [ ] **Step 4: Commit**

```bash
git add config/instruments.yaml config/optimizer_instruments.yaml
git commit -m "feat: add MNQ/MYM instrument configs and optimizer instrument queue"
```

---

### Task 3: Context Embedder — 64-dim Vector Builder

**Files:**
- Create: `src/optimization/context_embedder.py`
- Create: `tests/test_context_embedder.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_context_embedder.py`:

```python
"""Tests for the 64-dim context embedder."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone


def _make_sample_data(n=5000, start_price=100.0):
    """Generate minimal 1M OHLCV data for testing."""
    rng = np.random.default_rng(42)
    prices = start_price + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "open": prices,
        "high": prices + rng.uniform(0, 1, n),
        "low": prices - rng.uniform(0, 1, n),
        "close": prices + rng.normal(0, 0.3, n),
        "volume": rng.integers(10, 500, n).astype(float),
    }, index=idx)


class TestMarketEmbedding:
    def test_returns_20_dim_vector(self):
        from src.optimization.context_embedder import ContextEmbedder
        data = _make_sample_data()
        embedder = ContextEmbedder()
        vec = embedder.embed_market(data, tick_size=0.01, tick_value_usd=1.0,
                                    contract_size=100, point_value=100.0)
        assert vec.shape == (20,)
        assert vec.dtype == np.float64

    def test_values_normalized_zero_to_one(self):
        from src.optimization.context_embedder import ContextEmbedder
        data = _make_sample_data()
        embedder = ContextEmbedder()
        vec = embedder.embed_market(data, tick_size=0.01, tick_value_usd=1.0,
                                    contract_size=100, point_value=100.0)
        assert np.all(vec >= 0.0), f"Min value {vec.min()} < 0"
        assert np.all(vec <= 1.0), f"Max value {vec.max()} > 1"

    def test_different_data_different_embeddings(self):
        from src.optimization.context_embedder import ContextEmbedder
        embedder = ContextEmbedder()
        data1 = _make_sample_data(start_price=100.0)
        data2 = _make_sample_data(start_price=5000.0)
        vec1 = embedder.embed_market(data1, tick_size=0.01, tick_value_usd=1.0,
                                     contract_size=100, point_value=100.0)
        vec2 = embedder.embed_market(data2, tick_size=0.10, tick_value_usd=1.0,
                                     contract_size=10, point_value=10.0)
        assert not np.allclose(vec1, vec2)


class TestParamsEmbedding:
    def test_returns_24_dim_vector(self):
        from src.optimization.context_embedder import ContextEmbedder
        embedder = ContextEmbedder()
        params = {
            "risk": {"initial_risk_pct": 0.5, "reduced_risk_pct": 0.75,
                     "daily_circuit_breaker_pct": 4.5, "max_concurrent_positions": 3},
            "exit": {"tp_r_multiple": 1.5},
            "strategies": {
                "sss": {"swing_lookback_n": 2, "min_swing_pips": 0.5, "min_stop_pips": 10.0,
                         "min_confluence_score": 2, "rr_ratio": 2.0, "entry_mode": "cbc_only",
                         "spread_multiplier": 2.0},
                "ichimoku": {"ichimoku": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52},
                             "adx": {"threshold": 20}, "atr": {"stop_multiplier": 2.5},
                             "signal": {"min_confluence_score": 1, "tier_c": 1}},
                "asian_breakout": {"min_range_pips": 3, "max_range_pips": 80, "rr_ratio": 2.0},
                "ema_pullback": {"min_ema_angle_deg": 2, "pullback_candles_max": 20, "rr_ratio": 2.0},
            },
        }
        vec = embedder.embed_params(params)
        assert vec.shape == (24,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)


class TestOutcomeEmbedding:
    def test_returns_20_dim_vector(self):
        from src.optimization.context_embedder import ContextEmbedder
        embedder = ContextEmbedder()
        outcome = {
            "win_rate": 0.4, "profit_factor": 1.5, "total_return_pct": 6.0,
            "sharpe_ratio": 1.2, "max_drawdown_pct": 3.5, "total_trades": 30,
            "avg_r_multiple": 0.3, "best_trade_r": 3.0, "worst_trade_r": -1.0,
            "avg_trade_duration_bars": 50, "passed": True, "final_balance": 53000,
            "distance_to_target": 0, "best_day_profit": 1500, "consistency_ratio": 0.3,
            "p_value": 0.02, "n_permutations_beaten": 19,
            "edge_filtered_pct": 0.1, "signals_entered_pct": 0.05,
            "win_rate_long": 0.45, "win_rate_short": 0.35,
        }
        vec = embedder.embed_outcome(outcome)
        assert vec.shape == (20,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)


class TestFullEmbedding:
    def test_full_64_dim(self):
        from src.optimization.context_embedder import ContextEmbedder
        embedder = ContextEmbedder()
        data = _make_sample_data()
        market = embedder.embed_market(data, tick_size=0.01, tick_value_usd=1.0,
                                       contract_size=100, point_value=100.0)
        params_dict = {
            "risk": {"initial_risk_pct": 0.5, "reduced_risk_pct": 0.75,
                     "daily_circuit_breaker_pct": 4.5, "max_concurrent_positions": 3},
            "exit": {"tp_r_multiple": 1.5},
            "strategies": {
                "sss": {"swing_lookback_n": 2, "min_swing_pips": 0.5, "min_stop_pips": 10.0,
                         "min_confluence_score": 2, "rr_ratio": 2.0, "entry_mode": "cbc_only",
                         "spread_multiplier": 2.0},
                "ichimoku": {"ichimoku": {"tenkan_period": 9}, "adx": {"threshold": 20},
                             "atr": {"stop_multiplier": 2.5},
                             "signal": {"min_confluence_score": 1, "tier_c": 1}},
                "asian_breakout": {"min_range_pips": 3, "max_range_pips": 80, "rr_ratio": 2.0},
                "ema_pullback": {"min_ema_angle_deg": 2, "pullback_candles_max": 20, "rr_ratio": 2.0},
            },
        }
        params_vec = embedder.embed_params(params_dict)
        outcome = {"win_rate": 0.4, "profit_factor": 1.5, "total_return_pct": 6.0,
                    "sharpe_ratio": 1.2, "max_drawdown_pct": 3.5, "total_trades": 30,
                    "avg_r_multiple": 0.3, "best_trade_r": 3.0, "worst_trade_r": -1.0,
                    "avg_trade_duration_bars": 50, "passed": True, "final_balance": 53000,
                    "distance_to_target": 0, "best_day_profit": 1500, "consistency_ratio": 0.3,
                    "p_value": 0.02, "n_permutations_beaten": 19,
                    "edge_filtered_pct": 0.1, "signals_entered_pct": 0.05,
                    "win_rate_long": 0.45, "win_rate_short": 0.35}
        outcome_vec = embedder.embed_outcome(outcome)
        full = np.concatenate([market, params_vec, outcome_vec])
        assert full.shape == (64,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.optimization.context_embedder'`

- [ ] **Step 3: Implement ContextEmbedder**

Write `src/optimization/context_embedder.py`:

```python
"""64-dim context embedder for the adaptive optimizer.

Produces three vectors:
  - Market context (20-dim): trend, volatility, price action, instrument identity
  - Strategy params (24-dim): normalized parameter snapshot
  - Outcome fingerprint (20-dim): performance metrics

All values normalized to [0, 1] for pgvector cosine similarity.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _safe(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _normalize(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return _clamp((val - lo) / (hi - lo))


def _log_normalize(val: float, lo: float, hi: float) -> float:
    if val <= 0:
        return 0.0
    return _normalize(math.log(val), math.log(max(lo, 1e-10)), math.log(max(hi, 1e-10)))


class ContextEmbedder:
    """Compute 64-dim embeddings (20 market + 24 params + 20 outcome)."""

    # --- Market context (20 dims) ---

    def embed_market(
        self,
        data: pd.DataFrame,
        tick_size: float,
        tick_value_usd: float,
        contract_size: float,
        point_value: float,
    ) -> np.ndarray:
        """Compute 20-dim market context from a 1M OHLCV DataFrame."""
        closes = data["close"].values.astype(float)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)

        returns = np.diff(np.log(np.maximum(closes, 1e-10)))
        returns = returns[np.isfinite(returns)]

        # ATR proxy (14-bar)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr_14 = pd.Series(tr).rolling(14).mean().dropna().values
        mean_price = np.nanmean(closes)

        # Trend structure (dims 0-3)
        # EMA slopes at ~1H (60 bars) and ~4H (240 bars)
        ema_60 = pd.Series(closes).ewm(span=60).mean().values
        ema_240 = pd.Series(closes).ewm(span=240).mean().values
        slope_1h = (ema_60[-1] - ema_60[-min(60, len(ema_60))]) / max(mean_price, 1) if len(ema_60) > 60 else 0
        slope_4h = (ema_240[-1] - ema_240[-min(240, len(ema_240))]) / max(mean_price, 1) if len(ema_240) > 240 else 0
        price_vs_ema60 = (closes[-1] - ema_60[-1]) / max(mean_price, 1) if len(ema_60) > 0 else 0
        price_vs_ema240 = (closes[-1] - ema_240[-1]) / max(mean_price, 1) if len(ema_240) > 0 else 0

        # Volatility (dims 4-7)
        atr_mean = np.mean(atr_14) / max(mean_price, 1) if len(atr_14) > 0 else 0
        atr_std_ratio = np.std(atr_14) / max(np.mean(atr_14), 1e-10) if len(atr_14) > 1 else 0
        atr_trend = 1.0 if len(atr_14) > 10 and atr_14[-1] > np.mean(atr_14[-10:]) else 0.0
        hl_range_mean = np.mean(highs - lows) / max(mean_price, 1)

        # Price action (dims 8-11)
        ret_mean = np.mean(returns) if len(returns) > 0 else 0
        ret_std = np.std(returns) if len(returns) > 1 else 0
        from scipy.stats import skew as _skew, kurtosis as _kurtosis
        ret_skew = float(_skew(returns)) if len(returns) > 10 else 0
        ret_kurt = float(_kurtosis(returns)) if len(returns) > 10 else 0

        # Drawdown/runup (dims 12-15)
        cummax = np.maximum.accumulate(closes)
        drawdowns = (closes - cummax) / np.maximum(cummax, 1e-10)
        max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        cummin = np.minimum.accumulate(closes)
        runups = (closes - cummin) / np.maximum(cummin, 1e-10)
        max_runup = np.max(runups) if len(runups) > 0 else 0
        ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 10 else 0
        ac5 = np.corrcoef(returns[:-5], returns[5:])[0, 1] if len(returns) > 20 else 0
        ac1 = 0.0 if not math.isfinite(ac1) else ac1
        ac5 = 0.0 if not math.isfinite(ac5) else ac5

        # Instrument identity (dims 16-19)
        vec = np.array([
            _normalize(slope_1h, -0.05, 0.05),          # 0
            _normalize(slope_4h, -0.05, 0.05),           # 1
            _normalize(price_vs_ema60, -0.05, 0.05),     # 2
            _normalize(price_vs_ema240, -0.10, 0.10),    # 3
            _normalize(atr_mean, 0, 0.02),               # 4
            _normalize(atr_std_ratio, 0, 2.0),           # 5
            atr_trend,                                    # 6
            _normalize(hl_range_mean, 0, 0.02),          # 7
            _normalize(ret_mean, -0.001, 0.001),         # 8
            _normalize(ret_std, 0, 0.01),                # 9
            _normalize(ret_skew, -2.0, 2.0),             # 10
            _normalize(ret_kurt, -2.0, 10.0),            # 11
            _normalize(max_dd, 0, 0.5),                  # 12
            _normalize(max_runup, 0, 0.5),               # 13
            _normalize(ac1, -1.0, 1.0),                  # 14
            _normalize(ac5, -1.0, 1.0),                  # 15
            _log_normalize(tick_size, 0.001, 10.0),      # 16
            _log_normalize(tick_value_usd, 0.01, 100.0), # 17
            _log_normalize(contract_size, 0.1, 1000.0),  # 18
            _log_normalize(point_value, 0.1, 1000.0),    # 19
        ], dtype=np.float64)

        return np.clip(vec, 0.0, 1.0)

    # --- Strategy params (24 dims) ---

    def embed_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Compute 24-dim normalized param vector."""
        risk = params.get("risk", {})
        exit_ = params.get("exit", {})
        strats = params.get("strategies", {})
        sss = strats.get("sss", {})
        ichi = strats.get("ichimoku", {})
        ab = strats.get("asian_breakout", {})
        ep = strats.get("ema_pullback", {})

        entry_mode_map = {"cbc_only": 0.0, "fifty_tap": 0.5, "combined": 1.0}
        ichi_scale = _safe(ichi.get("ichimoku", {}).get("tenkan_period"), 9) / 9.0

        vec = np.array([
            _normalize(_safe(risk.get("initial_risk_pct"), 0.5), 0.1, 3.0),     # 0
            _normalize(_safe(risk.get("reduced_risk_pct"), 0.75), 0.1, 3.0),    # 1
            _normalize(_safe(risk.get("daily_circuit_breaker_pct"), 4.5), 1, 5), # 2
            _normalize(_safe(risk.get("max_concurrent_positions"), 3), 1, 5),    # 3
            _normalize(_safe(exit_.get("tp_r_multiple"), 1.5), 0.5, 4.0),       # 4
            _normalize(_safe(sss.get("swing_lookback_n"), 2), 1, 6),             # 5
            _normalize(_safe(sss.get("min_swing_pips"), 0.5), 0.01, 5.0),       # 6
            _normalize(_safe(sss.get("min_stop_pips"), 10.0), 0.01, 15.0),      # 7
            _normalize(_safe(sss.get("min_confluence_score"), 2), 0, 6),         # 8
            _normalize(_safe(sss.get("rr_ratio"), 2.0), 1.0, 4.0),              # 9
            entry_mode_map.get(sss.get("entry_mode", "cbc_only"), 0.0),          # 10
            _normalize(_safe(sss.get("spread_multiplier"), 2.0), 0.5, 4.0),     # 11
            _normalize(ichi_scale, 0.7, 1.3),                                    # 12
            _normalize(_safe(ichi.get("adx", {}).get("threshold"), 20), 10, 40), # 13
            _normalize(_safe(ichi.get("atr", {}).get("stop_multiplier"), 2.5), 0.5, 4.0), # 14
            _normalize(_safe(ichi.get("signal", {}).get("min_confluence_score"), 1), 0, 6), # 15
            _normalize(_safe(ichi.get("signal", {}).get("tier_c"), 1), 1, 5),    # 16
            0.5,  # reserved for ichimoku timeframe_mode                          # 17
            _normalize(_safe(ab.get("min_range_pips"), 3), 0.01, 10.0),          # 18
            _normalize(_safe(ab.get("max_range_pips"), 80), 1, 200.0),           # 19
            _normalize(_safe(ab.get("rr_ratio"), 2.0), 1.0, 4.0),               # 20
            _normalize(_safe(ep.get("min_ema_angle_deg"), 2), 0.5, 10.0),        # 21
            _normalize(_safe(ep.get("pullback_candles_max"), 20), 3, 30),        # 22
            _normalize(_safe(ep.get("rr_ratio"), 2.0), 1.0, 4.0),               # 23
        ], dtype=np.float64)

        return np.clip(vec, 0.0, 1.0)

    # --- Outcome fingerprint (20 dims) ---

    def embed_outcome(self, outcome: Dict[str, Any]) -> np.ndarray:
        """Compute 20-dim outcome vector."""
        vec = np.array([
            _normalize(_safe(outcome.get("win_rate"), 0), 0, 1),                  # 0
            _normalize(_safe(outcome.get("profit_factor"), 0), 0, 5),             # 1
            _normalize(_safe(outcome.get("total_return_pct"), 0), -20, 30),       # 2
            _normalize(_safe(outcome.get("sharpe_ratio"), 0), -3, 5),             # 3
            _normalize(_safe(outcome.get("max_drawdown_pct"), 0), 0, 20),         # 4
            _normalize(math.log1p(_safe(outcome.get("total_trades"), 0)), 0, 6),  # 5
            _normalize(_safe(outcome.get("avg_r_multiple"), 0), -2, 3),           # 6
            _normalize(_safe(outcome.get("best_trade_r"), 0), 0, 5),              # 7
            _normalize(_safe(outcome.get("worst_trade_r"), 0), -3, 0),            # 8
            _normalize(_safe(outcome.get("avg_trade_duration_bars"), 50), 0, 500),# 9
            1.0 if outcome.get("passed") else 0.0,                                # 10
            _normalize(_safe(outcome.get("final_balance"), 50000), 45000, 60000), # 11
            _normalize(_safe(outcome.get("distance_to_target"), 3000), 0, 5000),  # 12
            _normalize(_safe(outcome.get("best_day_profit"), 0), 0, 3000),        # 13
            _normalize(_safe(outcome.get("consistency_ratio"), 0.5), 0, 1),       # 14
            _normalize(_safe(outcome.get("p_value"), 1.0), 0, 1),                 # 15
            _normalize(_safe(outcome.get("n_permutations_beaten"), 0), 0, 20),    # 16
            _normalize(_safe(outcome.get("edge_filtered_pct"), 0), 0, 1),         # 17
            _normalize(_safe(outcome.get("signals_entered_pct"), 0), 0, 1),       # 18
            _normalize(
                _safe(outcome.get("win_rate_long"), 0) - _safe(outcome.get("win_rate_short"), 0),
                -0.5, 0.5,
            ),                                                                     # 19
        ], dtype=np.float64)

        return np.clip(vec, 0.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context_embedder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimization/context_embedder.py tests/test_context_embedder.py
git commit -m "feat: add 64-dim context embedder (market + params + outcome)"
```

---

### Task 4: Experience Store — pgvector Persist and Query

**Files:**
- Create: `src/optimization/experience_store.py`
- Create: `tests/test_experience_store.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_experience_store.py`:

```python
"""Tests for experience store (mocked DB)."""
import uuid
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


class TestExperienceStorePersist:
    def test_persist_trial_inserts_row(self):
        from src.optimization.experience_store import ExperienceStore

        mock_pool = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_pool.get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        store = ExperienceStore(db_pool=mock_pool)
        run_id = store.persist_trial(
            instrument="MGC",
            data_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
            data_end=datetime(2026, 4, 9, tzinfo=timezone.utc),
            market_embedding=np.random.rand(20),
            params_embedding=np.random.rand(24),
            outcome_embedding=np.random.rand(20),
            full_params={"risk": {"initial_risk_pct": 0.5}},
            active_strategies=["sss", "ichimoku"],
            outcome={"win_rate": 0.4, "passed": True},
            passed_combine=True,
            passed_permutation=True,
            epoch=1,
        )
        assert isinstance(run_id, uuid.UUID)
        assert mock_cursor.execute.called


class TestExperienceStoreQuery:
    def test_find_similar_returns_list(self):
        from src.optimization.experience_store import ExperienceStore

        mock_pool = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"run_id": uuid.uuid4(), "full_params": '{"risk": {}}',
             "outcome": '{"win_rate": 0.5}', "similarity": 0.95}
        ]
        mock_pool.get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_pool.get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        store = ExperienceStore(db_pool=mock_pool)
        results = store.find_similar_successes(
            market_embedding=np.random.rand(20),
            limit=5,
        )
        assert len(results) == 1
        assert mock_cursor.execute.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_experience_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ExperienceStore**

Write `src/optimization/experience_store.py`:

```python
"""pgvector-backed experience store for optimization trials.

Persists every trial result with embeddings and queries for similar
past successes/failures to warm-start future optimization runs.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class ExperienceStore:
    """Persist and query optimization trial results via pgvector."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def persist_trial(
        self,
        instrument: str,
        data_start: datetime,
        data_end: datetime,
        market_embedding: np.ndarray,
        params_embedding: np.ndarray,
        outcome_embedding: np.ndarray,
        full_params: dict,
        active_strategies: list[str],
        outcome: dict,
        passed_combine: bool = False,
        passed_permutation: bool = False,
        epoch: int = 0,
    ) -> uuid.UUID:
        """Insert one optimization trial into the DB. Returns the run_id."""
        run_id = uuid.uuid4()
        proven = passed_combine and passed_permutation

        with self._pool.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO optimization_runs (
                    run_id, instrument, data_start, data_end,
                    market_embedding, params_embedding, outcome_embedding,
                    full_params, active_strategies, outcome,
                    passed_combine, passed_permutation, proven, epoch
                ) VALUES (
                    %s, %s, %s, %s,
                    %s::vector, %s::vector, %s::vector,
                    %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                (
                    str(run_id), instrument, data_start, data_end,
                    _vec_to_str(market_embedding),
                    _vec_to_str(params_embedding),
                    _vec_to_str(outcome_embedding),
                    json.dumps(full_params), active_strategies,
                    json.dumps(outcome, default=str),
                    passed_combine, passed_permutation, proven, epoch,
                ),
            )
        return run_id

    def find_similar_successes(
        self,
        market_embedding: np.ndarray,
        instrument: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find proven trials with similar market context."""
        where = "WHERE proven = TRUE"
        params: list = [_vec_to_str(market_embedding)]
        if instrument:
            where += " AND instrument = %s"
            params.append(instrument)
        params.append(_vec_to_str(market_embedding))
        params.append(limit)

        with self._pool.get_cursor() as cur:
            cur.execute(
                f"""
                SELECT run_id, full_params, outcome,
                       1 - (market_embedding <=> %s::vector) AS similarity
                FROM optimization_runs
                {where}
                ORDER BY market_embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        return [
            {
                "run_id": r["run_id"],
                "full_params": json.loads(r["full_params"]) if isinstance(r["full_params"], str) else r["full_params"],
                "outcome": json.loads(r["outcome"]) if isinstance(r["outcome"], str) else r["outcome"],
                "similarity": float(r["similarity"]),
            }
            for r in rows
        ]

    def find_similar_failures(
        self,
        market_embedding: np.ndarray,
        min_similarity: float = 0.8,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find failed trials in similar market conditions."""
        with self._pool.get_cursor() as cur:
            cur.execute(
                """
                SELECT full_params, outcome,
                       1 - (market_embedding <=> %s::vector) AS similarity
                FROM optimization_runs
                WHERE passed_combine = FALSE
                  AND 1 - (market_embedding <=> %s::vector) > %s
                ORDER BY market_embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    _vec_to_str(market_embedding),
                    _vec_to_str(market_embedding),
                    min_similarity,
                    _vec_to_str(market_embedding),
                    limit,
                ),
            )
            rows = cur.fetchall()

        return [
            {
                "full_params": json.loads(r["full_params"]) if isinstance(r["full_params"], str) else r["full_params"],
                "outcome": json.loads(r["outcome"]) if isinstance(r["outcome"], str) else r["outcome"],
                "similarity": float(r["similarity"]),
            }
            for r in rows
        ]

    def save_proven_config(
        self,
        instrument: str,
        run_id: uuid.UUID,
        params: dict,
        win_rate: float,
        total_return_pct: float,
        p_value: float,
        combine_passes: int,
        data_start: datetime,
        data_end: datetime,
    ) -> int:
        """Save a proven config and supersede any prior active config for this instrument."""
        with self._pool.get_cursor() as cur:
            # Supersede old configs
            cur.execute(
                """
                UPDATE proven_configs
                SET active = FALSE, superseded_at = NOW()
                WHERE instrument = %s AND active = TRUE
                """,
                (instrument,),
            )
            cur.execute(
                """
                INSERT INTO proven_configs (
                    instrument, run_id, params, win_rate, total_return_pct,
                    p_value, combine_passes, data_start, data_end
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    instrument, str(run_id), json.dumps(params),
                    win_rate, total_return_pct, p_value, combine_passes,
                    data_start, data_end,
                ),
            )
            row = cur.fetchone()
            return row["id"] if row else 0

    def get_proven_config(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get the active proven config for an instrument, if any."""
        with self._pool.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM proven_configs
                WHERE instrument = %s AND active = TRUE
                ORDER BY timestamp DESC LIMIT 1
                """,
                (instrument,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        result = dict(row)
        if isinstance(result.get("params"), str):
            result["params"] = json.loads(result["params"])
        return result

    def count_trials(self, instrument: Optional[str] = None) -> int:
        """Count total trials, optionally filtered by instrument."""
        with self._pool.get_cursor() as cur:
            if instrument:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM optimization_runs WHERE instrument = %s",
                    (instrument,),
                )
            else:
                cur.execute("SELECT COUNT(*) AS cnt FROM optimization_runs")
            row = cur.fetchone()
            return row["cnt"] if row else 0


def _vec_to_str(vec: np.ndarray) -> str:
    """Convert numpy array to pgvector literal string."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experience_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimization/experience_store.py tests/test_experience_store.py
git commit -m "feat: add pgvector experience store for optimization trials"
```

---

### Task 5: Signal Persister — Batch Insert Signal Logs

**Files:**
- Create: `src/optimization/signal_persister.py`

- [ ] **Step 1: Implement SignalPersister**

Write `src/optimization/signal_persister.py`:

```python
"""Batch-insert signal log rows to PostgreSQL.

Receives telemetry events from StrategyTelemetryCollector and
persists them to the signal_log table after each trial completes.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from psycopg2.extras import execute_values


class SignalPersister:
    """Batch-insert signal events to the signal_log table."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def persist_signals(
        self,
        run_id: uuid.UUID,
        events: List[Dict[str, Any]],
    ) -> int:
        """Insert signal events for a trial. Returns count inserted."""
        if not events:
            return 0

        rows = []
        for ev in events:
            rows.append((
                str(run_id),
                ev.get("timestamp_utc"),
                ev.get("strategy_name", "unknown"),
                ev.get("direction", "unknown"),
                int(ev.get("confluence_score", 0)),
                ev.get("price"),
                ev.get("stop_loss"),
                ev.get("take_profit"),
                ev.get("filtered_by") or ev.get("rejection_reason"),
                ev.get("entered", False),
                ev.get("realized_r"),
                ev.get("exit_reason"),
                ev.get("pnl_usd"),
                json.dumps({
                    "atr": ev.get("atr"),
                    "adx": ev.get("adx"),
                    "session": ev.get("session"),
                    "regime": ev.get("regime"),
                }),
            ))

        with self._pool.get_cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO signal_log (
                    run_id, timestamp, strategy_name, direction,
                    confluence_score, entry_price, stop_loss, take_profit,
                    filtered_by, entered, trade_result_r, exit_reason,
                    pnl_usd, market_snapshot
                ) VALUES %s
                """,
                rows,
                page_size=500,
            )
        return len(rows)
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/signal_persister.py
git commit -m "feat: add signal persister for batch signal_log inserts"
```

---

### Task 6: Guardrails — 3x Combine Pass + Permutation Test

**Files:**
- Create: `src/optimization/guardrails.py`
- Create: `tests/test_guardrails.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_guardrails.py`:

```python
"""Tests for optimization guardrails."""
import numpy as np
from unittest.mock import MagicMock, patch


class TestConsecutivePassCheck:
    def test_all_pass_returns_true(self):
        from src.optimization.guardrails import check_consecutive_passes

        def mock_backtest(data, config, instrument, initial_balance):
            return {"passed": True, "total_trades": 20, "win_rate": 0.4,
                    "final_balance": 53000, "status": "passed"}

        result = check_consecutive_passes(
            data=MagicMock(),
            config={},
            instrument="MGC",
            backtest_fn=mock_backtest,
            n_required=3,
        )
        assert result["all_passed"] is True
        assert len(result["attempts"]) == 3

    def test_one_fail_returns_false(self):
        from src.optimization.guardrails import check_consecutive_passes

        call_count = [0]
        def mock_backtest(data, config, instrument, initial_balance):
            call_count[0] += 1
            passed = call_count[0] != 2  # fail on second attempt
            return {"passed": passed, "total_trades": 20, "win_rate": 0.3,
                    "final_balance": 49000 if not passed else 53000,
                    "status": "passed" if passed else "pending"}

        result = check_consecutive_passes(
            data=MagicMock(),
            config={},
            instrument="MGC",
            backtest_fn=mock_backtest,
            n_required=3,
        )
        assert result["all_passed"] is False


class TestPermutationCheck:
    def test_significant_returns_true(self):
        from src.optimization.guardrails import check_permutation_significance

        def mock_backtest(data, config, instrument, initial_balance):
            return {"total_return_pct": 0.5}  # low return on shuffled data

        result = check_permutation_significance(
            real_return=18.0,
            data=MagicMock(),
            config={},
            instrument="MGC",
            backtest_fn=mock_backtest,
            n_permutations=5,
        )
        assert result["p_value"] == 0.0
        assert result["significant"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement guardrails**

Write `src/optimization/guardrails.py`:

```python
"""Guardrail checks for the adaptive optimizer.

Two gates:
1. check_consecutive_passes — run backtest on offset data windows
2. check_permutation_significance — shuffle candle returns and compare
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


def check_consecutive_passes(
    data: pd.DataFrame,
    config: dict,
    instrument: str,
    backtest_fn: Callable,
    n_required: int = 3,
    initial_balance: float = 50_000.0,
) -> Dict[str, Any]:
    """Run backtest on N offset windows. Returns dict with all_passed + attempts."""
    bars_per_day = 60 * 23
    offsets = [0] + [bars_per_day * (i * 3) for i in range(1, n_required)]

    attempts = []
    for i, offset in enumerate(offsets):
        window = data.iloc[offset:]
        if len(window) < 5000:
            window = data
        result = backtest_fn(window, config, instrument, initial_balance)
        attempts.append(result)

    all_passed = all(a.get("passed", False) for a in attempts)
    return {"all_passed": all_passed, "attempts": attempts}


def check_permutation_significance(
    real_return: float,
    data: pd.DataFrame,
    config: dict,
    instrument: str,
    backtest_fn: Callable,
    n_permutations: int = 20,
    p_threshold: float = 0.05,
    initial_balance: float = 50_000.0,
) -> Dict[str, Any]:
    """Shuffle candle returns N times and compare real performance."""
    perm_returns = []
    for i in range(n_permutations):
        perm_data = _permute_candles(data, seed=42 + i)
        result = backtest_fn(perm_data, config, instrument, initial_balance)
        perm_returns.append(result.get("total_return_pct", 0) if result else 0)

    beats = sum(1 for pr in perm_returns if pr >= real_return)
    p_value = beats / len(perm_returns) if perm_returns else 1.0

    return {
        "p_value": p_value,
        "significant": p_value < p_threshold,
        "real_return": real_return,
        "mean_permuted": float(np.mean(perm_returns)) if perm_returns else 0,
        "permuted_returns": perm_returns,
    }


def _permute_candles(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle bar-to-bar log returns, reconstruct OHLC prices."""
    rng = np.random.default_rng(seed)
    closes = df["close"].values.astype(float)
    log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))
    shuffled = log_returns.copy()
    rng.shuffle(shuffled)
    new_closes = np.empty(len(closes))
    new_closes[0] = closes[0]
    for i in range(len(shuffled)):
        new_closes[i + 1] = new_closes[i] * np.exp(shuffled[i])
    ratio = new_closes / closes
    new_df = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in new_df.columns:
            new_df[col] = (df[col].values * ratio).astype(float)
    new_df["high"] = new_df[["open", "high", "close"]].max(axis=1)
    new_df["low"] = new_df[["open", "low", "close"]].min(axis=1)
    return new_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_guardrails.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optimization/guardrails.py tests/test_guardrails.py
git commit -m "feat: add guardrail checks (consecutive passes + permutation test)"
```

---

### Task 7: Data Manager — Download and Update Per Instrument

**Files:**
- Create: `src/optimization/data_manager.py`

- [ ] **Step 1: Implement DataManager**

Write `src/optimization/data_manager.py`:

```python
"""Download and manage 1M OHLCV data for each instrument from ProjectX API.

Wraps the existing download_projectx_gold.py pattern into a reusable class
that handles contract discovery, calendar rolling, and incremental updates.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard CME futures month codes
_MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


class DataManager:
    """Download and cache 1M data per instrument."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or _PROJECT_ROOT / "data"
        self._data_dir.mkdir(exist_ok=True)

    def load_instruments(self) -> list[dict]:
        """Load instrument list from config/optimizer_instruments.yaml."""
        cfg_path = _PROJECT_ROOT / "config" / "optimizer_instruments.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Instrument config not found: {cfg_path}")
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("instruments", [])

    def get_data(self, instrument: dict) -> Optional[pd.DataFrame]:
        """Load existing data for an instrument. Downloads if not available."""
        symbol = instrument["symbol"]
        data_file = instrument.get("data_file")

        # Try existing file first
        if data_file:
            path = _PROJECT_ROOT / data_file
            if path.exists():
                return self._load_parquet(path)

        # Try auto-named file
        auto_path = self._data_dir / f"projectx_{symbol.lower()}_1m.parquet"
        if auto_path.exists():
            return self._load_parquet(auto_path)

        # Download from API
        logger.info("No cached data for %s, attempting download...", symbol)
        return self.download(instrument)

    def download(
        self,
        instrument: dict,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Download 1M data from ProjectX API for an instrument."""
        symbol = instrument["symbol"]
        symbol_id = instrument["symbol_id"]

        if end is None:
            end = datetime.now(tz=timezone.utc)
        if start is None:
            start = end - timedelta(days=90)  # Default: last 90 days

        logger.info("Downloading %s data from %s to %s", symbol, start.date(), end.date())

        try:
            self._load_env()
            from src.config.loader import load_config
            from src.providers import build_projectx_stack

            cfg = load_config()
            client, _, _, _ = build_projectx_stack(cfg.provider.projectx, cfg.instruments)

            # Discover contracts
            contracts = self._discover_contracts(client, symbol_id, start.year, end.year)
            if not contracts:
                logger.warning("No contracts found for %s", symbol)
                return None

            # Download bars from each contract
            all_chunks = []
            step = timedelta(minutes=1)
            chunk_span = step * 20000

            for contract in contracts:
                cid = contract["id"]
                current = start
                while current < end:
                    chunk_end = min(end, current + chunk_span)
                    bars = self._fetch_bars(client, cid, current, chunk_end)
                    if bars is not None and not bars.empty:
                        bars["contract_id"] = cid
                        all_chunks.append(bars)
                        current = bars.index[-1].to_pydatetime() + step
                    else:
                        current = chunk_end + step
                    time.sleep(0.5)

            if not all_chunks:
                logger.warning("No data downloaded for %s", symbol)
                return None

            merged = pd.concat(all_chunks).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]

            # Save
            out_path = self._data_dir / f"projectx_{symbol.lower()}_1m.parquet"
            merged.to_parquet(out_path)
            logger.info("Saved %d bars to %s", len(merged), out_path)
            return merged

        except Exception as exc:
            logger.error("Download failed for %s: %s", symbol, exc)
            return None

    def _discover_contracts(self, client, symbol_id: str, start_year: int, end_year: int) -> list:
        """Find all available contracts for a symbol."""
        # Extract product code from symbol_id (e.g., "F.US.MCLE" → "MCLE")
        product = symbol_id.split(".")[-1]
        contracts = []
        for year in range(start_year, end_year + 1):
            yy = year % 100
            for code in _MONTH_CODES:
                cid = f"CON.F.US.{product}.{code}{yy:02d}"
                try:
                    resp = client.search_contract_by_id(cid)
                    contract = resp.get("contract")
                    if contract:
                        contracts.append(contract)
                except Exception:
                    continue
                time.sleep(0.2)
        return contracts

    def _fetch_bars(self, client, contract_id: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch 1M bars for a contract window."""
        for attempt in range(5):
            try:
                resp = client.retrieve_bars(
                    contract_id=contract_id, live=False,
                    start_time=start, end_time=end,
                    unit=2, unit_number=1, limit=20000,
                    include_partial_bar=False,
                )
                bars = resp.get("bars", [])
                if not bars:
                    return None
                df = pd.DataFrame({
                    "open": [float(b["o"]) for b in bars],
                    "high": [float(b["h"]) for b in bars],
                    "low": [float(b["l"]) for b in bars],
                    "close": [float(b["c"]) for b in bars],
                    "volume": [float(b["v"]) for b in bars],
                }, index=pd.DatetimeIndex(
                    pd.to_datetime([b["t"] for b in bars], utc=True), name="time"
                ))
                return df.sort_index()
            except Exception as exc:
                if "429" in str(exc):
                    time.sleep(min(60, 10 * (attempt + 1)))
                else:
                    logger.debug("Fetch error for %s: %s", contract_id, exc)
                    return None
        return None

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.columns = df.columns.str.lower()
        return df

    @staticmethod
    def _load_env():
        env_path = _PROJECT_ROOT / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/data_manager.py
git commit -m "feat: add data manager for instrument data download and caching"
```

---

### Task 8: Adaptive Runner — Core Optimization Loop

**Files:**
- Create: `src/optimization/adaptive_runner.py`

- [ ] **Step 1: Implement AdaptiveRunner**

Write `src/optimization/adaptive_runner.py`:

```python
"""Core adaptive optimization loop.

Cycles through instruments: data → embed → query → optimize → persist → check.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import yaml

from src.optimization.context_embedder import ContextEmbedder
from src.optimization.experience_store import ExperienceStore
from src.optimization.signal_persister import SignalPersister
from src.optimization.guardrails import check_consecutive_passes, check_permutation_significance
from src.optimization.data_manager import DataManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INITIAL_BALANCE = 50_000.0


class AdaptiveRunner:
    """Continuously optimize strategies across instruments."""

    def __init__(
        self,
        db_pool,
        data_manager: Optional[DataManager] = None,
        trials_per_epoch: int = 50,
    ) -> None:
        self._pool = db_pool
        self._embedder = ContextEmbedder()
        self._store = ExperienceStore(db_pool=db_pool)
        self._signal_persister = SignalPersister(db_pool=db_pool)
        self._data_manager = data_manager or DataManager()
        self._trials_per_epoch = trials_per_epoch
        self._epoch = 0

    def run_forever(self) -> None:
        """Run the optimization loop forever."""
        while True:
            self._epoch += 1
            logger.info("=== EPOCH %d ===", self._epoch)
            instruments = self._data_manager.load_instruments()
            for instrument in instruments:
                self.optimize_instrument(instrument)
            logger.info("Epoch %d complete. Sleeping before next cycle.", self._epoch)
            time.sleep(60)  # Brief pause between epochs

    def run_once(self, instrument_filter: Optional[str] = None) -> Dict[str, Any]:
        """Run one optimization epoch. Returns results per instrument."""
        self._epoch += 1
        instruments = self._data_manager.load_instruments()
        results = {}
        for instrument in instruments:
            symbol = instrument["symbol"]
            if instrument_filter and symbol != instrument_filter:
                continue
            results[symbol] = self.optimize_instrument(instrument)
        return results

    def optimize_instrument(self, instrument: dict) -> Dict[str, Any]:
        """Run full optimization for one instrument."""
        symbol = instrument["symbol"]
        logger.info("--- Optimizing %s ---", symbol)

        # 1. Load data
        data = self._data_manager.get_data(instrument)
        if data is None or len(data) < 5000:
            logger.warning("Insufficient data for %s (%s bars)", symbol,
                          len(data) if data is not None else 0)
            return {"status": "insufficient_data"}

        # 2. Build base config
        base_config = self._build_base_config(symbol)
        if base_config is None:
            return {"status": "config_error"}

        inst_cfg = self._get_instrument_config(symbol)

        # 3. Embed market context
        market_embedding = self._embedder.embed_market(
            data,
            tick_size=inst_cfg.get("tick_size", 0.01),
            tick_value_usd=inst_cfg.get("tick_value_usd", 1.0),
            contract_size=inst_cfg.get("contract_size", 100),
            point_value=inst_cfg.get("tick_value_usd", 1.0) / max(inst_cfg.get("tick_size", 0.01), 1e-10),
        )

        # 4. Query experience store for warm starts
        successes = self._store.find_similar_successes(market_embedding, instrument=symbol, limit=10)
        failures = self._store.find_similar_failures(market_embedding, limit=20)
        logger.info("Found %d similar successes, %d similar failures for warm-start",
                    len(successes), len(failures))

        # 5. Run Optuna optimization
        best_result = self._run_optuna(
            data=data,
            base_config=base_config,
            instrument=symbol,
            market_embedding=market_embedding,
            warm_starts=[s["full_params"] for s in successes],
        )

        if best_result is None:
            return {"status": "no_viable_trials"}

        # 6. Check guardrails on best result
        if best_result["passed"]:
            logger.info("Best trial passed combine. Checking 3x consecutive...")
            passes_result = check_consecutive_passes(
                data=data,
                config=best_result["config"],
                instrument=symbol,
                backtest_fn=self._backtest_fn,
                n_required=3,
            )

            if passes_result["all_passed"]:
                logger.info("3x passes confirmed! Running permutation test...")
                perm_result = check_permutation_significance(
                    real_return=best_result["total_return_pct"],
                    data=data,
                    config=best_result["config"],
                    instrument=symbol,
                    backtest_fn=self._backtest_fn,
                    n_permutations=20,
                )

                if perm_result["significant"]:
                    # PROVEN!
                    run_id = best_result["run_id"]
                    self._store.save_proven_config(
                        instrument=symbol,
                        run_id=run_id,
                        params=best_result["config"],
                        win_rate=best_result["win_rate"],
                        total_return_pct=best_result["total_return_pct"],
                        p_value=perm_result["p_value"],
                        combine_passes=3,
                        data_start=data.index[0].to_pydatetime(),
                        data_end=data.index[-1].to_pydatetime(),
                    )
                    logger.info("PROVEN CONFIG for %s! p=%.4f, return=%.2f%%",
                               symbol, perm_result["p_value"], best_result["total_return_pct"])
                    return {"status": "proven", "p_value": perm_result["p_value"],
                            "return": best_result["total_return_pct"]}
                else:
                    logger.info("%s: passes combine but p=%.4f (not significant)",
                               symbol, perm_result["p_value"])
                    return {"status": "passes_but_not_significant",
                            "p_value": perm_result["p_value"]}
            else:
                n_passed = sum(1 for a in passes_result["attempts"] if a.get("passed"))
                logger.info("%s: %d/3 consecutive passes", symbol, n_passed)
                return {"status": "partial_passes", "passes": n_passed}
        else:
            logger.info("%s: best trial did not pass combine (status=%s)",
                       symbol, best_result.get("status"))
            return {"status": "no_pass", "best_balance": best_result.get("final_balance")}

    def _run_optuna(
        self,
        data: pd.DataFrame,
        base_config: dict,
        instrument: str,
        market_embedding: np.ndarray,
        warm_starts: List[dict],
    ) -> Optional[Dict[str, Any]]:
        """Run Optuna optimization with warm-starts from experience."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Seed with warm-start params
        for ws_params in warm_starts[:5]:
            try:
                study.enqueue_trial(ws_params)
            except Exception:
                pass  # Skip incompatible warm starts

        best_result_holder: list = [None]

        def objective(trial: optuna.Trial) -> float:
            from scripts.optimize_mcl_fast import objective as _  # noqa — avoid name collision
            params = self._suggest_params(trial, base_config)
            result = self._backtest_fn(data, params, instrument, INITIAL_BALANCE)

            if result is None or result.get("total_trades", 0) == 0:
                return -2.0

            # Persist trial to DB
            params_embedding = self._embedder.embed_params(params)
            outcome_dict = {
                "win_rate": result.get("win_rate", 0),
                "total_return_pct": result.get("total_return_pct", 0),
                "total_trades": result.get("total_trades", 0),
                "final_balance": result.get("final_balance", INITIAL_BALANCE),
                "passed": result.get("passed", False),
                "status": result.get("status", "pending"),
            }
            outcome_embedding = self._embedder.embed_outcome(outcome_dict)
            run_id = self._store.persist_trial(
                instrument=instrument,
                data_start=data.index[0].to_pydatetime(),
                data_end=data.index[-1].to_pydatetime(),
                market_embedding=market_embedding,
                params_embedding=params_embedding,
                outcome_embedding=outcome_embedding,
                full_params=params,
                active_strategies=params.get("active_strategies", []),
                outcome=outcome_dict,
                epoch=self._epoch,
            )

            # Score
            if result["passed"]:
                score = 1.0
            else:
                profit = result["final_balance"] - INITIAL_BALANCE
                score = max(-1.0, min(1.0, profit / 3000.0))
                status = result.get("status", "")
                if "failed_mll" in status:
                    score -= 0.5
                elif "failed_daily" in status:
                    score -= 0.3
            score += min(result["total_trades"], 30) / 200.0

            # Track best
            if best_result_holder[0] is None or score > best_result_holder[0].get("_score", -999):
                best_result_holder[0] = {**result, "config": params, "run_id": run_id, "_score": score}

            return score

        study.optimize(objective, n_trials=self._trials_per_epoch)
        return best_result_holder[0]

    def _suggest_params(self, trial: optuna.Trial, base: dict) -> dict:
        """Suggest params adapted to any instrument's price scale."""
        params = copy.deepcopy(base)

        params["risk"]["initial_risk_pct"] = trial.suggest_float("ri", 0.1, 2.0, step=0.1)
        params["risk"]["reduced_risk_pct"] = trial.suggest_float("rr", 0.1, 2.0, step=0.1)
        params["risk"]["daily_circuit_breaker_pct"] = trial.suggest_float("cb", 1.5, 5.0, step=0.5)
        params["risk"]["max_concurrent_positions"] = trial.suggest_int("mc", 1, 5)

        params["exit"]["tp_r_multiple"] = trial.suggest_float("tp", 1.0, 3.0, step=0.25)
        params["exit"]["breakeven_threshold_r"] = trial.suggest_float("be", 0.5, 1.5, step=0.25)

        # SSS — scale-adaptive via relative params
        params["strategies"]["sss"]["min_swing_pips"] = trial.suggest_float("ss", 0.01, 2.0, step=0.01)
        params["strategies"]["sss"]["min_stop_pips"] = trial.suggest_float("sst", 0.02, 5.0, step=0.02)
        params["strategies"]["sss"]["min_confluence_score"] = trial.suggest_int("sc", 0, 4)
        params["strategies"]["sss"]["rr_ratio"] = trial.suggest_float("sr", 1.5, 3.0, step=0.25)
        params["strategies"]["sss"]["entry_mode"] = trial.suggest_categorical("se", ["cbc_only", "fifty_tap", "combined"])
        params["strategies"]["sss"]["spread_multiplier"] = trial.suggest_float("sp", 0.5, 3.0, step=0.5)

        # Ichimoku
        scale = trial.suggest_float("is", 0.7, 1.3, step=0.05)
        params["strategies"]["ichimoku"]["ichimoku"]["tenkan_period"] = max(3, round(9 * scale))
        params["strategies"]["ichimoku"]["ichimoku"]["kijun_period"] = max(9, round(26 * scale))
        params["strategies"]["ichimoku"]["ichimoku"]["senkou_b_period"] = max(18, round(52 * scale))
        params["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = trial.suggest_float("ia", 1.0, 3.0, step=0.25)
        params["strategies"]["ichimoku"]["adx"]["threshold"] = trial.suggest_int("id", 10, 35)
        params["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = trial.suggest_int("ic", 1, 5)

        # Asian Breakout
        params["strategies"]["asian_breakout"]["min_range_pips"] = trial.suggest_float("am", 0.01, 5.0, step=0.01)
        params["strategies"]["asian_breakout"]["max_range_pips"] = trial.suggest_float("ax", 1.0, 100.0, step=1.0)
        params["strategies"]["asian_breakout"]["rr_ratio"] = trial.suggest_float("ar", 1.5, 3.0, step=0.25)

        # EMA Pullback
        params["strategies"]["ema_pullback"]["min_ema_angle_deg"] = trial.suggest_float("ea", 0.5, 10.0, step=0.5)
        params["strategies"]["ema_pullback"]["pullback_candles_max"] = trial.suggest_int("ep", 5, 30)
        params["strategies"]["ema_pullback"]["rr_ratio"] = trial.suggest_float("er", 1.5, 3.0, step=0.25)

        return params

    @staticmethod
    def _backtest_fn(
        data: pd.DataFrame,
        config: dict,
        instrument: str,
        initial_balance: float,
    ) -> Optional[Dict[str, Any]]:
        """Run a single backtest and return key metrics."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        try:
            bt = IchimokuBacktester(config=config, initial_balance=initial_balance)
            result = bt.run(candles_1m=data, instrument=instrument,
                           log_trades=False, enable_learning=False)
            m = result.metrics
            prop = result.prop_firm
            active = prop.get("active_tracker", prop)
            return {
                "total_trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "total_return_pct": m.get("total_return_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "profit_factor": m.get("profit_factor", 0),
                "final_balance": float(active.get("current_balance", initial_balance)),
                "status": str(active.get("status", "pending")),
                "passed": str(active.get("status", "")) == "passed",
                "pipeline": m.get("pipeline_counts", {}),
            }
        except Exception as exc:
            logger.debug("Backtest failed: %s", exc)
            return None

    def _build_base_config(self, symbol: str) -> Optional[dict]:
        """Build base backtest config for an instrument."""
        try:
            from src.config.loader import ConfigLoader
            loader = ConfigLoader(config_dir=str(_PROJECT_ROOT / "config"))
            app_config = loader.load()

            with (_PROJECT_ROOT / "config" / "strategy.yaml").open() as f:
                raw_strat = yaml.safe_load(f) or {}

            ss = app_config.strategy.model_dump()
            cfg: dict = {}
            if hasattr(app_config, "edges"):
                cfg["edges"] = app_config.edges.model_dump()
            cfg["active_strategies"] = raw_strat.get("active_strategies", ["ichimoku"])
            cfg["strategies"] = ss.get("strategies", {})
            for key in ("risk", "exit", "prop_firm"):
                if key in ss:
                    cfg[key] = ss[key]

            inst = app_config.instruments.get(symbol)
            if inst is None:
                logger.error("Instrument %s not found in instruments.yaml", symbol)
                return None
            cfg["instrument_class"] = inst.class_.value
            cfg["instrument"] = {
                "symbol": inst.symbol, "class": inst.class_.value,
                "tick_size": inst.tick_size, "tick_value_usd": inst.tick_value_usd,
                "contract_size": inst.contract_size,
                "commission_per_contract_round_trip": inst.commission_per_contract_round_trip,
                "session_open_ct": inst.session_open_ct,
                "session_close_ct": inst.session_close_ct,
                "daily_reset_hour_ct": inst.daily_reset_hour_ct,
            }
            cfg["prop_firm"] = cfg.get("prop_firm", {})
            return cfg
        except Exception as exc:
            logger.error("Config build failed for %s: %s", symbol, exc)
            return None

    def _get_instrument_config(self, symbol: str) -> dict:
        """Get instrument tick/contract specs."""
        try:
            from src.config.loader import ConfigLoader
            app_config = ConfigLoader(config_dir=str(_PROJECT_ROOT / "config")).load()
            inst = app_config.instruments.get(symbol)
            if inst:
                return {
                    "tick_size": inst.tick_size or 0.01,
                    "tick_value_usd": inst.tick_value_usd or 1.0,
                    "contract_size": inst.contract_size or 100,
                }
        except Exception:
            pass
        return {"tick_size": 0.01, "tick_value_usd": 1.0, "contract_size": 100}

    def status(self) -> Dict[str, Any]:
        """Return status summary for all instruments."""
        instruments = self._data_manager.load_instruments()
        result = {}
        for inst in instruments:
            symbol = inst["symbol"]
            proven = self._store.get_proven_config(symbol)
            trials = self._store.count_trials(symbol)
            result[symbol] = {
                "total_trials": trials,
                "proven": proven is not None,
                "proven_config": proven,
            }
        return result
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/adaptive_runner.py
git commit -m "feat: add adaptive runner — core optimization loop with warm-starts"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `scripts/run_adaptive_optimizer.py`

- [ ] **Step 1: Implement CLI**

Write `scripts/run_adaptive_optimizer.py`:

```python
"""CLI entry point for the adaptive strategy optimizer."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy sub-loggers
for name in ["src.backtesting.vectorbt_engine", "src.strategy", "src.risk",
             "src.edges", "src.learning", "src.monitoring",
             "src.backtesting.strategy_telemetry"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("adaptive_optimizer")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Strategy Optimizer")
    parser.add_argument("--instrument", type=str, default=None,
                       help="Optimize a single instrument (e.g., MGC, MCL)")
    parser.add_argument("--once", action="store_true",
                       help="Run one epoch only (no looping)")
    parser.add_argument("--status", action="store_true",
                       help="Show status of all instruments")
    parser.add_argument("--trials", type=int, default=50,
                       help="Optuna trials per instrument per epoch")
    args = parser.parse_args()

    # Load .env
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        import os
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # Init DB
    from src.database.connection import DatabasePool
    pool = DatabasePool()
    pool.initialise()

    from src.optimization.adaptive_runner import AdaptiveRunner
    runner = AdaptiveRunner(db_pool=pool, trials_per_epoch=args.trials)

    if args.status:
        status = runner.status()
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.once:
        results = runner.run_once(instrument_filter=args.instrument)
        for symbol, result in results.items():
            logger.info("%s: %s", symbol, result.get("status", "unknown"))
        return 0

    # Continuous loop
    logger.info("Starting continuous adaptive optimization loop...")
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        pool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_adaptive_optimizer.py
git commit -m "feat: add adaptive optimizer CLI entry point"
```

---

### Task 10: Download MNQ and MYM Data

**Files:** None (uses existing DataManager)

- [ ] **Step 1: Download MNQ (Micro Nasdaq) data**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.optimization.data_manager import DataManager
dm = DataManager()
data = dm.download({'symbol': 'MNQ', 'symbol_id': 'F.US.MNQE'})
print(f'MNQ: {len(data) if data is not None else 0} bars')
"
```

- [ ] **Step 2: Download MYM (Micro Dow) data**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.optimization.data_manager import DataManager
dm = DataManager()
data = dm.download({'symbol': 'MYM', 'symbol_id': 'F.US.MYME'})
print(f'MYM: {len(data) if data is not None else 0} bars')
"
```

- [ ] **Step 3: Update optimizer_instruments.yaml with data file paths**

Update the `data_file` fields in `config/optimizer_instruments.yaml` with the actual parquet paths from steps 1-2.

- [ ] **Step 4: Commit**

```bash
git add data/projectx_mnq_*.parquet data/projectx_mym_*.parquet config/optimizer_instruments.yaml
git commit -m "feat: download MNQ and MYM data, update instrument queue"
```

---

### Task 11: Integration Test — Run One Epoch

**Files:**
- Create: `tests/test_adaptive_runner.py`

- [ ] **Step 1: Write integration test**

Write `tests/test_adaptive_runner.py`:

```python
"""Integration test: run one optimization epoch on MGC with reduced trials."""
import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestAdaptiveRunnerIntegration:
    def test_one_epoch_mgc(self):
        """Run 5 trials on MGC to verify the full pipeline works end-to-end."""
        from src.database.connection import DatabasePool
        from src.optimization.adaptive_runner import AdaptiveRunner

        pool = DatabasePool()
        pool.initialise()

        try:
            runner = AdaptiveRunner(db_pool=pool, trials_per_epoch=5)
            results = runner.run_once(instrument_filter="MGC")

            assert "MGC" in results
            status = results["MGC"]["status"]
            # With only 5 trials we likely won't prove it, but it should complete
            assert status in ("proven", "passes_but_not_significant",
                            "partial_passes", "no_pass", "no_viable_trials",
                            "insufficient_data")
        finally:
            pool.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_adaptive_runner.py -v -m integration --timeout=600`
Expected: PASS (completes without errors, status is one of the valid outcomes)

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_runner.py
git commit -m "test: add integration test for adaptive optimization loop"
```

---

### Task 12: Smoke Test — Full Loop on All 4 Instruments

- [ ] **Step 1: Run the optimizer for one epoch with 10 trials each**

Run:
```bash
python scripts/run_adaptive_optimizer.py --once --trials 10
```

Expected: Completes for all 4 instruments (MGC, MCL, MNQ, MYM), persists trials to DB, reports status per instrument.

- [ ] **Step 2: Check DB state**

Run:
```bash
PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d tradebot -c "
SELECT instrument, COUNT(*) as trials, SUM(passed_combine::int) as passes, SUM(proven::int) as proven
FROM optimization_runs GROUP BY instrument ORDER BY instrument;
"
```

Expected: Rows for each instrument with trial counts.

- [ ] **Step 3: Check status command**

Run:
```bash
python scripts/run_adaptive_optimizer.py --status
```

Expected: JSON output showing trial counts and proven status per instrument.

- [ ] **Step 4: Commit any config fixes discovered during smoke test**

```bash
git add -A
git commit -m "fix: smoke test adjustments for adaptive optimizer"
```
