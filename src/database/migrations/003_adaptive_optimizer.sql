-- Migration 003: Adaptive optimizer tables
-- Creates optimization_runs, signal_log, and proven_configs tables
-- for the adaptive strategy optimization loop.

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 1. optimization_runs — one row per backtest / optimisation run
-- ============================================================
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

-- NOTE: IVFFlat indexes require the table to have rows before creation.
-- We skip the IVFFlat index here and will create it after the first
-- batch of data is inserted. Until then, vector queries use sequential scan.
-- The deferred index DDL (for reference):
--   CREATE INDEX idx_opt_runs_market_embed
--       ON optimization_runs USING ivfflat (market_embedding vector_cosine_ops)
--       WITH (lists = 20);

-- ============================================================
-- 2. signal_log — per-signal detail rows linked to a run
-- ============================================================
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

-- ============================================================
-- 3. proven_configs — graduated "known-good" param sets
-- ============================================================
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
