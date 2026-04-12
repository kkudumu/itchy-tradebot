-- 004_llm_meta_optimizer.sql
-- Adds trade_type and reasoning_summary to signal_log,
-- creates llm_analysis table for LLM reasoning history.

ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS trade_type TEXT;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS reasoning_summary TEXT;

CREATE TABLE IF NOT EXISTS llm_analysis (
    id                    SERIAL PRIMARY KEY,
    instrument            TEXT NOT NULL,
    epoch                 INTEGER NOT NULL,
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reasoning             TEXT NOT NULL,
    config_changes        JSONB,
    code_patches          JSONB,
    config_applied        BOOLEAN NOT NULL DEFAULT FALSE,
    code_merged           BOOLEAN NOT NULL DEFAULT FALSE,
    backtest_improvement  DOUBLE PRECISION,
    raw_output_path       TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_instrument
    ON llm_analysis (instrument, epoch DESC);
