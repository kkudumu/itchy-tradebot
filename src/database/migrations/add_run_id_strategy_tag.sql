-- Migration: add run_id and strategy_tag columns to trades
-- Required by the Agentic Self-Optimisation Loop (Task 1: Trade Persistence Pipeline)
--
-- These columns group all trades from one backtest run under a common run_id
-- and let the similarity search filter by strategy variant.
--
-- Idempotent: uses IF NOT EXISTS so it is safe to run multiple times.

ALTER TABLE trades ADD COLUMN IF NOT EXISTS run_id VARCHAR(50);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS strategy_tag VARCHAR(100);

CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades (run_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_tag ON trades (strategy_tag);
