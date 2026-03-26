-- =============================================================================
-- XAU/USD Ichimoku Trading Agent — Full Database Schema
-- Requires: PostgreSQL 14+, TimescaleDB 2.x, pgvector
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 1. candles_1m
--    Raw 1-minute OHLCV bars. Converted to a TimescaleDB hypertable so that
--    range queries on timestamp are routed to individual time chunks instead
--    of scanning the full table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS candles_1m (
    timestamp   TIMESTAMPTZ      NOT NULL,
    instrument  VARCHAR(20)      NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (timestamp, instrument)
);

-- Convert to hypertable partitioned by time, one chunk per week.
SELECT create_hypertable(
    'candles_1m',
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists       => TRUE
);

-- Compress chunks older than 30 days.  Segmenting by instrument keeps
-- per-symbol scans efficient; ordering by timestamp aligns with the most
-- common query pattern.
ALTER TABLE candles_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument',
    timescaledb.compress_orderby   = 'timestamp DESC'
);

SELECT add_compression_policy(
    'candles_1m',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Covering index for instrument-first lookups (e.g., fetch last N bars for a
-- given symbol without scanning all instruments in the same time chunk).
CREATE INDEX IF NOT EXISTS idx_candles_1m_instrument_time
    ON candles_1m (instrument, timestamp DESC);

-- ---------------------------------------------------------------------------
-- 2. trades
--    Every trade entry regardless of whether it came from a live session, a
--    paper-trading session, or a back-test run.  The source column lets the
--    application separate performance statistics cleanly.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id               SERIAL PRIMARY KEY,
    instrument       VARCHAR(20)      NOT NULL,
    source           VARCHAR(10)      NOT NULL CHECK (source IN ('backtest', 'live', 'paper')),
    direction        VARCHAR(5)       NOT NULL CHECK (direction IN ('long', 'short')),
    entry_time       TIMESTAMPTZ      NOT NULL,
    exit_time        TIMESTAMPTZ,
    entry_price      DOUBLE PRECISION NOT NULL,
    exit_price       DOUBLE PRECISION,
    stop_loss        DOUBLE PRECISION NOT NULL,
    take_profit      DOUBLE PRECISION,
    lot_size         DOUBLE PRECISION NOT NULL,
    risk_pct         DOUBLE PRECISION NOT NULL,
    r_multiple       DOUBLE PRECISION,        -- realised R at close
    pnl              DOUBLE PRECISION,        -- dollar P&L
    pnl_pct          DOUBLE PRECISION,        -- % of account at risk
    status           VARCHAR(10)      DEFAULT 'open'
                         CHECK (status IN ('open', 'partial', 'closed', 'cancelled')),
    confluence_score INTEGER,                 -- 0–100 composite score at entry
    signal_tier      VARCHAR(5),              -- e.g. 'A', 'B', 'C'
    created_at       TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_instrument_entry
    ON trades (instrument, entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_source_status
    ON trades (source, status);

-- ---------------------------------------------------------------------------
-- 3. market_context
--    Point-in-time snapshot of the market regime at the moment a trade
--    decision was evaluated.  The context_embedding column stores a dense
--    vector summary used for similarity search (Task 5 / pattern matching).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_context (
    id                      SERIAL PRIMARY KEY,
    trade_id                INTEGER          REFERENCES trades(id),
    timestamp               TIMESTAMPTZ      NOT NULL,
    instrument              VARCHAR(20)      NOT NULL,

    -- Ichimoku cloud state per timeframe
    cloud_direction_4h      VARCHAR(10),     -- 'bullish' | 'bearish' | 'neutral'
    cloud_direction_1h      VARCHAR(10),
    tk_cross_15m            VARCHAR(10),     -- 'bullish' | 'bearish' | 'none'
    chikou_confirmation     BOOLEAN,
    cloud_thickness_4h      DOUBLE PRECISION,

    -- Supporting indicator values at decision time
    adx_value               DOUBLE PRECISION,
    atr_value               DOUBLE PRECISION,
    rsi_value               DOUBLE PRECISION,
    bb_width_percentile     DOUBLE PRECISION, -- percentile rank of current BB width

    -- Session label (london, newyork, asian, overlap)
    session                 VARCHAR(10),

    -- Nearest S/R zone data
    nearest_sr_distance     DOUBLE PRECISION, -- distance in price from nearest zone
    zone_confluence_score   INTEGER,          -- number of confluent factors at that zone

    -- 64-dimensional embedding of the full context snapshot
    context_embedding       VECTOR(64),

    created_at              TIMESTAMPTZ      DEFAULT NOW()
);

-- HNSW index for approximate nearest-neighbour queries using cosine
-- similarity.  m=16 / ef_construction=64 is a balanced default for a
-- medium-sized dataset; tune upward for recall if needed.
CREATE INDEX IF NOT EXISTS idx_market_context_embedding
    ON market_context
    USING hnsw (context_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_market_context_trade
    ON market_context (trade_id);

CREATE INDEX IF NOT EXISTS idx_market_context_instrument_ts
    ON market_context (instrument, timestamp DESC);

-- ---------------------------------------------------------------------------
-- 4. pattern_signatures
--    One record per stored pattern.  The embedding column is used for
--    vector search; outcome_r and win summarise the historical result so
--    the agent can weight similar patterns by past performance.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pattern_signatures (
    id            SERIAL PRIMARY KEY,
    context_id    INTEGER          REFERENCES market_context(id),
    trade_id      INTEGER          REFERENCES trades(id),
    embedding     VECTOR(64)       NOT NULL,
    outcome_r     DOUBLE PRECISION,  -- R-multiple achieved
    win           BOOLEAN,
    cluster_label INTEGER,           -- optional k-means / HDBSCAN label
    created_at    TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pattern_signatures_embedding
    ON pattern_signatures
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_pattern_signatures_cluster
    ON pattern_signatures (cluster_label);

-- ---------------------------------------------------------------------------
-- 5. screenshots
--    Chart screenshots associated with a trade.  Multiple screenshots can
--    belong to one trade, each tagged with a phase and the timeframe shown.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screenshots (
    id         SERIAL PRIMARY KEY,
    trade_id   INTEGER     REFERENCES trades(id),
    phase      VARCHAR(10) CHECK (phase IN ('pre_entry', 'entry', 'during', 'exit')),
    file_path  TEXT        NOT NULL,
    timeframe  VARCHAR(5),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_screenshots_trade
    ON screenshots (trade_id);

-- ---------------------------------------------------------------------------
-- 6. zones
--    Support / resistance / supply / demand zones detected across timeframes.
--    Zones persist across sessions; the status column tracks their lifecycle.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zones (
    id          SERIAL PRIMARY KEY,
    instrument  VARCHAR(20)      NOT NULL,
    zone_type   VARCHAR(20)      NOT NULL CHECK (zone_type IN (
                    'support', 'resistance', 'supply', 'demand', 'pivot'
                )),
    price_high  DOUBLE PRECISION NOT NULL,
    price_low   DOUBLE PRECISION NOT NULL,
    timeframe   VARCHAR(5)       NOT NULL,
    strength    DOUBLE PRECISION DEFAULT 0,   -- composite strength score
    touch_count INTEGER          DEFAULT 0,
    status      VARCHAR(10)      DEFAULT 'active'
                    CHECK (status IN ('active', 'tested', 'invalidated')),
    first_seen  TIMESTAMPTZ      NOT NULL,
    last_tested TIMESTAMPTZ,
    created_at  TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_zones_instrument_type
    ON zones (instrument, zone_type, status);

CREATE INDEX IF NOT EXISTS idx_zones_price_range
    ON zones (instrument, price_low, price_high);

-- ---------------------------------------------------------------------------
-- 7. zone_confluence
--    One row per confluent factor observed at a zone (e.g., Ichimoku cloud
--    edge aligns, Fibonacci level, round number).  Aggregating these rows
--    gives the zone_confluence_score stored in market_context.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zone_confluence (
    id               SERIAL PRIMARY KEY,
    zone_id          INTEGER     REFERENCES zones(id),
    confluence_type  VARCHAR(30) NOT NULL,  -- e.g. 'fibonacci_618', 'cloud_edge', 'round_number'
    value            DOUBLE PRECISION,
    timeframe        VARCHAR(5),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_zone_confluence_zone
    ON zone_confluence (zone_id);

-- ---------------------------------------------------------------------------
-- 8. decisions
--    Every decision made by the agent (enter, skip, exit, etc.) with the
--    full context captured as JSONB.  This provides a detailed audit trail
--    and feeds the edge_stats materialized view.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id               SERIAL PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    instrument       VARCHAR(20) NOT NULL,
    action           VARCHAR(12) NOT NULL CHECK (action IN (
                         'enter', 'skip', 'exit', 'partial_exit', 'modify'
                     )),
    trade_id         INTEGER     REFERENCES trades(id),
    signal_data      JSONB,     -- raw signal values that triggered the decision
    edge_results     JSONB,     -- per-edge pass/fail results (keyed by edge name)
    similarity_data  JSONB,     -- nearest-neighbour results from pattern search
    confluence_score INTEGER,
    reasoning        TEXT,      -- human-readable explanation
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decisions_trade
    ON decisions (trade_id);

CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
    ON decisions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_decisions_action
    ON decisions (action, instrument);

-- GIN index on edge_results so we can efficiently query for specific edge keys
CREATE INDEX IF NOT EXISTS idx_decisions_edge_results
    ON decisions USING gin (edge_results);

-- =============================================================================
-- Continuous Aggregates (higher timeframes derived from candles_1m)
-- =============================================================================
-- Real-time aggregation is enabled so that queries always include data up to
-- the most recent complete or partial bucket.

-- 5-minute bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_5m
WITH (timescaledb.continuous, timescaledb.materialized_only = FALSE) AS
SELECT
    time_bucket('5 minutes', timestamp) AS timestamp,
    instrument,
    first(open,  timestamp) AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close,  timestamp) AS close,
    sum(volume)             AS volume
FROM candles_1m
GROUP BY time_bucket('5 minutes', timestamp), instrument
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'candles_5m',
    start_offset  => INTERVAL '1 day',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- 15-minute bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_15m
WITH (timescaledb.continuous, timescaledb.materialized_only = FALSE) AS
SELECT
    time_bucket('15 minutes', timestamp) AS timestamp,
    instrument,
    first(open,  timestamp) AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close,  timestamp) AS close,
    sum(volume)             AS volume
FROM candles_1m
GROUP BY time_bucket('15 minutes', timestamp), instrument
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'candles_15m',
    start_offset  => INTERVAL '2 days',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- 1-hour bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_1h
WITH (timescaledb.continuous, timescaledb.materialized_only = FALSE) AS
SELECT
    time_bucket('1 hour', timestamp) AS timestamp,
    instrument,
    first(open,  timestamp) AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close,  timestamp) AS close,
    sum(volume)             AS volume
FROM candles_1m
GROUP BY time_bucket('1 hour', timestamp), instrument
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'candles_1h',
    start_offset  => INTERVAL '7 days',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- 4-hour bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_4h
WITH (timescaledb.continuous, timescaledb.materialized_only = FALSE) AS
SELECT
    time_bucket('4 hours', timestamp) AS timestamp,
    instrument,
    first(open,  timestamp) AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close,  timestamp) AS close,
    sum(volume)             AS volume
FROM candles_1m
GROUP BY time_bucket('4 hours', timestamp), instrument
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'candles_4h',
    start_offset  => INTERVAL '14 days',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- =============================================================================
-- edge_stats — Materialized View
-- Aggregates per-edge win rate and P&L across all closed trades so the agent
-- can monitor which edges contribute most to profitability.
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS edge_stats AS
SELECT
    e.key                                              AS edge_name,
    COUNT(*)                                           AS total_trades,
    AVG(CASE WHEN t.r_multiple > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
    AVG(t.r_multiple)                                  AS avg_r,
    SUM(t.pnl)                                         AS total_pnl
FROM trades t
JOIN decisions d   ON d.trade_id = t.id AND d.action = 'enter'
CROSS JOIN LATERAL jsonb_each(d.edge_results) e
WHERE t.status = 'closed'
GROUP BY e.key;

-- Allow CONCURRENTLY refresh (requires at least one unique index)
CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_stats_name
    ON edge_stats (edge_name);

-- Wrapper function so callers do not need to remember the SQL syntax
CREATE OR REPLACE FUNCTION refresh_edge_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY edge_stats;
END;
$$ LANGUAGE plpgsql;
