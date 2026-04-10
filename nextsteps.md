# Next Steps — End-to-End Testing & Optimization

## Completed (2026-03-28)

| Step | Status |
|------|--------|
| Created `.env` with DB + MT5 credential placeholders | Done |
| Created Python 3.13 virtual environment (`.venv/`) | Done |
| Installed all dependencies (core + dev + MT5) | Done |
| Fixed Windows port-binding bug in live dashboard server | Done |
| Ran full test suite — **1216 passed, 0 failed** | Done |
| Ran synthetic backtest (0 trades — correct, noise has no signals) | Done |
| Generated dashboard report at `reports/dashboard_20260328_160700.html` | Done |

## 1. Fill in your `.env` credentials

Edit `.env` and fill in your actual values:

```env
DB_PASSWORD=your_postgres_password
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=The5ers-Demo
```

## 2. Set up PostgreSQL + TimescaleDB

Install PostgreSQL 16+ and the TimescaleDB extension, then run:

```sql
CREATE USER trader WITH PASSWORD 'your_password_here';
CREATE DATABASE trading OWNER trader;
\c trading
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
```

## 3. Download real XAU/USD data

Once the DB is running:

```bash
.venv/Scripts/python scripts/download_history.py \
    --instrument XAUUSD \
    --start 2019-01-01 \
    --end 2024-01-01
```

## 4. Run backtest on real data

```bash
.venv/Scripts/python scripts/run_demo_challenge.py \
    --mode backtest \
    --data-file data/xauusd_1m.parquet \
    --initial-balance 10000 \
    --output-dir reports
```

## 5. Run full validation pipeline (go/no-go)

This runs walk-forward analysis + Monte Carlo simulation + overfitting detection:

```bash
.venv/Scripts/python scripts/run_validation.py \
    --data-file data/xauusd_1m.parquet \
    --wf-trials 200 \
    --mc-sims 10000 \
    --initial-balance 10000 \
    --haircut 25 \
    --output-dir reports
```

### Validation targets

| Metric | Target | What to do if failing |
|--------|--------|----------------------|
| WFE | > 0.50 | Parameters are overfitting — reduce search space |
| DSR | > 0.95 | Sharpe is inflated by multiple testing |
| MC pass rate | > 50% | Strategy doesn't have enough edge |
| Haircut Sharpe | > 0.80 | Apply more conservative parameters |

## 6. Tune and iterate

Based on validation results, tune these files and re-run step 5:

- `config/strategy.yaml` — ADX threshold, ATR multiplier, risk %, confluence score
- `config/edges.yaml` — disable edges with negative marginal Sharpe
- `config/instruments.yaml` — broker-specific spread/ATR overrides

## 7. Paper trade on demo account

Once validation gives **GO** or strong **BORDERLINE**:

```bash
.venv/Scripts/python scripts/run_demo_challenge.py \
    --mode live \
    --mt5-login YOUR_LOGIN \
    --mt5-password YOUR_PASSWORD \
    --mt5-server The5ers-Demo \
    --initial-balance 10000
```

Run for **at least 2 weeks** on demo before funding a real challenge.

## 8. Go-live checklist

- [ ] All 1216 unit tests pass
- [ ] Walk-forward efficiency (WFE) > 0.50
- [ ] Deflated Sharpe Ratio (DSR) > 0.95
- [ ] Monte Carlo pass rate > 50%
- [ ] 2+ weeks of demo trading matches backtest expectations
- [ ] Circuit breaker enabled (2% daily DD halt)
- [ ] Friday close edge enabled (no weekend risk)
- [ ] News filter enabled (no red-flag event trading)
