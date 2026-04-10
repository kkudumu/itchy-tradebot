## Backtest & Optimization Workflow

When the user says "backtest and optimize [strategy] for [date range]":

1. **Validate data** — Check if `data/xauusd_1m_*.parquet` covers the requested date range. If not, download real data using the `histdata` Python package (pip install histdata) which pulls from https://github.com/philipperemy/FX-1-Minute-Data — our baseline data source. XAU/USD available from 2009.
   ```python
   from histdata import download_hist_data as dl
   from histdata.api import Platform as P, TimeFrame as TF
   # Past years: month=None for full year
   dl(year='2024', month=None, pair='xauusd', platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
   ```
   Then combine CSVs into parquet (semicolon-delimited, datetime format YYYYMMDD HHMMSS in EST → add 5h for UTC).
   **NEVER use `--synthetic-data` or generated fake data. Always use real market data.**

2. **Set active strategy** — Update `config/strategy.yaml` → `active_strategies` to only include the requested strategy.

3. **Run the full validation pipeline** with the live dashboard:
   ```bash
   python scripts/run_demo_challenge.py --mode validate \
       --data-file data/xauusd_1m_<range>.parquet \
       --wf-trials 100 --mc-sims 5000
   ```
   This runs: backtest with live dashboard (http://localhost:8501) → 22-day rolling window challenge simulation (Phase 1: 8% target, Phase 2: 5% target) → Monte Carlo (sequence robustness) → go/no-go verdict with 25% haircut.

4. **Do NOT** write a custom backtest script. The existing `run_demo_challenge.py` already has the live dashboard, charts, trades, equity curve, optimization tab — everything wired up.

## Strategy Learnings
- Run opt_iter_001_20260330_173002: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_002_20260330_174717: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_003_20260330_180204: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_004_20260330_180847: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_005_20260330_181517: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_006_20260330_182136: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_007_20260330_182755: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_008_20260330_183423: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_009_20260330_184110: enabled, min_confluence_score, tier_c, london_entry_end_utc, min_range_pips → WinRate 6.7%→6.7% (kept)
- Run opt_iter_001_20260330_203156: (no changes) → WinRate 12.5%→12.5% (kept)
- Run opt_iter_001_20260331_032138: enabled, candle_limit, stop_multiplier, tp_r_multiple → WinRate 7.9%→3.5% (reverted)
- Run opt_iter_002_20260331_043156: initial_risk_pct, reduced_risk_pct → WinRate 2.5%→2.5% (kept)
- Run opt_iter_003_20260331_052148: kijun_proximity_atr, initial_risk_pct, reduced_risk_pct, # Parse London entry window from config (default 06, start_str = self._cfg.get("london_entry_start_utc", "06 → WinRate 2.5%→2.5% (reverted)
