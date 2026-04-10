## Futures Workflow (TopstepX / ProjectX)

The project supports two trading profiles simultaneously — forex/the5ers
(legacy) and futures/TopstepX — switched via the instrument's ``class:``
field in ``config/instruments.yaml`` and the ``prop_firm.style`` field
in ``config/strategy.yaml``. Both paths share the same engine, strategy
blend, and dashboard; only the cost model, daily reset, sizing, and
tracker differ.

### Switching profiles

```yaml
# config/instruments.yaml
instruments:
  - symbol: "XAUUSD"
    class: futures          # ← forex | futures
    provider: "projectx"
    tick_size: 0.10
    tick_value_usd: 1.0
    contract_size: 10       # MGC = 10 troy oz
    commission_per_contract_round_trip: 1.40
    ...
```

```yaml
# config/strategy.yaml
prop_firm:
  style: topstep_combine_dollar   # ← the5ers_pct_phased | topstep_combine_dollar
  account_size: 50000
  profit_target_usd: 3000
  max_loss_limit_usd_trailing: 2000
  daily_loss_limit_usd: 1000
  consistency_pct: 50.0
  daily_reset_tz: "America/Chicago"
  daily_reset_hour: 17
```

To switch back to forex/the5ers: flip `class: forex` and the commented
legacy the5ers prop_firm block in strategy.yaml.

### Running a TopstepX backtest

The existing ``run_demo_challenge.py`` is profile-aware — it reads
``prop_firm.style`` and dispatches to the right tracker automatically.

```bash
python scripts/run_demo_challenge.py --mode validate \
    --data-file data/projectx_mgc_1m_20260101_20260409.parquet
```

The live dashboard at http://localhost:8501 shows TopstepX metrics
(current balance, MLL, distance to MLL, distance to target, best-day /
total consistency) when the active tracker is dollar-based.

### Profile-aware components

* ``src/config/profile.py`` — InstrumentClass, ProfileConfig, price_distance
* ``config/profiles/{forex,futures}.yaml`` — class-wide defaults
* ``src/risk/topstep_tracker.py`` — TopstepCombineTracker (trailing MLL)
* ``src/risk/session_clock.py`` — 5pm CT vs midnight UTC rollover
* ``src/risk/instrument_sizer.py`` — ForexLotSizer vs FuturesContractSizer
* ``src/backtesting/topstep_simulator.py`` — TopstepCombineSimulator
* ``src/backtesting/strategy_telemetry.py`` — StrategyTelemetryCollector

### Telemetry output

Every run writes ``reports/<run_id>/strategy_telemetry.parquet`` and
``strategy_telemetry_summary.json`` with per-strategy / per-session /
per-pattern aggregates. Task 17's retuning workflow reads this
directly. Task 27's mega-vision training data pipeline also consumes it.

### Optimization loop under TopstepX

```bash
python scripts/run_optimization_loop.py
```

When ``prop_firm.style == "topstep_combine_dollar"``, the loop auto-routes
to the ``topstep_combine_pass_score`` objective instead of the legacy
pct-based ``pass_rate``. Score is in [0, 1] (clamped); Optuna maximises.

### Mega-vision agent modes (planned — see docs/mega_vision_design.md)

```yaml
# config/mega_vision.yaml
mode: disabled          # disabled | shadow | authority
shadow_model: "claude-opus-4-6"
live_model: "claude-haiku-4-5"
cost_budget_usd: 10.00
kill_switch_env_var: "MEGA_VISION_KILL_SWITCH"
```

* **disabled** — native blender only, no agent involvement
* **shadow** — agent consulted on every signal but native execution is
  byte-identical to disabled mode; agent decisions are logged to
  ``reports/<run_id>/mega_vision_shadow.parquet`` for offline eval
* **authority** — agent filters the native signal set; safety gates
  enforce prop firm rules + position caps + kill switch even then
* **kill switch** — ``MEGA_VISION_KILL_SWITCH=1`` falls back to the
  native blender immediately without any agent calls

Mega-vision uses the ``claude-agent-sdk`` Python package authenticated
via the installed Claude CLI subscription — not direct API keys.

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
