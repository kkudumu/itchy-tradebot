# XAU/USD Ichimoku Trading Agent — Intent

## Vision
Autonomous gold trading agent for The5ers prop firm challenge, using Ichimoku Kinko Hyo as the primary indicator framework with multi-timeframe confluence analysis, adaptive risk management, and machine learning-enhanced decision making. The system targets >50% challenge pass rate through rigorous pre-challenge validation with Monte Carlo simulation and walk-forward analysis.

## Architecture Decisions
| Decision | Choice | Reasoning |
|---|---|---|
| Primary indicator framework | Ichimoku Kinko Hyo (9/26/52) | Provides trend direction, momentum, support/resistance, and entry timing in a single framework — reduces indicator redundancy |
| Multi-timeframe hierarchy | 4H -> 1H -> 15M -> 5M | Top-down filtering reduces false signals; 4H provides trend context, 5M provides precise entry timing |
| Lookahead prevention | `.shift(1)` on all higher-TF indicators | Prevents future data leakage in backtests — the most common source of unrealistic backtest results |
| Backtesting approach | Custom event loop instead of vectorbt Portfolio.from_orders() | Portfolio.from_orders() cannot handle partial exits (50% at 2R, trail remainder) — custom loop provides full control over exit management |
| Risk parameters | All hard-coded, not optimizable | Prevents optimization from discovering risk parameters that overfit to historical data — risk rules must be fixed by trading policy |
| Exit strategy | Hybrid 50/50 (partial TP + Kijun trail) | Locks in profit at 2R while allowing winners to run via Kijun-Sen trailing stop — balances certainty with upside |
| No breakeven before 1R | Hard-coded anti-trap rule | Moving stop to breakeven too early gets stopped out by noise before the trade can develop |
| Edge filter architecture | 12 independent toggleable filters | Each edge can be isolated, tested, and disabled independently — enables systematic edge contribution analysis |
| Feature vectors | 64-dim manual vectors, not learned embeddings | Manual feature engineering provides interpretable dimensions; 64 dims balances expressiveness with similarity search performance |
| MT5 bridge | Lazy import, fully mocked on Linux | MetaTrader5 SDK is Windows-only; lazy import allows the full codebase to run on Linux for development and CI |
| Database | TimescaleDB + pgvector on PostgreSQL | Time-series optimized storage for OHLCV data + vector similarity search for trade pattern matching in a single database |
| Learning phases | Mechanical (0-99) -> Statistical (100-499) -> Similarity (500+) | Prevents the system from adapting before it has sufficient data — mechanical phase forces discipline during initial trades |
| Validation approach | 25% haircut on all magnitude metrics | Assumes backtest results are 25% optimistic — conservative adjustment before go/no-go decision |

## Module Map
| Module | Purpose | Key Relationships |
|---|---|---|
| src/config | YAML configuration loading with Pydantic v2 validation | Depended by all modules for settings |
| src/data | Dukascopy .bi5 historical data download, normalization, and bulk loading | Depends on config, database |
| src/database | PostgreSQL/TimescaleDB connection pooling and schema management | Depended by data, learning, engine |
| src/indicators | Ichimoku calculator, confluence indicators (ADX/ATR/RSI/BB), session identification, divergence detection | Depended by strategy, backtesting |
| src/zones | Support/resistance cluster detection (DBSCAN), supply/demand zones, pivot points, confluence density scoring | Depended by strategy, backtesting |
| src/strategy | Multi-timeframe signal engine with confluence scoring and lookahead prevention | Depends on indicators, zones; depended by engine, backtesting |
| src/edges | 12 toggleable edge optimization filters with pipeline manager | Depends on config; depended by engine, backtesting |
| src/risk | Position sizing, hybrid exit management, circuit breaker, trade manager | Depended by engine, backtesting |
| src/execution | MT5 bridge for live trading, order management, screenshots, account monitoring | Depends on risk; depended by engine |
| src/engine | Central decision engine orchestrating scan -> signal -> edges -> execute -> log loop | Depends on strategy, edges, risk, execution, learning |
| src/learning | Adaptive 3-phase learning, feature vectors, pgvector embeddings, similarity search | Depends on database; depended by engine, backtesting |
| src/optimization | Optuna parameter optimization, walk-forward analysis, overfitting detection, edge isolation testing | Depends on backtesting |
| src/simulation | Monte Carlo simulation with fat-tailed distributions and block bootstrapping | Depends on risk config; depended by validation |
| src/validation | Pre-challenge go/no-go pipeline with Wilson CI, haircut, threshold checking | Depends on optimization, simulation |
| src/backtesting | Event-driven backtesting engine with prop firm tracking, metrics, dashboards | Depends on strategy, edges, risk, learning |

## Cross-Cutting Decisions
- **Immutable risk rules**: Position sizing caps, circuit breaker thresholds, and exit R-multiples are hard-coded constants across all modules — they cannot be overridden by configuration or optimization
- **ATR-based normalization**: All distance measurements (stop loss, zone proximity, position sizing) use ATR as the normalizing unit — makes the system adaptive to volatility without parameter changes
- **Vectorized computation**: All indicator and signal calculations use numpy/pandas vectorized operations — no Python loops over price bars
- **Structured reasoning traces**: Every trade decision includes a human-readable reasoning string documenting why the signal was taken or rejected — enables post-hoc analysis
- **Source column on trades**: Every trade record carries a `source` field ('backtest'/'live'/'paper') — enables filtering and prevents mixing backtest results with live performance
