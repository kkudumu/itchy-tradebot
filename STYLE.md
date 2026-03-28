# Style Guide — XAU/USD Ichimoku Trading Agent

## Hard Limits (enforced)

- **Python 3.11+** required
- **Type hints** on all public function signatures
- **Pydantic v2** for all configuration models
- **No mutable default arguments** in function signatures
- **No global mutable state** — all state in class instances or function-local
- **Risk parameters are HARD-CODED constants** — never optimizable, never configurable at runtime
- **`.shift(1)` on ALL higher-timeframe indicators** before merge to lower TF — prevents lookahead bias
- **No bare `except:`** — always catch specific exceptions
- **No `print()` in library code** — use `logging` module

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `IchimokuBacktester`, `EdgeManager`)
- **Functions/methods**: `snake_case` (e.g., `check_entry`, `calculate_position_size`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `_MAX_RISK_PCT`, `_ABSOLUTE_MAX_DAILY_LOSS`)
- **Private members**: single underscore prefix `_` (e.g., `_scan_for_signal`)
- **Edge filter classes**: `{Name}Filter` or `{Name}Amplifier` (e.g., `TimeOfDayFilter`, `BBSqueezeAmplifier`)

## Architecture Patterns

- **Edge filters**: Inherit `EdgeFilter` base, implement `should_allow(context) -> EdgeResult`
- **Configuration**: YAML files in `config/`, loaded by `ConfigLoader`, validated by Pydantic models
- **Database access**: Dependency injection via constructor parameters (e.g., `db_pool=None`)
- **MT5 bridge**: Lazy import pattern — `MetaTrader5` only imported when actually called, not at module load
- **Backtesting**: Custom event-driven loop over 5M bars, not vectorbt's built-in Portfolio methods
- **Learning phases**: Mechanical (0-99 trades) -> Statistical (100-499) -> Similarity (500+)
- **Exit strategy**: Hybrid 50/50 — partial TP at 2R fixed, trail remainder with Kijun-Sen

## Testing Standards

- **pytest** as test runner
- **No database or network in unit tests** — mock all external I/O
- **Hand-computed reference values** for indicator tests (not just "output is not None")
- **Boundary condition tests** for all threshold-based logic
- **Edge filter tests**: test pass, fail, boundary, and disabled states for each filter
- **Risk tests**: verify hard-coded limits cannot be overridden

## Import Style

- Standard library first, then third-party, then `src.*` imports
- Prefer explicit imports: `from src.risk.position_sizer import AdaptivePositionSizer`
- Use `__init__.py` to expose public API of each package
- Lazy imports for heavy/optional dependencies (e.g., matplotlib, MetaTrader5)

## Code Organization

- One primary class per file (small helper classes/dataclasses in same file OK)
- Each `src/` subdirectory is a package with `__init__.py`
- Scripts in `scripts/` — entry points for CLI operations
- Tests mirror source structure: `src/risk/position_sizer.py` -> `tests/test_risk.py`
- Integration tests in `tests/integration/`
