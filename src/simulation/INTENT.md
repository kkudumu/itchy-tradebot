# Simulation — Intent

## Functions

### ChallengeOutcome (dataclass)
- **Does**: Holds the outcome of a single simulated prop firm challenge attempt including pass/fail, failure reason, equity curve, and drawdown stats.
- **Why**: Provides a structured record of each simulation so aggregation and visualization can operate on typed data rather than raw dicts.
- **Relationships**: Produced by MonteCarloSimulator._simulate_single, consumed by MCResult, MCVisualizer.
- **Decisions**: Stores full per-day equity_curve for downstream fan-chart visualization despite memory cost.

### MCResult (dataclass)
- **Does**: Aggregates statistics from all Monte Carlo simulation outcomes including pass rate, failure breakdowns, and convergence metadata.
- **Why**: Provides a single object that downstream consumers (validation pipeline, visualizer, reports) can query without re-computing stats.
- **Relationships**: Produced by MonteCarloSimulator.run, consumed by GoNoGoValidator, MCVisualizer, ValidationReportGenerator.
- **Decisions**: Includes running_pass_rates list for convergence visualization even though it duplicates derivable data.

### MonteCarloSimulator.__init__(self, initial_balance: float, profit_target_pct: float, max_daily_dd_pct: float, max_total_dd_pct: float, daily_circuit_breaker_pct: float, time_limit_days: int, initial_risk_pct: float, reduced_risk_pct: float, phase_threshold_pct: float) -> None
- **Does**: Configures the simulator with The5ers-style prop firm challenge parameters and phased risk sizing thresholds.
- **Why**: Encapsulates all challenge rules in one place so simulations accurately reflect real prop firm constraints.
- **Relationships**: Called by GoNoGoValidator.__init__, user code.
- **Decisions**: Validates all parameters eagerly to fail fast on misconfiguration.

### MonteCarloSimulator.run(self, trade_results: List[dict], n_simulations: int, use_fat_tails: bool, block_bootstrap: bool, seed: Optional[int]) -> MCResult
- **Does**: Runs n_simulations challenge attempts by resampling trade results with optional fat-tail distributions and block bootstrap, then aggregates outcomes.
- **Why**: Estimates the probability of passing a prop firm challenge from backtest trade data, accounting for statistical uncertainty and tail risk.
- **Relationships**: Calls _simulate_single, _resample_trades, _check_convergence, _aggregate; calls TradeDistribution.fit/sample and BlockBootstrapper.resample.
- **Decisions**: Uses early convergence detection to confirm 10,000 sims is sufficient; fits fat-tail distribution once and samples per sim for efficiency.

### MonteCarloSimulator._simulate_single(self, trades: List[dict], rng: np.random.Generator) -> ChallengeOutcome
- **Does**: Simulates one prop firm challenge by processing trades day-by-day with phased risk sizing, circuit breaker, and drawdown limits.
- **Why**: Models a realistic challenge attempt with all The5ers constraints enforced in the correct order.
- **Relationships**: Called by run; calls _group_by_day_index.
- **Decisions**: Checks daily DD before total DD; circuit breaker suspends trading for the day without failing the challenge.

### MonteCarloSimulator._resample_trades(self, trades: List[dict], rng: np.random.Generator, block_bootstrap: bool, bootstrapper: Optional[BlockBootstrapper], dist: Optional[TradeDistribution], use_fat_tails: bool, target_count: int) -> List[dict]
- **Does**: Generates a resampled trade sequence for one simulation using either block bootstrap or IID sampling, optionally replacing R-multiples with fat-tail samples.
- **Why**: Creates statistically varied trade sequences that stress-test the strategy beyond the historical sample.
- **Relationships**: Called by run; calls BlockBootstrapper.resample, TradeDistribution.sample.
- **Decisions**: Block bootstrap path resamples day-blocks then optionally overlays fat-tail R-multiples to combine correlation preservation with tail stress testing.

### MonteCarloSimulator._check_convergence(running_pass_rates: List[float], n_done: int) -> Tuple[bool, int]
- **Does**: Tests whether the running pass rate has stabilized by checking if the standard deviation over a trailing window is below 1 percentage point.
- **Why**: Confirms that the simulation count is sufficient for a reliable pass-rate estimate without running unnecessary iterations.
- **Relationships**: Called by run.
- **Decisions**: Uses 1% tolerance and 1000-simulation window after a 2000-simulation minimum to balance accuracy with early termination.

### MonteCarloSimulator._aggregate(outcomes: List[ChallengeOutcome], n_simulations: int, running_pass_rates: List[float], convergence_reached: bool, convergence_at: int) -> MCResult
- **Does**: Computes summary statistics (pass rate, failure breakdowns, day statistics) from all simulation outcomes.
- **Why**: Centralizes aggregation logic so run() stays clean and results are consistent.
- **Relationships**: Called by run; produces MCResult.
- **Decisions**: None.

### MonteCarloSimulator._avg_trades_per_day(trades: List[dict]) -> float
- **Does**: Estimates the average number of trades per calendar day from entry_time fields in the trade list.
- **Why**: Determines how many trades to generate per simulation to match historical trading frequency.
- **Relationships**: Called by run.
- **Decisions**: Falls back to 1.0 when no date information is available.

### MonteCarloSimulator._group_by_day_index(trades: List[dict]) -> List[List[dict]]
- **Does**: Groups a sequenced trade list into per-day buckets using entry_time dates, or chunks of 3 when dates are unavailable.
- **Why**: Enables day-by-day simulation with daily drawdown tracking which requires knowing which trades belong to the same calendar day.
- **Relationships**: Called by _simulate_single.
- **Decisions**: Uses chunk size of 3 as a reasonable pseudo-day estimate when timestamps are missing.

### FittedDistribution (dataclass)
- **Does**: Stores the fitted non-central t-distribution parameters, win probability, empirical fallback arrays, and fit quality metrics.
- **Why**: Separates distribution state from fitting logic so the fitted model can be passed around and inspected independently.
- **Relationships**: Produced by TradeDistribution.fit, consumed by TradeDistribution.sample.
- **Decisions**: Stores loss-side parameters as private attributes to keep the dataclass interface focused on the primary (win) side.

### TradeDistribution.__init__(self, tail_df_cap: float, min_df: float) -> None
- **Does**: Configures bounds on degrees of freedom for the non-central t-distribution fit.
- **Why**: Prevents degenerate fits by clamping df to a range that ensures meaningful tail behavior.
- **Relationships**: Called by MonteCarloSimulator.run (indirectly).
- **Decisions**: Default min_df=2.1 keeps finite variance; cap=30 prevents near-normal tails that would understate risk.

### TradeDistribution.fit(self, r_multiples: List[float]) -> FittedDistribution
- **Does**: Fits separate non-central t-distributions to the winning and losing sides of observed R-multiples.
- **Why**: Models the fat-tailed, asymmetric distribution of real trade outcomes more accurately than a normal distribution.
- **Relationships**: Called by MonteCarloSimulator.run; calls _fit_nct.
- **Decisions**: Fits wins and losses separately because their tail shapes differ; negates losses before fitting to keep the fitter in positive domain.

### TradeDistribution.sample(self, n: int, rng: np.random.Generator) -> np.ndarray
- **Does**: Draws n synthetic R-multiples from the fitted distribution, respecting the historical win probability.
- **Why**: Generates realistic synthetic trade outcomes for Monte Carlo simulations.
- **Relationships**: Called by MonteCarloSimulator._resample_trades; calls _sample_side.
- **Decisions**: Determines win/loss per sample via Bernoulli draw using fitted win_probability rather than sampling from a single combined distribution.

### BlockBootstrapper.__init__(self, trades: List[dict], date_key: str) -> None
- **Does**: Groups trades into calendar-day blocks for block bootstrap resampling.
- **Why**: Preserves intra-day trade correlations (e.g., session volatility clusters) that IID bootstrap would destroy.
- **Relationships**: Called by MonteCarloSimulator.run.
- **Decisions**: Assigns trades without parseable dates to unique singleton buckets to avoid silently dropping them.

### BlockBootstrapper.n_days (property) -> int
- **Does**: Returns the number of distinct trading days in the dataset.
- **Why**: Allows callers to assess dataset breadth before resampling.
- **Relationships**: Called by MonteCarloSimulator.run for logging.
- **Decisions**: None.

### BlockBootstrapper.resample(self, n_days: int, rng: np.random.Generator) -> List[dict]
- **Does**: Resamples day-blocks with replacement to produce n_days worth of trades as a flat list.
- **Why**: Generates a resampled trade sequence that preserves within-day correlations for Monte Carlo input.
- **Relationships**: Called by MonteCarloSimulator._resample_trades.
- **Decisions**: Returns a flat list rather than grouped blocks because downstream simulation handles day grouping itself.

### BlockBootstrapper.resample_n_trades(self, n_trades: int, rng: np.random.Generator) -> List[dict]
- **Does**: Resamples whole day-blocks until at least n_trades have been collected, then truncates to exactly n_trades.
- **Why**: Provides a fixed-count trade sequence when the simulation needs a specific number of trades rather than a specific number of days.
- **Relationships**: Available for alternate simulation modes.
- **Decisions**: Truncates at exactly n_trades which may split a day-block, but this is acceptable since the block structure already served its correlation-preservation purpose.

### MCVisualizer.__init__(self, style: str, figsize_default: tuple) -> None
- **Does**: Configures the matplotlib style and default figure size for all chart methods.
- **Why**: Centralizes visual styling so all Monte Carlo charts share a consistent appearance.
- **Relationships**: Called by user code and report generators.
- **Decisions**: Uses seaborn-v0_8-darkgrid with automatic fallback for older matplotlib versions.

### MCVisualizer.equity_fan(self, outcomes: List[ChallengeOutcome], n_sample: int, show_percentile_bands: bool) -> Figure
- **Does**: Renders a fan chart of sampled equity curves with pass/fail coloring and percentile bands.
- **Why**: Visualizes the spread of possible outcomes to help assess strategy robustness and risk.
- **Relationships**: Consumes ChallengeOutcome.equity_curve from MCResult.outcomes.
- **Decisions**: Samples a balanced mix of pass/fail curves to avoid visual bias toward the majority outcome.

### MCVisualizer.pass_rate_convergence(self, result: MCResult) -> Figure
- **Does**: Plots the running pass rate versus simulation number with convergence marker and tolerance band.
- **Why**: Confirms that the simulation count was sufficient for a stable estimate and diagnoses any instability.
- **Relationships**: Consumes MCResult.running_pass_rates and convergence metadata.
- **Decisions**: None.

### MCVisualizer.failure_breakdown(self, result: MCResult, chart_type: str) -> Figure
- **Does**: Renders a bar or pie chart decomposing all outcomes into pass/fail categories by failure reason.
- **Why**: Identifies which constraint (daily DD, total DD, timeout) is the primary risk driver so the strategy can be tuned.
- **Relationships**: Consumes MCResult failure rates.
- **Decisions**: Defaults to horizontal bar chart because it conveys proportions more clearly than pie charts.

### MCVisualizer.daily_dd_distribution(self, outcomes: List[ChallengeOutcome], bins: int) -> Figure
- **Does**: Renders a histogram of the worst daily drawdown observed in each simulation.
- **Why**: Shows how close the strategy typically comes to the daily drawdown limit, revealing hidden fragility.
- **Relationships**: Consumes ChallengeOutcome.max_daily_dd from MCResult.outcomes.
- **Decisions**: Hardcodes circuit breaker and limit lines at 2% and 5% since MCVisualizer does not have access to simulator parameters.
