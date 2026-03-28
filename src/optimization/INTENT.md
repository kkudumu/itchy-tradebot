# Optimization — Intent

## Functions

### OptunaOptimizer.__init__(self, data: pd.DataFrame, config: Optional[dict], initial_balance: float) -> None
- **Does**: Creates an optimizer that manages Optuna studies for single-objective and multi-objective strategy optimization.
- **Why**: Provides a high-level interface to Optuna that handles study creation, sampler configuration, and backtester lifecycle.
- **Relationships**: Uses PropFirmObjective and MultiObjective as callables; lazily creates IchimokuBacktester.
- **Decisions**: Lazily constructs the backtester to avoid requiring a live database connection at instantiation.

### OptunaOptimizer.optimize_single(self, n_trials: int, storage: Optional[str], study_name: str, n_jobs: int) -> optuna.Study
- **Does**: Runs single-objective TPE optimization maximising the prop firm composite score with median pruning.
- **Why**: Finds the parameter set that best balances Sharpe ratio with prop firm drawdown and return constraints.
- **Relationships**: Creates PropFirmObjective; returns completed Optuna study consumed by get_best_params and WalkForwardAnalyzer.
- **Decisions**: Uses TPE sampler with seed=42 for reproducibility; MedianPruner with 20 warm-up trials to avoid premature discarding.

### OptunaOptimizer.optimize_multi(self, n_trials: int, storage: Optional[str], study_name: str, n_jobs: int) -> optuna.Study
- **Does**: Runs NSGA-II multi-objective optimization across Sortino ratio, drawdown safety, and Calmar ratio.
- **Why**: Explores the Pareto front of return quality vs drawdown safety vs risk-adjusted return, giving the operator a range of viable parameter sets.
- **Relationships**: Creates MultiObjective; returns study with Pareto-optimal trials accessible via study.best_trials.
- **Decisions**: All three objectives are maximised; drawdown safety is encoded as (20 - max_dd_pct) so higher = safer.

### OptunaOptimizer.get_best_params(self, study: optuna.Study) -> dict
- **Does**: Extracts and validates the best parameter set from a completed study, converting optimizer keys to backtester config keys.
- **Why**: Bridges the gap between Optuna's sampled parameter names and the backtester's expected configuration format.
- **Relationships**: Called after optimize_single or optimize_multi; output used by WalkForwardAnalyzer and EdgeIsolationTester.
- **Decisions**: For multi-objective studies, selects the Pareto trial with the highest composite score sum.

### PropFirmObjective.__init__(self, backtester: IchimokuBacktester, data: pd.DataFrame, initial_balance: float) -> None
- **Does**: Creates a single-objective Optuna callable that evaluates strategy parameter combinations against prop firm constraints.
- **Why**: Encapsulates the scoring logic that penalises drawdown breaches, insufficient return, and low win rate into a single Optuna-compatible callable.
- **Relationships**: Used by OptunaOptimizer.optimize_single; delegates backtesting to IchimokuBacktester.
- **Decisions**: None.

### PropFirmObjective.__call__(self, trial: optuna.Trial) -> float
- **Does**: Evaluates one parameter combination by running a backtest and returning a composite score based on Sharpe ratio with prop firm penalty multipliers.
- **Why**: Guides Optuna toward parameter sets that pass prop firm challenges by penalising daily DD >5%, total DD >10%, return <8%, and win rate <45%.
- **Relationships**: Calls suggest_params and _run_backtest; called by Optuna's study.optimize.
- **Decisions**: Total DD >10% is a hard fail (returns 0.0); daily DD >5% is a 50% penalty; insufficient return scales proportionally.

### PropFirmObjective.suggest_params(self, trial: optuna.Trial) -> dict
- **Does**: Defines and samples the parameter search space for one trial, deriving all Ichimoku periods from a single scale factor.
- **Why**: Preserves the canonical 1:3:6 Ichimoku period ratio across the search space while exploring other strategy parameters.
- **Relationships**: Called by __call__ and MultiObjective.__call__; output passed to IchimokuBacktester.
- **Decisions**: Single ichimoku_scale factor (0.7-1.3) instead of independent period sampling to maintain period ratio integrity.

### MultiObjective.__init__(self, backtester: IchimokuBacktester, data: pd.DataFrame, initial_balance: float) -> None
- **Does**: Creates an NSGA-II multi-objective callable returning (Sortino, drawdown safety, Calmar) tuples.
- **Why**: Enables Pareto-front exploration of the trade-off between return quality, drawdown safety, and risk-adjusted return.
- **Relationships**: Delegates parameter sampling and backtesting to PropFirmObjective for consistency.
- **Decisions**: Reuses PropFirmObjective internally to keep the parameter space identical across single and multi-objective modes.

### MultiObjective.__call__(self, trial: optuna.Trial) -> tuple[float, float, float]
- **Does**: Returns (Sortino, drawdown safety, Calmar) for NSGA-II, with (0, 0, 0) for failed or zero-trade backtests.
- **Why**: Provides three complementary risk/return metrics that Optuna's NSGA-II sampler uses to build the Pareto front.
- **Relationships**: Calls PropFirmObjective.suggest_params and _run_backtest; called by Optuna's study.optimize.
- **Decisions**: All values clamped to non-negative to avoid sign confusion in the maximisation framework.

### WFWindow(window_index: int, is_start: pd.Timestamp, is_end: pd.Timestamp, oos_start: pd.Timestamp, oos_end: pd.Timestamp, is_sharpe: float, oos_sharpe: float, best_params: dict, oos_metrics: dict, oos_trades: list, oos_prop_firm: dict) -> WFWindow
- **Does**: Dataclass holding results for a single walk-forward window including IS/OOS boundaries, Sharpe ratios, and best parameters.
- **Why**: Provides per-window granularity for walk-forward analysis so each window can be inspected individually.
- **Relationships**: Created by WalkForwardAnalyzer.run; aggregated into WFResult.
- **Decisions**: None.

### WFResult(windows: List[WFWindow], wfe: float, oos_trades: list, oos_metrics: dict, is_sharpes: List[float], oos_sharpes: List[float]) -> WFResult
- **Does**: Dataclass holding aggregate results from a complete walk-forward run including WFE and concatenated OOS trades.
- **Why**: Provides the final output of walk-forward validation with both summary metrics and per-window detail.
- **Relationships**: Created by WalkForwardAnalyzer._aggregate_results; consumed by OverfitDetector.check_all.
- **Decisions**: WFE > 0.5 indicates acceptable out-of-sample generalisation.

### WalkForwardAnalyzer.__init__(self, is_months: int, oos_months: int, initial_balance: float) -> None
- **Does**: Creates a rolling walk-forward optimizer/validator with configurable in-sample and out-of-sample window lengths.
- **Why**: Tests whether optimised parameters generalise to unseen data by systematically validating on out-of-sample slices.
- **Relationships**: Uses OptunaOptimizer for IS optimization; uses IchimokuBacktester for OOS evaluation.
- **Decisions**: Default 12-month IS and 3-month OOS windows based on standard walk-forward practice for daily trading strategies.

### WalkForwardAnalyzer.run(self, data: pd.DataFrame, n_trials: int, storage: Optional[str]) -> WFResult
- **Does**: Executes the full rolling walk-forward analysis: slides windows through the dataset, optimizes on IS slices, evaluates on OOS slices, and computes WFE.
- **Why**: Provides the definitive test of whether the strategy's optimised parameters are robust or overfit to historical data.
- **Relationships**: Calls _build_windows, _optimize_is, _evaluate_oos, and _aggregate_results.
- **Decisions**: Windows slide forward by oos_months on each iteration so consecutive windows share no out-of-sample data.

### WalkForwardAnalyzer.walk_forward_efficiency(self, is_sharpe: List[float], oos_sharpe: List[float]) -> float
- **Does**: Computes Walk-Forward Efficiency as mean(OOS Sharpe) / mean(IS Sharpe).
- **Why**: Provides a single scalar metric for how well in-sample performance transfers to out-of-sample data.
- **Relationships**: Called by _aggregate_results; WFE consumed by OverfitDetector.check_all.
- **Decisions**: Returns 0.0 when either list is empty or IS mean is zero, indicating inability to assess.

### OverfitReport(dsr: float, dsr_pass: bool, wfe: float, wfe_pass: bool, plateau_pass: bool, plateau_cv: float, overall_pass: bool, notes: List[str]) -> OverfitReport
- **Does**: Dataclass consolidating pass/fail results from three overfitting checks: Deflated Sharpe Ratio, Walk-Forward Efficiency, and Plateau Test.
- **Why**: Provides a single object that summarises whether optimised parameters are likely overfit.
- **Relationships**: Created by OverfitDetector.check_all; consumed by the operator for go/no-go decisions.
- **Decisions**: overall_pass requires all three individual checks to pass.

### OverfitDetector.__init__(self) -> None
- **Does**: Creates a detector that runs statistical overfitting checks on Optuna studies and walk-forward results.
- **Why**: Guards against deploying over-fitted strategy parameters by applying three complementary statistical tests.
- **Relationships**: Consumes Optuna Study and WFResult; produces OverfitReport.
- **Decisions**: None.

### OverfitDetector.deflated_sharpe_ratio(self, sharpe: float, n_trials: int, var_sharpe: float, skew: float, kurtosis: float, T: int) -> float
- **Does**: Computes the Deflated Sharpe Ratio using the Bailey & Lopez de Prado (2014) formula, adjusting for multiple testing and non-normality.
- **Why**: Determines whether the best observed Sharpe ratio is statistically significant after accounting for the number of parameter combinations tested.
- **Relationships**: Called by check_all; uses _inv_normal and _normal_cdf helper functions.
- **Decisions**: DSR >= 0.95 threshold chosen as the standard significance level for strategy validation.

### OverfitDetector.plateau_test(self, study: optuna.Study, top_pct: float) -> tuple[bool, float]
- **Does**: Checks whether near-optimal parameters form a broad performance plateau by computing the coefficient of variation of top-decile trial scores.
- **Why**: Detects narrow performance spikes that indicate fragile, overfit parameter sets versus robust, flat optima.
- **Relationships**: Called by check_all; analyses completed trials from the Optuna study.
- **Decisions**: CV threshold of 0.30 balances sensitivity; skipped with fewer than 5 trials.

### OverfitDetector.check_all(self, study: optuna.Study, wf_result, n_trials: int) -> OverfitReport
- **Does**: Runs all three overfitting checks (DSR, WFE, plateau) and returns a consolidated OverfitReport with pass/fail flags and diagnostic notes.
- **Why**: Provides a single entry point for comprehensive overfitting assessment before deploying optimised parameters.
- **Relationships**: Calls deflated_sharpe_ratio, plateau_test, and reads wf_result.wfe.
- **Decisions**: Uses conservative return distribution assumptions (skew=-0.5, kurtosis=1.0) when raw return series are unavailable.

### EdgeImpact(edge_name: str, base_sharpe: float, with_edge_sharpe: float, marginal_impact: float, base_trades: int, with_edge_trades: int, recommended: bool) -> EdgeImpact
- **Does**: Dataclass holding the marginal OOS performance impact of a single strategy edge.
- **Why**: Quantifies each edge's contribution by comparing Sharpe with and without the edge on out-of-sample data.
- **Relationships**: Created by EdgeIsolationTester.test_single_edge; consumed by the operator for edge configuration decisions.
- **Decisions**: recommended is True when marginal_impact > 0.

### EdgeIsolationTester.__init__(self, initial_balance: float) -> None
- **Does**: Creates a tester that measures each strategy edge's marginal contribution by running paired backtests on OOS data.
- **Why**: Provides empirical evidence for which edges add value and which should be disabled, complementing the EdgeReviewer's statistical analysis.
- **Relationships**: Uses IchimokuBacktester for each paired test; results complement EdgeReviewer suggestions.
- **Decisions**: Operates on OOS data only to avoid contamination by in-sample optimization.

### EdgeIsolationTester.test_all_edges(self, data: pd.DataFrame, base_config: dict) -> Dict[str, EdgeImpact]
- **Does**: Evaluates every named edge's marginal OOS contribution by running baseline (disabled) and enabled backtests for each.
- **Why**: Produces a complete map of edge values so the operator can make informed enable/disable decisions across all 11 edges.
- **Relationships**: Calls _test_single_edge for each edge in _EDGE_NAMES.
- **Decisions**: Tests all edges against the same base_config for fair comparison.

### EdgeIsolationTester.test_single_edge(self, data: pd.DataFrame, base_config: dict, edge_name: str) -> EdgeImpact
- **Does**: Evaluates the marginal impact of one specific edge by running paired backtests with and without it.
- **Why**: Allows targeted testing of a single edge without the overhead of testing all edges.
- **Relationships**: Public wrapper around _test_single_edge; called directly or by test_all_edges.
- **Decisions**: None.
