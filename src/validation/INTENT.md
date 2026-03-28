# Validation — Intent

## Functions

### FullValidationResult (dataclass)
- **Does**: Holds the complete output of the pre-challenge validation pipeline including threshold results, overfitting checks, Monte Carlo results, OOS metrics, verdict, and recommendations.
- **Why**: Bundles all validation artifacts into a single object so downstream report generators and decision logic can access everything without re-running the pipeline.
- **Relationships**: Produced by GoNoGoValidator.run_full_validation, consumed by ValidationReportGenerator.
- **Decisions**: Includes both raw oos_metrics dict and structured validation_result to serve both programmatic and display consumers.

### GoNoGoValidator.__init__(self, data: pd.DataFrame, config: Optional[dict], initial_balance: float, haircut_pct: float) -> None
- **Does**: Initializes the validation pipeline orchestrator with walk-forward, overfit, Monte Carlo, and threshold checker components.
- **Why**: Centralizes all pre-challenge validation dependencies so the pipeline can be run with a single method call.
- **Relationships**: Creates WalkForwardAnalyzer, OverfitDetector, MonteCarloSimulator, ThresholdChecker.
- **Decisions**: Accepts optional config dict to override default challenge parameters (profit target, DD limits, window sizes).

### GoNoGoValidator.run_full_validation(self, n_wf_trials: int, n_mc_sims: int, storage: Optional[str], seed: Optional[int]) -> FullValidationResult
- **Does**: Executes the complete 7-step validation pipeline: walk-forward analysis, OOS trade collection, metrics calculation, Wilson CI, overfitting checks, Monte Carlo simulation, and threshold checking.
- **Why**: Provides a single go/no-go decision for whether the strategy is ready for a funded prop firm challenge, with full auditability.
- **Relationships**: Calls WalkForwardAnalyzer.run, ThresholdChecker.check_all, MonteCarloSimulator.run, OverfitDetector.check_all, PerformanceMetrics.calculate; produces FullValidationResult.
- **Decisions**: Returns immediate NO-GO when zero OOS trades are collected rather than proceeding with meaningless downstream analysis.

### GoNoGoValidator._collect_oos_trades(self, wf_result: WFResult) -> List[dict]
- **Does**: Extracts and concatenates OOS trades from all walk-forward windows into a single flat list.
- **Why**: Creates the unified OOS trade set that all subsequent metrics and simulations operate on.
- **Relationships**: Called by run_full_validation; reads from WFResult.oos_trades.
- **Decisions**: None.

### GoNoGoValidator._calculate_oos_metrics(self, oos_trades: List[dict], wf_result: WFResult) -> dict
- **Does**: Builds a synthetic equity curve from OOS R-multiples and calculates comprehensive performance metrics using PerformanceMetrics.
- **Why**: Converts raw trade outcomes into the metric keys that ThresholdChecker expects, using a neutral 1% risk normalization.
- **Relationships**: Called by run_full_validation; calls PerformanceMetrics.calculate.
- **Decisions**: Uses monotonically increasing 5-minute synthetic timestamps to avoid duplicate-label errors when multiple trades share dates.

### GoNoGoValidator._run_overfit_checks(self, wf_result: WFResult, n_trials: int) -> OverfitReport
- **Does**: Runs DSR and WFE overfitting checks by creating a synthetic Optuna study from walk-forward OOS Sharpe values.
- **Why**: Detects whether the strategy's in-sample performance is inflated by overfitting, without requiring a live optimization session.
- **Relationships**: Called by run_full_validation; calls OverfitDetector.check_all.
- **Decisions**: Creates a placeholder Optuna study because OverfitDetector requires one, even though DSR is computed from OOS Sharpe distribution.

### GoNoGoValidator._run_monte_carlo(self, oos_trades: List[dict], n_mc_sims: int, seed: Optional[int]) -> MCResult
- **Does**: Runs the Monte Carlo simulation from OOS trade results with fat tails and block bootstrap enabled.
- **Why**: Estimates the probability of passing a real challenge using the validated OOS trade distribution.
- **Relationships**: Called by run_full_validation; calls MonteCarloSimulator.run.
- **Decisions**: Returns a zero-pass-rate MCResult on failure to trigger NO-GO on the MC threshold rather than crashing the pipeline.

### GoNoGoValidator._determine_verdict(self, validation_result: ValidationResult) -> str
- **Does**: Derives the final GO/NO-GO/BORDERLINE verdict from threshold results, marking BORDERLINE when all pass but margins are within 10% of thresholds.
- **Why**: Adds a nuanced middle ground between pass and fail so thin-margin results get additional human review before committing capital.
- **Relationships**: Called by run_full_validation; reads ValidationResult.
- **Decisions**: Uses 10% relative margin as the BORDERLINE boundary to flag metrics that could easily flip under slightly different conditions.

### GoNoGoValidator._build_recommendations(self, validation_result: ValidationResult, overfit_report: OverfitReport, mc_result: MCResult, verdict: str) -> List[str]
- **Does**: Generates actionable text recommendations based on which thresholds failed, overfitting signals, and Monte Carlo results.
- **Why**: Translates numerical results into specific next steps so the user knows exactly what to improve.
- **Relationships**: Called by run_full_validation.
- **Decisions**: Always includes the demo challenge reminder regardless of verdict as a safety guardrail.

### GoNoGoValidator._immediate_no_go(self, timestamp: datetime, wf_result: WFResult, reason: str) -> FullValidationResult
- **Does**: Returns a fully populated FullValidationResult with NO-GO verdict and empty/zero metrics when the pipeline cannot proceed.
- **Why**: Provides a clean early-exit path that still returns a valid result object, avoiding None checks in downstream code.
- **Relationships**: Called by run_full_validation when zero OOS trades.
- **Decisions**: None.

### ThresholdResult (dataclass)
- **Does**: Stores the pass/fail verdict for a single metric threshold including raw value, haircutted value, threshold, margin, and direction.
- **Why**: Provides per-metric auditability so users can see exactly why each threshold passed or failed.
- **Relationships**: Produced by ThresholdChecker.check_metric, consumed by ValidationResult.
- **Decisions**: None.

### ValidationResult (dataclass)
- **Does**: Aggregates pass/fail results across all thresholds with overall verdict, counts, and critical failure names.
- **Why**: Provides a single object for the final threshold-level decision with easy access to failures.
- **Relationships**: Produced by ThresholdChecker.check_all, consumed by GoNoGoValidator._determine_verdict.
- **Decisions**: None.

### ThresholdChecker.__init__(self, haircut_pct: float) -> None
- **Does**: Configures the 25% haircut percentage applied to magnitude-based performance metrics before threshold comparison.
- **Why**: Makes threshold checking conservative by penalizing backtest-derived metrics to account for real-world slippage and overfitting.
- **Relationships**: Called by GoNoGoValidator.__init__.
- **Decisions**: Validates haircut_pct is in [0, 100) to prevent nonsensical negative or total haircuts.

### ThresholdChecker.apply_haircut(self, value: float, direction: str) -> float
- **Does**: Applies the configured haircut to a metric value, reducing "higher is better" metrics and inflating "lower is better" metrics.
- **Why**: Makes the threshold test more conservative by assuming real performance will be worse than backtest performance.
- **Relationships**: Called by check_metric.
- **Decisions**: Directional haircut application ensures both min-thresholds and max-thresholds become harder to pass.

### ThresholdChecker.check_metric(self, name: str, raw_value: float, spec: Optional[dict]) -> ThresholdResult
- **Does**: Checks a single metric against its threshold specification, applying the haircut if configured.
- **Why**: Encapsulates the per-metric logic (haircut, direction, margin calculation) for reuse and testability.
- **Relationships**: Called by check_all; produces ThresholdResult.
- **Decisions**: Defaults to "min" direction and no haircut when spec is missing, failing safely toward passing.

### ThresholdChecker.check_all(self, metrics: dict) -> ValidationResult
- **Does**: Checks all 9 go/no-go thresholds against the provided metrics dict, treating missing keys as 0.0.
- **Why**: Runs the complete threshold battery in a single call, producing the aggregate pass/fail verdict.
- **Relationships**: Called by GoNoGoValidator.run_full_validation; calls check_metric for each threshold.
- **Decisions**: Treats missing metrics as 0.0 rather than skipping them, so absent data fails rather than passes silently.

### ThresholdChecker.wilson_score_ci(wins: int, total: int, confidence: float) -> Tuple[float, float]
- **Does**: Computes the Wilson score confidence interval for a binomial proportion (win rate).
- **Why**: Provides a statistically rigorous win-rate confidence interval that works well for small samples, unlike the normal approximation.
- **Relationships**: Called by GoNoGoValidator.run_full_validation.
- **Decisions**: Uses a custom inverse-normal approximation (_inv_normal) to avoid a scipy dependency in this module.

### ValidationReportGenerator.__init__(self, title: str) -> None
- **Does**: Configures the report title for HTML and text output generation.
- **Why**: Allows customization of the report header for different instruments or strategies.
- **Relationships**: Called by user code and pipeline runners.
- **Decisions**: None.

### ValidationReportGenerator.generate_html(self, result: FullValidationResult) -> str
- **Does**: Generates a self-contained HTML report with verdict banner, threshold table, metrics summary, embedded charts, and recommendations.
- **Why**: Produces a single-file shareable report that requires no external assets, suitable for archiving and review.
- **Relationships**: Consumes FullValidationResult; calls internal _html component builders and _fig_to_base64.
- **Decisions**: Embeds matplotlib charts as base64 PNG data URIs to keep the report fully self-contained.

### ValidationReportGenerator.generate_text(self, result: FullValidationResult) -> str
- **Does**: Generates a plain-text summary of the validation results suitable for console output or logging.
- **Why**: Provides a quick-read format for terminal users and CI/CD logs where HTML is impractical.
- **Relationships**: Consumes FullValidationResult.
- **Decisions**: Uses fixed-width formatting for alignment in monospace terminals.

### ValidationReportGenerator.save_report(self, html: str, path: str) -> str
- **Does**: Writes the HTML report string to a file, creating directories as needed, and returns the absolute path.
- **Why**: Handles file I/O and directory creation so callers only need to provide the desired path.
- **Relationships**: Called by user code after generate_html.
- **Decisions**: Overwrites existing files silently since validation reports are timestamped and disposable.
