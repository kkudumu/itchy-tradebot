# Learning — Intent

## Functions

### PreTradeInsight(similar_trades: List[SimilarTrade], expected_win_rate: float, expected_avg_r: float, expected_expectancy: float, confidence: float, confidence_adjustment: float, statistical_filters: Dict[str, bool], recommendation: str, reasoning: str) -> PreTradeInsight
- **Does**: Dataclass holding the advisory output of a pre-trade analysis, including similar trades, expected performance, filter flags, and a recommendation.
- **Why**: Provides a structured, self-contained package of learning insights that the decision engine can act on without understanding the learning internals.
- **Relationships**: Created by AdaptiveLearningEngine.pre_trade_analysis; consumed by DecisionEngine.
- **Decisions**: All fields are advisory only — the trading engine decides whether to act on each piece.

### AdaptiveLearningEngine.__init__(self, similarity_search: Optional[SimilaritySearch], embedding_engine: Optional[EmbeddingEngine], db_pool, stats_analyzer: Optional[StatsAnalyzer]) -> None
- **Does**: Initialises the three-phase adaptive learning engine with optional similarity search, embedding, and statistics components.
- **Why**: Serves as the central learning coordinator that improves strategy execution over time through mechanical, statistical, and similarity-based phases.
- **Relationships**: Composes SimilaritySearch, EmbeddingEngine, and StatsAnalyzer; called by the decision engine.
- **Decisions**: Creates default EmbeddingEngine and StatsAnalyzer when not provided; similarity_search is optional since it requires pgvector.

### AdaptiveLearningEngine.get_phase(self, total_trades: Optional[int]) -> str
- **Does**: Returns the current learning phase name based on the total trade count: "mechanical" (0-99), "statistical" (100-499), or "similarity" (500+).
- **Why**: Controls which learning features are active, preventing premature optimization on insufficient data.
- **Relationships**: Called by pre_trade_analysis, should_filter, and get_confidence_adjustment.
- **Decisions**: Conservative thresholds ensure a clean baseline dataset before applying any learning adjustments.

### AdaptiveLearningEngine.set_total_trades(self, n: int) -> None
- **Does**: Manually sets the total trade counter for bootstrapping or testing.
- **Why**: Allows the engine to resume from a known trade count after restart without re-counting from the database.
- **Relationships**: Called during engine initialization or by tests.
- **Decisions**: Clamps to non-negative integer.

### AdaptiveLearningEngine.pre_trade_analysis(self, context: dict) -> PreTradeInsight
- **Does**: Analyses a candidate trade setup before entry, returning similarity data, statistical filter flags, and a recommendation of "proceed", "caution", or "skip".
- **Why**: Provides the decision engine with historical context so it can make informed entry decisions without the learning logic leaking into the engine.
- **Relationships**: Calls _compute_statistical_filters, _run_similarity_search, _compute_confidence_adjustment, and _build_insight.
- **Decisions**: Returns "proceed" with no adjustments during the mechanical phase to build a clean baseline.

### AdaptiveLearningEngine.post_trade_analysis(self, trade_result: dict) -> None
- **Does**: Increments the internal trade counter after a trade closes.
- **Why**: Advances the learning phase progression and tracks lifetime trade count.
- **Relationships**: Called by the decision engine after each trade exit.
- **Decisions**: Currently only increments the counter; future versions may persist outcome statistics.

### AdaptiveLearningEngine.get_confidence_adjustment(self, context: dict, base_confluence: int) -> float
- **Does**: Suggests a confidence/sizing adjustment in [-0.5, +0.5] based on similarity search results for the given trade context.
- **Why**: Allows the engine to modestly increase or decrease position sizing based on historical performance of similar setups.
- **Relationships**: Calls _run_similarity_search and _compute_confidence_adjustment; called by the decision engine.
- **Decisions**: Returns 0.0 during the mechanical phase or when similarity data is unavailable, enforcing a hard cap of 0.5.

### AdaptiveLearningEngine.should_filter(self, context: dict) -> Tuple[bool, str]
- **Does**: Decides whether to skip a setup based on statistical history of the current session and ADX regime.
- **Why**: Prevents trading in historically poor-performing conditions once enough data has been collected.
- **Relationships**: Calls _compute_statistical_filters; called by the decision engine.
- **Decisions**: Inactive during the mechanical phase; requires minimum 20 trades per category before filtering.

### StatsAnalyzer.__init__(self, db_pool) -> None
- **Does**: Creates a statistical analyser that queries closed trade history from the database.
- **Why**: Provides win-rate breakdowns by session, regime, confluence tier, and day of week to drive statistical filtering.
- **Relationships**: Used by AdaptiveLearningEngine and ReportGenerator; queries the trades and market_context tables.
- **Decisions**: Accepts None db_pool for testing via _inject_trade_cache.

### StatsAnalyzer.win_rate_by_session(self, min_trades: int) -> Dict[str, dict]
- **Does**: Computes win rate, trade count, and average R-multiple for each trading session with at least min_trades.
- **Why**: Identifies sessions (London, New York, Asian, Overlap) where the strategy underperforms so they can be filtered.
- **Relationships**: Called by should_filter_session and ReportGenerator.weekly_report.
- **Decisions**: Requires min_trades threshold to avoid reacting to noise in small samples.

### StatsAnalyzer.win_rate_by_regime(self, min_trades: int) -> Dict[str, dict]
- **Does**: Computes win rate by ADX trend regime (low <20, medium 20-35, high 35+).
- **Why**: Identifies market conditions (trending vs ranging) where the Ichimoku strategy underperforms.
- **Relationships**: Called by should_filter_regime and ReportGenerator.weekly_report.
- **Decisions**: ADX boundaries chosen to align with standard technical analysis conventions.

### StatsAnalyzer.win_rate_by_confluence(self, min_trades: int) -> Dict[str, dict]
- **Does**: Computes win rate by confluence score tier (A+, B, C).
- **Why**: Validates whether higher confluence scores actually produce better outcomes.
- **Relationships**: Called by ReportGenerator.weekly_report.
- **Decisions**: Tier boundaries align with the confluence_scoring edge tiers.

### StatsAnalyzer.win_rate_by_day(self, min_trades: int) -> Dict[str, dict]
- **Does**: Computes win rate by day of week (Monday through Friday).
- **Why**: Identifies specific days where the strategy historically underperforms.
- **Relationships**: Called by ReportGenerator.weekly_report.
- **Decisions**: None.

### StatsAnalyzer.performance_heatmap(self) -> pd.DataFrame
- **Does**: Builds a session-by-regime performance matrix showing win rate and trade count in each cell.
- **Why**: Provides a two-dimensional view of performance to identify specific session/regime combinations to avoid.
- **Relationships**: Standalone analysis method; not currently used by automated filters.
- **Decisions**: Returns string-formatted cells for display readability.

### StatsAnalyzer.should_filter_session(self, session: str, min_wr: float, min_trades: int) -> bool
- **Does**: Returns True when a session's historical win rate is below the threshold and has sufficient trade count.
- **Why**: Provides the statistical filter predicate that AdaptiveLearningEngine uses to skip poor sessions.
- **Relationships**: Called by AdaptiveLearningEngine._compute_statistical_filters.
- **Decisions**: Returns False (do not filter) when insufficient data exists, erring on the side of trading.

### StatsAnalyzer.should_filter_regime(self, adx: float, min_wr: float, min_trades: int) -> bool
- **Does**: Returns True when the ADX regime's historical win rate is below the threshold.
- **Why**: Prevents trading in ADX regimes where the strategy has consistently lost.
- **Relationships**: Called by AdaptiveLearningEngine._compute_statistical_filters.
- **Decisions**: Converts raw ADX to regime label before lookup.

### StatsAnalyzer.get_all_stats(self, min_trades_session: int, min_trades_regime: int, min_trades_confluence: int, min_trades_day: int) -> dict
- **Does**: Returns all statistical breakdowns (session, regime, confluence, day) in a single dict.
- **Why**: Convenience method that avoids multiple separate calls when all breakdowns are needed.
- **Relationships**: Called by the adaptive engine and report generator.
- **Decisions**: None.

### EdgeReviewResult(edge_name: str, total_trades_affected: int, win_rate_when_active: float, avg_r_when_active: float, filter_rate: float, marginal_impact: float) -> EdgeReviewResult
- **Does**: Dataclass holding performance metrics for a single strategy edge.
- **Why**: Provides a structured summary of each edge's real-world contribution for review and reporting.
- **Relationships**: Created by EdgeReviewer.review_all_edges; consumed by ReportGenerator and EdgeSuggestion logic.
- **Decisions**: Includes marginal_impact (with-edge vs without-edge avg_r difference) as the key value metric.

### EdgeSuggestion(edge_name: str, current_state: bool, suggested_state: bool, reason: str, confidence: float) -> EdgeSuggestion
- **Does**: Dataclass representing a suggested change to an edge's enabled/disabled state.
- **Why**: Makes optimization suggestions explicit and auditable, requiring human approval before any configuration change.
- **Relationships**: Created by EdgeReviewer.suggest_edge_changes; included in WeeklyReport.
- **Decisions**: Advisory only by design — never auto-applied.

### EdgeReviewer.__init__(self, db_pool) -> None
- **Does**: Creates an edge performance reviewer that analyses trade history to assess each edge's contribution.
- **Why**: Enables data-driven decisions about which strategy edges to keep, disable, or re-enable.
- **Relationships**: Queries the trades table; results feed into ReportGenerator and operator review.
- **Decisions**: Pre-populates all edges as enabled (conservative default).

### EdgeReviewer.review_all_edges(self) -> List[EdgeReviewResult]
- **Does**: Calculates performance metrics (win rate, avg R, filter rate, marginal impact) for every tracked edge.
- **Why**: Provides a comprehensive view of each edge's real-world value so underperforming edges can be identified.
- **Relationships**: Called by suggest_edge_changes and ReportGenerator.weekly_report.
- **Decisions**: Returns zero-filled results for edges with no trade data rather than omitting them.

### EdgeReviewer.suggest_edge_changes(self) -> List[EdgeSuggestion]
- **Does**: Proposes enabling or disabling edges based on their win rate relative to threshold values, requiring human approval.
- **Why**: Automates the identification of edges that should be toggled, but keeps human authority over configuration changes.
- **Relationships**: Calls review_all_edges and _load_edge_states; suggestions included in WeeklyReport.
- **Decisions**: Requires 30+ trades before emitting suggestions; confidence ramps from 0 to 1 over 100 trades.

### WeeklyReport(period_start: datetime, period_end: datetime, total_trades: int, win_rate: float, pnl: float, ...) -> WeeklyReport
- **Does**: Dataclass consolidating all weekly performance data: period summary, session/regime breakdowns, edge review, similarity quality, and advisory suggestions.
- **Why**: Provides a single structured object for rendering reports in text or HTML format.
- **Relationships**: Created by ReportGenerator.weekly_report; rendered by to_text and to_html.
- **Decisions**: Includes the learning phase and lifetime trade count for context.

### ReportGenerator.__init__(self, stats_analyzer: Optional[StatsAnalyzer], edge_reviewer: Optional[EdgeReviewer], db_pool) -> None
- **Does**: Creates a report generator that consolidates output from StatsAnalyzer, EdgeReviewer, and period trade data.
- **Why**: Centralises report assembly so consumers get a complete performance picture from a single call.
- **Relationships**: Composes StatsAnalyzer and EdgeReviewer; queries the trades and pre_trade_insights tables.
- **Decisions**: Creates default StatsAnalyzer and EdgeReviewer when not provided.

### ReportGenerator.weekly_report(self, period_start: Optional[datetime], period_end: Optional[datetime]) -> WeeklyReport
- **Does**: Generates a weekly performance summary with period trades, session/regime breakdowns, edge review, and similarity quality metrics.
- **Why**: Produces the primary operator-facing report for monitoring strategy health and learning phase progression.
- **Relationships**: Calls StatsAnalyzer methods, EdgeReviewer.review_all_edges, and EdgeReviewer.suggest_edge_changes.
- **Decisions**: Defaults to the last 7 days when period boundaries are not specified.

### ReportGenerator.to_text(self, report: WeeklyReport) -> str
- **Does**: Renders a WeeklyReport as a plain-text summary suitable for logging or email.
- **Why**: Provides a human-readable report format that works in any text-based medium.
- **Relationships**: Called with the output of weekly_report.
- **Decisions**: Uses fixed-width formatting for alignment in monospace contexts.

### ReportGenerator.to_html(self, report: WeeklyReport) -> str
- **Does**: Renders a WeeklyReport as a complete HTML document with color-coded win rates.
- **Why**: Provides a visually rich report format for browser or email distribution.
- **Relationships**: Called with the output of weekly_report.
- **Decisions**: Color-codes win rates green (>=55%), orange (40-55%), red (<40%).

### FeatureVectorBuilder.__init__(self) -> None
- **Does**: Creates a builder that converts trade context dictionaries into 64-dimensional normalised feature vectors.
- **Why**: Standardises the encoding of heterogeneous market context data into a fixed-size vector suitable for similarity search via pgvector.
- **Relationships**: Used by EmbeddingEngine.create_embedding; output stored in market_context.context_embedding.
- **Decisions**: 64 dimensions with reserved slots for forward compatibility; all values normalised to [0, 1].

### FeatureVectorBuilder.build(self, context: Dict[str, Any]) -> np.ndarray
- **Does**: Builds a 64-dim feature vector from a context dictionary, encoding Ichimoku state, trend/momentum, zone context, session/time, signal quality, and market regime.
- **Why**: Transforms raw market context into a compact, comparable vector for nearest-neighbour similarity search.
- **Relationships**: Called by EmbeddingEngine.create_embedding and embed_trade.
- **Decisions**: Missing keys silently default to 0.0 so partial contexts still produce valid vectors; cyclical features use sin/cos pairs.

### EmbeddingEngine.__init__(self, feature_builder: Optional[FeatureVectorBuilder]) -> None
- **Does**: Creates an embedding engine that wraps FeatureVectorBuilder for single, batch, and DB-ready embedding creation.
- **Why**: Provides a higher-level interface over raw feature vector construction with convenience methods for serialisation and batch processing.
- **Relationships**: Used by EngineTradeLogger and AdaptiveLearningEngine; delegates to FeatureVectorBuilder.
- **Decisions**: Creates a default FeatureVectorBuilder when none is provided.

### EmbeddingEngine.create_embedding(self, context: Dict[str, Any]) -> np.ndarray
- **Does**: Creates a 64-dim embedding from a trade context dictionary.
- **Why**: Primary entry point for generating embeddings used in similarity search and trade logging.
- **Relationships**: Delegates to FeatureVectorBuilder.build; called by EngineTradeLogger.log_trade_entry and AdaptiveLearningEngine._run_similarity_search.
- **Decisions**: None.

### EmbeddingEngine.embed_trade(self, trade_context: Dict[str, Any], trade_result: Optional[Dict[str, Any]]) -> Dict[str, Any]
- **Does**: Creates an embedding dict with context_embedding, outcome_r, and win flag, ready for database insertion.
- **Why**: Packages the embedding alongside trade outcome data in a single structure for storage.
- **Relationships**: Calls FeatureVectorBuilder.build; output inserted into market_context table.
- **Decisions**: Derives win flag from r_multiple when not explicitly provided.

### EmbeddingEngine.batch_embed(self, contexts: List[Dict[str, Any]]) -> List[np.ndarray]
- **Does**: Embeds multiple contexts independently and returns a list of arrays in input order.
- **Why**: Enables efficient bulk embedding for backtest result processing.
- **Relationships**: Calls FeatureVectorBuilder.build for each context.
- **Decisions**: Processes each context independently (no cross-context normalization).

### EmbeddingEngine.embedding_to_list(self, embedding: np.ndarray) -> List[float]
- **Does**: Converts a numpy embedding to a plain Python list for JSON serialisation.
- **Why**: Bridges the numpy/Python boundary for database and API transport.
- **Relationships**: Used during DB insertion workflows.
- **Decisions**: None.

### EmbeddingEngine.embedding_from_list(self, values: List[float]) -> np.ndarray
- **Does**: Reconstructs a numpy array from a stored list of floats, validating the expected 64-dim shape.
- **Why**: Enables retrieval of stored embeddings back into numpy for similarity computation.
- **Relationships**: Used when loading embeddings from the database.
- **Decisions**: Raises ValueError if the dimension count does not match VECTOR_DIM.

### SimilarTrade(trade_id: int, similarity: float, r_multiple: float, win: bool, context: Dict[str, Any]) -> SimilarTrade
- **Does**: Dataclass representing a single historical trade that is contextually similar to a query embedding.
- **Why**: Provides a structured result from similarity search with both the similarity score and trade outcome.
- **Relationships**: Created by SimilaritySearch.find_similar_trades; consumed by get_performance_stats and PreTradeInsight.
- **Decisions**: Similarity is cosine similarity in [0, 1] where 1 means identical.

### PerformanceStats(win_rate: float, avg_r: float, expectancy: float, n_trades: int, confidence: float, avg_win_r: float, avg_loss_r: float) -> PerformanceStats
- **Does**: Dataclass holding aggregated performance statistics over a set of similar trades.
- **Why**: Summarises the expected performance of a trade setup based on historical similar trades.
- **Relationships**: Created by SimilaritySearch.get_performance_stats; consumed by AdaptiveLearningEngine._compute_confidence_adjustment.
- **Decisions**: Confidence ramps linearly from 0 to 1 over 20 similar trades.

### SimilaritySearch.__init__(self, db_pool) -> None
- **Does**: Creates a similarity search engine that queries pgvector for nearest-neighbour trade embeddings.
- **Why**: Enables the similarity-based learning phase by finding historically similar trade setups.
- **Relationships**: Used by AdaptiveLearningEngine and DecisionEngine; queries the market_context and trades tables via pgvector.
- **Decisions**: Uses the cosine distance operator (<=>) with an HNSW index for O(log n) lookup.

### SimilaritySearch.find_similar_trades(self, context_embedding: np.ndarray, k: int, min_similarity: float, source_filter: Optional[str]) -> List[SimilarTrade]
- **Does**: Finds the k most similar historical trades via pgvector cosine similarity, optionally filtered by trade source.
- **Why**: Core method that retrieves the nearest-neighbour cohort for pre-trade performance estimation.
- **Relationships**: Called by AdaptiveLearningEngine._run_similarity_search and DecisionEngine.scan; results feed into get_performance_stats.
- **Decisions**: Supports source_filter to isolate backtest vs live vs paper trades.

### SimilaritySearch.get_performance_stats(self, similar_trades: List[SimilarTrade]) -> PerformanceStats
- **Does**: Calculates win rate, average R, expectancy, and confidence from a list of similar trades.
- **Why**: Aggregates individual similar-trade outcomes into actionable statistics for the confidence adjustment.
- **Relationships**: Called after find_similar_trades; output consumed by AdaptiveLearningEngine._compute_confidence_adjustment.
- **Decisions**: Uses the formula expectancy = win_rate * avg_win_r + loss_rate * avg_loss_r.

### SimilaritySearch.get_confidence(n_similar: int) -> float
- **Does**: Computes data confidence as min(1.0, n_similar / 20).
- **Why**: Scales the influence of similarity results based on how many matching trades exist, preventing overreaction to sparse data.
- **Relationships**: Called by get_performance_stats.
- **Decisions**: Ramp denominator of 20 chosen as the minimum for baseline statistical reliability.

### SimilaritySearch.cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
- **Does**: Computes cosine similarity between two vectors in pure Python/numpy.
- **Why**: Provides an offline alternative to pgvector for testing and local analysis.
- **Relationships**: Used in tests and offline analysis; not used in the live pgvector path.
- **Decisions**: Returns 0.0 for zero-norm vectors rather than raising an error.
