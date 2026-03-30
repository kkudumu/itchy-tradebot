"""
Base classes and registries for the three-layer strategy abstraction.

Layer 1 — Evaluators:  stateless per-TF indicator modules (Evaluator ABC).
Layer 2 — Strategies:  confluence logic that combines evaluator results (Strategy ABC).
Layer 3 — TradingModes: exit / position-management logic (TradingMode ABC).

Registries are populated automatically via ``__init_subclass__``:
  EVALUATOR_REGISTRY[key] → Evaluator subclass
  STRATEGY_REGISTRY[key]  → Strategy subclass

Typical access patterns
~~~~~~~~~~~~~~~~~~~~~~~
::

    eval_matrix.get('ichimoku_4H')  # → EvaluatorResult or None
    'ichimoku_4H' in eval_matrix    # → bool
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

EVALUATOR_REGISTRY: dict[str, type] = {}
STRATEGY_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Evaluator result types
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorResult:
    """Normalised output from a single evaluator on a single timeframe.

    Attributes
    ----------
    direction:
        -1.0 (fully bearish) … 0.0 (neutral) … +1.0 (fully bullish).
    confidence:
        Signal strength, 0.0 (no conviction) to 1.0 (maximum conviction).
    metadata:
        Arbitrary indicator values for logging / debugging (e.g. raw
        Ichimoku component states, ADX value).
    """

    direction: float    # -1.0 to +1.0
    confidence: float   # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)


class EvalMatrix:
    """Container for evaluator results keyed by ``'{eval_name}_{tf}'``.

    Typical key format:  ``'ichimoku_4H'``, ``'adx_1H'``, etc.

    Usage
    -----
    ::

        matrix = EvalMatrix()
        matrix.set('ichimoku_4H', EvaluatorResult(direction=1.0, confidence=0.8))
        result = matrix.get('ichimoku_4H')   # EvaluatorResult or None
        'ichimoku_4H' in matrix              # True
    """

    def __init__(self) -> None:
        self._results: dict[str, EvaluatorResult] = {}

    def set(self, key: str, result: EvaluatorResult) -> None:
        """Store *result* under *key*."""
        self._results[key] = result

    def get(self, key: str) -> Optional[EvaluatorResult]:
        """Return the result for *key*, or ``None`` if not present."""
        return self._results.get(key)

    def keys(self):
        """Return a view of all stored keys."""
        return self._results.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._results

    def __repr__(self) -> str:
        keys = list(self._results.keys())
        return f"EvalMatrix(keys={keys!r})"


# ---------------------------------------------------------------------------
# Evaluator requirement declaration
# ---------------------------------------------------------------------------

@dataclass
class EvalRequirement:
    """Declares which evaluator should be run on which timeframes.

    Used by the orchestrator to determine which evaluators to invoke and
    which timeframes to fetch data for before calling ``Strategy.decide``.

    Attributes
    ----------
    evaluator_name:
        Key in ``EVALUATOR_REGISTRY`` (e.g. ``'ichimoku'``).
    timeframes:
        List of timeframe strings to run the evaluator on
        (e.g. ``['4H', '1H', '15M']``).
    """

    evaluator_name: str
    timeframes: list[str]


# ---------------------------------------------------------------------------
# Confluence result
# ---------------------------------------------------------------------------

@dataclass
class ConfluenceResult:
    """Confluence scoring output from ``Strategy.score_confluence``.

    Attributes
    ----------
    score:
        Raw integer confluence score.
    quality_tier:
        One of ``'A+'``, ``'B'``, ``'C'``, or ``'no_trade'``.
    breakdown:
        Per-component details used for logging and optimisation.
    """

    score: int
    quality_tier: str   # 'A+', 'B', 'C', 'no_trade'
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator ABC
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    """Abstract base for all per-timeframe indicator evaluators.

    Sub-classes declare themselves with a ``key`` keyword argument:
    ::

        class IchimokuEvaluator(Evaluator, key='ichimoku'):
            def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
                ...

    The ``key`` is stored as ``cls.name`` and registered in
    ``EVALUATOR_REGISTRY`` at class-creation time.

    Attributes
    ----------
    name:
        Class attribute set by ``__init_subclass__`` from the ``key`` argument.
    """

    name: str  # set by __init_subclass__ from the key= argument

    def __init_subclass__(cls, key: str = '', **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if key:
            if key in EVALUATOR_REGISTRY:
                raise ValueError(
                    f"Duplicate evaluator key: {key!r}. "
                    f"Already registered by {EVALUATOR_REGISTRY[key].__name__}."
                )
            EVALUATOR_REGISTRY[key] = cls
            cls.name = key

    @abstractmethod
    def evaluate(self, ohlcv: 'pd.DataFrame') -> EvaluatorResult:
        """Compute indicator values and return a normalised result.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame for a single timeframe (indexed by timestamp).
            At minimum: columns ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        EvaluatorResult
            Normalised direction and confidence for this TF.
        """


# ---------------------------------------------------------------------------
# Exit decision
# ---------------------------------------------------------------------------

@dataclass
class ExitDecision:
    """Decision returned by ``TradingMode.check_exit``.

    Attributes
    ----------
    action:
        One of ``'hold'``, ``'partial_exit'``, ``'trail_update'``,
        ``'full_exit'``.
    new_stop:
        Updated stop-loss level (absolute price).  ``None`` means no change.
    close_pct:
        Fraction of the position to close (0.0 = nothing, 1.0 = all).
        Only meaningful when *action* is ``'partial_exit'``.
    reason:
        Human-readable explanation logged by the engine.
    """

    action: str              # 'hold' | 'partial_exit' | 'trail_update' | 'full_exit'
    new_stop: Optional[float] = None
    close_pct: float = 0.0
    reason: str = ''


# ---------------------------------------------------------------------------
# TradingMode ABC
# ---------------------------------------------------------------------------

class TradingMode(ABC):
    """Abstract base for exit and position-management logic.

    A ``TradingMode`` instance is attached to a :class:`Strategy` and is
    called on every bar while a trade is open.

    Sub-classes implement ``check_exit`` to decide whether to hold, trail,
    partially close, or fully exit the trade.
    """

    @abstractmethod
    def check_exit(
        self,
        trade: object,
        current_data: dict,
        eval_results: EvalMatrix,
    ) -> ExitDecision:
        """Evaluate whether the open trade should be exited or modified.

        Parameters
        ----------
        trade:
            The active trade object (engine-specific, passed through
            without coupling to a concrete type).
        current_data:
            Dictionary of current market data for the primary timeframe
            (e.g. ``{'close': 1920.5, 'atr': 3.2, ...}``).
        eval_results:
            Current ``EvalMatrix`` snapshot (may be partial / empty for
            simple exit modes that don't need evaluator data).

        Returns
        -------
        ExitDecision
            Instruction for the engine on how to handle the open trade.
        """


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract base for all multi-timeframe trading strategies.

    Sub-classes declare themselves with a ``key`` keyword argument and set
    the class-level attributes:
    ::

        class IchimokuStrategy(Strategy, key='ichimoku_mtf'):
            required_evaluators = [
                EvalRequirement('ichimoku', ['4H', '1H', '15M', '5M']),
            ]
            config_model = IchimokuConfig   # Pydantic model
            warmup_bars = 200

            def decide(self, eval_matrix: EvalMatrix) -> Optional[Signal]:
                ...

            def score_confluence(self, eval_matrix: EvalMatrix) -> ConfluenceResult:
                ...

    Class Attributes
    ----------------
    name:
        Set automatically by ``__init_subclass__`` from the ``key`` argument.
    required_evaluators:
        List of :class:`EvalRequirement` declaring which evaluators the
        orchestrator must run before calling :meth:`decide`.
    config_model:
        Optional Pydantic model class used to validate strategy config.
    trading_mode:
        Optional :class:`TradingMode` instance attached at construction time.
    warmup_bars:
        Minimum number of bars required before the strategy can emit
        signals (used by the backtesting engine to skip the warm-up period).
    """

    name: str
    required_evaluators: list[EvalRequirement] = []
    config_model: type = None
    trading_mode: Optional[TradingMode] = None
    warmup_bars: int = 0

    def __init_subclass__(cls, key: str = '', **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if key:
            if key in STRATEGY_REGISTRY:
                raise ValueError(
                    f"Duplicate strategy key: {key!r}. "
                    f"Already registered by {STRATEGY_REGISTRY[key].__name__}."
                )
            STRATEGY_REGISTRY[key] = cls
            cls.name = key

    @abstractmethod
    def decide(self, eval_matrix: EvalMatrix) -> 'Optional[Signal]':
        """Produce a trade signal or ``None`` from the evaluation matrix.

        Parameters
        ----------
        eval_matrix:
            Populated matrix of evaluator results for all required
            timeframes.

        Returns
        -------
        Signal or None
            A fully-qualified trade signal, or ``None`` if conditions are
            not met.
        """

    @abstractmethod
    def score_confluence(self, eval_matrix: EvalMatrix) -> ConfluenceResult:
        """Score signal confluence from the evaluation matrix.

        Parameters
        ----------
        eval_matrix:
            Populated matrix of evaluator results.

        Returns
        -------
        ConfluenceResult
            Score, quality tier, and per-component breakdown.
        """

    def suggest_params(self, trial: object) -> dict:
        """Suggest Optuna trial parameters for hyper-parameter optimisation.

        Override this in concrete strategies to expose tunable parameters.
        The default implementation returns an empty dict (no-op).

        Parameters
        ----------
        trial:
            An ``optuna.Trial`` instance used to call ``trial.suggest_*``.

        Returns
        -------
        dict
            Parameter name → suggested value mapping.
        """
        return {}

    def populate_edge_context(self, eval_matrix: EvalMatrix) -> dict:
        """Return indicator values dict for populating an ``EdgeContext``.

        Override this to expose strategy-specific indicator values (e.g.
        ``cloud_thickness``, ``kijun_value``) that are required by edge
        filters.  The default implementation returns an empty dict.

        Parameters
        ----------
        eval_matrix:
            Populated matrix of evaluator results.

        Returns
        -------
        dict
            Indicator key → value mapping consumed by the engine when
            building an ``EdgeContext`` for the edge pipeline.
        """
        return {}
