"""
EdgeManager — loads, chains, and routes all 12 edge filters.

Entry edges gate whether a new position can be opened.
Exit edges signal that an open position should be closed.
Modifier edges adjust position size without blocking the trade.

The manager separates these three categories and provides dedicated
evaluation methods so the trading engine can call the appropriate
pipeline at each decision point.
"""

from __future__ import annotations

from typing import Any

from .base import EdgeContext, EdgeFilter, EdgeResult
from .time_of_day import TimeOfDayFilter
from .day_of_week import DayOfWeekFilter
from .london_open_delay import LondonOpenDelayFilter
from .candle_close_confirmation import CandleCloseConfirmationFilter
from .spread_filter import SpreadFilter
from .news_filter import NewsFilter, NewsCalendar
from .friday_close import FridayCloseFilter
from .regime_filter import RegimeFilter
from .time_stop import TimeStopFilter
from .bb_squeeze import BBSqueezeAmplifier
from .confluence_scoring import ConfluenceScoringFilter
from .equity_curve import EquityCurveFilter


# Mapping from config key → (constructor, category)
# category: 'entry' | 'exit' | 'modifier' | 'entry+modifier'
_REGISTRY: dict[str, tuple[type[EdgeFilter], str]] = {
    "time_of_day":               (TimeOfDayFilter,                "entry"),
    "day_of_week":               (DayOfWeekFilter,                "entry"),
    "london_open_delay":         (LondonOpenDelayFilter,          "entry"),
    "candle_close_confirmation": (CandleCloseConfirmationFilter,  "entry"),
    "spread_filter":             (SpreadFilter,                   "entry"),
    "news_filter":               (NewsFilter,                     "entry"),
    "regime_filter":             (RegimeFilter,                   "entry"),
    "friday_close":              (FridayCloseFilter,              "exit"),
    "time_stop":                 (TimeStopFilter,                 "exit"),
    "bb_squeeze":                (BBSqueezeAmplifier,             "modifier"),
    "confluence_scoring":        (ConfluenceScoringFilter,        "entry+modifier"),
    "equity_curve":              (EquityCurveFilter,              "modifier"),
}


def _normalise_config(raw: Any) -> dict:
    """Convert a Pydantic EdgeConfig model or a plain dict to a usable dict.

    Pydantic v2 models expose ``.model_dump()``; v1 uses ``.dict()``.
    A plain dict is returned as-is.
    """
    if isinstance(raw, dict):
        return raw
    # Pydantic v2
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    # Pydantic v1
    if hasattr(raw, "dict"):
        return raw.dict()
    raise TypeError(f"Unsupported config type: {type(raw)}")


class EdgeManager:
    """Load enabled edges from config and provide pipeline evaluation.

    Parameters
    ----------
    edge_configs:
        Either an ``EdgeConfig`` Pydantic model (from models.py) or a
        plain dict mapping edge names to their config sub-dicts.
        Each sub-dict must contain at least ``{"enabled": bool, "params": {...}}``.
    news_calendar:
        Optional NewsCalendar implementation. Injected into NewsFilter
        after construction if provided.

    Examples
    --------
    >>> from src.config.models import EdgeConfig
    >>> manager = EdgeManager(EdgeConfig())
    >>> result_ok, results = manager.check_entry(context)
    """

    def __init__(
        self,
        edge_configs: Any,
        news_calendar: NewsCalendar | None = None,
    ) -> None:
        configs = _normalise_config(edge_configs)

        self.entry_edges: list[EdgeFilter] = []
        self.exit_edges: list[EdgeFilter] = []
        self.modifier_edges: list[EdgeFilter] = []

        self._all_edges: dict[str, EdgeFilter] = {}

        for key, (cls, category) in _REGISTRY.items():
            # Pydantic dump uses the field name; fall back to a default if absent
            raw_cfg = configs.get(key, {})
            cfg = _normalise_config(raw_cfg) if not isinstance(raw_cfg, dict) else raw_cfg

            # Ensure required keys are present
            if not isinstance(cfg, dict):
                cfg = {}

            # When no config provided for this edge, default to disabled
            if not cfg:
                cfg = {"enabled": False}

            edge = cls(cfg)  # type: ignore[call-arg]
            self._all_edges[key] = edge

            if category == "entry+modifier":
                # Dual-role: blocks below min score AND provides size modifier
                self.entry_edges.append(edge)
                self.modifier_edges.append(edge)
            elif category == "entry":
                self.entry_edges.append(edge)
            elif category == "exit":
                self.exit_edges.append(edge)
            else:
                self.modifier_edges.append(edge)

        # Inject news calendar if provided
        if news_calendar is not None:
            news: EdgeFilter | None = self._all_edges.get("news_filter")
            if news is not None and hasattr(news, "set_calendar"):
                news.set_calendar(news_calendar)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Pipeline evaluation
    # ------------------------------------------------------------------

    def check_entry(
        self, context: EdgeContext
    ) -> tuple[bool, list[EdgeResult]]:
        """Run all entry edge filters in sequence.

        Short-circuits on the first failure so that the reason for
        rejection is clear in the result list.

        Returns
        -------
        (all_passed, results)
            all_passed: True only when every enabled edge allowed the action.
            results: list of EdgeResult, one per evaluated edge.
        """
        results: list[EdgeResult] = []

        for edge in self.entry_edges:
            if not edge.enabled:
                continue
            result = edge.should_allow(context)
            results.append(result)
            if not result.allowed:
                # Return immediately on first failure (short-circuit)
                return False, results

        return True, results

    def check_exit(
        self, context: EdgeContext
    ) -> tuple[bool, list[EdgeResult]]:
        """Run all exit edge filters.

        Any edge returning ``allowed=False`` signals that the open position
        should be closed. All exit edges are evaluated (no short-circuit) so
        the full picture is captured in the results.

        Returns
        -------
        (exit_triggered, results)
            exit_triggered: True when at least one exit edge returned False.
            results: list of EdgeResult for all evaluated exit edges.
        """
        results: list[EdgeResult] = []
        exit_triggered = False

        for edge in self.exit_edges:
            if not edge.enabled:
                continue
            result = edge.should_allow(context)
            results.append(result)
            if not result.allowed:
                exit_triggered = True

        return exit_triggered, results

    def get_modifiers(self, context: EdgeContext) -> dict[str, float]:
        """Evaluate all modifier edges and return their modifier values.

        Returns
        -------
        dict mapping edge name → modifier float value.
        Missing modifiers default to 1.0 (neutral / no change).

        Typical keys: 'bb_squeeze', 'confluence_scoring', 'equity_curve'.
        """
        modifiers: dict[str, float] = {}

        for edge in self.modifier_edges:
            if not edge.enabled:
                modifiers[edge.name] = 1.0
                continue
            result = edge.should_allow(context)
            modifiers[edge.name] = result.modifier if result.modifier is not None else 1.0

        return modifiers

    def get_combined_size_multiplier(self, context: EdgeContext) -> float:
        """Return a single position size multiplier by combining all modifier edges.

        The combination rule: take the minimum modifier from confluence_scoring
        and equity_curve (size-reduction modifiers), then add the bb_squeeze
        boost to the confluence score indirectly. Direct size multipliers are
        combined multiplicatively to avoid double-penalising.

        The bb_squeeze modifier is additive to confluence but expressed as an
        integer boost; it does not directly scale size. Only size-related
        modifiers (confluence_scoring, equity_curve) are multiplied together.

        Returns
        -------
        float in range [0.0, 1.0].
        """
        modifiers = self.get_modifiers(context)

        size_keys = {"confluence_scoring", "equity_curve"}
        size_multiplier = 1.0
        for key in size_keys:
            if key in modifiers:
                size_multiplier *= modifiers[key]

        return max(0.0, min(1.0, size_multiplier))

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def get_enabled_edges(self) -> list[str]:
        """Return names of all currently enabled edges."""
        return [name for name, edge in self._all_edges.items() if edge.enabled]

    def toggle_edge(self, edge_name: str, enabled: bool) -> None:
        """Enable or disable a named edge at runtime.

        Parameters
        ----------
        edge_name:
            One of the 12 registered edge keys (e.g. 'time_of_day').
        enabled:
            True to enable, False to disable.

        Raises
        ------
        KeyError
            If ``edge_name`` is not a recognised edge.
        """
        if edge_name not in self._all_edges:
            raise KeyError(
                f"Unknown edge '{edge_name}'. "
                f"Available edges: {sorted(self._all_edges.keys())}"
            )
        self._all_edges[edge_name].enabled = enabled

    def get_edge(self, name: str) -> EdgeFilter:
        """Return the EdgeFilter instance for the given edge name.

        Parameters
        ----------
        name:
            Registered edge key (e.g. 'regime_filter').

        Raises
        ------
        KeyError
            If ``name`` is not a recognised edge.
        """
        if name not in self._all_edges:
            raise KeyError(
                f"Unknown edge '{name}'. "
                f"Available edges: {sorted(self._all_edges.keys())}"
            )
        return self._all_edges[name]

    def set_edge_param(self, edge_name: str, param_name: str, value: Any) -> None:
        """Call a setter method on a named edge at runtime.

        Delegates to the edge's setter method named ``set_<param_name>``.
        Used by AdaptiveRelaxer to adjust filter parameters without
        direct access to the edge instances.

        Parameters
        ----------
        edge_name:
            Registered edge key (e.g. 'regime_filter').
        param_name:
            Parameter to set — must match a ``set_<param_name>`` method on the edge.
        value:
            New value to pass to the setter.

        Raises
        ------
        KeyError
            If ``edge_name`` is not recognised.
        AttributeError
            If the edge has no ``set_<param_name>`` method.
        """
        edge = self.get_edge(edge_name)
        setter_name = f"set_{param_name}"
        setter = getattr(edge, setter_name, None)
        if setter is None:
            raise AttributeError(
                f"Edge '{edge_name}' has no setter '{setter_name}'. "
                f"Available methods: {[m for m in dir(edge) if m.startswith('set_')]}"
            )
        if isinstance(value, (list,)):
            setter(value)
        else:
            setter(value)

    def set_edge_params(self, edge_name: str, params: dict) -> None:
        """Update parameters for a named edge at runtime.

        Updates the edge's config dict and any matching instance attributes.

        Parameters
        ----------
        edge_name:
            Registered edge key (e.g. 'regime_filter').
        params:
            Dict of parameter names to new values.

        Raises
        ------
        KeyError
            If ``edge_name`` is not recognised.
        """
        if edge_name not in self._all_edges:
            raise KeyError(
                f"Unknown edge '{edge_name}'. "
                f"Available edges: {sorted(self._all_edges.keys())}"
            )
        edge = self._all_edges[edge_name]
        cfg_params = edge.config.get("params", {})
        cfg_params.update(params)
        edge.config["params"] = cfg_params
        for k, v in params.items():
            if hasattr(edge, k):
                setattr(edge, k, v)

    def get_all_config(self) -> dict:
        """Return full configuration for all edges.

        Returns
        -------
        dict mapping edge name to ``{"enabled": bool, "params": dict}``.
        """
        result: dict[str, dict] = {}
        for name, edge in self._all_edges.items():
            result[name] = {
                "enabled": edge.enabled,
                "params": dict(edge.config.get("params", {})),
            }
        return result

    def __repr__(self) -> str:
        enabled = self.get_enabled_edges()
        return (
            f"EdgeManager("
            f"entry={len(self.entry_edges)}, "
            f"exit={len(self.exit_edges)}, "
            f"modifier={len(self.modifier_edges)}, "
            f"enabled={enabled})"
        )
