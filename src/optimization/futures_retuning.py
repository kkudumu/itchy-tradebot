"""Futures strategy retuning workflow (plan Task 17).

Reads a telemetry parquet produced by :class:`StrategyTelemetryCollector`,
analyzes per-strategy distributions, and suggests parameter adjustments
that better fit the MGC futures characteristics. Does NOT run Optuna
sweeps by default — that's an expensive step the caller opts into
explicitly via :func:`run_optuna_sweep`.

The workflow:
  1. :func:`analyze_telemetry` reads a parquet and returns a
     :class:`RetuningReport` per strategy.
  2. :func:`suggest_param_adjustments` takes a report and returns a
     concrete dict of parameter overrides suitable for pasting into
     ``config/profiles/futures.yaml``.
  3. :func:`apply_overrides_to_profile` merges the suggestions into
     the profile YAML file.
  4. :func:`run_optuna_sweep` (optional, slow) runs a per-strategy
     Optuna study with the topstep_combine_pass_score objective.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StrategyTelemetryStats:
    """Per-strategy summary of telemetry activity."""

    strategy_name: str
    signals_generated: int = 0
    signals_entered: int = 0
    trades_exited: int = 0
    entry_rate_pct: float = 0.0
    top_rejection_stages: Dict[str, int] = field(default_factory=dict)
    sessions: Dict[str, int] = field(default_factory=dict)
    confluence_score_mean: Optional[float] = None
    confluence_score_median: Optional[float] = None
    atr_mean: Optional[float] = None
    planned_stop_pips_mean: Optional[float] = None
    planned_stop_pips_median: Optional[float] = None


@dataclass
class RetuningReport:
    """Per-strategy retuning recommendation bundle."""

    per_strategy: Dict[str, StrategyTelemetryStats] = field(default_factory=dict)
    total_events: int = 0
    source_parquet: Optional[str] = None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_telemetry(parquet_path: str | Path) -> RetuningReport:
    """Read a telemetry parquet and return a per-strategy stats bundle.

    Handles an empty parquet gracefully (returns a report with zero
    total_events so the caller can detect "nothing to analyze" without
    a special case).
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"telemetry parquet not found: {path}")

    df = pd.read_parquet(path)
    report = RetuningReport(source_parquet=str(path), total_events=len(df))
    if df.empty:
        return report

    for name, group in df.groupby("strategy_name"):
        stats = StrategyTelemetryStats(strategy_name=str(name))

        gen = group[group["event_type"] == "signal_generated"]
        ent = group[group["event_type"] == "signal_entered"]
        ext = group[group["event_type"] == "trade_exited"]

        stats.signals_generated = int(len(gen))
        stats.signals_entered = int(len(ent))
        stats.trades_exited = int(len(ext))
        stats.entry_rate_pct = round(
            (stats.signals_entered / stats.signals_generated * 100.0)
            if stats.signals_generated
            else 0.0,
            2,
        )

        rej = group[group["event_type"].astype(str).str.startswith("signal_rejected")]
        if not rej.empty and "filter_stage" in rej.columns:
            counts = rej["filter_stage"].value_counts().head(5).to_dict()
            stats.top_rejection_stages = {str(k): int(v) for k, v in counts.items()}

        if "session" in group.columns:
            session_counts = group["session"].value_counts().to_dict()
            stats.sessions = {str(k): int(v) for k, v in session_counts.items()}

        if not gen.empty:
            if "confluence_score" in gen.columns:
                scores = pd.to_numeric(gen["confluence_score"], errors="coerce").dropna()
                if not scores.empty:
                    stats.confluence_score_mean = round(float(scores.mean()), 3)
                    stats.confluence_score_median = round(float(scores.median()), 3)
            if "atr" in gen.columns:
                atrs = pd.to_numeric(gen["atr"], errors="coerce").dropna()
                if not atrs.empty:
                    stats.atr_mean = round(float(atrs.mean()), 4)
            if "planned_stop_pips" in gen.columns:
                stops = pd.to_numeric(gen["planned_stop_pips"], errors="coerce").dropna()
                if not stops.empty:
                    stats.planned_stop_pips_mean = round(float(stops.mean()), 3)
                    stats.planned_stop_pips_median = round(float(stops.median()), 3)

        report.per_strategy[str(name)] = stats

    return report


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def suggest_param_adjustments(report: RetuningReport) -> Dict[str, Dict[str, Any]]:
    """Turn a RetuningReport into concrete parameter overrides.

    Heuristics (intentionally conservative — the goal is unblocking
    the backtest, not squeezing the last drop of performance):

    * If entry_rate_pct < 0.5% — the signal filters are too strict.
      Loosen the strategy's lowest-tier threshold (``min_confluence_score``
      for ichimoku/sss, ``min_range_pips`` for asian_breakout,
      ``min_ema_angle_deg`` for ema_pullback).
    * If entry_rate_pct > 5% and signals_generated > 500 — too loose,
      tighten the opposite knob.
    * If top rejection stage is ``edge.spread`` — widen the spread cap
      in instrument config (we can't change that here; just note it).
    """
    suggestions: Dict[str, Dict[str, Any]] = {}

    for name, stats in report.per_strategy.items():
        strat_sugg: Dict[str, Any] = {}

        if stats.signals_generated < 5:
            # Not enough data — skip, nothing to base a decision on
            continue

        if stats.entry_rate_pct < 0.5:
            if name == "ichimoku":
                strat_sugg.setdefault("signal", {})["min_confluence_score"] = 0
            elif name == "sss":
                strat_sugg["min_swing_pips"] = 0.3
                strat_sugg["min_confluence_score"] = 0
            elif name == "asian_breakout":
                strat_sugg["min_range_pips"] = 1
            elif name == "ema_pullback":
                strat_sugg["min_ema_angle_deg"] = 0.5

        if stats.entry_rate_pct > 5.0 and stats.signals_generated > 500:
            if name == "ichimoku":
                strat_sugg.setdefault("signal", {})["min_confluence_score"] = 2
            elif name == "sss":
                strat_sugg["min_swing_pips"] = 1.0
            elif name == "asian_breakout":
                strat_sugg["min_range_pips"] = 10

        # Planned-stop distribution check: if the average planned stop
        # is tiny on futures, strategies are likely misinterpreting pip
        # semantics. Bump min_stop_pips upward.
        if stats.planned_stop_pips_mean is not None and stats.planned_stop_pips_mean < 5.0:
            if name == "sss":
                strat_sugg["min_stop_pips"] = 10.0

        if strat_sugg:
            suggestions[name] = strat_sugg

    return suggestions


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_overrides_to_profile(
    overrides: Dict[str, Dict[str, Any]],
    profile_path: str | Path,
) -> None:
    """Merge strategy overrides into a profile YAML file.

    Writes the overrides under a top-level ``strategy_overrides:`` key.
    Existing keys are merged shallowly per strategy (profile values
    win when the caller passes a smaller dict — use ``replace=True``
    on the caller side if that's not what you want).
    """
    import yaml

    path = Path(profile_path)
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    existing = data.get("strategy_overrides") or {}
    for strat, vals in overrides.items():
        if strat not in existing:
            existing[strat] = vals
        else:
            # Shallow merge — new values replace old keys
            existing[strat].update(vals)
    data["strategy_overrides"] = existing

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_report(report: RetuningReport, out_path: str | Path) -> Path:
    """Serialize a RetuningReport to JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_events": report.total_events,
        "source_parquet": report.source_parquet,
        "per_strategy": {
            name: asdict(stats)
            for name, stats in report.per_strategy.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Optuna sweep (opt-in, slow)
# ---------------------------------------------------------------------------


def run_optuna_sweep(
    strategy_name: str,
    data_file: str | Path,
    n_trials: int = 50,
    initial_balance: float = 50_000.0,
) -> Dict[str, Any]:
    """Run an Optuna sweep for a single strategy against MGC data.

    This is an expensive step — each trial runs a full backtest. The
    return value is a dict of the best parameters found, suitable for
    merging into ``config/profiles/futures.yaml``.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("optuna is required for run_optuna_sweep") from exc

    import pandas as pd

    from src.backtesting.vectorbt_engine import IchimokuBacktester
    from src.optimization.objectives import topstep_combine_pass_score

    df = pd.read_parquet(data_file)

    def _objective(trial: "optuna.Trial") -> float:
        params = _suggest_params_for_strategy(trial, strategy_name)
        cfg = _build_config_with_params(strategy_name, params)
        bt = IchimokuBacktester(config=cfg, initial_balance=initial_balance)
        try:
            result = bt.run(df, instrument="MGC", log_trades=False, enable_learning=False)
        except Exception as exc:
            logger.warning("trial crashed: %s", exc)
            return -2.0
        return topstep_combine_pass_score(result)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_score": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
    }


def _suggest_params_for_strategy(trial, strategy_name: str) -> Dict[str, Any]:
    """Tiny per-strategy search space for initial sweeps."""
    if strategy_name == "sss":
        return {
            "min_swing_pips": trial.suggest_float("min_swing_pips", 0.2, 3.0),
            "ss_candle_min": trial.suggest_int("ss_candle_min", 4, 15),
            "iss_candle_min": trial.suggest_int("iss_candle_min", 2, 6),
            "min_stop_pips": trial.suggest_float("min_stop_pips", 5.0, 30.0),
        }
    if strategy_name == "asian_breakout":
        return {
            "min_range_pips": trial.suggest_int("min_range_pips", 1, 15),
            "max_range_pips": trial.suggest_int("max_range_pips", 50, 300),
            "rr_ratio": trial.suggest_float("rr_ratio", 1.5, 3.0),
        }
    if strategy_name == "ema_pullback":
        return {
            "min_ema_angle_deg": trial.suggest_float("min_ema_angle_deg", 0.5, 5.0),
            "max_ema_angle_deg": trial.suggest_float("max_ema_angle_deg", 30.0, 95.0),
        }
    return {}


def _build_config_with_params(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal backtest config patching in the trial's params."""
    from src.config.loader import load_config

    app_cfg = load_config()
    cfg: Dict[str, Any] = {
        "active_strategies": [strategy_name],
        "active_strategy": "ichimoku",
        "strategies": {strategy_name: dict(params)},
        "prop_firm": app_cfg.strategy.prop_firm.model_dump()
        if app_cfg.strategy.prop_firm
        else {"style": "topstep_combine_dollar"},
        "instrument": {
            "class": "futures",
            "tick_size": 0.10,
            "tick_value_usd": 1.0,
            "commission_per_contract_round_trip": 1.40,
            "slippage_ticks": 1,
        },
        "max_concurrent_positions": 3,
    }
    return cfg
