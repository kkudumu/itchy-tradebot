"""YAML config loader with Pydantic validation.

Usage
-----
    from src.config import load_config

    cfg = load_config()                      # reads from <project_root>/config/
    cfg = load_config(config_dir="/custom")  # explicit path

    # Typed access
    cfg.edges.time_of_day_filter.enabled     # bool
    cfg.strategy.ichimoku.tenkan_period      # int
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config.models import (
    AppConfig,
    DatabaseConfig,
    EdgeConfig,
    InstrumentsConfig,
    ProviderConfig,
    StrategyConfig,
)
from src.config.profile import InstrumentClass, ProfileConfig, load_profile

# Default config directory is <project_root>/config/
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict.

    Returns an empty dict when the file does not exist, allowing optional
    config files to be omitted without raising errors.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive copy)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class ConfigLoader:
    """Loads, merges, and validates configuration from the YAML files.

    File resolution order (later files take precedence):
        1. config/edges.yaml
        2. config/strategy.yaml
        3. config/database.yaml
        4. config/instruments.yaml
        5. config/provider.yaml
        6. config/profiles/<class>.yaml for every class referenced by an
           instrument (forex / futures)

    Environment variable ``CONFIG_DIR`` overrides the default directory.
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is not None:
            self._dir = Path(config_dir).resolve()
        elif "CONFIG_DIR" in os.environ:
            self._dir = Path(os.environ["CONFIG_DIR"]).resolve()
        else:
            self._dir = _DEFAULT_CONFIG_DIR

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> AppConfig:
        """Read all YAML files and return a fully validated :class:`AppConfig`."""
        edges_data = _load_yaml(self._dir / "edges.yaml")
        strategy_data = _load_yaml(self._dir / "strategy.yaml")
        database_data = _load_yaml(self._dir / "database.yaml")
        instruments_data = _load_yaml(self._dir / "instruments.yaml")
        provider_data = _load_yaml(self._dir / "provider.yaml")

        instruments = InstrumentsConfig.model_validate(instruments_data)

        # Load profile defaults for every class referenced by an instrument,
        # plus both standard classes so downstream code can always look up
        # either profile regardless of which instruments are configured.
        profile_dir = self._dir / "profiles"
        profiles: dict[InstrumentClass, ProfileConfig] = {}
        classes_in_use = {inst.class_ for inst in instruments.instruments}
        classes_in_use.update({InstrumentClass.FOREX, InstrumentClass.FUTURES})
        for cls in classes_in_use:
            profiles[cls] = load_profile(cls, profile_dir=profile_dir)

        # Merge profile-level strategy_overrides into strategy_data so
        # forex and futures get independent strategy tuning without
        # duplicating the whole strategy.yaml. The active instrument's
        # class determines which profile's overrides apply. When
        # multiple instruments with different classes are configured,
        # the FIRST instrument's class wins (single-instrument is the
        # normal case).
        active_class = InstrumentClass.FOREX
        for inst in instruments.instruments:
            active_class = inst.class_
            break
        active_profile = profiles.get(active_class)
        if active_profile and active_profile.strategy_overrides:
            strategies_block = strategy_data.get("strategies", {})
            for strat_name, overrides in active_profile.strategy_overrides.items():
                if strat_name in strategies_block and isinstance(overrides, dict):
                    strategies_block[strat_name] = _deep_merge(
                        strategies_block[strat_name], overrides
                    )
                else:
                    strategies_block[strat_name] = overrides
            strategy_data["strategies"] = strategies_block

        return AppConfig(
            edges=EdgeConfig.model_validate(edges_data),
            strategy=StrategyConfig.model_validate(strategy_data),
            database=DatabaseConfig.model_validate(database_data),
            instruments=instruments,
            provider=ProviderConfig.model_validate(provider_data),
            profiles=profiles,
        )

    # ------------------------------------------------------------------
    # Helpers exposed for testing / tooling
    # ------------------------------------------------------------------

    @property
    def config_dir(self) -> Path:
        return self._dir

    def reload(self) -> AppConfig:
        """Re-read all YAML files from disk — useful during live development."""
        return self.load()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    """Load and return a validated :class:`AppConfig`.

    Parameters
    ----------
    config_dir:
        Path to the directory that contains the YAML config files.
        Defaults to ``<project_root>/config/``.
    """
    return ConfigLoader(config_dir=config_dir).load()
