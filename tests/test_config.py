"""Tests for the configuration loading and validation system."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader, load_config
from src.config.models import (
    AppConfig,
    EdgeConfig,
    InstrumentsConfig,
    ProviderConfig,
    StrategyConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_config_dir() -> Path:
    """Return the real config/ directory bundled with the project."""
    return Path(__file__).resolve().parents[1] / "config"


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return an empty temporary directory for isolated config tests."""
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Package-level smoke test
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_without_error(self, project_config_dir: Path) -> None:
        cfg = load_config(config_dir=project_config_dir)
        assert isinstance(cfg, AppConfig)

    def test_returns_app_config_type(self, project_config_dir: Path) -> None:
        cfg = load_config(config_dir=project_config_dir)
        assert isinstance(cfg.edges, EdgeConfig)
        assert isinstance(cfg.strategy, StrategyConfig)
        assert isinstance(cfg.instruments, InstrumentsConfig)
        assert isinstance(cfg.provider, ProviderConfig)


# ---------------------------------------------------------------------------
# 2. Edge toggle access pattern: config.edges.<name>.enabled
# ---------------------------------------------------------------------------


class TestEdgeToggles:
    """Verify all 12 edge toggles are accessible and return bool values."""

    EDGE_NAMES = [
        "time_of_day",
        "day_of_week",
        "london_open_delay",
        "candle_close_confirmation",
        "spread_filter",
        "news_filter",
        "friday_close",
        "regime_filter",
        "time_stop",
        "bb_squeeze",
        "confluence_scoring",
        "equity_curve",
    ]

    @pytest.fixture(autouse=True)
    def cfg(self, project_config_dir: Path) -> None:
        self._cfg = load_config(config_dir=project_config_dir)

    def test_all_12_edges_present(self) -> None:
        for name in self.EDGE_NAMES:
            assert hasattr(self._cfg.edges, name), f"Missing edge: {name}"

    def test_all_edges_have_enabled_bool(self) -> None:
        for name in self.EDGE_NAMES:
            edge = getattr(self._cfg.edges, name)
            assert isinstance(edge.enabled, bool), (
                f"{name}.enabled must be bool, got {type(edge.enabled)}"
            )

    def test_all_edges_have_params_dict(self) -> None:
        for name in self.EDGE_NAMES:
            edge = getattr(self._cfg.edges, name)
            assert isinstance(edge.params, dict), (
                f"{name}.params must be dict, got {type(edge.params)}"
            )

    # Individual edge defaults from edges.yaml
    def test_time_of_day_defaults(self) -> None:
        e = self._cfg.edges.time_of_day
        assert e.enabled is False  # Disabled: strategies have built-in session logic
        assert e.params["start_utc"] == "06:00"
        assert e.params["end_utc"] == "21:00"

    def test_day_of_week_defaults(self) -> None:
        e = self._cfg.edges.day_of_week
        assert e.enabled is False
        assert e.params["allowed_days"] == [1, 2, 3]  # Tue/Wed/Thu

    def test_london_open_delay_defaults(self) -> None:
        e = self._cfg.edges.london_open_delay
        assert e.enabled is False
        assert e.params["delay_minutes"] == 30

    def test_spread_filter_defaults(self) -> None:
        e = self._cfg.edges.spread_filter
        assert e.enabled is False
        assert e.params["max_spread_points"] == 30

    def test_news_filter_defaults(self) -> None:
        e = self._cfg.edges.news_filter
        assert e.enabled is False
        assert e.params["block_minutes_before"] == 30
        assert e.params["block_minutes_after"] == 30
        assert "red" in e.params["impact_levels"]

    def test_friday_close_defaults(self) -> None:
        e = self._cfg.edges.friday_close
        assert e.enabled is True
        assert e.params["close_time_utc"] == "20:00"

    def test_regime_filter_defaults(self) -> None:
        e = self._cfg.edges.regime_filter
        assert e.enabled is False
        assert e.params["adx_min"] == 28

    def test_time_stop_defaults(self) -> None:
        e = self._cfg.edges.time_stop
        assert e.enabled is False
        assert e.params["candle_limit"] == 12
        assert e.params["breakeven_r_threshold"] == pytest.approx(0.5)

    def test_bb_squeeze_defaults(self) -> None:
        e = self._cfg.edges.bb_squeeze
        assert e.enabled is False
        assert e.params["bb_period"] == 20

    def test_confluence_scoring_defaults(self) -> None:
        e = self._cfg.edges.confluence_scoring
        assert e.enabled is False
        assert e.params["min_score"] == 4
        assert e.params["tier_a_plus_threshold"] == 7

    def test_equity_curve_defaults(self) -> None:
        e = self._cfg.edges.equity_curve
        assert e.enabled is False  # Disabled by default in edges.yaml
        assert e.params["lookback_trades"] == 20

    def test_toggle_edge_off_via_yaml(self, tmp_config_dir: Path) -> None:
        """Disabling an edge in YAML must propagate through the loader."""
        write_yaml(
            tmp_config_dir / "edges.yaml",
            {"time_of_day": {"enabled": False, "params": {}}},
        )
        cfg = load_config(config_dir=tmp_config_dir)
        assert cfg.edges.time_of_day.enabled is False


# ---------------------------------------------------------------------------
# 3. Strategy config values
# ---------------------------------------------------------------------------


class TestStrategyConfig:
    @pytest.fixture(autouse=True)
    def cfg(self, project_config_dir: Path) -> None:
        self._cfg = load_config(config_dir=project_config_dir)

    def test_ichimoku_periods(self) -> None:
        ich = self._cfg.strategy.ichimoku
        assert ich.tenkan_period == 9
        assert ich.kijun_period == 26
        assert ich.senkou_b_period == 52

    def test_adx_threshold(self) -> None:
        assert self._cfg.strategy.adx.threshold == 20

    def test_risk_params(self) -> None:
        risk = self._cfg.strategy.risk
        assert risk.initial_risk_pct == pytest.approx(0.5)
        assert risk.reduced_risk_pct == pytest.approx(0.75)
        assert risk.phase_threshold_pct == pytest.approx(4.0)
        assert risk.daily_circuit_breaker_pct == pytest.approx(4.5)
        assert risk.max_concurrent_positions == 3

    def test_exit_params(self) -> None:
        ex = self._cfg.strategy.exit
        assert ex.strategy == "hybrid_50_50"
        assert ex.tp_r_multiple == pytest.approx(1.5)
        assert ex.trail_type == "kijun"
        assert ex.breakeven_threshold_r == pytest.approx(1.0)

    def test_signal_tiers(self) -> None:
        sig = self._cfg.strategy.signal
        assert sig.min_confluence_score == 1
        assert sig.tier_a_plus == 7
        assert sig.tier_b == 5
        assert sig.tier_c == 1

    def test_signal_timeframes(self) -> None:
        tf = self._cfg.strategy.signal.timeframes
        assert "15M" in tf
        assert "5M" in tf


# ---------------------------------------------------------------------------
# 4. Instruments config
# ---------------------------------------------------------------------------


class TestInstrumentsConfig:
    @pytest.fixture(autouse=True)
    def cfg(self, project_config_dir: Path) -> None:
        self._cfg = load_config(config_dir=project_config_dir)

    def test_xauusd_is_configured(self) -> None:
        inst = self._cfg.instruments.get("XAUUSD")
        assert inst is not None

    def test_xauusd_symbol(self) -> None:
        inst = self._cfg.instruments.get("XAUUSD")
        assert inst.symbol == "XAUUSD"
        assert inst.provider == "projectx"

    def test_unknown_symbol_returns_none(self) -> None:
        assert self._cfg.instruments.get("EURUSD") is None


# ---------------------------------------------------------------------------
# 5. Config loader behaviour
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_missing_files_use_defaults(self, tmp_config_dir: Path) -> None:
        """Loading from an empty directory must succeed with Pydantic defaults."""
        cfg = load_config(config_dir=tmp_config_dir)
        assert isinstance(cfg, AppConfig)

    def test_partial_override_preserves_defaults(self, tmp_config_dir: Path) -> None:
        """Supplying only one key in a YAML should not break other defaults."""
        write_yaml(tmp_config_dir / "strategy.yaml", {"ichimoku": {"tenkan_period": 5}})
        cfg = load_config(config_dir=tmp_config_dir)
        # Overridden
        assert cfg.strategy.ichimoku.tenkan_period == 5
        # Defaults preserved
        assert cfg.strategy.ichimoku.kijun_period == 26

    def test_loader_exposes_config_dir(self, project_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=project_config_dir)
        assert loader.config_dir == project_config_dir

    def test_reload_returns_fresh_config(self, tmp_config_dir: Path) -> None:
        loader = ConfigLoader(config_dir=tmp_config_dir)
        cfg1 = loader.load()

        # Write a change and reload
        write_yaml(tmp_config_dir / "strategy.yaml", {"ichimoku": {"tenkan_period": 7}})
        cfg2 = loader.reload()

        assert cfg2.strategy.ichimoku.tenkan_period == 7
        # First load should still show the default (9)
        assert cfg1.strategy.ichimoku.tenkan_period == 9

    def test_provider_yaml_is_loaded(self, project_config_dir: Path) -> None:
        cfg = load_config(config_dir=project_config_dir)
        assert cfg.provider.provider == "projectx"
        assert cfg.provider.projectx.api_base_url.startswith("https://api.")
