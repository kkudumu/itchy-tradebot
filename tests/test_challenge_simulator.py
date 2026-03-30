"""Tests for rolling window + Monte Carlo challenge simulation."""
from src.backtesting.challenge_simulator import ChallengeSimulator, ChallengeSimulationResult


class TestRollingWindows:
    def test_correct_window_count(self):
        # ~750 trading days, spacing=22 -> ~34 windows
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5, "day_index": i}
                  for i in range(750)]
        sim = ChallengeSimulator(rolling_window_spacing_days=22)
        result = sim.run_rolling(trades, total_trading_days=750)
        assert 30 <= result.total_windows <= 36

    def test_winning_trades_pass_phase_1(self):
        # 20 trades, each +1R at 1.5% risk = +30% total -> passes 8%
        trades = [{"r_multiple": 1.0, "risk_pct": 1.5, "day_index": i * 2}
                  for i in range(20)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=40)
        assert result.phase_1_pass_count > 0

    def test_all_losers_fail(self):
        trades = [{"r_multiple": -1.0, "risk_pct": 1.5, "day_index": i * 2}
                  for i in range(50)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=100)
        assert result.full_pass_count == 0
        assert result.pass_rate == 0.0

    def test_failure_breakdown_populated(self):
        trades = [{"r_multiple": -1.0, "risk_pct": 1.5, "day_index": i}
                  for i in range(100)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=100)
        assert isinstance(result.failure_breakdown, dict)
        assert sum(result.failure_breakdown.values()) > 0


class TestMonteCarlo:
    def test_monte_carlo_runs_n_simulations(self):
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5} for _ in range(100)]
        sim = ChallengeSimulator()
        result = sim.run_monte_carlo(trades, n_simulations=1000)
        assert result.total_windows == 1000

    def test_monte_carlo_deterministic_with_seed(self):
        trades = [{"r_multiple": 0.5 if i % 2 == 0 else -0.8, "risk_pct": 1.5}
                  for i in range(100)]
        sim = ChallengeSimulator()
        r1 = sim.run_monte_carlo(trades, n_simulations=100, seed=42)
        r2 = sim.run_monte_carlo(trades, n_simulations=100, seed=42)
        assert r1.pass_rate == r2.pass_rate


class TestCombinedResult:
    def test_combined_has_both_metrics(self):
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5, "day_index": i}
                  for i in range(200)]
        sim = ChallengeSimulator()
        result = sim.run(trades, total_trading_days=200)
        assert hasattr(result, "rolling_pass_rate")
        assert hasattr(result, "monte_carlo_pass_rate")
        assert 0.0 <= result.pass_rate <= 1.0
