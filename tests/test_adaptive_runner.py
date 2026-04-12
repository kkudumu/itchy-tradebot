"""Integration test: run one optimization epoch on MGC with reduced trials.

Verifies the full adaptive optimization pipeline end-to-end:
  load data -> embed market -> query DB for warm-starts ->
  Optuna optimize -> backtest -> persist trials -> check guardrails.

The guardrail functions (consecutive passes + permutation significance) are
patched to return fast synthetic results so that the test completes in
under 10 minutes.  The guardrails themselves have dedicated unit tests in
test_guardrails.py.
"""
import logging
import sys
from unittest.mock import patch

import pytest


def _fast_consecutive_passes(*args, **kwargs):
    """Stub that pretends all consecutive passes succeeded."""
    return {"all_passed": True, "results": []}


def _fast_permutation_significance(*args, **kwargs):
    """Stub that pretends the permutation test is significant."""
    return {"p_value": 0.01, "significant": True, "n_permutations": 1}


@pytest.mark.integration
@pytest.mark.slow
class TestAdaptiveRunnerIntegration:
    def test_one_epoch_mgc(self):
        """Run 1 Optuna trial on MGC to verify the full pipeline.

        Each trial backtests ~27K 1-minute bars across multiple strategies,
        so a single trial takes 3-5 minutes on commodity hardware.

        Guardrails are patched out to avoid an additional 20+ re-backtests
        that would push total runtime past an hour.
        """
        # Enable INFO logging so progress is visible with pytest -s
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            force=True,
        )

        from src.database.connection import DatabasePool, DBConfig
        from src.optimization.adaptive_runner import AdaptiveRunner

        config = DBConfig(
            host="localhost",
            port=5433,
            dbname="trading",
            user="postgres",
            password="postgres",
        )
        pool = DatabasePool(config=config)
        pool.initialise()

        try:
            with patch(
                "src.optimization.adaptive_runner.check_consecutive_passes",
                side_effect=lambda **kw: _fast_consecutive_passes(**kw),
            ), patch(
                "src.optimization.adaptive_runner.check_permutation_significance",
                side_effect=lambda **kw: _fast_permutation_significance(**kw),
            ):
                runner = AdaptiveRunner(db_pool=pool, trials_per_epoch=1)
                results = runner.run_once(instrument_filter="MGC")

            # The result should be keyed by the optimizer symbol "MGC"
            assert "MGC" in results, (
                f"Expected 'MGC' in results, got {list(results.keys())}"
            )

            result = results["MGC"]

            # Should not have errored out
            assert "error" not in result, (
                f"Optimization errored: {result.get('error', '')}"
            )

            # Basic structure checks on the status dict returned by
            # optimize_instrument()
            assert result["symbol"] == "MGC"
            assert result["engine_symbol"] == "XAUUSD"
            assert result["epoch"] == 1
            assert result["n_trials"] == 1
            assert isinstance(result["best_score"], (int, float))
            assert isinstance(result["best_passed"], bool)
            assert isinstance(result["total_trials_in_db"], int)
            assert result["total_trials_in_db"] >= 1  # at least our 1 trial
            assert result["data_bars"] > 0

            # Guardrail fields should always be present (even if False)
            assert "consecutive_passes" in result
            assert "proven_config_saved" in result
        finally:
            pool.close()
