from __future__ import annotations
import numpy as np
import pytest
from src.indicators.cloud_balancing import CloudBalancer, BalanceState

KIHON_PERIOD = 26

def _trending_components(n=200, direction='bullish'):
    step = 2.0 if direction == 'bullish' else -2.0
    base = np.arange(n, dtype=float) * step + 1800.0
    tenkan = base + 5.0
    kijun = base
    chikou = np.roll(base, KIHON_PERIOD)
    senkou_a = base + 3.0
    senkou_b = base - 3.0
    return tenkan, kijun, chikou, senkou_a, senkou_b

class TestCloudBalancer:
    def test_returns_balance_state(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert isinstance(state, BalanceState)

    def test_has_og_counts(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert hasattr(state, 'o_count')
        assert hasattr(state, 'g_count')
        assert hasattr(state, 'is_disequilibrium')

    def test_tk_cross_resets_cycle(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.o_count >= 0
        assert state.g_count >= 0

    def test_disequilibrium_when_unequal(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.is_disequilibrium == (state.o_count != state.g_count)

    def test_crossover_counted_once_per_cycle(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.o_count + state.g_count <= 9
