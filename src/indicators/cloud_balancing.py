"""Wu Xing Five Elements O/G counting system.

Elements: Fire=Tenkan, Water=Kijun, Wood=Chikou, Metal=SpanA, Earth=SpanB

Crossover O/G classification per course:
  Tenkan x SpanA = O,  Tenkan x SpanB = G
  Kijun x SpanA = G,   Kijun x SpanB = O
  Chikou x Tenkan = G,  Chikou x Kijun = G
  Chikou x SpanA = O,   Chikou x SpanB = G
  SpanA x SpanB (Kumo twist) = G (always)

Rules:
  - TK cross starts new cycle, resets count
  - Each crossover type counted only ONCE per cycle
  - O == G → equilibrium → DO NOT TRADE
  - O != G → disequilibrium → CAN TRADE in TK cross direction
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

_CROSS_TABLE = {
    ('tenkan', 'senkou_a'): 'O',
    ('tenkan', 'senkou_b'): 'G',
    ('kijun', 'senkou_a'): 'G',
    ('kijun', 'senkou_b'): 'O',
    ('chikou', 'tenkan'): 'G',
    ('chikou', 'kijun'): 'G',
    ('chikou', 'senkou_a'): 'O',
    ('chikou', 'senkou_b'): 'G',
    ('senkou_a', 'senkou_b'): 'G',
}

@dataclass
class BalanceState:
    o_count: int = 0
    g_count: int = 0
    is_disequilibrium: bool = False
    tk_direction: int = 0
    cycle_start_bar: int = 0
    crossover_log: list[dict] = field(default_factory=list)

class CloudBalancer:
    def calculate(
        self, tenkan: np.ndarray, kijun: np.ndarray,
        chikou: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray,
    ) -> BalanceState:
        n = len(tenkan)
        components = {
            'tenkan': tenkan, 'kijun': kijun, 'chikou': chikou,
            'senkou_a': senkou_a, 'senkou_b': senkou_b,
        }
        tk_crosses = []
        for i in range(1, n):
            if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1]):
                continue
            prev = tenkan[i-1] - kijun[i-1]
            curr = tenkan[i] - kijun[i]
            if prev <= 0 < curr:
                tk_crosses.append((i, 1))
            elif prev >= 0 > curr:
                tk_crosses.append((i, -1))
        if not tk_crosses:
            return BalanceState()
        cycle_bar, tk_dir = tk_crosses[-1]
        o_count, g_count = 0, 0
        seen: set[str] = set()
        log: list[dict] = []
        for (c1_name, c2_name), og_type in _CROSS_TABLE.items():
            c1, c2 = components.get(c1_name), components.get(c2_name)
            if c1 is None or c2 is None:
                continue
            pair_key = f"{c1_name}_{c2_name}"
            if pair_key in seen:
                continue
            for i in range(cycle_bar + 1, n):
                if i < 1 or np.isnan(c1[i]) or np.isnan(c2[i]) or np.isnan(c1[i-1]) or np.isnan(c2[i-1]):
                    continue
                if (c1[i-1] - c2[i-1]) * (c1[i] - c2[i]) < 0:
                    seen.add(pair_key)
                    if og_type == 'O':
                        o_count += 1
                    else:
                        g_count += 1
                    log.append({'pair': pair_key, 'type': og_type, 'bar': i})
                    break
        return BalanceState(
            o_count=o_count, g_count=g_count,
            is_disequilibrium=(o_count != g_count),
            tk_direction=tk_dir, cycle_start_bar=cycle_bar,
            crossover_log=log,
        )
