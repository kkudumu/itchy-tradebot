"""Ichimoku Time Theory — Kihon Suchi projection + Taito Suchi detection."""
from __future__ import annotations
from dataclasses import dataclass, field

KIHON_NUMBERS = [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200, 226, 257]


def is_kihon_number(bars: int, tolerance: int = 1) -> bool:
    return any(abs(bars - ks) <= tolerance for ks in KIHON_NUMBERS)


def nearest_kihon(bars: int) -> int:
    for ks in KIHON_NUMBERS:
        if ks >= bars:
            return ks
    return KIHON_NUMBERS[-1]


def project_from_pivot(pivot_bar: int, total_bars: int) -> list[dict]:
    targets = []
    for ks in KIHON_NUMBERS:
        target = pivot_bar + ks
        if target < total_bars:
            targets.append({'target_bar': target, 'kihon_number': ks})
    return targets


def find_active_cycles(current_bar: int, pivots: list[dict], tolerance: int = 1) -> list[dict]:
    hits = []
    for p in pivots:
        elapsed = current_bar - p['bar_index']
        if elapsed > 0 and is_kihon_number(elapsed, tolerance):
            hits.append({
                'source_bar': p['bar_index'], 'source_price': p['price'],
                'bars_elapsed': elapsed, 'matched_kihon': nearest_kihon(elapsed),
            })
    return hits


def detect_taito_suchi(pivots: list[dict], tolerance: int = 1) -> list[dict]:
    cycles = []
    if len(pivots) < 3:
        return cycles
    sorted_p = sorted(pivots, key=lambda p: p['bar_index'])
    for i in range(len(sorted_p) - 2):
        d1 = sorted_p[i + 1]['bar_index'] - sorted_p[i]['bar_index']
        d2 = sorted_p[i + 2]['bar_index'] - sorted_p[i + 1]['bar_index']
        if abs(d1 - d2) <= tolerance:
            cycles.append({'bar_count': d1, 'matches_kihon': is_kihon_number(d1, tolerance)})
    return cycles
