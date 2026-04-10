"""N-wave detection, E/V/N/NT price targets, I/V/N/P/Y wave classification,
and rules-based 5-wave Elliott counting."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def n_value(a, b, c, direction):
    return c + (b - a) if direction == 'bullish' else c - (a - b)


def v_value(a, b, c, direction):
    return b + (b - c) if direction == 'bullish' else b - (c - b)


def e_value(a, b, c, direction):
    return b + (b - a) if direction == 'bullish' else b - (a - b)


def nt_value(a, b, c, direction):
    return c + (c - a) if direction == 'bullish' else c - (a - c)


def compute_all_targets(a, b, c, direction):
    return {
        'nt_value': nt_value(a, b, c, direction),
        'n_value': n_value(a, b, c, direction),
        'v_value': v_value(a, b, c, direction),
        'e_value': e_value(a, b, c, direction),
    }


class WaveAnalyzer:
    def build_swing_sequence(self, bulls, bears):
        all_s = [{'type': 'high', 'price': f.price, 'bar': f.bar_index} for f in bulls]
        all_s += [{'type': 'low', 'price': f.price, 'bar': f.bar_index} for f in bears]
        all_s.sort(key=lambda s: s['bar'])
        filtered = []
        for s in all_s:
            if filtered and filtered[-1]['type'] == s['type']:
                if s['type'] == 'high' and s['price'] > filtered[-1]['price']:
                    filtered[-1] = s
                elif s['type'] == 'low' and s['price'] < filtered[-1]['price']:
                    filtered[-1] = s
            else:
                filtered.append(s)
        return filtered

    def classify(self, swings, current_price):
        if len(swings) < 2:
            d = 'bullish' if len(swings) == 0 or current_price > swings[0]['price'] else 'bearish'
            return {'wave_type': 'I', 'direction': d, 'position': 'impulse', 'targets': {}}

        if len(swings) >= 4:
            s = swings[-4:]
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low' and s[3]['type'] == 'high':
                if s[3]['price'] > s[1]['price'] and s[2]['price'] > s[0]['price']:
                    targets = compute_all_targets(s[0]['price'], s[1]['price'], s[2]['price'], 'bullish')
                    pos = 'impulse' if current_price > s[2]['price'] else 'correction'
                    return {'wave_type': 'N', 'direction': 'bullish', 'position': pos,
                            'targets': targets, 'A': s[0]['price'], 'B': s[1]['price'], 'C': s[2]['price']}
            if s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high' and s[3]['type'] == 'low':
                if s[3]['price'] < s[1]['price'] and s[2]['price'] < s[0]['price']:
                    targets = compute_all_targets(s[0]['price'], s[1]['price'], s[2]['price'], 'bearish')
                    pos = 'impulse' if current_price < s[2]['price'] else 'correction'
                    return {'wave_type': 'N', 'direction': 'bearish', 'position': pos,
                            'targets': targets, 'A': s[0]['price'], 'B': s[1]['price'], 'C': s[2]['price']}

        if len(swings) >= 4:
            highs = [s['price'] for s in swings if s['type'] == 'high']
            lows = [s['price'] for s in swings if s['type'] == 'low']
            if len(highs) >= 2 and len(lows) >= 2:
                if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
                    return {'wave_type': 'P', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}
                if highs[-1] > highs[-2] and lows[-1] < lows[-2]:
                    return {'wave_type': 'Y', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}

        if len(swings) >= 4:
            prices = [s['price'] for s in swings[-6:]]
            rng = max(prices) - min(prices)
            if np.mean(prices) > 0 and rng / np.mean(prices) < 0.01:
                return {'wave_type': 'box', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}

        if len(swings) >= 3:
            s = swings[-3:]
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low':
                return {'wave_type': 'V', 'direction': 'bearish', 'position': 'correction', 'targets': {}}
            if s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high':
                return {'wave_type': 'V', 'direction': 'bullish', 'position': 'correction', 'targets': {}}

        d = 'bullish' if current_price > swings[0]['price'] else 'bearish'
        return {'wave_type': 'I', 'direction': d, 'position': 'impulse', 'targets': {}}


def count_elliott(swing_prices, direction):
    if len(swing_prices) < 4:
        return None
    is_bull = direction == 'bullish'
    p = swing_prices
    w1 = abs(p[1] - p[0])
    w2_retrace = abs(p[2] - p[1])
    if w2_retrace >= w1:
        return None
    confidence = 0.0
    w2_pct = w2_retrace / w1 if w1 > 0 else 0
    if 0.50 <= w2_pct <= 0.786:
        confidence += 0.25
    if len(swing_prices) >= 4:
        w3 = abs(p[3] - p[2])
        if w1 > 0 and w3 / w1 >= 1.382:
            confidence += 0.25
    if len(swing_prices) >= 5:
        w3 = abs(p[3] - p[2])
        w4_retrace = abs(p[4] - p[3])
        if is_bull and p[4] <= p[1]:
            return None
        if not is_bull and p[4] >= p[1]:
            return None
        w4_pct = w4_retrace / w3 if w3 > 0 else 0
        if 0.236 <= w4_pct <= 0.50:
            confidence += 0.25
    if len(swing_prices) >= 6:
        w3 = abs(p[3] - p[2])
        w5 = abs(p[5] - p[4])
        if w3 < w1 and w3 < w5:
            return None
        if w1 > 0 and 0.8 <= w5 / w1 <= 1.2:
            confidence += 0.25
    wave_number = min(len(swing_prices) - 1, 5)
    return {
        'wave_number': wave_number,
        'confidence': confidence,
        'is_complete': wave_number >= 5,
        'direction': direction,
    }
