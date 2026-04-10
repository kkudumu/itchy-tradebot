"""IchimokuExitManager — 3-mode configurable exit with HA confirmation.

Modes:
  trailing — Kijun + fractal trailing, no fixed target
  targets  — fixed TP at N/V/E-value, full exit
  hybrid   — 50% partial at N-value, trail remainder with Kijun/fractal

Heikin Ashi confirmation:
  In trailing/hybrid, only trail-update when HA shows weak candle (has
  opposite wick). Strong HA candle = hold position tighter.
"""
from __future__ import annotations

from ..base import TradingMode, ExitDecision, EvalMatrix
from ...indicators.heikin_ashi import compute_heikin_ashi, ha_candle_at, ha_trend_signal


class IchimokuExitManager(TradingMode):
    def __init__(self, mode='hybrid', entry_tf='1H', kijun_buffer=5.0,
                 partial_close_pct=0.5, target_price=None,
                 move_stop_to_entry=True, use_heikin_ashi=True):
        self.mode = mode
        self.tf = entry_tf
        self.kijun_buffer = kijun_buffer
        self.partial_pct = partial_close_pct
        self.target = target_price
        self.move_be = move_stop_to_entry
        self.use_ha = use_heikin_ashi
        self._partial_taken = False

    def check_exit(self, trade, current_data, eval_results):
        close = current_data['close']
        is_long = trade.direction == 'long'

        # Extract Kijun
        ichi = eval_results.get(f'ichimoku_{self.tf}')
        kijun = ichi.metadata.get('kijun') if ichi else None

        # Extract nearest opposite fractal
        frac = eval_results.get(f'fractal_{self.tf}')
        opp_frac = None
        if frac:
            key = 'last_bear_fractal' if is_long else 'last_bull_fractal'
            fd = frac.metadata.get(key)
            if fd and hasattr(fd, 'price'):
                opp_frac = fd.price

        # Kijun close check — price wrong side of Kijun = full exit
        if kijun is not None:
            if is_long and close < kijun - self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Close below Kijun-sen')
            if not is_long and close > kijun + self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Close above Kijun-sen')

        # HA confirmation for trail decisions
        ha_signal = None
        if self.use_ha and 'open' in current_data:
            try:
                import numpy as np
                o = np.array(current_data.get('open_series', [current_data.get('open', close)]))
                h = np.array(current_data.get('high_series', [current_data.get('high', close)]))
                l = np.array(current_data.get('low_series', [current_data.get('low', close)]))
                c = np.array(current_data.get('close_series', [close]))
                if len(o) >= 2:
                    ha = compute_heikin_ashi(o, h, l, c)
                    candle = ha_candle_at(ha, -1)
                    ha_signal = ha_trend_signal(candle)
            except Exception:
                pass

        if self.mode == 'targets':
            return self._check_targets(close, is_long)
        elif self.mode == 'hybrid':
            return self._check_hybrid(close, is_long, trade, kijun, opp_frac, ha_signal)
        else:
            return self._check_trailing(close, is_long, trade, kijun, opp_frac, ha_signal)

    def _check_trailing(self, close, is_long, trade, kijun, opp_frac, ha_signal):
        new_stop = trade.stop_loss
        candidates = []
        if kijun is not None:
            buf = self.kijun_buffer if is_long else -self.kijun_buffer
            candidates.append(kijun - buf if is_long else kijun + abs(buf))
        if opp_frac is not None:
            candidates.append(opp_frac)

        if candidates:
            best = max(candidates) if is_long else min(candidates)
            # Only trail if HA doesn't show strong trend (strong = hold tighter)
            if ha_signal and 'strong' in ha_signal:
                pass  # hold — don't widen the trail
            elif (is_long and best > trade.stop_loss) or (not is_long and best < trade.stop_loss):
                new_stop = best

        if new_stop != trade.stop_loss:
            return ExitDecision(action='trail_update', new_stop=new_stop, reason='Kijun/fractal trail')
        return ExitDecision(action='hold')

    def _check_targets(self, close, is_long):
        if self.target is None:
            return ExitDecision(action='hold')
        if (is_long and close >= self.target) or (not is_long and close <= self.target):
            return ExitDecision(action='full_exit', close_pct=1.0, reason=f'Target hit {self.target}')
        return ExitDecision(action='hold')

    def _check_hybrid(self, close, is_long, trade, kijun, opp_frac, ha_signal):
        # Phase 1: partial at target
        if not self._partial_taken and self.target is not None:
            hit = (is_long and close >= self.target) or (not is_long and close <= self.target)
            if hit:
                self._partial_taken = True
                new_stop = trade.entry_price if self.move_be else trade.stop_loss
                return ExitDecision(
                    action='partial_exit', close_pct=self.partial_pct,
                    new_stop=new_stop, reason=f'Partial close at N-value {self.target}',
                )
        # Phase 2: trail remainder
        return self._check_trailing(close, is_long, trade, kijun, opp_frac, ha_signal)
