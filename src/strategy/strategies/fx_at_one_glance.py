"""FXAtOneGlance — faithful FXAOG course implementation.

9 trade types, 5-point checklist gate, configurable Five Elements and
Time Theory filters, 8 TF modes, 0-15 confluence scoring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..base import Strategy, EvalMatrix, EvalRequirement, ConfluenceResult
from ..signal_engine import Signal


TF_MODES = {
    'swing':      {'bias': 'D',  'entry': '4H', 'chikou': True,  'ichi_entry': True},
    'intraday':   {'bias': '4H', 'entry': '1H', 'chikou': True,  'ichi_entry': True},
    'hybrid':     {'bias': '4H', 'entry': '15M','chikou': False, 'ichi_entry': True},
    'scalp':      {'bias': '1H', 'entry': '15M','chikou': False, 'ichi_entry': True},
    'hyperscalp_m15_m5': {'bias': '15M','entry': '5M', 'chikou': False, 'ichi_entry': 'stripped'},
    'hyperscalp_h1_m5':  {'bias': '1H', 'entry': '5M', 'chikou': False, 'ichi_entry': 'stripped'},
    'hyperscalp_m5_m1':  {'bias': '5M', 'entry': '1M', 'chikou': False, 'ichi_entry': False},
    'hyperscalp_h1_m1':  {'bias': '1H', 'entry': '1M', 'chikou': False, 'ichi_entry': False},
}


class FXAtOneGlance(Strategy, key='fx_at_one_glance'):

    warmup_bars = 200

    def __init__(self, config=None, instrument='XAUUSD'):
        cfg = config or {}
        self._instrument = instrument
        self._tf_mode_name = cfg.get('tf_mode', 'intraday')
        mode = TF_MODES[self._tf_mode_name]
        self._bias_tf = mode['bias']
        self._entry_tf = mode['entry']
        self._use_chikou = mode['chikou']
        self._ichi_entry = mode['ichi_entry']

        self._fe_mode = cfg.get('five_elements_mode', 'hard_gate')
        self._tt_mode = cfg.get('time_theory_mode', 'soft_filter')
        self._min_score = cfg.get('min_confluence_score', 6)
        self._min_tier = cfg.get('min_tier', 'B')
        self._max_kijun_dist = cfg.get('max_kijun_distance_pips', 200.0)
        self._kijun_buffer = cfg.get('kijun_buffer_pips', 5.0)
        self._max_stop = cfg.get('max_stop_pips', 100.0)
        self._min_rr = cfg.get('min_rr_ratio', 1.5)
        self._primary_target_key = cfg.get('primary_target', 'n_value')
        self._exit_mode = cfg.get('exit_mode', 'hybrid')
        self._partial_pct = cfg.get('partial_close_pct', 0.5)

        # Build evaluator requirements based on TF mode
        both = [self._bias_tf, self._entry_tf]
        reqs = [
            EvalRequirement('ichimoku', both),
            EvalRequirement('fractal', both),
            EvalRequirement('wave', both),
            EvalRequirement('price_action', [self._entry_tf]),
            EvalRequirement('cloud_balance', [self._bias_tf]),
            EvalRequirement('kihon_suchi', [self._bias_tf]),
            EvalRequirement('adx', [self._entry_tf]),
            EvalRequirement('atr', [self._entry_tf]),
        ]
        if not self._use_chikou:
            reqs.append(EvalRequirement('rsi', [self._entry_tf]))
        reqs.append(EvalRequirement('divergence', [self._entry_tf]))
        self.required_evaluators = reqs

        # Exit manager wired in Task 10
        self.trading_mode = None

    # --- 5-Point Checklist ---
    def _checklist_direction(self, matrix):
        """Check 5-point on entry TF. Returns 'long'/'short' or None.

        Ichimoku metadata uses int values:
            cloud_position: 1=above, -1=below, 0=inside
            tk_cross: 1=bullish, -1=bearish, 0=none
            chikou_confirmed: 1=bullish, -1=bearish, 0=neutral
        """
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return None
        meta = ichi.metadata

        # Chikou gate: inside (0) = ranging = block
        if self._use_chikou and meta.get('chikou_confirmed', 0) == 0:
            return None
        # RSI gate on lower TFs
        if not self._use_chikou:
            rsi = matrix.get(f'rsi_{self._entry_tf}')
            if rsi and abs(rsi.metadata.get('rsi', 50) - 50) < 5:
                return None  # RSI too close to 50 = no conviction

        if ichi.direction > 0:
            direction = 'long'
        elif ichi.direction < 0:
            direction = 'short'
        else:
            return None

        # Bias TF alignment (cloud_position is int: 1=above, -1=below, 0=inside)
        bias_ichi = matrix.get(f'ichimoku_{self._bias_tf}')
        if bias_ichi:
            bias_pos = bias_ichi.metadata.get('cloud_position', 0)
            if direction == 'long' and bias_pos == -1:  # below cloud
                return None
            if direction == 'short' and bias_pos == 1:  # above cloud
                return None
        return direction

    # --- Filters ---
    def _filter_five_elements(self, matrix):
        if self._fe_mode == 'disabled':
            return True
        fe = matrix.get(f'cloud_balance_{self._bias_tf}')
        if not fe:
            return self._fe_mode != 'hard_gate'
        if self._fe_mode == 'hard_gate':
            return fe.metadata.get('is_disequilibrium', False)
        return True

    def _filter_time_theory(self, matrix):
        if self._tt_mode in ('disabled', 'soft_filter'):
            return True
        tt = matrix.get(f'kihon_suchi_{self._bias_tf}')
        if not tt:
            return False
        return tt.metadata.get('is_cycle_date', False)

    def _filter_range(self, matrix):
        """Triple range filter: ADX>20 + cloud balance disequilibrium + cloud thickness."""
        adx = matrix.get(f'adx_{self._entry_tf}')
        if adx and adx.metadata.get('adx', 0) < 20:
            return False
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if ichi and ichi.metadata.get('cloud_thickness', 999) < 0.50:
            return False
        return True

    # --- Helper: compute derived ichimoku values ---
    def _ichi_derived(self, matrix):
        """Compute derived values from raw ichimoku metadata that the strategy needs."""
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return {}
        meta = ichi.metadata
        derived = {}
        # Kijun distance in pips (close vs kijun)
        kijun = meta.get('kijun')
        tenkan = meta.get('tenkan')
        if kijun is not None and tenkan is not None:
            derived['kijun_distance_pips'] = abs(tenkan - kijun) * 10  # approx for gold
        else:
            derived['kijun_distance_pips'] = 9999
        # TK cross (int: 1=bullish, -1=bearish, 0=none)
        derived['tk_cross'] = meta.get('tk_cross', 0)
        # Cloud position (int: 1=above, -1=below, 0=inside)
        derived['cloud_position'] = meta.get('cloud_position', 0)
        # Kijun value
        derived['kijun'] = kijun
        derived['tenkan'] = tenkan
        return derived

    # --- 9 Trade Types (priority order) ---
    def _detect_trade_type(self, matrix, direction, derived):
        checks = [
            self._check_walking_dragon,
            self._check_tk_crossover,
            self._check_kumo_breakout,
            self._check_kijun_bounce,
            self._check_kijun_break,
            self._check_ffo,
            self._check_fractal_breakout,
            self._check_kumo_twist,
            self._check_rolling_dragon,
        ]
        for check in checks:
            result = check(matrix, direction, derived)
            if result:
                return result
        return None

    def _check_walking_dragon(self, matrix, direction, derived):
        # Walking dragon: recent TK cross with strong angle
        tk = derived.get('tk_cross', 0)
        expected = 1 if direction == 'long' else -1
        if tk == expected and derived.get('kijun_distance_pips', 9999) <= 100:
            return 'walking_dragon'

    def _check_tk_crossover(self, matrix, direction, derived):
        expected = 1 if direction == 'long' else -1
        if derived.get('tk_cross', 0) == expected and derived.get('kijun_distance_pips', 9999) <= self._max_kijun_dist:
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa and pa.direction != 0.0:
                return 'tk_crossover'

    def _check_kumo_breakout(self, matrix, direction, derived):
        expected_pos = 1 if direction == 'long' else -1
        if derived.get('cloud_position', 0) == expected_pos:
            return 'kumo_breakout'

    def _check_kijun_bounce(self, matrix, direction, derived):
        if derived.get('kijun_distance_pips', 9999) <= 50:
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa:
                pm = pa.metadata
                if pm.get('inside_bar_breakout') != 'none':
                    return 'kijun_bounce'
                if direction == 'long' and pm.get('engulfing_bullish'):
                    return 'kijun_bounce'
                if direction == 'short' and pm.get('engulfing_bearish'):
                    return 'kijun_bounce'

    def _check_kijun_break(self, matrix, direction, derived):
        pa = matrix.get(f'price_action_{self._entry_tf}')
        if pa:
            pm = pa.metadata
            if direction == 'long' and pm.get('engulfing_bullish'):
                return 'kijun_break'
            if direction == 'short' and pm.get('engulfing_bearish'):
                return 'kijun_break'

    def _check_ffo(self, matrix, direction, derived):
        frac = matrix.get(f'fractal_{self._entry_tf}')
        if not frac:
            return None
        expected = 'uptrend' if direction == 'long' else 'downtrend'
        if frac.metadata.get('structure') == expected:
            if frac.metadata.get('momentum_trend') == 'strengthening':
                return 'ffo'

    def _check_fractal_breakout(self, matrix, direction, derived):
        frac = matrix.get(f'fractal_{self._entry_tf}')
        if not frac:
            return None
        expected = 'uptrend' if direction == 'long' else 'downtrend'
        if frac.metadata.get('structure') == expected and frac.metadata.get('momentum_trend') != 'weakening':
            return 'fractal_breakout'

    def _check_kumo_twist(self, matrix, direction, derived):
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return None
        # cloud_direction is int: 1=bullish, -1=bearish, 0=neutral
        cd = ichi.metadata.get('cloud_direction', 0)
        expected = 1 if direction == 'long' else -1
        if cd == expected:
            return 'kumo_twist'

    def _check_rolling_dragon(self, matrix, direction, derived):
        if derived.get('kijun_distance_pips', 0) > 50:
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa:
                pm = pa.metadata
                if direction == 'long' and pm.get('pin_bar_bullish'):
                    return 'rolling_dragon'
                if direction == 'short' and pm.get('pin_bar_bearish'):
                    return 'rolling_dragon'

    # --- Confluence Scoring (0-15) ---
    def score_confluence(self, eval_matrix, direction=None):
        score, breakdown = 0, {}

        # 1. Checklist (0-5) — count aligned ichimoku dimensions
        ichi = eval_matrix.get(f'ichimoku_{self._entry_tf}')
        checklist = 0
        if ichi:
            meta = ichi.metadata
            # Each aligned dimension = 1 point
            if meta.get('tk_cross', 0) != 0:
                checklist += 1
            if meta.get('cloud_position', 0) != 0:
                checklist += 1
            if meta.get('chikou_confirmed', 0) != 0:
                checklist += 1
            if meta.get('cloud_thickness', 0) > 0:
                checklist += 1
            if meta.get('cloud_direction', 0) != 0:
                checklist += 1
        score += checklist
        breakdown['checklist'] = checklist

        # 2. Signal strength — cloud position (0-2)
        ss = 2 if ichi and meta.get('cloud_position', 0) != 0 else 0
        score += ss
        breakdown['signal_strength'] = ss

        # 3. Price action quality (0-2)
        pa = eval_matrix.get(f'price_action_{self._entry_tf}')
        pa_pts = 0
        if pa:
            if pa.metadata.get('engulfing_bullish') or pa.metadata.get('engulfing_bearish'):
                pa_pts = 2
            elif pa.metadata.get('inside_bar_breakout') != 'none':
                pa_pts = 2
            elif pa.metadata.get('pin_bar_bullish') or pa.metadata.get('pin_bar_bearish'):
                pa_pts = 1
        score += pa_pts
        breakdown['price_action'] = pa_pts

        # 4. O/G disequilibrium (0-2)
        fe = eval_matrix.get(f'cloud_balance_{self._bias_tf}')
        fe_pts = 0
        if fe and fe.metadata.get('is_disequilibrium'):
            fe_pts = 2 if (fe.metadata.get('o_count', 0) + fe.metadata.get('g_count', 0)) >= 3 else 1
        score += fe_pts
        breakdown['five_elements'] = fe_pts

        # 5. Kihon Suchi (0-2)
        tt = eval_matrix.get(f'kihon_suchi_{self._bias_tf}')
        tt_pts = 0
        if tt:
            if tt.metadata.get('double_confirmation'):
                tt_pts = 2
            elif tt.metadata.get('is_cycle_date'):
                tt_pts = 1
        score += tt_pts
        breakdown['time_theory'] = tt_pts

        # 6. Wave context (0-1) — not in correction
        wa = eval_matrix.get(f'wave_{self._bias_tf}')
        wa_pts = 1 if wa and not wa.metadata.get('is_correction') else 0
        score += wa_pts
        breakdown['wave_context'] = wa_pts

        # 7. Fractal momentum (0-1)
        frac = eval_matrix.get(f'fractal_{self._entry_tf}')
        fm = 1 if frac and frac.metadata.get('momentum_trend') == 'strengthening' else 0
        score += fm
        breakdown['fractal_momentum'] = fm

        # Tier
        if score >= 12:
            tier = 'A+'
        elif score >= 9:
            tier = 'A'
        elif score >= 6:
            tier = 'B'
        elif score >= 4:
            tier = 'C'
        else:
            tier = 'no_trade'

        return ConfluenceResult(score=score, quality_tier=tier, breakdown=breakdown)

    # --- Main decide() ---
    def decide(self, eval_matrix):
        direction = self._checklist_direction(eval_matrix)
        if not direction:
            return None
        if not self._filter_five_elements(eval_matrix):
            return None
        if not self._filter_time_theory(eval_matrix):
            return None
        if not self._filter_range(eval_matrix):
            return None

        derived = self._ichi_derived(eval_matrix)
        if not derived:
            return None

        trade_type = self._detect_trade_type(eval_matrix, direction, derived)
        if not trade_type:
            return None

        confluence = self.score_confluence(eval_matrix, direction)
        if confluence.score < self._min_score:
            return None
        tier_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'no_trade': 4}
        if tier_order.get(confluence.quality_tier, 4) > tier_order.get(self._min_tier, 2):
            return None

        # Entry, SL, TP
        kijun = derived.get('kijun', 0)
        entry_price = derived.get('tenkan', kijun)
        atr_eval = eval_matrix.get(f'atr_{self._entry_tf}')
        atr = atr_eval.metadata.get('atr', 10.0) if atr_eval else 10.0

        # Stop from fractal or kijun
        frac = eval_matrix.get(f'fractal_{self._entry_tf}')
        stop_loss = None
        if frac:
            frac_key = 'last_bear_fractal' if direction == 'long' else 'last_bull_fractal'
            frac_data = frac.metadata.get(frac_key)
            if frac_data and hasattr(frac_data, 'price'):
                stop_loss = (frac_data.price - self._kijun_buffer) if direction == 'long' else (frac_data.price + self._kijun_buffer)
        if stop_loss is None and kijun:
            stop_loss = (kijun - self._kijun_buffer) if direction == 'long' else (kijun + self._kijun_buffer)
        if stop_loss is None:
            return None

        # TP from wave targets
        wave = eval_matrix.get(f'wave_{self._entry_tf}')
        target = None
        if wave and wave.metadata.get('targets'):
            target = wave.metadata['targets'].get(self._primary_target_key)
        if target is None:
            target = entry_price + (entry_price - stop_loss) * 2  # fallback 2R

        # R:R and max stop checks
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        if risk <= 0 or reward / risk < self._min_rr or risk > self._max_stop:
            return None

        # Wire exit manager target if present
        if self.trading_mode is not None:
            try:
                self.trading_mode.target = target
                self.trading_mode._partial_taken = False
            except AttributeError:
                pass

        return Signal(
            timestamp=datetime.now(timezone.utc),
            instrument=self._instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=target,
            confluence_score=confluence.score,
            quality_tier=confluence.quality_tier,
            atr=atr,
            strategy_name='fx_at_one_glance',
            reasoning={
                'trade_type': trade_type,
                'tf_mode': self._tf_mode_name,
                'confluence_score': confluence.score,
                'quality_tier': confluence.quality_tier,
                **confluence.breakdown,
            },
        )

    def suggest_params(self, trial):
        return {
            'min_confluence_score': trial.suggest_int('min_confluence_score', 4, 12),
            'min_tier': trial.suggest_categorical('min_tier', ['A_plus', 'A', 'B']),
            'exit_mode': trial.suggest_categorical('exit_mode', ['trailing', 'targets', 'hybrid']),
            'primary_target': trial.suggest_categorical('primary_target', ['n_value', 'v_value']),
            'five_elements_mode': trial.suggest_categorical('five_elements_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'time_theory_mode': trial.suggest_categorical('time_theory_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'min_rr_ratio': trial.suggest_float('min_rr_ratio', 1.0, 3.0, step=0.5),
            'tf_mode': trial.suggest_categorical('tf_mode', ['swing', 'intraday', 'hybrid', 'scalp']),
        }

    def populate_edge_context(self, eval_matrix):
        ichi = eval_matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return {}
        return {
            'kijun': ichi.metadata.get('kijun'),
            'cloud_thickness': ichi.metadata.get('cloud_thickness'),
        }
