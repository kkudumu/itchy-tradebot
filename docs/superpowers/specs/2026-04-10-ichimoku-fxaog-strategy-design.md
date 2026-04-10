# Ichimoku FXAOG Strategy Design

> Full replacement of the existing Ichimoku strategy with a faithful implementation
> of the FX At One Glance course methodology (First Glance + Advanced Japanese Techniques).

**Date:** 2026-04-10
**Status:** Approved
**Replaces:** `src/strategy/strategies/ichimoku.py` (key: `ichimoku`)

---

## Table of Contents

1. [Design Decisions](#design-decisions)
2. [Architecture Overview](#architecture-overview)
3. [Evaluators](#evaluators)
4. [Strategy — Trade Types & Signal Selection](#strategy--trade-types--signal-selection)
5. [Exit Management & Position Sizing](#exit-management--position-sizing)
6. [Configuration Schema](#configuration-schema)
7. [File Structure](#file-structure)
8. [System Wiring](#system-wiring)
9. [Testing Strategy](#testing-strategy)
10. [Backtest & Optimization Integration](#backtest--optimization-integration)

---

## Design Decisions

| Decision | Choice |
|---|---|
| Existing strategy | **Replace** entirely (old code moved to `_legacy/` or left importable) |
| Trade setups | **All 7**: TK crossover, Kumo breakout, Kijun bounce/break, FFO, fractal breakout, Walking Dragon, Rolling Dragon |
| Advanced techniques | **All**: Five Elements O/G, Time Theory (Kihon Suchi + Tato Suchi), Wave Analysis (I/V/N/P/Y), Price Range targets (E/V/N/NT) |
| Timeframes | **Adaptive** — configurable primary/secondary TF pairs |
| Entry triggers | **Full price-action gating** (tweezer + inside-out + engulfing) |
| Exit modes | **Configurable** (trailing / targets / hybrid) |
| O/G counting | **Configurable** (hard gate or soft filter) |
| Time Theory | **Full** (Kihon Suchi + Tato Suchi detection) |
| Architecture | **Multi-Evaluator + Strategy Coordinator** (Approach 1) |

---

## Architecture Overview

```
Bar arrives on secondary TF
  -> All evaluators run on secondary TF data
  -> Primary TF evaluators run if primary bar also closed
  -> Strategy.decide(eval_matrix):
      1. Read IchimokuCoreEval from both TFs
      2. Check 5-point checklist on both TFs (or primary Kumo bias only)
      3. Check FiveElementsEval gate (if hard_gate mode)
      4. Check TimeTheoryEval (boost or gate)
      5. Check each trade type in priority order:
         Walking Dragon -> TK Cross -> Kumo Breakout -> Kijun Bounce ->
         Kijun Break -> FFO -> Fractal Breakout -> Rolling Dragon
      6. First qualifying trade type -> score_confluence()
      7. If score >= min_confluence_score and tier >= min_tier -> emit Signal
      8. Signal includes: direction, entry_price, stop_loss, targets[],
         confluence_score, quality_tier, trade_type, reasoning{}

Signal emitted
  -> ExitManager attached to the position
  -> ExitManager.check_exit() called each bar:
      - Checks active exit mode (trailing / targets / hybrid)
      - Reads Kijun, fractal, HA state
      - Returns ExitDecision (hold / partial_close / full_close)
```

---

## Evaluators

7 evaluators, each registered in the existing `EVALUATOR_REGISTRY`.

### 1. IchimokuCoreEval (key: `ichimoku_core`)

Replaces the existing `ichimoku` evaluator. Wraps the existing `IchimokuCalculator` and produces a structured state snapshot per bar.

**Output state:**

```
IchimokuState:
  tenkan, kijun, senkou_a, senkou_b, chikou
  cloud_position: above | below | inside
  cloud_direction: bullish | bearish  (Span A vs Span B)
  tk_cross: bullish | bearish | none  (on this bar)
  tk_cross_bars_ago: int  (bars since last TK cross)
  chikou_vs_price: above | below | inside
  chikou_vs_kumo: above | below | inside
  kumo_future_direction: bullish | bearish  (26 bars ahead)
  kumo_thickness: float (ATR-normalized)
  kijun_flat: bool  (Kijun unchanged for N bars)
  kijun_distance_pips: float  (price distance from Kijun)
  tenkan_kijun_angle: float  (slope direction/strength)
```

Returns `EvaluatorResult` with direction based on the 5-point checklist. Confidence = count of aligned signals / 5.

### 2. PriceActionEval (key: `price_action`)

Scans recent bars for candlestick patterns.

**Output state:**

```
PriceActionState:
  tweezer_bottom: bool  (red + green at same low)
  tweezer_top: bool  (green + red at same high)
  inside_bar_count: int  (consecutive bars inside mother bar body)
  inside_bar_breakout: up | down | none
  engulfing_bullish: bool
  engulfing_bearish: bool
  pin_bar_bullish: bool  (hammer / inverted hammer)
  pin_bar_bearish: bool  (shooting star / hanging man)
  doji: bool
  mother_bar_high: float
  mother_bar_low: float
```

**Key logic:**
- Tweezer: two consecutive opposite-color candles with matching lows (bottom) or highs (top), within a tolerance of N ticks
- Inside bar: each subsequent candle's **body** (not wick) fits within the mother bar's body
- Mother bar fallback: if doji/hammer forms, use previous full-body candle as mother
- Outside break: candle closes beyond the mother bar body range
- Engulfing: candle body fully contains previous candle body — immediate entry trigger (no inside-bar wait)

### 3. FractalEval (key: `fractal`)

Detects Bill Williams 5-bar fractals and tracks market structure.

**Output state:**

```
FractalState:
  bull_fractals: list[FractalLevel]  (price, bar_index)
  bear_fractals: list[FractalLevel]
  last_broken_direction: bull | bear
  fractal_momentum: list[float]  (distances between successive same-type fractals)
  momentum_trend: strengthening | weakening | flat
  first_fractal_out_bull: FractalLevel | None  (first bull fractal above Kumo)
  first_fractal_out_bear: FractalLevel | None  (first bear fractal below Kumo)
  nearest_bull_fractal: FractalLevel
  nearest_bear_fractal: FractalLevel
```

Requires Kumo boundaries from `IchimokuCoreEval` to identify FFO levels. Fractal = bar with higher high (or lower low) than 2 bars on each side (5-bar minimum, no repaint).

### 4. FiveElementsEval (key: `five_elements`)

Implements the O/G equilibrium counting system from the Advanced course.

**Output state:**

```
FiveElementsState:
  cycle_start_bar: int
  cycle_direction: bullish | bearish
  overcoming_count: int
  generating_count: int
  is_disequilibrium: bool  (O != G)
  total_signals: int  (O + G)
  crossover_log: list[CrossoverEvent]
```

**Crossover table:**

| Cross | Type |
|---|---|
| Tenkan x Kijun | O (starts new cycle + resets counts) |
| Tenkan x Span A | O |
| Tenkan x Span B | G |
| Kijun x Span A | G |
| Kijun x Span B | O |
| Chikou x Tenkan | G |
| Chikou x Kijun | G |
| Chikou x Span A | O |
| Chikou x Span B | G |
| Span A x Span B (Kumo twist) | G (always) |

Each crossover counted once per cycle. Cycle resets on every TK cross.

### 5. TimeTheoryEval (key: `time_theory`)

Operates on the primary (analysis) timeframe only. Projects Kihon Suchi numbers from fractal extremes.

**Output state:**

```
TimeTheoryState:
  kihon_suchi_numbers: [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200]
  active_projections: list[TimeProjection]
    - source_fractal: FractalLevel
    - bars_elapsed: int
    - next_kihon_suchi: int
    - is_on_kihon_suchi: bool  (+/- 1 tolerance)
  tato_suchi_detected: bool
  tato_suchi_cycles: list[TatoSuchiCycle]
    - fractal_a, fractal_b: FractalLevel
    - bar_count: int
    - matching_kihon_suchi: int | None
  double_confirmation: bool  (two projections from different sources hit same day)
  confluence_score: float  (0.0-1.0)
```

### 6. WaveAnalysisEval (key: `wave_analysis`)

Classifies the current market wave pattern using fractal swing points.

**Output state:**

```
WaveAnalysisState:
  current_wave_type: I | V | N | P | Y | box
  wave_position: impulse_leg | correction_leg | breakout_pending
  n_wave_points: {A, B, C, D_projected}
  wave_direction: bullish | bearish
  higher_tf_wave_type: I | V | N | P | Y | box
  is_trading_correction: bool  (V-wave within larger N-wave)
```

**Detection logic:**
- I-wave: single directional move between two fractals
- V-wave: two I-waves (up-down or down-up) — always a correction
- N-wave: three legs (I + V + I) — THE primary pattern (A-B-C-D)
- P-wave: contracting — lower highs + higher lows (converging fractals)
- Y-wave: expanding — higher highs + lower lows (diverging fractals)
- Box: fractals clustering at same horizontal levels

### 7. PriceTargetEval (key: `price_target`)

Calculates E/V/N/NT target levels from identified 1-2-3 (A-B-C) patterns.

**Output state:**

```
PriceTargetState:
  has_valid_pattern: bool
  point_a: float  (origin of move)
  point_b: float  (end of impulse)
  point_c: float  (end of correction)
  nt_value: float  (C + (C - A), Fib 0.927)
  n_value: float   (C + (B - A), Fib 1.0 -- THE primary target)
  v_value: float   (B + (B - C), Fib 1.221)
  e_value: float   (B + (B - A), Fib 1.611)
  targets_hit: dict[str, bool]
```

**Target hierarchy (closest to farthest):**
1. Nearest opposite fractal (immediate obstacle)
2. NT-value (0.927)
3. N-value (1.0) — most reliable, default primary target
4. V-value (1.221)
5. E-value (1.611)

---

## Strategy -- Trade Types & Signal Selection

### `IchimokuFXAOGStrategy` (key: `ichimoku_fxaog`)

### 5-Point Checklist (Universal Gate)

All trade types must pass before any entry logic runs:

| # | Check | Long | Short |
|---|---|---|---|
| 1 | Price vs Kumo | Above | Below |
| 2 | Tenkan vs Kijun | Tenkan above | Tenkan below |
| 3 | Chikou vs Price (26 back) | Above | Below |
| 4 | Chikou vs Kumo | Above | Below |
| 5 | Kumo Future (26 ahead) | Bullish | Bearish |

If any fail, no trade. If Chikou is **inside** price (ranging), all trades blocked (whipsaw filter).

Exception: Kijun Bounce allows check #2 to be momentarily neutral as price touches Kijun.

### Configurable Filters

Applied after the 5-point checklist:

- `five_elements_mode`: hard_gate | soft_filter | disabled
- `time_theory_mode`: hard_gate | soft_filter | disabled
- `max_kijun_distance_pips`: skip if price too far from Kijun
- `min_kumo_thickness_atr`: skip if cloud too thin
- `signal_strength_min`: strong | neutral | weak (position relative to Kumo)

### 7 Trade Types (checked in priority order)

#### Priority 1: Walking the Dragon

Trend continuation after TK cross pullback.

**Conditions:**
1. TK cross occurred within last 5-10 bars
2. Price pulled back inside Tenkan-sen after the cross
3. Price closes back outside Tenkan-sen in the cross direction
4. Both Tenkan and Kijun angling strongly in trade direction

**Entry:** Close of the bar that exits Tenkan.
**Stop:** Below Kijun-sen or below the lowest fractal of the pullback.

#### Priority 2: TK Crossover

**Conditions:**
1. TK cross on current bar
2. Price close to Kijun-sen (within `max_kijun_distance_pips`)
3. Both Tenkan and Kijun moving in trade direction (not flat)
4. Price action confirmation: engulfing OR tweezer + inside-out breakout

**Entry:** Close of the confirmation candle.
**Stop:** Below Kijun-sen or below nearest opposite fractal.

#### Priority 3: Kumo Breakout

**Conditions:**
1. Price closes outside Kumo for the first time
2. Price on correct side of Kijun-sen
3. All 5-point checklist met
4. The breakout candle itself is the trigger (no additional PA required)

**Entry:** Close of the breakout candle.
**Stop:** Breakout candle open, Kumo edge, or Kijun-sen (configurable).

#### Priority 4: Kijun Bounce

The instructor's favorite. Mean-reversion to equilibrium.

**Conditions:**
1. Price on correct side of Kumo
2. Price pulls back TO Kijun-sen (within `kijun_proximity_atr` tolerance)
3. Price action at Kijun: tweezer + inside-out breakout OR engulfing
4. Kijun is NOT flat (flat = range = whipsaw risk, unless `reject_flat_kijun: false`)

**Entry:** Outside break of inside-bar range, or close of engulfing candle.
**Stop:** Other side of Kijun-sen, Kumo edge, or fractal level.

#### Priority 5: Kijun Break

When bounce fails — price breaks through Kijun, gets supported by Kumo, then crosses back through.

**Conditions:**
1. Price on correct side of Kumo
2. Price breaks and closes through Kijun-sen in trade direction
3. Kumo future confirms direction

**Entry:** Close of the candle that breaks back through Kijun.
**Stop:** Kumo edge or nearest fractal.

#### Priority 6: First Fractal Out (FFO)

Catches the trend when Kumo breakout was missed.

**Conditions:**
1. Price above Kumo and Kijun (longs) / below both (shorts)
2. First bull/bear fractal formed outside the Kumo identified by `FractalEval`
3. Price breaks beyond that fractal level

**Entry:** Break of the FFO fractal level.
**Stop:** Fractal candle open, Kumo edge, or Kijun.

#### Priority 7: Fractal Breakout

Trend continuation via market structure.

**Conditions:**
1. Price above Kumo and Kijun (longs) / below both (shorts)
2. `last_broken_direction` confirms trend direction
3. Price breaks the next fractal in the trend direction
4. Fractal momentum not weakening

**Entry:** Break of the fractal level.
**Stop:** Opposite fractal, Kijun, or fractal candle open.

#### Priority 8: Rolling the Dragon (C-Clamp)

Mean-reversion when price overextends from Kijun.

**Conditions:**
1. Tenkan/Kijun were close, then separated significantly (gap > `c_clamp_threshold_atr`)
2. Kijun-sen has flattened
3. Price action rejection candle at the extreme
4. Target = Kijun-sen level (not a runner)

**Entry:** Beyond the rejection candle in the direction of Kijun.
**Stop:** Above/below the extreme before Kijun flattened.

### Signal Arbitration

- Highest priority trade type wins when multiple fire on the same bar
- Only one signal per bar per instrument
- If same priority (shouldn't happen), highest confluence score wins

### Confluence Scoring (0-15 scale)

| Component | Points | Source |
|---|---|---|
| 5-point checklist aligned | 0-5 | IchimokuCoreEval |
| Signal strength (strong/neutral/weak) | 0-2 | Position relative to Kumo |
| Price action quality | 0-2 | Engulfing=2, inside-out=2, pin bar=1, bare signal=0 |
| O/G disequilibrium | 0-2 | 3+ signals imbalanced=2, any imbalance=1, balanced=0 |
| Kihon Suchi alignment | 0-2 | Double confirmation=2, single=1, none=0 |
| Wave context favorable | 0-1 | Trading with N-wave=1, against=0 |
| Fractal momentum aligned | 0-1 | Strengthening=1, weakening=0 |

**Quality tiers:**
- A+: 12-15
- A: 9-11
- B: 6-8
- C: 0-5

### MTA Integration

Evaluators run on both primary and secondary timeframes. Config option `mta_mode`:
- `kumo_bias`: primary Kumo position filters secondary entries (Method 1)
- `full_alignment`: all 4 signals must align on both TFs (Method 2)

---

## Exit Management & Position Sizing

### Exit Modes (configurable via `exit.mode`)

#### Mode 1: `trailing`

- Trail stop with Kijun-sen (with `kijun_buffer_pips` buffer)
- Trail stop with fractal levels (each new opposite fractal)
- Use whichever is tighter (closer to price)
- Exit triggers: price closes wrong side of Kijun, breaks opposite fractal, or opposite TK cross
- Walking the Dragon special exit: price closes back inside Tenkan
- Rolling the Dragon special exit: price reaches Kijun (target achieved)

#### Mode 2: `targets`

- Take profit at configured `primary_target` (default: N-value/1.0 expansion)
- Target hierarchy: nearest fractal -> NT (0.927) -> N (1.0) -> V (1.221) -> E (1.611)
- Fixed SL and TP, no trailing
- `respect_fractal_obstacle: true` exits if nearest fractal blocks path to target

#### Mode 3: `hybrid`

Course-recommended approach for multi-lot positions:
1. Enter with full position, trail with Kijun/fractal
2. At `primary_target` (default N-value), close `partial_close_pct` (default 50%)
3. Move SL to entry (breakeven) on remainder
4. Trail remainder with Kijun/fractal until exit trigger

### Stop Loss Placement (universal)

Preference order (configurable): fractal -> kijun -> kumo_edge -> candle_open

- `kijun_buffer_pips`: 5 (never place SL exactly on Kijun)
- `max_stop_pips`: 100 (hard cap — skip trade if exceeded)
- `min_rr_ratio`: 1.5 (skip if target_distance / stop_distance below this)

### Position Sizing

- `max_risk_pct`: 2.0% per trade
- `max_total_open_risk_pct`: 2.0% across all open positions
- Quality tier multipliers: A+=1.5x, A=1.0x, B=0.5x, C=0x (no trade)
- Integrated with existing `InstrumentSizer`

### Heikin Ashi Integration

Regular candles for entry, HA for exit management (configurable):
- `use_for: exit_only` (default)
- HA color change or doji/spinning top = additional exit signal
- No wick on momentum side = hold/add; wick appearing = tighten; small body = prepare to exit

### Session Filters

- `close_before_weekend: true` — flatten before Friday close
- `friday_cutoff_utc: "18:00"` — no new trades after this
- `news_blackout_minutes: 30` — requires external calendar (disabled if unavailable)

---

## Configuration Schema

Full config block for `config/strategy.yaml`:

```yaml
active_strategy: ichimoku_fxaog

strategies:
  ichimoku_fxaog:
    # --- Ichimoku Core ---
    ichimoku:
      tenkan_period: 9
      kijun_period: 26
      senkou_b_period: 52
      chikou_offset: 26

    # --- Timeframe Pair (Adaptive) ---
    timeframes:
      primary: "4H"
      secondary: "1H"
      mta_mode: full_alignment  # kumo_bias | full_alignment

    # --- Trade Types ---
    trade_types:
      walking_dragon:
        enabled: true
        pullback_window: [5, 10]
        min_angle_threshold: 0.3
      tk_crossover:
        enabled: true
        max_kijun_distance_pips: 200
        require_both_sloping: true
      kumo_breakout:
        enabled: true
      kijun_bounce:
        enabled: true
        kijun_proximity_atr: 0.5
        reject_flat_kijun: true
      kijun_break:
        enabled: true
      ffo:
        enabled: true
      fractal_breakout:
        enabled: true
        reject_weakening_momentum: true
      rolling_dragon:
        enabled: true
        c_clamp_threshold_atr: 2.0
        require_flat_kijun: true

    # --- Price Action Gating ---
    price_action:
      tweezer_tick_tolerance: 2
      inside_bar_use_body_only: true
      min_inside_bars: 1
      engulfing_immediate_entry: true

    # --- Five Elements O/G ---
    five_elements:
      mode: hard_gate  # hard_gate | soft_filter | disabled
      min_total_signals: 3
      count_once_per_cycle: true

    # --- Time Theory ---
    time_theory:
      mode: soft_filter  # hard_gate | soft_filter | disabled
      kihon_suchi: [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200]
      tolerance_bars: 1
      max_fractal_sources: 5
      tato_suchi_enabled: true
      apply_on_timeframe: primary

    # --- Wave Analysis ---
    wave_analysis:
      enabled: true
      warn_trading_correction: true
      correction_score_penalty: 2

    # --- Price Targets ---
    price_targets:
      enabled: true
      fib_levels:
        nt_value: 0.927
        n_value: 1.0
        v_value: 1.221
        e_value: 1.611

    # --- Exit Management ---
    exit:
      mode: hybrid  # trailing | targets | hybrid
      partial_close_pct: 50
      primary_target: n_value
      trail_remainder: true
      move_stop_to_entry: true

    # --- Stop Loss ---
    stop_loss:
      preference_order: [fractal, kijun, kumo_edge, candle_open]
      kijun_buffer_pips: 5
      max_stop_pips: 100
      min_rr_ratio: 1.5

    # --- Heikin Ashi ---
    heikin_ashi:
      enabled: true
      use_for: exit_only

    # --- Quality / Sizing ---
    quality:
      min_tier: B
      a_plus_size_mult: 1.5
      a_size_mult: 1.0
      b_size_mult: 0.5

    # --- Session Filters ---
    session_filters:
      close_before_weekend: true
      friday_cutoff_utc: "18:00"
      news_blackout_minutes: 30

    # --- Signal ---
    signal:
      max_signals_per_bar: 1
      min_confluence_score: 6
```

---

## File Structure

```
src/
  indicators/
    ichimoku.py              # KEEP -- IchimokuCalculator already correct
    heikin_ashi.py           # NEW -- HA candle computation
    fractals.py              # NEW -- Bill Williams 5-bar fractal detection
    price_action.py          # NEW -- candlestick pattern detection engine
  strategy/
    evaluators/
      ichimoku_eval.py       # REPLACE -- new IchimokuCoreEval
      price_action_eval.py   # NEW
      fractal_eval.py        # NEW
      five_elements_eval.py  # NEW
      time_theory_eval.py    # NEW
      wave_analysis_eval.py  # NEW
      price_target_eval.py   # NEW
    strategies/
      ichimoku.py            # REPLACE -- new IchimokuFXAOGStrategy
    exits/
      ichimoku_exit_manager.py  # NEW -- all 3 exit modes + HA integration
config/
  strategy.yaml              # UPDATE -- new config block
```

**Total: 8 new files, 3 replaced files, 1 updated file.**

---

## System Wiring

### Registration

Each evaluator uses the existing decorator/registry pattern:

```python
class IchimokuCoreEval(Evaluator, key='ichimoku_core'):
    ...

class PriceActionEval(Evaluator, key='price_action'):
    ...
```

### Dynamic Evaluator Requirements

```python
class IchimokuFXAOGStrategy(Strategy, key='ichimoku_fxaog'):

    @property
    def required_evaluators(self) -> list[EvalRequirement]:
        primary = self.config.timeframes.primary
        secondary = self.config.timeframes.secondary
        both = [primary, secondary]

        reqs = [
            EvalRequirement('ichimoku_core', both),
            EvalRequirement('price_action', [secondary]),
            EvalRequirement('fractal', both),
        ]

        if self.config.five_elements.mode != 'disabled':
            reqs.append(EvalRequirement('five_elements', [primary]))

        if self.config.time_theory.mode != 'disabled':
            reqs.append(EvalRequirement('time_theory', [primary]))

        if self.config.wave_analysis.enabled:
            reqs.append(EvalRequirement('wave_analysis', both))

        if self.config.price_targets.enabled:
            reqs.append(EvalRequirement('price_target', [secondary]))

        return reqs
```

### Warmup Bars

```python
@property
def warmup_bars(self) -> int:
    base = 52  # Senkou B period
    if self.config.time_theory.mode != 'disabled':
        return max(base, 200) * self._bar_multiplier(self.config.timeframes.primary)
    return base * self._bar_multiplier(self.config.timeframes.primary)
```

### Backward Compatibility

- Existing `ichimoku` strategy key remains in the registry
- `active_strategy: ichimoku_fxaog` selects the new strategy
- Switching back: `active_strategy: ichimoku`
- Existing `KijunExitMode` not deleted, just superseded

---

## Testing Strategy

### Unit Tests (per evaluator)

```
tests/
  strategy/
    evaluators/
      test_ichimoku_core_eval.py
      test_price_action_eval.py
      test_fractal_eval.py
      test_five_elements_eval.py
      test_time_theory_eval.py
      test_wave_analysis_eval.py
      test_price_target_eval.py
    strategies/
      test_ichimoku_fxaog.py
    exits/
      test_ichimoku_exit_manager.py
  indicators/
    test_heikin_ashi.py
    test_fractals.py
    test_price_action.py
```

### Critical Test Cases per Evaluator

| Evaluator | Critical tests |
|---|---|
| IchimokuCoreEval | 5-point checklist all-bullish/bearish/mixed, Chikou inside (blocked), flat Kijun, Kijun distance |
| PriceActionEval | Tweezer detection, inside bar body-only counting, outside break, engulfing, doji/hammer mother-bar fallback |
| FractalEval | 5-bar detection (not 3), no repaint, FFO vs Kumo, momentum measurement, last-broken-direction |
| FiveElementsEval | O/G per crossover table, cycle reset on TK cross, count-once rule, equilibrium vs disequilibrium |
| TimeTheoryEval | Kihon Suchi at 9/17/26, +/-1 tolerance, Tato Suchi equal cycles, double confirmation, no false positives |
| WaveAnalysisEval | N-wave (3-leg), V-wave as correction, P-wave converging, Y-wave diverging, box clustering |
| PriceTargetEval | N=C+(B-A), V=B+(B-C), E=B+(B-A), NT=C+(C-A), correct direction, targets_hit tracking |

### Strategy Integration Tests

| Test | Verifies |
|---|---|
| TK cross + 5-point + engulfing | Signal emitted with correct direction, SL, tier |
| TK cross + Chikou inside | No signal (whipsaw filter) |
| Kumo breakout + wrong future | No signal (checklist #5 fails) |
| Kijun bounce + tweezer + 3 inside + outside break | Signal on outside-break bar only |
| Walking Dragon: cross -> pullback -> re-exit | Signal on re-exit bar, SL at pullback fractal |
| Rolling Dragon: C-clamp + flat Kijun + rejection | Signal with target = Kijun |
| O/G hard gate: balanced blocks valid setup | No signal |
| O/G soft filter: balanced drops score below min | No signal |
| Multiple trade types same bar | Highest priority wins |
| Hybrid exit: partial at N-value | Position reduced, SL to entry |
| Trailing exit: Kijun break | Full close |
| HA exit: color change | Full close |

---

## Backtest & Optimization Integration

### Backtest

Plugs into existing `run_demo_challenge.py` with zero runner changes:

```bash
python scripts/run_demo_challenge.py --mode validate \
    --data-file data/projectx_mgc_1m_20260101_20260409.parquet
```

### Telemetry

Enriched `reasoning` dict in signals for telemetry and dashboard:

```python
reasoning={
    'trade_type': 'kijun_bounce',
    'confluence_score': 11,
    'quality_tier': 'A',
    'checklist': {1: True, 2: True, 3: True, 4: True, 5: True},
    'five_elements': {'o_count': 3, 'g_count': 1, 'disequilibrium': True},
    'time_theory': {'kihon_suchi': 26, 'source_fractal': '2026-03-15', 'tato_suchi': False},
    'wave_context': {'type': 'N', 'position': 'impulse_leg', 'direction': 'bullish'},
    'price_targets': {'n_value': 2385.50, 'v_value': 2401.20, 'e_value': 2425.80},
    'price_action': {'trigger': 'inside_out_breakout', 'inside_bars': 3},
    'ha_state': {'color': 'green', 'momentum': 'strong'},
    'stop_loss_type': 'fractal',
    'stop_distance_pips': 25,
    'target_distance_pips': 62,
    'rr_ratio': 2.48,
}
```

### Optimization

Exposes tunable parameters to Optuna via `suggest_params()`:

```python
def suggest_params(self, trial):
    return {
        'min_confluence_score': trial.suggest_int('min_confluence_score', 4, 12),
        'min_tier': trial.suggest_categorical('min_tier', ['A_plus', 'A', 'B']),
        'exit_mode': trial.suggest_categorical('exit_mode', ['trailing', 'targets', 'hybrid']),
        'primary_target': trial.suggest_categorical('primary_target', ['n_value', 'v_value']),
        'partial_close_pct': trial.suggest_int('partial_close_pct', 30, 70, step=10),
        'max_kijun_distance_pips': trial.suggest_int('max_kijun_distance_pips', 100, 300, step=50),
        'kijun_proximity_atr': trial.suggest_float('kijun_proximity_atr', 0.3, 1.0, step=0.1),
        'five_elements_mode': trial.suggest_categorical('five_elements_mode',
            ['hard_gate', 'soft_filter', 'disabled']),
        'time_theory_mode': trial.suggest_categorical('time_theory_mode',
            ['hard_gate', 'soft_filter', 'disabled']),
        'min_rr_ratio': trial.suggest_float('min_rr_ratio', 1.0, 3.0, step=0.5),
        'timeframe_primary': trial.suggest_categorical('tf_primary', ['D1', '4H', '1H']),
        'timeframe_secondary': trial.suggest_categorical('tf_secondary', ['4H', '1H', '15M']),
        'heikin_ashi_enabled': trial.suggest_categorical('ha_enabled', [True, False]),
    }
```

### Validation Checklist

1. All evaluator unit tests pass
2. Strategy integration tests pass for each trade type
3. Exit manager tests pass for all 3 modes
4. Backtest runs end-to-end with `run_demo_challenge.py --mode validate`
5. Dashboard displays correctly with new strategy metadata
6. Optimization loop can tune new strategy parameters
7. Telemetry parquet contains enriched reasoning fields
8. Switching back to `active_strategy: ichimoku` still works
