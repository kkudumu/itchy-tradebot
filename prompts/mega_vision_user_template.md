Decision request at {timestamp_utc}.

## Candidate signals from native blender

{candidate_signals_summary}

## Current prop firm state

- Style: {prop_firm_style}
- Distance to MLL: ${distance_to_mll_usd}
- Daily loss used: ${daily_loss_used_usd} / ${daily_loss_limit_usd}
- Distance to profit target: ${distance_to_target_usd}

## Risk state

- Open positions: {open_positions_count}
- Equity: ${equity_usd}

## Instructions

Investigate the state with your tools (start with `view_chart_screenshot`, then consult `get_strategy_performance_buckets` for the current session/regime), then call `record_strategy_pick` with your decision.

If you're unsure, picking an empty list is always safe.
