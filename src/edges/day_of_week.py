"""
Day-of-week entry filter.

Monday and Friday exhibit reduced directional consistency in gold due to
pre-weekend positioning and range-setting behaviour. Restricting to
Tuesday–Thursday (default) improves signal quality.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DayOfWeekFilter(EdgeFilter):
    """Allow entries only on configured weekdays.

    Default allowed days: Tuesday, Wednesday, Thursday (indices 1, 2, 3).

    Config keys (via ``params``):
        allowed_days: list[int]  — 0=Mon … 6=Sun. Default [1, 2, 3].
    """

    def __init__(self, config: dict) -> None:
        super().__init__("day_of_week", config)
        params = config.get("params", {})
        self._allowed_days: set[int] = set(params.get("allowed_days", [1, 2, 3]))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        day = context.day_of_week
        day_name = _DAY_NAMES[day] if 0 <= day <= 6 else str(day)
        allowed_names = [_DAY_NAMES[d] for d in sorted(self._allowed_days) if 0 <= d <= 6]

        if day in self._allowed_days:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=f"{day_name} is in allowed days: {', '.join(allowed_names)}",
            )

        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"{day_name} is not in allowed days: {', '.join(allowed_names)}"
            ),
        )
