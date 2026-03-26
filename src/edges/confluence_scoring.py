"""
Confluence scoring filter and position size tier modifier.

This edge enforces a minimum confluence score before allowing entry, and
maps the score to a position size tier. Higher-quality setups (A+) get
full risk allocation; lower-quality setups (B, C) are reduced to conserve
capital for high-conviction opportunities.

This edge operates as both a BINARY FILTER (blocks entries below min_score)
and a MODIFIER (returns size multiplier via modifier field).
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class ConfluenceScoringFilter(EdgeFilter):
    """Enforce minimum confluence score and return tiered size modifier.

    Config keys (via ``params``):
        min_score:              int   — Floor score required. Default 4.
        tier_a_plus_threshold:  int   — Score ≥ this → A+ (full size). Default 7.
        tier_b_threshold:       int   — Score ≥ this → B (75% size). Default 5.
        tier_c_threshold:       int   — Score ≥ this → C (50% size). Default 4.
        tier_b_size_pct:        float — Size fraction for B trades. Default 0.75.
        tier_c_size_pct:        float — Size fraction for C trades. Default 0.50.

    Returns
    -------
    EdgeResult
        ``allowed=False`` when score < min_score.
        ``modifier`` is the position size multiplier: 1.0 (A+), 0.75 (B), 0.50 (C).
    """

    def __init__(self, config: dict) -> None:
        super().__init__("confluence_scoring", config)
        params = config.get("params", {})
        self._min_score: int = int(params.get("min_score", 4))
        self._a_plus: int = int(params.get("tier_a_plus_threshold", 7))
        self._b: int = int(params.get("tier_b_threshold", 5))
        self._c: int = int(params.get("tier_c_threshold", 4))
        self._b_size: float = float(params.get("tier_b_size_pct", 0.75))
        self._c_size: float = float(params.get("tier_c_size_pct", 0.50))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="confluence_scoring disabled — no score filter applied",
                modifier=1.0,
            )

        score = context.confluence_score

        if score < self._min_score:
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Confluence score {score} below minimum {self._min_score} — "
                    f"trade blocked"
                ),
                modifier=0.0,
            )

        # Determine tier and size multiplier
        if score >= self._a_plus:
            tier = "A+"
            size = 1.0
        elif score >= self._b:
            tier = "B"
            size = self._b_size
        else:
            tier = "C"
            size = self._c_size

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=(
                f"Confluence score {score} → Tier {tier} "
                f"(size multiplier {size:.2f})"
            ),
            modifier=size,
        )
