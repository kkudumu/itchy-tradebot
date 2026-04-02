
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class HighADXFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX low")
