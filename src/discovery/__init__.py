"""Creative Pattern Discovery Agent.

Strategy-agnostic edge discovery via XGBoost/SHAP analysis,
chart pattern detection, visual analysis, and validated rule absorption.
"""

from src.discovery.chart_patterns import ChartPattern, PatternDetector
from src.discovery.pattern_hook import PatternHook
from src.discovery.pattern_confluence import PatternConfluenceScorer, PatternConfluenceEdge
from src.discovery.screenshot_selector import ScreenshotSelector
from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

def __getattr__(name: str):
    """Lazy imports for heavy modules (shap/xgboost dependency chain)."""
    if name == "build_training_data":
        from src.discovery.xgb_analyzer import build_training_data
        return build_training_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_training_data",
    "ChartPattern",
    "PatternDetector",
    "PatternHook",
    "PatternConfluenceScorer",
    "PatternConfluenceEdge",
    "ScreenshotSelector",
    "build_visual_prompt",
    "parse_visual_response",
]
