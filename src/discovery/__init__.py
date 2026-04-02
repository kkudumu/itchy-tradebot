"""Creative Pattern Discovery Agent.

Strategy-agnostic edge discovery via XGBoost/SHAP analysis,
hypothesis generation, and validated rule absorption.
"""

from src.discovery.xgb_analyzer import build_training_data

__all__ = ["build_training_data"]
