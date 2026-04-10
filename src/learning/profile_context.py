"""Profile context for the adaptive learning engine (plan Task 20).

Produces a small metadata dict describing the active instrument's
profile (class, prop firm style, account / risk limits, tick scale)
and converts it into a fixed-length numeric feature vector suitable
for concatenation with the existing 64-dim trade embeddings.

Used by:
  * ``src/learning/embeddings.py`` — appends the profile features to
    the trade embedding so similarity search respects profile
    context (e.g. a futures trade won't match a forex trade purely
    on price-action similarity).
  * ``src/learning/adaptive_engine.py`` (or equivalent) — carries
    the metadata on trade context dicts so pre-trade analysis can
    filter by profile.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.config.profile import InstrumentClass


PROFILE_FEATURE_DIM: int = 8
"""Length of the numeric vector returned by :func:`profile_embedding_features`."""


def profile_metadata(
    instrument_config: Any,
    prop_firm_config: Any = None,
) -> Dict[str, Any]:
    """Return a metadata dict describing the profile of a trade.

    Accepts anything with ``class_``/``instrument_class`` on the
    instrument side and an optional prop firm config (pydantic model
    or plain dict). All fields are nullable — fields that don't apply
    to a given profile stay ``None`` so downstream code can tell "not
    set" from "zero".
    """
    cls = _resolve_class(instrument_config)

    meta: Dict[str, Any] = {
        "instrument_class": cls.value if cls else None,
        "prop_firm_style": None,
        "account_size_usd": None,
        "max_loss_limit_usd": None,
        "daily_loss_limit_usd": None,
        "tick_size": _maybe_float(getattr(instrument_config, "tick_size", None)),
        "tick_value_usd": _maybe_float(
            getattr(instrument_config, "tick_value_usd", None)
            or getattr(instrument_config, "tick_value", None)
        ),
        "pip_size": _maybe_float(getattr(instrument_config, "pip_size", None)),
        "pip_value_per_lot": _maybe_float(
            getattr(instrument_config, "pip_value_per_lot", None)
            or getattr(instrument_config, "pip_value_usd", None)
        ),
    }

    if prop_firm_config is not None:
        meta.update(_extract_prop_firm_fields(prop_firm_config))

    return meta


def profile_embedding_features(metadata: Dict[str, Any]) -> np.ndarray:
    """Turn profile metadata into a deterministic 8-dim float vector.

    Layout (all normalised to roughly [-1, 1] where possible):
      [0] is_futures (1.0 if class is futures, 0.0 otherwise)
      [1] is_forex (1.0 if class is forex, 0.0 otherwise)
      [2] account_size_log10 / 6 (e.g. $50K → 4.7/6 ≈ 0.78)
      [3] max_loss_usd / account_size (headroom fraction, e.g. 0.04)
      [4] daily_loss_usd / account_size (e.g. 0.02)
      [5] tick_scale_log10 / 4 (e.g. 0.10 → -1/4 = -0.25)
      [6] 1.0 if prop_firm_style == "topstep_combine_dollar" else 0.0
      [7] 1.0 if prop_firm_style == "the5ers_pct_phased" else 0.0
    """
    vec = np.zeros(PROFILE_FEATURE_DIM, dtype=np.float32)

    cls = metadata.get("instrument_class")
    vec[0] = 1.0 if cls == "futures" else 0.0
    vec[1] = 1.0 if cls == "forex" else 0.0

    account = _maybe_float(metadata.get("account_size_usd"))
    if account and account > 0:
        vec[2] = float(np.log10(account) / 6.0)

        max_loss = _maybe_float(metadata.get("max_loss_limit_usd")) or 0.0
        daily_loss = _maybe_float(metadata.get("daily_loss_limit_usd")) or 0.0
        vec[3] = float(max_loss / account) if account else 0.0
        vec[4] = float(daily_loss / account) if account else 0.0

    tick = _maybe_float(metadata.get("tick_size")) or _maybe_float(
        metadata.get("pip_size")
    )
    if tick and tick > 0:
        vec[5] = float(np.log10(tick) / 4.0)

    style = metadata.get("prop_firm_style") or ""
    vec[6] = 1.0 if style == "topstep_combine_dollar" else 0.0
    vec[7] = 1.0 if style == "the5ers_pct_phased" else 0.0

    return vec


def pad_embedding_with_profile(
    base_embedding: np.ndarray,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """Append profile features to a base (e.g. 64-dim) embedding.

    Backward-compat helper for the ``EmbeddingEngine`` — old 64-dim
    vectors stored before this pass get 8 extra zeroed dims when
    read back, so similarity search still works.
    """
    profile_feats = profile_embedding_features(metadata)
    return np.concatenate([np.asarray(base_embedding, dtype=np.float32), profile_feats])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_class(inst: Any) -> Optional[InstrumentClass]:
    cls = getattr(inst, "class_", None) or getattr(inst, "instrument_class", None)
    if isinstance(cls, InstrumentClass):
        return cls
    if isinstance(cls, str):
        try:
            return InstrumentClass(cls)
        except ValueError:
            return None
    return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_prop_firm_fields(pf: Any) -> Dict[str, Any]:
    """Pull account_size / loss limits / style out of a pydantic model or dict."""
    out: Dict[str, Any] = {}

    def _get(field: str, default: Any = None) -> Any:
        if hasattr(pf, field):
            return getattr(pf, field)
        if isinstance(pf, dict):
            return pf.get(field, default)
        return default

    style = _get("style") or "unknown"
    out["prop_firm_style"] = str(style)
    out["account_size_usd"] = _maybe_float(_get("account_size"))

    if style == "topstep_combine_dollar":
        out["max_loss_limit_usd"] = _maybe_float(_get("max_loss_limit_usd_trailing"))
        out["daily_loss_limit_usd"] = _maybe_float(_get("daily_loss_limit_usd"))
    elif style == "the5ers_pct_phased":
        # Convert pct to dollars for consistent feature layout
        account = out["account_size_usd"] or 0.0
        p1 = _get("phase_1") or {}
        max_loss_pct = _maybe_float(
            p1.get("max_loss_pct") if isinstance(p1, dict) else getattr(p1, "max_loss_pct", None)
        )
        daily_loss_pct = _maybe_float(
            p1.get("daily_loss_pct") if isinstance(p1, dict) else getattr(p1, "daily_loss_pct", None)
        )
        if account and max_loss_pct is not None:
            out["max_loss_limit_usd"] = account * (max_loss_pct / 100.0)
        if account and daily_loss_pct is not None:
            out["daily_loss_limit_usd"] = account * (daily_loss_pct / 100.0)

    return out
