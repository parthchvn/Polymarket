"""Explicit fixed-model context counterfactuals.

These are FEATURE-SPACE INTERVENTIONS scored under a FIXED fitted model:
Delta_CF = log P(D_obs | C) - log P(D_obs | C_intervened).

They are deliberately distinct from the refit group ablation in
``reasoning.py`` (which retrains without a channel and answers "was this
channel informative to the model?"); an intervention answers "under the
fitted model, what if this specific context had looked uninformative?".
Both are stored separately and never conflated.

Missingness semantics are preserved: interventions set explicit
missing-indicator coding (indicator -> 1, value -> neutral fill) or the
training reference (standardized mean), never a blind post-
standardization zero for every feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from polymarket.analysis.features import FEATURE_GROUPS
from polymarket.analysis.models import LogisticModel
from polymarket.analysis.reasoning import (
    ATTRIBUTION_FEATURES,
    ATTRIBUTION_GROUPS,
    _log_prob_observed,
)

Intervention = Callable[[dict[str, float], LogisticModel], dict[str, float]]


def _training_mean(model: LogisticModel, name: str) -> float:
    index = model.feature_names.index(name)
    return float(model.standardizer.mean[index])


def remove_fresh_news(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    out = dict(features)
    for name in ATTRIBUTION_GROUPS["fresh_news"]:
        out[name] = 0.0
    out["news_recent_missing"] = 1.0
    return out


def remove_persistent_news(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    out = dict(features)
    for name in ATTRIBUTION_GROUPS["persistent_news"]:
        out[name] = 0.0
    out["news_decay_missing"] = 1.0
    out["news_missing"] = 1.0
    return out


def remove_all_news(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    return remove_persistent_news(remove_fresh_news(features, model), model)


def flatten_market_trend(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    """Remove directional trend information while keeping the price level
    observed (the market still exists; it just is not moving)."""
    out = dict(features)
    for name in ("mkt_return_short", "mkt_return_long", "mkt_volatility"):
        out[name] = 0.0
    return out


def neutralise_position(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    out = dict(features)
    for name in FEATURE_GROUPS["position"]:
        out[name] = 0.0
    # a flat book is a COMPLETE history of nothing, not missing data
    out["pos_history_incomplete"] = 0.0
    return out


def replace_liquidity_with_training_reference(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    out = dict(features)
    for name in ATTRIBUTION_GROUPS["liquidity"]:
        if name.endswith("_missing"):
            continue
        out[name] = _training_mean(model, name)
    return out


def remove_actor_history(
    features: dict[str, float], model: LogisticModel
) -> dict[str, float]:
    out = dict(features)
    out["act_recent_trade_count"] = 0.0
    out["act_recent_gross_volume"] = 0.0
    out["act_recent_net_prop_change"] = 0.0
    out["act_time_since_last_trade"] = 86400.0
    out["act_time_since_last_trade_missing"] = 1.0
    out["act_category_trade_count"] = 0.0
    out["base_actor_positive_rate"] = 0.5
    out["base_actor_positive_rate_missing"] = 1.0
    return out


INTERVENTIONS: dict[str, Intervention] = {
    "remove_fresh_news": remove_fresh_news,
    "remove_persistent_news": remove_persistent_news,
    "remove_all_news": remove_all_news,
    "flatten_market_trend": flatten_market_trend,
    "neutralise_position": neutralise_position,
    "replace_liquidity_with_training_reference": (
        replace_liquidity_with_training_reference
    ),
    "remove_actor_history": remove_actor_history,
}

COUNTERFACTUAL_NAMES = tuple(INTERVENTIONS)

# materiality threshold for "removing X must materially reduce the
# observed-action probability" — recorded in the manifest
DEFAULT_MATERIALITY_DELTA = 0.02


@dataclass
class CounterfactualResult:
    deltas: dict[str, float]
    baseline_log_prob: float


def run_counterfactuals(
    model: LogisticModel,
    features: dict[str, float],
    label: float,
) -> CounterfactualResult:
    """Score all interventions for one decision under a fixed model."""
    x = np.asarray(
        [features[n] for n in ATTRIBUTION_FEATURES], dtype=float
    ).reshape(1, -1)
    p = float(model.predict_proba(x)[0])
    baseline = _log_prob_observed(p, label)
    deltas: dict[str, float] = {}
    for name, intervention in INTERVENTIONS.items():
        intervened = intervention(features, model)
        x_cf = np.asarray(
            [intervened[n] for n in ATTRIBUTION_FEATURES], dtype=float
        ).reshape(1, -1)
        p_cf = float(model.predict_proba(x_cf)[0])
        deltas[name] = baseline - _log_prob_observed(p_cf, label)
    return CounterfactualResult(deltas=deltas, baseline_log_prob=baseline)


def required_counterfactuals_pass(
    template_name: str,
    result: CounterfactualResult,
    *,
    materiality_delta: float = DEFAULT_MATERIALITY_DELTA,
) -> tuple[bool, list[str]]:
    """Check the template's declared required counterfactuals: each must
    MATERIALLY reduce the observed-action probability (delta above the
    threshold).  A failed required counterfactual disqualifies the
    template as primary."""
    from polymarket.analysis.reasoning_templates import TEMPLATES

    template = TEMPLATES[template_name]
    failures: list[str] = []
    for name in template.required_counterfactuals:
        if result.deltas.get(name, float("-inf")) < materiality_delta:
            failures.append(name)
    return not failures, failures


def counterfactual_report(result: CounterfactualResult) -> dict[str, Any]:
    return {
        "kind": "fixed_model_context_intervention",
        "note": (
            "distinct from refit group ablation; interventions preserve "
            "missingness semantics"
        ),
        "baseline_log_prob": result.baseline_log_prob,
        "deltas": result.deltas,
        "materiality_delta": DEFAULT_MATERIALITY_DELTA,
    }
