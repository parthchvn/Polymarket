"""Reproducible SHA-256 identities for feature and reasoning configuration.

A change to any decay horizon, feature group, attribution channel,
acceptance threshold, template definition, counterfactual definition or
model hyperparameter changes the hash.  PARSER_VERSION is never reused
as a feature version.
"""

from __future__ import annotations

import hashlib
from typing import Any

from polymarket.collection.canonical import canonical_json

SYNTHETIC_GENERATOR_VERSION = "reasoning-worlds-1.0.0"
REASONING_METHOD_BASE = "reasoning-reconstruction-1.0.0"


def _hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def feature_version_hash(extra: dict[str, Any] | None = None) -> str:
    from polymarket.analysis.features import feature_manifest
    from polymarket.analysis.reasoning import ATTRIBUTION_GROUPS

    payload = {
        "feature_manifest": feature_manifest(),
        "attribution_channels": ATTRIBUTION_GROUPS,
        "preprocessing": {
            "standardization": "training_only_zscore",
            "binary_suffixes_zero_preserving": ["_missing", "_incomplete"],
            "market_sources": {
                "prices": "market_series_policy:book_preferred",
                "activity": "canonical_executions",
                "books": "order_book_snapshots",
            },
        },
    }
    if extra:
        payload["extra"] = extra
    return f"feat-{_hash(payload)}"


def reasoning_method_version_hash(
    attribution_config: Any,
    posterior_config: Any,
    counterfactual_names: tuple[str, ...],
    model_hyperparameters: dict[str, Any],
) -> str:
    import dataclasses

    payload = {
        "base": REASONING_METHOD_BASE,
        "attribution_config": dataclasses.asdict(attribution_config),
        "posterior_config": dataclasses.asdict(posterior_config),
        "counterfactuals": list(counterfactual_names),
        "model_hyperparameters": model_hyperparameters,
    }
    return f"reason-{_hash(payload)}"


def template_ontology_version_hash() -> str:
    import dataclasses

    from polymarket.analysis.reasoning_templates import TEMPLATES

    payload = {
        name: dataclasses.asdict(template)
        for name, template in TEMPLATES.items()
    }
    return f"ontology-{_hash(payload)}"


def version_block(
    attribution_config: Any,
    posterior_config: Any,
    counterfactual_names: tuple[str, ...],
    model_hyperparameters: dict[str, Any],
) -> dict[str, str]:
    return {
        "feature_version": feature_version_hash(),
        "reasoning_method_version": reasoning_method_version_hash(
            attribution_config, posterior_config, counterfactual_names,
            model_hyperparameters,
        ),
        "template_ontology_version": template_ontology_version_hash(),
        "synthetic_generator_version": SYNTHETIC_GENERATOR_VERSION,
    }
