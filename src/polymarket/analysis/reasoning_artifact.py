"""Serializable reasoning-model artifact for reproducible inference.

The artifact carries everything needed to run template inference on real
data: classes, weights, feature names, training standardization,
calibration temperature, both configurations, version hashes and the
world seeds it was trained/calibrated on.  ``load_reasoning_model``
REFUSES inference when the artifact's feature hash differs from the
current feature manifest, so a stale model can never silently score
fresh features.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any

import numpy as np

from polymarket.analysis.reasoning import AttributionConfig
from polymarket.analysis.reasoning_posterior import (
    POSTERIOR_FEATURES,
    PosteriorConfig,
    TemplateModel,
)
from polymarket.analysis.versioning import feature_version_hash

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactVersionMismatch(RuntimeError):
    """The artifact was trained against a different feature manifest."""


def build_artifact_payload(
    model: TemplateModel,
    *,
    attribution_config: AttributionConfig,
    posterior_config: PosteriorConfig,
    versions: dict[str, str],
    train_world_seeds: tuple[int, ...],
    validation_world_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if model.weights is None or not model.fit_success:
        raise ValueError("refusing to serialize an unfitted or failed model")
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": time.time(),
        "classes": list(model.classes),
        "feature_names": list(model.feature_names),
        "weights": model.weights.tolist(),
        "mean": model.mean.tolist(),
        "std": model.std.tolist(),
        "temperature": model.temperature,
        "calibration_version": model.calibration_version,
        "posterior_config": dataclasses.asdict(posterior_config),
        "attribution_config": dataclasses.asdict(attribution_config),
        "versions": dict(versions),
        "train_world_seeds": list(train_world_seeds),
        "validation_world_seeds": list(validation_world_seeds),
    }


def save_reasoning_model(payload: dict[str, Any], path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_reasoning_model(
    path: str,
) -> tuple[TemplateModel, dict[str, Any]]:
    """Load and VERIFY an artifact.  Raises ArtifactVersionMismatch when
    the artifact's feature hash or posterior feature list differs from
    the current code's manifest."""
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactVersionMismatch(
            "unsupported artifact schema version: "
            f"{payload.get('artifact_schema_version')!r}"
        )
    current_hash = feature_version_hash()
    stored_hash = payload.get("versions", {}).get("feature_version")
    if stored_hash != current_hash:
        raise ArtifactVersionMismatch(
            f"artifact feature_version {stored_hash!r} does not match the "
            f"current feature manifest {current_hash!r}; retrain the "
            "reasoning model before running inference"
        )
    if payload["feature_names"] != list(POSTERIOR_FEATURES):
        raise ArtifactVersionMismatch(
            "artifact posterior feature names differ from the current "
            "POSTERIOR_FEATURES definition; retrain the reasoning model"
        )
    model = TemplateModel(
        classes=list(payload["classes"]),
        feature_names=list(payload["feature_names"]),
        weights=np.asarray(payload["weights"], dtype=float),
        mean=np.asarray(payload["mean"], dtype=float),
        std=np.asarray(payload["std"], dtype=float),
        temperature=float(payload["temperature"]),
        calibration_version=str(payload["calibration_version"]),
        fit_success=True,
    )
    return model, payload
