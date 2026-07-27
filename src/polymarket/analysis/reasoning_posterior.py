"""Calibrated posterior over reasoning templates.

P(template | D, C, layer1_attribution) via deterministic L2-regularized
multinomial logistic regression with training-only standardization and
scalar temperature calibration fitted on held-out synthetic VALIDATION
worlds (test worlds are never used for tuning).

Inputs are strictly pre-decision or decision-time: Layer 1 ablation
deltas and logit contributions, attribution stability, decision
direction, pre-decision position and exposure change, fresh/persistent
news evidence and alignment, pre-decision market trend and liquidity,
actor-history features and missingness indicators.  Post-decision price
movement, market outcomes, future wallet trades, later news and
resolution results are never inputs.

The posterior is behavioural: it names the hypothesis the decision is
most consistent with under the fitted model, never the actor's private
mental state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from polymarket.analysis.decisions import DecisionEpisode
from polymarket.analysis.reasoning import ATTRIBUTION_GROUPS, DriverAttribution
from polymarket.analysis.reasoning_templates import TEMPLATE_NAMES

_DUST = 1.0


@dataclass(frozen=True)
class PosteriorConfig:
    """Primary-template gating.  Recorded in every manifest."""

    min_top_probability: float = 0.40
    min_top_margin: float = 0.10
    max_entropy: float = 1.8
    l2: float = 0.5

    def ambiguity_threshold(self) -> float:
        return self.max_entropy


DEFAULT_POSTERIOR_CONFIG = PosteriorConfig()

POSTERIOR_FEATURES = [
    "dir_fresh_align", "fresh_mag", "dir_persist_align",
    "dir_persist_excess", "recent_news_present", "any_news_present",
    "dir_trend_align_short", "dir_trend_align_long", "trend_mag",
    "exposure_reduce", "exposure_build", "position_mag",
    "liquidity_tight", "book_present", "book_depth",
    "dir_prior_align", "prior_history_n",
    "aged_news_only", "dir_aged_news_align",
    "ev_present", "ev_latest_age", "ev_latest_align",
    "ev_latest_fresh", "ev_latest_aged",
    "ev_fresh_align", "ev_aged_align",
    *[f"l1_delta_{c}" for c in ATTRIBUTION_GROUPS],
    "l1_available", "l1_stability",
]


def posterior_features(
    features: dict[str, float],
    episode: DecisionEpisode,
    layer1: DriverAttribution | None = None,
    evidence: list[dict] | None = None,
) -> dict[str, float]:
    """Deterministic template-indicator features from strict context."""
    direction = (
        1.0 if episode.direction == "positive"
        else -1.0 if episode.direction == "negative" else 0.0
    )
    net = features["pos_net_proposition"]
    net_sign = 1.0 if net > _DUST else -1.0 if net < -_DUST else 0.0
    spread_present = 1.0 - features["mkt_spread_missing"]
    tight = 0.0
    if spread_present:
        tight = max(0.0, (0.05 - features["mkt_spread"]) / 0.05)
    x = {
        "dir_fresh_align": direction * features["news_decay_signed_6h"],
        "fresh_mag": abs(features["news_decay_signed_6h"]),
        "dir_persist_align": direction * features["news_decay_signed_72h"],
        "dir_persist_excess": direction * (
            features["news_decay_signed_72h"]
            - features["news_decay_signed_6h"]
        ),
        "recent_news_present": 1.0 - features["news_recent_missing"],
        "any_news_present": 1.0 - features["news_decay_missing"],
        "dir_trend_align_short": direction * features["mkt_return_short"] * 50.0,
        "dir_trend_align_long": direction * features["mkt_return_long"] * 20.0,
        "trend_mag": abs(features["mkt_return_long"]) * 20.0,
        "exposure_reduce": 1.0 if net_sign and direction == -net_sign else 0.0,
        "exposure_build": 1.0 if net_sign and direction == net_sign else 0.0,
        "position_mag": min(abs(net), 20.0) / 20.0,
        "liquidity_tight": tight,
        "book_present": spread_present,
        "book_depth": min(features["mkt_depth"], 500.0) / 500.0,
        "dir_prior_align": direction
        * (2.0 * features["base_actor_positive_rate"] - 1.0)
        * (1.0 - features["base_actor_positive_rate_missing"]),
        "prior_history_n": min(features["act_category_trade_count"], 10.0) / 10.0,
        "l1_available": 0.0,
        "l1_stability": 0.0,
    }
    x["aged_news_only"] = x["any_news_present"] * (
        1.0 - x["recent_news_present"]
    )
    x["dir_aged_news_align"] = x["dir_persist_align"] * x["aged_news_only"]
    # per-family evidence from the strict context (youngest family)
    x["ev_present"] = 0.0
    x["ev_latest_age"] = 1.0
    x["ev_latest_align"] = 0.0
    x["ev_latest_fresh"] = 0.0
    x["ev_latest_aged"] = 0.0
    x["ev_fresh_align"] = 0.0
    x["ev_aged_align"] = 0.0
    if evidence:
        youngest = min(evidence, key=lambda e: e["age_hours"])
        age_h = float(youngest["age_hours"])
        x["ev_present"] = 1.0
        x["ev_latest_age"] = min(age_h, 200.0) / 200.0
        x["ev_latest_align"] = direction * float(youngest["direction"])
        x["ev_latest_fresh"] = 1.0 if age_h < 6.0 else 0.0
        x["ev_latest_aged"] = 1.0 if 24.0 < age_h < 28 * 24.0 else 0.0
    x["ev_fresh_align"] = x["ev_latest_fresh"] * x["ev_latest_align"]
    x["ev_aged_align"] = x["ev_latest_aged"] * x["ev_latest_align"]
    for channel in ATTRIBUTION_GROUPS:
        x[f"l1_delta_{channel}"] = 0.0
    if layer1 is not None and layer1.group_attributions:
        x["l1_available"] = 1.0
        if np.isfinite(layer1.attribution_stability):
            x["l1_stability"] = layer1.attribution_stability
        for channel, delta in layer1.group_attributions.items():
            x[f"l1_delta_{channel}"] = float(np.clip(delta, -3.0, 3.0))
    return x


# ---------------------------------------------------------------------------
@dataclass
class TemplatePosterior:
    probabilities: dict[str, float]
    primary_template: str | None
    entropy: float
    top_margin: float
    calibration_version: str


@dataclass
class TemplateModel:
    classes: list[str]
    feature_names: list[str]
    weights: np.ndarray | None = None      # (n_classes, n_features + 1)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    temperature: float = 1.0
    calibration_version: str = "uncalibrated"
    fit_success: bool = False

    # -- helpers -----------------------------------------------------------
    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.std
        return np.column_stack([np.ones(len(Xs)), Xs])

    def _logits(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.weights.T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._logits(X) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / expz.sum(axis=1, keepdims=True)

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, labels: list[str], l2: float = 0.5) -> "TemplateModel":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        design = self._design(X)
        n, d = design.shape
        k = len(self.classes)
        class_index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([class_index[label] for label in labels])
        Y = np.eye(k)[y]
        counts = np.bincount(y, minlength=k).astype(float)
        row_weight = n / (k * counts[y])  # class-balanced, deterministic

        def unpack(w):
            return w.reshape(k, d)

        def loss(w):
            W = unpack(w)
            z = design @ W.T
            z -= z.max(axis=1, keepdims=True)
            logsum = np.log(np.exp(z).sum(axis=1))
            nll = float(np.sum(row_weight * (logsum - z[np.arange(n), y])))
            reg = 0.5 * l2 * float(np.sum(W[:, 1:] ** 2))
            return nll + reg

        def grad(w):
            W = unpack(w)
            z = design @ W.T
            z -= z.max(axis=1, keepdims=True)
            expz = np.exp(z)
            p = expz / expz.sum(axis=1, keepdims=True)
            g = ((p - Y) * row_weight[:, None]).T @ design
            g[:, 1:] += l2 * W[:, 1:]
            return g.ravel()

        res = minimize(
            loss, np.zeros(k * d), jac=grad, method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.weights = unpack(res.x)
        self.fit_success = bool(res.success) and bool(
            np.all(np.isfinite(res.x))
        )
        return self

    def calibrate(self, X_val: np.ndarray, labels_val: list[str]) -> "TemplateModel":
        """Scalar temperature minimizing validation NLL.  Must be fitted
        on VALIDATION worlds only — never on test worlds."""
        class_index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([class_index[label] for label in labels_val])
        logits = self._logits(X_val)

        def nll(temperature: float) -> float:
            z = logits / temperature
            z = z - z.max(axis=1, keepdims=True)
            logsum = np.log(np.exp(z).sum(axis=1))
            return float(np.sum(logsum - z[np.arange(len(y)), y]))

        res = minimize_scalar(nll, bounds=(0.25, 8.0), method="bounded")
        self.temperature = float(res.x)
        self.calibration_version = f"temperature-{self.temperature:.4f}"
        return self


def train_template_model(
    feature_dicts: list[dict[str, float]],
    labels: list[str],
    *,
    classes: tuple[str, ...] = TEMPLATE_NAMES,
    l2: float = DEFAULT_POSTERIOR_CONFIG.l2,
) -> TemplateModel:
    present = [c for c in classes if c in set(labels)]
    X = np.asarray(
        [[row[n] for n in POSTERIOR_FEATURES] for row in feature_dicts],
        dtype=float,
    )
    return TemplateModel(
        classes=present, feature_names=list(POSTERIOR_FEATURES)
    ).fit(X, labels, l2=l2)


def infer_posterior(
    model: TemplateModel,
    feature_dict: dict[str, float],
    *,
    layer1_status: str = "accepted",
    coverage_complete: bool = True,
    config: PosteriorConfig = DEFAULT_POSTERIOR_CONFIG,
) -> tuple[TemplatePosterior, str]:
    """Return (posterior, status).  The posterior is always kept; the
    primary template is withheld (None) when gating fails."""
    x = np.asarray(
        [[feature_dict[n] for n in POSTERIOR_FEATURES]], dtype=float
    )
    probs = model.predict_proba(x)[0]
    probabilities = {
        c: float(p) for c, p in zip(model.classes, probs)
    }
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    top_name, top_p = ordered[0]
    second_p = ordered[1][1] if len(ordered) > 1 else 0.0
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    margin = top_p - second_p

    status = "accepted"
    primary: str | None = top_name
    if not coverage_complete:
        status, primary = "insufficient_context", None
    elif layer1_status not in ("accepted",):
        status, primary = "ambiguous", None
    elif top_name == "MIXED_OR_UNRESOLVED":
        # MIXED means "no single hypothesis": it is a resolution status,
        # never an accepted primary template
        status, primary = "ambiguous", None
    elif (
        top_p < config.min_top_probability
        or margin < config.min_top_margin
        or entropy > config.ambiguity_threshold()
    ):
        status, primary = "ambiguous", None
    return (
        TemplatePosterior(
            probabilities=probabilities,
            primary_template=primary,
            entropy=entropy,
            top_margin=margin,
            calibration_version=model.calibration_version,
        ),
        status,
    )
