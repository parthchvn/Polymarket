"""Nested model suite and chronological evaluation.

Models (nested):
* M0: intercept + base rates
* M1: + actor and market metadata
* M2: + market state and position
* M3: + news features

Headline evaluation is blocked expanding-window with an embargo — never a
random split.  All preprocessing (standardization) is fitted on training
data only.  Zero-preserving coding is kept for sparse news variables:
missingness indicators are binary and are not centered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from polymarket.analysis.features import FEATURE_GROUPS

MODEL_GROUPS = {
    "M0": ["base"],
    "M1": ["base", "actor"],
    "M2": ["base", "actor", "market", "position"],
    "M3": ["base", "actor", "market", "position", "news"],
}

_BINARY_SUFFIXES = ("_missing", "_incomplete")


def model_feature_names(model: str) -> list[str]:
    return [n for g in MODEL_GROUPS[model] for n in FEATURE_GROUPS[g]]


# ---------------------------------------------------------------------------
@dataclass
class Standardizer:
    """Training-only standardization.  Binary indicator columns keep their
    zero-preserving coding (not centered/scaled)."""

    names: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        for i, name in enumerate(self.names):
            if name.endswith(_BINARY_SUFFIXES):
                self.mean[i] = 0.0
                self.std[i] = 1.0
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer not fitted")
        return (X - self.mean) / self.std


@dataclass
class LogisticModel:
    """L2-regularized logistic regression via scipy L-BFGS (deterministic)."""

    feature_names: list[str]
    l2: float = 1.0
    weights: np.ndarray | None = None
    standardizer: Standardizer | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.standardizer = Standardizer(self.feature_names).fit(X)
        Xs = np.column_stack(
            [np.ones(len(X)), self.standardizer.transform(X)]
        )

        def loss(w: np.ndarray) -> float:
            z = Xs @ w
            # log(1+exp(-y*z)) stable
            m = -y * z
            nll = np.sum(np.logaddexp(0.0, m))
            reg = 0.5 * self.l2 * np.sum(w[1:] ** 2)
            return float(nll + reg)

        def grad(w: np.ndarray) -> np.ndarray:
            z = Xs @ w
            p = 1.0 / (1.0 + np.exp(-z))
            g = Xs.T @ (p - (y + 1) / 2)
            g[1:] += self.l2 * w[1:]
            return g

        w0 = np.zeros(Xs.shape[1])
        res = minimize(loss, w0, jac=grad, method="L-BFGS-B")
        self.weights = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.standardizer is None:
            raise RuntimeError("model not fitted")
        Xs = np.column_stack(
            [np.ones(len(X)), self.standardizer.transform(X)]
        )
        return 1.0 / (1.0 + np.exp(-(Xs @ self.weights)))


# ---------------------------------------------------------------------------
def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y01 = (y + 1) / 2
    return float(np.mean((p - y01) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y01 = (y + 1) / 2
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(((p >= 0.5).astype(int) * 2 - 1) == y))


def calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> list[dict]:
    y01 = (y + 1) / 2
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() == 0:
            bins.append(
                {"lo": float(lo), "hi": float(hi), "count": 0,
                 "mean_pred": None, "mean_observed": None}
            )
        else:
            bins.append(
                {"lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
                 "mean_pred": float(p[mask].mean()),
                 "mean_observed": float(y01[mask].mean())}
            )
    return bins


def expected_calibration_error(y: np.ndarray, p: np.ndarray) -> float:
    total = len(y)
    ece = 0.0
    for b in calibration_bins(y, p):
        if b["count"]:
            ece += b["count"] / total * abs(b["mean_pred"] - b["mean_observed"])
    return float(ece)


# ---------------------------------------------------------------------------
@dataclass
class Fold:
    fold_index: int
    train_end: float
    eval_start: float
    eval_end: float
    train_indices: np.ndarray
    eval_indices: np.ndarray


def chronological_folds(
    times: np.ndarray,
    *,
    n_folds: int = 3,
    embargo_seconds: float = 0.0,
    min_train_fraction: float = 0.3,
) -> list[Fold]:
    """Blocked expanding-window folds with an embargo between training and
    evaluation.  Never a random split."""
    order = np.argsort(times, kind="stable")
    n = len(times)
    if n < 4:
        raise ValueError("not enough observations for chronological folds")
    first_eval = max(2, int(n * min_train_fraction))
    eval_pool = order[first_eval:]
    blocks = np.array_split(eval_pool, n_folds)
    folds: list[Fold] = []
    for i, block in enumerate(blocks):
        if len(block) == 0:
            continue
        eval_start_time = times[block[0]]
        train_cutoff = eval_start_time - embargo_seconds
        train_idx = order[times[order] < train_cutoff]
        if len(train_idx) < 2:
            continue
        folds.append(
            Fold(
                fold_index=i,
                train_end=float(train_cutoff),
                eval_start=float(eval_start_time),
                eval_end=float(times[block[-1]]),
                train_indices=train_idx,
                eval_indices=block,
            )
        )
    return folds


@dataclass
class EvaluationResult:
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_decision: list[dict] = field(default_factory=list)
    folds: list[dict] = field(default_factory=list)
    improvements: dict[str, float] = field(default_factory=dict)


def evaluate_nested_models(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    times: list[float],
    decision_ids: list[str],
    *,
    n_folds: int = 3,
    embargo_seconds: float = 0.0,
    l2: float = 1.0,
) -> EvaluationResult:
    y = np.asarray(labels, dtype=float)
    t = np.asarray(times, dtype=float)
    result = EvaluationResult()
    folds = chronological_folds(t, n_folds=n_folds, embargo_seconds=embargo_seconds)
    result.folds = [
        {"fold_index": f.fold_index, "train_end": f.train_end,
         "eval_start": f.eval_start, "eval_end": f.eval_end,
         "train_n": int(len(f.train_indices)),
         "eval_n": int(len(f.eval_indices))}
        for f in folds
    ]
    predictions: dict[str, dict[int, float]] = {m: {} for m in MODEL_GROUPS}
    for model_name in MODEL_GROUPS:
        names = model_feature_names(model_name)
        X = np.asarray(
            [[row[n] for n in names] for row in feature_rows], dtype=float
        )
        for fold in folds:
            model = LogisticModel(feature_names=names, l2=l2).fit(
                X[fold.train_indices], y[fold.train_indices]
            )
            p = model.predict_proba(X[fold.eval_indices])
            for idx, prob in zip(fold.eval_indices, p):
                predictions[model_name][int(idx)] = float(prob)

    eval_indices = sorted(predictions["M0"].keys())
    if not eval_indices:
        raise ValueError("no evaluation predictions produced")
    ye = y[eval_indices]
    for model_name in MODEL_GROUPS:
        pe = np.asarray([predictions[model_name][i] for i in eval_indices])
        result.metrics[model_name] = {
            "brier": brier_score(ye, pe),
            "log_loss": log_loss(ye, pe),
            "accuracy": accuracy(ye, pe),
            "ece": expected_calibration_error(ye, pe),
            "n": len(eval_indices),
        }
    result.improvements = {
        "m2_to_m3_log_loss": (
            result.metrics["M2"]["log_loss"] - result.metrics["M3"]["log_loss"]
        ),
        "m2_to_m3_brier": (
            result.metrics["M2"]["brier"] - result.metrics["M3"]["brier"]
        ),
    }
    for i in eval_indices:
        row = {
            "decision_id": decision_ids[i],
            "time": float(t[i]),
            "label": float(y[i]),
        }
        for model_name in MODEL_GROUPS:
            row[f"p_{model_name}"] = predictions[model_name][i]
        result.per_decision.append(row)
    return result
