"""Latent reasoning scaffold — Track B of the real reasoning dataset.

Instead of forcing every decision into a named template, learn a small
set of LATENT behavioral primitives that must EARN their existence by
predicting real decisions out of sample.  This is the first, honest
rung of q(R|D,C):

    encoder   z = W' x            (rank-K bottleneck over context C)
    decoder   P(direction=+|C,R) = sigma(z'v + b)

i.e. a rank-constrained logistic decision model: the K latent
dimensions are exactly the directions of context space that carry
decision-relevant structure.  Deliberately NOT named: dimensions are
``latent_0..latent_{K-1}``; candidate interpretations (belief-update
direction/magnitude, news horizon, trend sensitivity, inventory
pressure, liquidity sensitivity, urgency, confidence) may be attached
only after clusters prove stable across seeds and folds — the
open-set rule from the acceptance gates.

Evaluation is the whole point:

* held-out ACTOR split and held-out TIME split (never random rows);
* the latent model must beat the C-only full-rank baseline's held-out
  log loss to claim anything (gate 1: predictive value);
* cluster stability across seeds is measured (centroid matching), not
  asserted.

The synthetic-trained template classifier remains the diagnostic
benchmark; nothing here retires it.  The full variational
q_phi(R|D,C) / p_theta(D|C,R,actor,market) with KL-regularized
D-informed posteriors is the successor once real gold labels (Track A)
and multi-week data exist — this scaffold defines the interfaces and
the evaluation harness it must beat.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np

from polymarket.analysis.features import ALL_FEATURES

LATENT_DIM_DEFAULT = 8
CANDIDATE_PRIMITIVES = (
    "belief_update_direction", "belief_update_magnitude",
    "news_horizon", "trend_sensitivity", "inventory_pressure",
    "liquidity_sensitivity", "urgency", "confidence",
)


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def assemble_decision_matrix(
    conn: sqlite3.Connection,
    *,
    end_time: float,
    mode_run_id: str | None = None,
) -> dict:
    """Features + direction labels + split keys for every clean
    directional decision, through the strict context path."""
    from polymarket.analysis.context import build_context
    from polymarket.analysis.decisions import build_decision_episodes
    from polymarket.analysis.features import compute_features
    from polymarket.analysis.reader import SQLiteNormalizedReader

    reader = SQLiteNormalizedReader(conn)
    episodes = [
        e for e in build_decision_episodes(reader, end_time=end_time)
        if e.direction in ("positive", "negative")
    ]
    rows, y, actors, times, decision_ids = [], [], [], [], []
    for episode in episodes:
        context = build_context(
            reader, episode, mode_run_id=mode_run_id,
            relevance_availability="online_scored",
        )
        features = compute_features(context, episode)
        rows.append([features[name] for name in ALL_FEATURES])
        y.append(1.0 if episode.direction == "positive" else 0.0)
        actors.append(episode.actor_id)
        times.append(episode.anchor_time)
        decision_ids.append(episode.decision_id)
    return {
        "X": np.asarray(rows, dtype=float),
        "y": np.asarray(y, dtype=float),
        "actors": np.asarray(actors),
        "times": np.asarray(times, dtype=float),
        "decision_ids": decision_ids,
        "feature_names": list(ALL_FEATURES),
    }


def _standardize_train_only(
    X: np.ndarray, train_mask: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    """Training-only z-scores; ``*_missing`` indicators stay 0/1."""
    out = X.copy()
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std == 0] = 1.0
    for j, name in enumerate(feature_names):
        if name.endswith("_missing") or name.endswith("_incomplete"):
            continue
        out[:, j] = (X[:, j] - mean[j]) / std[j]
    return np.clip(out, -10, 10)


def heldout_splits(
    actors: np.ndarray, times: np.ndarray, *, seed: int = 7,
    holdout_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Never random rows: held-out ACTORS (transfer across people) and
    held-out final TIME slice (transfer across periods)."""
    rng = np.random.default_rng(seed)
    unique_actors = np.array(sorted(set(actors.tolist())))
    rng.shuffle(unique_actors)
    k = max(1, int(len(unique_actors) * holdout_fraction))
    held_actors = set(unique_actors[:k].tolist())
    actor_test = np.array([a in held_actors for a in actors])
    cutoff = np.quantile(times, 1.0 - holdout_fraction)
    time_test = times > cutoff
    return {"actor": actor_test, "time": time_test}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
    steps: int = 400, lr: float = 0.1,
) -> np.ndarray:
    """Full-rank baseline: plain L2 logistic regression (the C-only
    model the latent bottleneck must beat out of sample)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * grad
    return w


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)


class LatentReasoningModel:
    """Rank-K bottleneck decision model, trained by deterministic
    gradient descent.  ``encode`` yields the latent vector per
    decision; latent dims are unnamed by construction."""

    def __init__(self, latent_dim: int = LATENT_DIM_DEFAULT,
                 *, l2: float = 1.0, steps: int = 800,
                 lr: float = 0.05, seed: int = 13):
        self.latent_dim = latent_dim
        self.l2 = l2
        self.steps = steps
        self.lr = lr
        self.seed = seed
        self.W: np.ndarray | None = None   # (d, K) encoder
        self.v: np.ndarray | None = None   # (K,) decoder
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LatentReasoningModel":
        rng = np.random.default_rng(self.seed)
        d, K = X.shape[1], self.latent_dim
        W = rng.normal(scale=0.05, size=(d, K))
        v = rng.normal(scale=0.05, size=K)
        b = 0.0
        n = len(y)
        for _ in range(self.steps):
            Z = X @ W
            p = _sigmoid(Z @ v + b)
            err = (p - y) / n
            grad_v = Z.T @ err + self.l2 * v / n
            grad_W = np.outer(X.T @ err, v) + self.l2 * W / n
            grad_b = float(err.sum())
            v -= self.lr * grad_v
            W -= self.lr * grad_W
            b -= self.lr * grad_b
        self.W, self.v, self.b = W, v, b
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.encode(X) @ self.v + self.b)


# ---------------------------------------------------------------------------
# clustering + stability
# ---------------------------------------------------------------------------
def _kmeans_once(Z, k, rng, iterations):
    centroids = Z[rng.choice(len(Z), size=k, replace=False)].copy()
    labels = np.zeros(len(Z), dtype=int)
    for _ in range(iterations):
        distances = ((Z[:, None, :] - centroids[None, :, :]) ** 2
                     ).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for j in range(k):
            members = Z[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    inertia = float(((Z - centroids[labels]) ** 2).sum())
    return labels, centroids, inertia


def kmeans(Z: np.ndarray, k: int, *, seed: int = 3,
           iterations: int = 50, restarts: int = 8,
           ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic multi-restart k-means (lowest inertia wins) so a
    single unlucky initialization cannot masquerade as instability."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        labels, centroids, inertia = _kmeans_once(
            Z, k, rng, iterations
        )
        if best is None or inertia < best[2]:
            best = (labels, centroids, inertia)
    return best[0], best[1]


def cluster_stability(
    Z_a: np.ndarray, Z_b: np.ndarray, k: int,
) -> float:
    """Fraction of points whose cluster assignment is preserved under
    the best cluster matching between two runs — the 'attach names
    only after clusters prove stable' measurement.

    Matching is by PARTITION OVERLAP (contingency counts), never by
    centroid geometry: the rank-K latent space is identified only up
    to sign/rotation, so centroids from different seeds live in
    different coordinate frames and geometric matching produces
    spurious zero stability on perfectly reproducible partitions."""
    labels_a, _ = kmeans(Z_a, k, seed=3)
    labels_b, _ = kmeans(Z_b, k, seed=17)
    contingency = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        contingency[a, b] += 1
    remaining = list(range(k))
    matched = 0
    # greedy maximum-overlap assignment (k is small)
    for i in np.argsort(-contingency.max(axis=1)):
        best = max(
            remaining, key=lambda j: contingency[i, j]
        )
        matched += contingency[i, best]
        remaining.remove(best)
    return float(matched / len(labels_a))


# ---------------------------------------------------------------------------
# the evaluation harness (acceptance gate 1)
# ---------------------------------------------------------------------------
def evaluate_latent_reasoning(
    data: dict,
    *,
    latent_dim: int = LATENT_DIM_DEFAULT,
    seed: int = 13,
    n_clusters: int = 4,
) -> dict:
    """Held-out actor and time evaluation of the latent model against
    the C-only full-rank baseline, plus cluster stability across
    seeds.  Refuses (returns refusal notes) when the sample is too
    small for the split to mean anything."""
    X, y = data["X"], data["y"]
    if len(y) < 60 or len(set(data["actors"].tolist())) < 8:
        return {
            "status": "refused",
            "note": (
                f"{len(y)} decisions across "
                f"{len(set(data['actors'].tolist()))} actors: too few "
                "for held-out actor/time evaluation; collect more "
                "real decisions"
            ),
        }
    splits = heldout_splits(data["actors"], data["times"], seed=seed)
    report: dict = {"status": "evaluated", "latent_dim": latent_dim,
                    "n_decisions": int(len(y)), "splits": {}}
    for split_name, test_mask in splits.items():
        train_mask = ~test_mask
        if test_mask.sum() < 15 or train_mask.sum() < 30:
            report["splits"][split_name] = {
                "status": "refused",
                "note": "split too small",
            }
            continue
        Xs = _standardize_train_only(
            X, train_mask, data["feature_names"]
        )
        baseline = fit_logistic(Xs[train_mask], y[train_mask])
        p_base = predict_logistic(Xs[test_mask], baseline)
        latent = LatentReasoningModel(
            latent_dim=latent_dim, seed=seed
        ).fit(Xs[train_mask], y[train_mask])
        p_latent = latent.predict(Xs[test_mask])
        base_rate = y[train_mask].mean()
        p_null = np.full(test_mask.sum(), base_rate)
        stability = cluster_stability(
            LatentReasoningModel(latent_dim=latent_dim, seed=seed)
            .fit(Xs[train_mask], y[train_mask]).encode(Xs),
            LatentReasoningModel(latent_dim=latent_dim, seed=seed + 100)
            .fit(Xs[train_mask], y[train_mask]).encode(Xs),
            n_clusters,
        )
        loss_null = _log_loss(y[test_mask], p_null)
        loss_base = _log_loss(y[test_mask], p_base)
        loss_latent = _log_loss(y[test_mask], p_latent)
        report["splits"][split_name] = {
            "status": "evaluated",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "log_loss_null": loss_null,
            "log_loss_c_only": loss_base,
            "log_loss_latent": loss_latent,
            "latent_beats_c_only": bool(loss_latent < loss_base),
            "latent_beats_null": bool(loss_latent < loss_null),
            "cluster_stability": stability,
        }
    evaluated = [
        block for block in report["splits"].values()
        if block.get("status") == "evaluated"
    ]
    # the gate requires BOTH: beating the C-only baseline AND the
    # base-rate null on every evaluated split — beating a baseline
    # that itself loses to the null is regularization, not extracted
    # decision structure
    report["gate_1_predictive_value"] = bool(evaluated) and all(
        block["latent_beats_c_only"] and block["latent_beats_null"]
        for block in evaluated
    )
    if evaluated and not report["gate_1_predictive_value"] and all(
        block["latent_beats_c_only"] for block in evaluated
    ):
        report["gate_1_note"] = (
            "latent beats the C-only baseline on every split but "
            "loses to the base-rate null on at least one: a "
            "regularization advantage on thin data, NOT evidence of "
            "extracted decision structure on that split"
        )
    report["naming_policy"] = (
        "latent dimensions are UNNAMED (latent_0..latent_"
        f"{latent_dim - 1}); candidate primitives "
        f"{list(CANDIDATE_PRIMITIVES)} may be attached only after "
        "cluster stability holds across seeds and folds"
    )
    return report


def template_agreement_diagnostic(
    latent_labels: np.ndarray,
    template_labels: list[str | None],
) -> dict:
    """How the unnamed clusters line up against the synthetic-trained
    template classifier — a DIAGNOSTIC, not supervision."""
    table: dict[int, dict] = {}
    for cluster, template in zip(latent_labels, template_labels):
        bucket = table.setdefault(int(cluster), {})
        key = template or "NONE"
        bucket[key] = bucket.get(key, 0) + 1
    purity = []
    for bucket in table.values():
        total = sum(bucket.values())
        purity.append(max(bucket.values()) / total)
    return {
        "cross_table": table,
        "mean_cluster_purity": (
            sum(purity) / len(purity) if purity else None
        ),
        "note": "diagnostic only; templates are not supervision here",
    }


def entropy_of_labels(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p))
                 ) if len(p) > 1 else 0.0
