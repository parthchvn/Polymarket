"""Bootstrap uncertainty for model-comparison metrics.

Provides actor-clustered, market-clustered and moving-block bootstraps
behind one interface.  Pooled predictive gain does NOT automatically
imply a homogeneous actor-level effect; intervals here quantify sampling
uncertainty of the pooled comparison only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    statistic: str
    method: str
    point: float
    lower: float
    upper: float
    n_bootstrap: int
    seed: int
    note: str = (
        "pooled predictive gain does not imply a homogeneous "
        "actor-level effect"
    )


def _percentile_ci(samples: list[float], point: float, **meta) -> BootstrapCI:
    arr = np.asarray([s for s in samples if np.isfinite(s)])
    if len(arr) == 0:
        return BootstrapCI(point=point, lower=float("nan"),
                           upper=float("nan"), **meta)
    return BootstrapCI(
        point=point,
        lower=float(np.percentile(arr, 2.5)),
        upper=float(np.percentile(arr, 97.5)),
        **meta,
    )


def _per_unit_loss_difference(per_decision: list[dict]) -> list[dict]:
    """Per-decision log-loss difference (M2 - M3)."""
    out = []
    eps = 1e-12
    for row in per_decision:
        y01 = (row["label"] + 1) / 2
        losses = {}
        for model in ("M2", "M3"):
            p = min(max(row[f"p_{model}"], eps), 1 - eps)
            losses[model] = -(y01 * np.log(p) + (1 - y01) * np.log(1 - p))
        out.append({**row, "loss_diff": float(losses["M2"] - losses["M3"])})
    return out


def cluster_bootstrap(
    per_decision: list[dict],
    cluster_key: str,
    clusters: dict[str, str],
    *,
    statistic: str = "m2_to_m3_log_loss",
    n_bootstrap: int = 500,
    seed: int = 2024,
) -> BootstrapCI:
    """Resample whole clusters (actors or markets) with replacement."""
    rows = _per_unit_loss_difference(per_decision)
    by_cluster: dict[str, list[float]] = {}
    for row in rows:
        cluster = clusters.get(row["decision_id"], "?")
        by_cluster.setdefault(cluster, []).append(row["loss_diff"])
    point = float(np.mean([r["loss_diff"] for r in rows]))
    keys = sorted(by_cluster)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        chosen = [keys[rng.randrange(len(keys))] for _ in keys]
        values = [v for k in chosen for v in by_cluster[k]]
        if values:
            samples.append(float(np.mean(values)))
    return _percentile_ci(
        samples, point, statistic=statistic,
        method=f"{cluster_key}_clustered_bootstrap",
        n_bootstrap=n_bootstrap, seed=seed,
    )


def moving_block_bootstrap(
    per_decision: list[dict],
    *,
    block_size: int = 3,
    statistic: str = "m2_to_m3_log_loss",
    n_bootstrap: int = 500,
    seed: int = 2024,
) -> BootstrapCI:
    """Moving-block bootstrap over time-ordered decisions."""
    rows = sorted(_per_unit_loss_difference(per_decision), key=lambda r: r["time"])
    diffs = [r["loss_diff"] for r in rows]
    n = len(diffs)
    point = float(np.mean(diffs)) if diffs else float("nan")
    if n == 0:
        return _percentile_ci([], point, statistic=statistic,
                              method="moving_block_bootstrap",
                              n_bootstrap=n_bootstrap, seed=seed)
    block_size = min(block_size, n)
    starts = list(range(0, n - block_size + 1))
    rng = random.Random(seed)
    samples = []
    n_blocks = max(1, n // block_size)
    for _ in range(n_bootstrap):
        values: list[float] = []
        for _ in range(n_blocks):
            s = starts[rng.randrange(len(starts))]
            values.extend(diffs[s:s + block_size])
        samples.append(float(np.mean(values)))
    return _percentile_ci(
        samples, point, statistic=statistic,
        method="moving_block_bootstrap",
        n_bootstrap=n_bootstrap, seed=seed,
    )
