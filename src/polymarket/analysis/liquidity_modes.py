"""Two-state liquidity jump model — 'Towards Systematic Intraday News
Screening' (arXiv 2304.05115), adapted to prediction markets.

The paper clusters five-minute liquidity vectors x_t = (phi, V, sigma,
B) — spread in ticks, turnover, volatility, best-level book size —
into K=2 modes by minimizing

    J = sum_t || x_t - theta_{m_t} ||^2  +  lambda * sum_t 1{m_t != m_{t+1}}

(their eq. 3.1): a K-means fit with a temporal switch penalty, solved
by alternating centroid updates with dynamic-programming mode
assignment.  The lower-volatility mode is CALM, the other EVENT.

Adaptations for Polymarket, all versioned:

* **Stationarization.**  The paper removes intraday seasonality per
  stock using time-of-day location/IQR over ~750 reference days.
  Prediction markets trade continuously and our histories are far
  shorter, so reference cells are (condition, UTC hour): for each cell,
  the median and IQR of log(x + eps) over TRAINING bars only; each
  observation is transformed as (log(x+eps) - median) / IQR.  Cells
  with too few observations fall back to the per-condition global
  statistics, and degenerate IQRs fall back to a floor — both recorded.
* **Lambda selection** uses training data only: the smallest candidate
  lambda whose fitted training mode sequence reaches a minimum mean
  mode duration (persistence target), which mirrors the paper's
  observation that lambda trades off clustering fit against mode
  persistence.  A fixed lambda can be supplied instead.
* **Determinism.**  Centroids initialize from the volatility median
  split (lower-half mean vs upper-half mean), so identical inputs give
  identical fits — no random restarts.

Only ``coverage_complete`` bars participate: incomplete bars break DP
chains into separate segments rather than being interpolated.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass, field

from polymarket.collection.canonical import canonical_json

MODEL_VERSION = "liquidity-jump-1.1.0"
VARIABLES = ("spread_ticks", "turnover", "volatility", "best_book_size")
LOG_EPS = 1e-9
IQR_FLOOR = 1e-6
# Turnover decomposes into PRESENCE (the binary indicator dimension)
# and LEVEL GIVEN PRESENCE: reference stats for turnover are fitted on
# positive-turnover bars only, and zero-turnover bars contribute level
# z = 0.  Otherwise, in sparse markets where most five-minute bars
# have no execution, presence-vs-absence dominates every distance and
# a lone trade in calm conditions gets forced into event mode.
# Minimum meaningful dispersion per variable, in TRANSFORM units — a
# degenerate IQR falls back to these instead of a numerical epsilon:
MIN_SCALE = {
    "spread_ticks": 0.05,
    "turnover": 1.0,        # log1p units: one unit ~ factor e
    "volatility": 0.05,
    "best_book_size": 0.05,
}
WINSOR_Z = 10.0
TURNOVER_PRESENT_WEIGHT = 1.0  # binary 5th dimension


@dataclass(frozen=True)
class JumpModelConfig:
    bin_seconds: float = 300.0
    lambda_candidates: tuple[float, ...] = (
        0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0
    )
    fixed_lambda: float | None = None
    min_mean_duration_bins: float = 3.0   # persistence target (15 min)
    min_cell_observations: int = 20
    max_iterations: int = 50


@dataclass
class BarVector:
    condition_id: str
    bin_start: float
    values: tuple[float, float, float, float]


@dataclass
class FittedJumpModel:
    mode_run_id: str
    centroids: list[list[float]]
    calm_mode: int
    lambda_penalty: float
    lambda_selection: str
    reference_stats: dict
    train_bar_count: int
    config: JumpModelConfig
    assignments: dict[tuple[str, float], int] = field(default_factory=dict)
    assignments_online: dict[tuple[str, float], int] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# raw bar loading
# ---------------------------------------------------------------------------
def _load_raw_bars(
    conn: sqlite3.Connection, config: JumpModelConfig,
    end: float | None = None,
) -> list[dict]:
    sql = (
        "SELECT condition_id, bin_start, spread_ticks_mean, spread_mean, "
        "turnover_notional, realized_variance, best_book_size_mean, "
        "feature_version "
        "FROM liquidity_bars WHERE bin_seconds = ? AND coverage_complete = 1"
    )
    args: list = [config.bin_seconds]
    if end is not None:
        sql += " AND bin_start < ?"
        args.append(end)
    sql += " ORDER BY condition_id, bin_start"
    out = []
    for row in conn.execute(sql, args):
        spread = (
            row["spread_ticks_mean"]
            if row["spread_ticks_mean"] is not None else row["spread_mean"]
        )
        if (spread is None or row["realized_variance"] is None
                or row["best_book_size_mean"] is None):
            continue
        out.append({
            "condition_id": row["condition_id"],
            "bin_start": row["bin_start"],
            "spread_ticks": spread,
            "turnover": row["turnover_notional"] or 0.0,
            "volatility": math.sqrt(max(row["realized_variance"], 0.0)),
            "best_book_size": row["best_book_size_mean"],
            "feature_version": row["feature_version"],
        })
    return out


# ---------------------------------------------------------------------------
# stationarization (training-only reference cells)
# ---------------------------------------------------------------------------
def _utc_hour(bin_start: float) -> int:
    return int(bin_start // 3600) % 24


def _log_transform(variable: str, value: float) -> float:
    """Zero-aware transforms: turnover uses log1p (five-minute
    prediction-market bars legitimately have zero executions, and
    log(0+eps) would explode the standardized value when the IQR is
    tiny); the strictly-positive variables keep log(x+eps)."""
    if variable == "turnover":
        return math.log1p(max(value, 0.0))
    return math.log(value + LOG_EPS)


def fit_reference_stats(
    train_bars: list[dict], config: JumpModelConfig
) -> dict:
    """Median/IQR of transformed variables per (condition, UTC hour)
    cell, fitted on TRAINING bars only; fallback chain: cell ->
    per-condition global -> POOLED global (all conditions), so markets
    absent from training still standardize instead of vanishing."""
    cells: dict[str, dict[str, list[float]]] = {}
    for bar in train_bars:
        cell = f"{bar['condition_id']}|{_utc_hour(bar['bin_start'])}"
        bucket = cells.setdefault(cell, {v: [] for v in VARIABLES})
        for variable in VARIABLES:
            if variable == "turnover" and bar[variable] <= 0:
                continue  # level-given-presence: zeros excluded
            bucket[variable].append(
                _log_transform(variable, bar[variable])
            )
    globals_: dict[str, dict[str, list[float]]] = {}
    pooled: dict[str, list[float]] = {v: [] for v in VARIABLES}
    for bar in train_bars:
        bucket = globals_.setdefault(
            bar["condition_id"], {v: [] for v in VARIABLES}
        )
        for variable in VARIABLES:
            if variable == "turnover" and bar[variable] <= 0:
                continue  # level-given-presence: zeros excluded
            value = _log_transform(variable, bar[variable])
            bucket[variable].append(value)
            pooled[variable].append(value)

    def summarize(values: list[float], variable: str) -> dict:
        if not values:  # e.g. no traded bar in this cell
            return {"median": 0.0, "iqr": MIN_SCALE.get(
                variable, IQR_FLOOR), "n": 0, "iqr_floored": True}
        med = statistics.median(values)
        qs = statistics.quantiles(values, n=4) if len(values) >= 4 else None
        iqr = (qs[2] - qs[0]) if qs else 0.0
        floor = MIN_SCALE.get(variable, IQR_FLOOR)
        return {"median": med, "iqr": max(iqr, floor),
                "n": len(values), "iqr_floored": bool(iqr < floor)}

    stats = {
        "cells": {
            cell: {v: summarize(vals[v], v) for v in VARIABLES}
            for cell, vals in cells.items()
        },
        "globals": {
            condition: {v: summarize(vals[v], v) for v in VARIABLES}
            for condition, vals in globals_.items()
        },
        "pooled": {v: summarize(pooled[v], v) for v in VARIABLES},
        "min_cell_observations": config.min_cell_observations,
        "log_eps": LOG_EPS,
        "turnover_transform": "log1p_level_given_presence",
        "min_scale": MIN_SCALE,
        "winsor_z": WINSOR_Z,
    }
    return stats


def stationarize(bar: dict, stats: dict, config: JumpModelConfig
                 ) -> tuple[float, ...] | None:
    """Standardized vector, winsorized at +/- WINSOR_Z, with a binary
    turnover-present indicator appended as the fifth dimension so
    sparse turnover carries bounded, meaningful signal."""
    cell_key = f"{bar['condition_id']}|{_utc_hour(bar['bin_start'])}"
    cell = stats["cells"].get(cell_key)
    condition_global = stats["globals"].get(bar["condition_id"])
    pooled = stats.get("pooled")
    if condition_global is None and pooled is None:
        return None
    out = []
    for variable in VARIABLES:
        if variable == "turnover" and bar[variable] <= 0:
            out.append(0.0)  # level undefined absent presence
            continue
        if (cell is not None
                and cell[variable]["n"] >= config.min_cell_observations):
            source = cell[variable]
        elif (condition_global is not None
              and condition_global[variable]["n"] > 0):
            source = condition_global[variable]
        else:
            source = pooled[variable]
        value = _log_transform(variable, bar[variable])
        z = (value - source["median"]) / source["iqr"]
        out.append(max(-WINSOR_Z, min(WINSOR_Z, z)))
    out.append(
        TURNOVER_PRESENT_WEIGHT if bar["turnover"] > 0 else 0.0
    )
    return tuple(out)


# ---------------------------------------------------------------------------
# jump model core
# ---------------------------------------------------------------------------
def _sq_dist(x: tuple[float, ...], theta: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(x, theta))


def online_assign(
    X: list[tuple[float, ...]],
    centroids: list[list[float]],
    lam: float,
) -> list[int]:
    """ONLINE (filtered) mode sequence: the mode at t uses observations
    only through t — the forward pass of the DP, with m_t taken as the
    argmin of the forward cost at t.  No backtracking from the future,
    so an assignment claimed available at the end of bin t truly was."""
    if not X:
        return []
    K = len(centroids)
    forward = [_sq_dist(X[0], centroids[k]) for k in range(K)]
    modes = [min(range(K), key=lambda k: forward[k])]
    for t in range(1, len(X)):
        forward = [
            min(
                forward[j] + (0.0 if j == k else lam)
                for j in range(K)
            ) + _sq_dist(X[t], centroids[k])
            for k in range(K)
        ]
        modes.append(min(range(K), key=lambda k: forward[k]))
    return modes


def dp_assign(
    X: list[tuple[float, ...]],
    centroids: list[list[float]],
    lam: float,
) -> list[int]:
    """Optimal mode sequence for fixed centroids: DP over the switch
    penalty (Viterbi with uniform transitions penalized by lambda)."""
    if not X:
        return []
    K = len(centroids)
    cost = [[_sq_dist(x, centroids[k]) for k in range(K)] for x in X]
    best = [cost[0][:]]
    back: list[list[int]] = []
    for t in range(1, len(X)):
        row, back_row = [], []
        for k in range(K):
            candidates = [
                best[t - 1][j] + (0.0 if j == k else lam)
                for j in range(K)
            ]
            j_star = min(range(K), key=lambda j: candidates[j])
            row.append(candidates[j_star] + cost[t][k])
            back_row.append(j_star)
        best.append(row)
        back.append(back_row)
    modes = [0] * len(X)
    modes[-1] = min(range(K), key=lambda k: best[-1][k])
    for t in range(len(X) - 2, -1, -1):
        modes[t] = back[t][modes[t + 1]]
    return modes


def _update_centroids(
    X: list[tuple[float, ...]], modes: list[int], K: int,
    previous: list[list[float]],
) -> list[list[float]]:
    out = []
    for k in range(K):
        members = [x for x, m in zip(X, modes) if m == k]
        if members:
            out.append([
                sum(x[d] for x in members) / len(members)
                for d in range(len(X[0]))
            ])
        else:
            out.append(previous[k])  # empty mode keeps its centroid
    return out


def _volatility_split_init(
    X: list[tuple[float, ...]]
) -> list[list[float]]:
    sigma_index = VARIABLES.index("volatility")
    # dimensionality follows the data (stationarized vectors carry the
    # turnover-present indicator as an extra component)
    ranked = sorted(X, key=lambda x: x[sigma_index])
    half = max(1, len(ranked) // 2)
    lower, upper = ranked[:half], ranked[half:] or ranked[-1:]

    def mean(group):
        return [
            sum(x[d] for x in group) / len(group) for d in range(len(X[0]))
        ]

    return [mean(lower), mean(upper)]


def fit_modes_for_lambda(
    segments: list[list[tuple[float, ...]]], lam: float,
    config: JumpModelConfig,
) -> tuple[list[list[float]], list[list[int]], float]:
    """Alternating minimization over all contiguous segments jointly:
    shared centroids, per-segment DP chains (incomplete bars break the
    temporal chain rather than being bridged)."""
    X_all = [x for segment in segments for x in segment]
    centroids = _volatility_split_init(X_all)
    modes_per_segment = [[0] * len(seg) for seg in segments]
    for _ in range(config.max_iterations):
        new_modes = [
            dp_assign(segment, centroids, lam) for segment in segments
        ]
        flat = [m for seg in new_modes for m in seg]
        centroids = _update_centroids(X_all, flat, 2, centroids)
        if new_modes == modes_per_segment:
            break
        modes_per_segment = new_modes
    objective = 0.0
    for segment, modes in zip(segments, modes_per_segment):
        for x, m in zip(segment, modes):
            objective += _sq_dist(x, centroids[m])
        objective += lam * sum(
            1 for a, b in zip(modes, modes[1:]) if a != b
        )
    return centroids, modes_per_segment, objective


def _mean_mode_duration(modes_per_segment: list[list[int]]) -> float:
    durations = []
    for modes in modes_per_segment:
        if not modes:
            continue
        run = 1
        for a, b in zip(modes, modes[1:]):
            if a == b:
                run += 1
            else:
                durations.append(run)
                run = 1
        durations.append(run)
    return sum(durations) / len(durations) if durations else 0.0


def _contiguous_segments(
    bars: list[dict], stats: dict, config: JumpModelConfig
) -> tuple[list[list[tuple[float, ...]]], list[list[dict]]]:
    segments: list[list[tuple[float, ...]]] = []
    segment_bars: list[list[dict]] = []
    current_x: list[tuple[float, ...]] = []
    current_b: list[dict] = []
    previous: dict | None = None
    for bar in bars:
        x = stationarize(bar, stats, config)
        if x is None:
            continue
        contiguous = (
            previous is not None
            and bar["condition_id"] == previous["condition_id"]
            and bar["bin_start"] - previous["bin_start"]
            == config.bin_seconds
        )
        if not contiguous and current_x:
            segments.append(current_x)
            segment_bars.append(current_b)
            current_x, current_b = [], []
        current_x.append(x)
        current_b.append(bar)
        previous = bar
    if current_x:
        segments.append(current_x)
        segment_bars.append(current_b)
    return segments, segment_bars


# ---------------------------------------------------------------------------
def fit_jump_model(
    conn: sqlite3.Connection,
    *,
    fit_cutoff: float,
    config: JumpModelConfig = JumpModelConfig(),
) -> FittedJumpModel:
    """Fit on all complete bars strictly before ``fit_cutoff`` (across
    markets jointly, per the reviewer's instruction not to fit one
    short market in isolation), then assign every complete bar."""
    train_bars = _load_raw_bars(conn, config, end=fit_cutoff)
    if len(train_bars) < 10:
        raise ValueError(
            f"only {len(train_bars)} complete training bars before the "
            "fit cutoff; the jump model needs more coverage"
        )
    stats = fit_reference_stats(train_bars, config)
    train_segments, _ = _contiguous_segments(train_bars, stats, config)

    if config.fixed_lambda is not None:
        lam, selection = config.fixed_lambda, "fixed"
        centroids, train_modes, _ = fit_modes_for_lambda(
            train_segments, lam, config
        )
    else:
        lam, selection = config.lambda_candidates[-1], "persistence_target"
        centroids = None
        for candidate in config.lambda_candidates:
            c, train_modes, _ = fit_modes_for_lambda(
                train_segments, candidate, config
            )
            if _mean_mode_duration(train_modes) \
                    >= config.min_mean_duration_bins:
                lam, centroids = candidate, c
                break
        if centroids is None:  # target never reached: largest candidate
            centroids, train_modes, _ = fit_modes_for_lambda(
                train_segments, lam, config
            )

    sigma_index = VARIABLES.index("volatility")
    calm_mode = min(
        (0, 1), key=lambda k: centroids[k][sigma_index]
    )
    import hashlib

    train_identity = [
        (
            bar["condition_id"], bar["bin_start"], bar["spread_ticks"],
            bar["turnover"], bar["volatility"], bar["best_book_size"],
            bar["feature_version"],
        )
        for bar in train_bars
    ]
    fingerprint = hashlib.sha256(canonical_json({
        "reference_stats": stats,
        "config": {
            "bin_seconds": config.bin_seconds,
            "lambda": lam,
            "lambda_candidates": list(config.lambda_candidates),
            "min_mean_duration_bins": config.min_mean_duration_bins,
            "min_cell_observations": config.min_cell_observations,
            "max_iterations": config.max_iterations,
        },
        "model_version": MODEL_VERSION,
        "variables": list(VARIABLES),
        "train_bars": train_identity,
        "fit_cutoff": fit_cutoff,
        "centroids": centroids,
        "calm_mode": calm_mode,
    }).encode()).hexdigest()
    mode_run_id = f"modes-{fingerprint[:32]}"
    model = FittedJumpModel(
        mode_run_id=mode_run_id, centroids=centroids,
        calm_mode=calm_mode, lambda_penalty=lam,
        lambda_selection=selection, reference_stats=stats,
        train_bar_count=len(train_bars), config=config,
    )
    # assign ALL complete bars (training and after) under the frozen fit
    all_bars = _load_raw_bars(conn, config)
    segments, segment_bars = _contiguous_segments(all_bars, stats, config)
    for xs, bars in zip(segments, segment_bars):
        smoothed = dp_assign(xs, centroids, lam)
        filtered = online_assign(xs, centroids, lam)
        for bar, mode_s, mode_o in zip(bars, smoothed, filtered):
            key = (bar["condition_id"], bar["bin_start"])
            model.assignments[key] = mode_s
            model.assignments_online[key] = mode_o
    return model


def persist_jump_model(
    conn: sqlite3.Connection, model: FittedJumpModel, fit_cutoff: float
) -> None:
    now = time.time()
    conn.execute(
        """
        INSERT OR REPLACE INTO liquidity_mode_runs
            (mode_run_id, fit_cutoff, bin_seconds, lambda_penalty,
             lambda_selection, centroids_json, reference_stats_json,
             calm_mode, train_bar_count, config_json, model_version,
             created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model.mode_run_id, fit_cutoff, model.config.bin_seconds,
            model.lambda_penalty, model.lambda_selection,
            canonical_json(model.centroids),
            canonical_json(model.reference_stats), model.calm_mode,
            model.train_bar_count,
            canonical_json({
                "lambda_candidates": list(model.config.lambda_candidates),
                "min_mean_duration_bins":
                    model.config.min_mean_duration_bins,
                "min_cell_observations":
                    model.config.min_cell_observations,
                "variables": list(VARIABLES),
            }),
            MODEL_VERSION, now,
        ),
    )
    for key, mode in model.assignments.items():
        condition_id, bin_start = key
        mode_online = model.assignments_online[key]
        conn.execute(
            """
            INSERT OR REPLACE INTO liquidity_mode_assignments
                (mode_run_id, condition_id, bin_start, mode, mode_label,
                 mode_online, mode_label_online, in_training,
                 assigned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model.mode_run_id, condition_id, bin_start,
                mode, "calm" if mode == model.calm_mode else "event",
                mode_online,
                "calm" if mode_online == model.calm_mode else "event",
                int(bin_start < fit_cutoff), now,
            ),
        )
    conn.commit()


def load_jump_model_run(
    conn: sqlite3.Connection, mode_run_id: str
) -> dict:
    row = conn.execute(
        "SELECT * FROM liquidity_mode_runs WHERE mode_run_id = ?",
        (mode_run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown mode run: {mode_run_id}")
    record = dict(row)
    record["centroids"] = json.loads(record.pop("centroids_json"))
    record["reference_stats"] = json.loads(
        record.pop("reference_stats_json")
    )
    record["config"] = json.loads(record.pop("config_json"))
    return record
