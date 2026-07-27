"""Underreaction tests: future-drift regressions, event-level
absorption, and placebos — 'Pervasive Underreaction' adapted to
prediction markets.

Core regression, per horizon h (a local-projection adaptation of the
paper's daily design, run at the interval level):

    R_future(m, j, h) = a_h + bN_h * r_news(m, j)
                            + bNN_h * r_nonnews(m, j)
                            + g_h' Z(m, j) + e

with market fixed effects (within transformation) and standard errors
clustered one-way by market AND, separately, by UTC day — both
reported; the paper's central finding corresponds to bN_h > 0 (the
initial news response continues) while non-news moves do not.

Controls Z: probability level (logit close), mean spread, log1p
turnover, and trailing realized volatility.

Event-level absorption (OUR adaptation, marked as such):

    initial_e = l(tau + h0) - l(tau-)
    drift_e,h = l(tau + h) - l(tau + h0)
    A_e = |initial| / (|initial| + |drift|)

with same-direction continuation flags and explicit coverage flags —
missing closes yield missing outcomes, never zeros.

The analyst-revision mechanism of the paper requires an external
expectations series this pipeline does not have; it is UNTESTED here,
not replicated.
"""

from __future__ import annotations

import math
import random
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from polymarket.analysis.news_returns import (
    DecompositionConfig,
    IntervalRecord,
    build_interval_records,
)

HORIZONS_SECONDS = (3600.0, 6 * 3600.0, 24 * 3600.0,
                    72 * 3600.0, 7 * 86400.0)


# ---------------------------------------------------------------------------
# close-series lookups
# ---------------------------------------------------------------------------
class CloseSeries:
    """Per-condition (bin_end -> logit_close) lookup with as-of reads."""

    def __init__(self, conn: sqlite3.Connection, bin_seconds: float):
        self._times: dict[str, list[float]] = {}
        self._values: dict[str, list[float]] = {}
        rows = conn.execute(
            "SELECT condition_id, bin_start, logit_close FROM "
            "liquidity_bars WHERE bin_seconds = ? AND "
            "coverage_complete = 1 AND logit_close IS NOT NULL "
            "ORDER BY condition_id, bin_start",
            (bin_seconds,),
        ).fetchall()
        for row in rows:
            self._times.setdefault(row["condition_id"], []).append(
                row["bin_start"] + bin_seconds
            )
            self._values.setdefault(row["condition_id"], []).append(
                row["logit_close"]
            )
        self.bin_seconds = bin_seconds

    def close_asof(self, condition_id: str, ts: float,
                   max_staleness: float | None = None) -> float | None:
        times = self._times.get(condition_id)
        if not times:
            return None
        index = bisect_right(times, ts) - 1
        if index < 0:
            return None
        if (max_staleness is not None
                and ts - times[index] > max_staleness):
            return None
        return self._values[condition_id][index]


# ---------------------------------------------------------------------------
# OLS with fixed effects and clustered standard errors
# ---------------------------------------------------------------------------
def ols_clustered(
    y: np.ndarray,
    X: np.ndarray,
    clusters: dict[str, np.ndarray],
) -> dict:
    """OLS with CR1 cluster-robust standard errors, one clustering per
    entry in ``clusters`` (each an integer label array)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta
    out = {"beta": beta, "n": int(n), "se": {}}
    for name, labels in clusters.items():
        groups = np.unique(labels)
        meat = np.zeros((k, k))
        for group in groups:
            mask = labels == group
            Xg = X[mask]
            ug = residuals[mask]
            score = Xg.T @ ug
            meat += np.outer(score, score)
        g = len(groups)
        correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
        cov = correction * XtX_inv @ meat @ XtX_inv
        out["se"][name] = np.sqrt(np.clip(np.diag(cov), 0, None))
    return out


def _within_demean(
    values: np.ndarray, entities: np.ndarray
) -> np.ndarray:
    out = values.astype(float).copy()
    for entity in np.unique(entities):
        mask = entities == entity
        out[mask] -= out[mask].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
@dataclass
class DriftRegressionResult:
    horizon_seconds: float
    spec: str
    n: int
    beta_news: float
    beta_nonnews: float
    se_news_by_cluster: dict[str, float]
    se_nonnews_by_cluster: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "horizon_seconds": self.horizon_seconds,
            "spec": self.spec,
            "n": self.n,
            "beta_news": self.beta_news,
            "beta_nonnews": self.beta_nonnews,
            "se_news": self.se_news_by_cluster,
            "se_nonnews": self.se_nonnews_by_cluster,
            "t_news": {
                name: (self.beta_news / se if se > 0 else None)
                for name, se in self.se_news_by_cluster.items()
            },
        }


def _trailing_vol(records: list[IntervalRecord], index: int,
                  window: int = 6) -> float:
    rets = [
        r.ret for r in records[max(0, index - window):index]
        if r.condition_id == records[index].condition_id
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    return math.sqrt(sum((x - mean) ** 2 for x in rets) / (len(rets) - 1))


def run_drift_regressions(
    conn: sqlite3.Connection,
    config: DecompositionConfig = DecompositionConfig(),
    spec: str = "all_relevant",
    horizons: tuple[float, ...] = HORIZONS_SECONDS,
    records: list[IntervalRecord] | None = None,
    placebo_seed: int | None = None,
) -> list[DriftRegressionResult]:
    """Interval-level local projections of future log-odds changes on
    the news / non-news decomposition.  ``placebo_seed`` circularly
    shifts news labels within each market (destroying the true
    news-return alignment while preserving both marginals)."""
    if records is None:
        records = build_interval_records(conn, config, spec)
    if placebo_seed is not None:
        records = _placebo_shift(records, placebo_seed)
    closes = CloseSeries(conn, config.bin_seconds)
    results = []
    for horizon in horizons:
        rows, y = [], []
        condition_labels, day_labels = [], []
        for index, record in enumerate(records):
            t_end = record.bin_start + config.bin_seconds
            base = closes.close_asof(record.condition_id, t_end)
            future = closes.close_asof(
                record.condition_id, t_end + horizon,
                max_staleness=horizon,
            )
            if base is None or future is None:
                continue  # missing coverage is missing, never zero
            rows.append([
                record.r_news, record.r_nonnews, record.close,
                record.spread_mean or 0.0,
                math.log1p(record.turnover),
                _trailing_vol(records, index),
            ])
            y.append(future - base)
            condition_labels.append(record.condition_id)
            day_labels.append(int(record.bin_start // 86400))
        if len(rows) < 20:
            continue
        X = np.asarray(rows)
        y_arr = np.asarray(y)
        entities = np.asarray(condition_labels)
        X = _within_demean(X, entities)
        y_arr = _within_demean(y_arr.reshape(-1, 1), entities).ravel()
        X = np.column_stack([np.ones(len(X)), X])
        fit = ols_clustered(
            y_arr, X,
            clusters={
                "market": np.unique(entities, return_inverse=True)[1],
                "utc_day": np.asarray(day_labels),
            },
        )
        results.append(DriftRegressionResult(
            horizon_seconds=horizon, spec=spec, n=fit["n"],
            beta_news=float(fit["beta"][1]),
            beta_nonnews=float(fit["beta"][2]),
            se_news_by_cluster={
                name: float(se[1]) for name, se in fit["se"].items()
            },
            se_nonnews_by_cluster={
                name: float(se[2]) for name, se in fit["se"].items()
            },
        ))
    return results


def _placebo_shift(
    records: list[IntervalRecord], seed: int
) -> list[IntervalRecord]:
    rng = random.Random(seed)
    by_condition: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_condition.setdefault(record.condition_id, []).append(index)
    out = list(records)
    for indices in by_condition.values():
        flags = [records[i].is_news for i in indices]
        shifted = list(flags)
        rng.shuffle(shifted)
        for position, index in enumerate(indices):
            record = records[index]
            out[index] = IntervalRecord(
                condition_id=record.condition_id,
                bin_start=record.bin_start, ret=record.ret,
                close=record.close, spread_mean=record.spread_mean,
                turnover=record.turnover, is_news=shifted[position],
                news_claims=[],
            )
    return out


# ---------------------------------------------------------------------------
# event-level absorption (our adaptation, marked as such)
# ---------------------------------------------------------------------------
def event_absorption(
    conn: sqlite3.Connection,
    config: DecompositionConfig = DecompositionConfig(),
    spec: str = "all_relevant",
    initial_horizon: float | None = None,
    drift_horizon: float = 24 * 3600.0,
) -> list[dict]:
    from polymarket.analysis.news_returns import _news_arrivals

    h0 = initial_horizon or config.bin_seconds
    closes = CloseSeries(conn, config.bin_seconds)
    events = []
    for condition_id, arrivals in _news_arrivals(
        conn, config, spec
    ).items():
        for tau, claim_id in sorted(arrivals):
            pre = closes.close_asof(
                condition_id, tau, max_staleness=2 * config.bin_seconds
            )
            at_h0 = closes.close_asof(
                condition_id, tau + h0,
                max_staleness=2 * config.bin_seconds,
            )
            at_h = closes.close_asof(
                condition_id, tau + drift_horizon,
                max_staleness=2 * config.bin_seconds,
            )
            coverage_complete = None not in (pre, at_h0, at_h)
            record = {
                "condition_id": condition_id, "claim_id": claim_id,
                "news_time": tau, "coverage_complete": coverage_complete,
                "initial_response": None, "later_drift": None,
                "same_direction_continuation": None,
                "absorption_fraction": None,
                "absorption_note": (
                    "A_e = |initial| / (|initial| + |drift|): a "
                    "Polymarket adaptation, NOT a statistic from the "
                    "paper"
                ),
            }
            if coverage_complete:
                initial = at_h0 - pre
                drift = at_h - at_h0
                record["initial_response"] = initial
                record["later_drift"] = drift
                record["same_direction_continuation"] = bool(
                    initial * drift > 0
                )
                denominator = abs(initial) + abs(drift)
                record["absorption_fraction"] = (
                    abs(initial) / denominator if denominator > 0
                    else None
                )
            events.append(record)
    return events
