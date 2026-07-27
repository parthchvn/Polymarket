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
MIN_MARKET_CLUSTERS = 5


class MarketCensor:
    """Prediction markets converge mechanically at resolution, so a
    horizon window is admissible only if the SAME contract stayed open
    and tradeable through the endpoint: no resolved/closed/disabled
    status effective inside the window, no contract-version change,
    and the window ends before any recorded resolution time."""

    def __init__(self, conn: sqlite3.Connection):
        self._timeline: dict[str, list[tuple[float, bool]]] = {}
        self._versions: dict[str, list[float]] = {}
        self._resolution: dict[str, float] = {}
        for row in conn.execute(
            "SELECT m.condition_id, s.effective_from, s.trading_enabled,"
            " s.closed, s.resolved FROM market_status_versions s "
            "JOIN markets m ON m.market_id = s.market_id"
        ):
            blocking = bool(
                (not row["trading_enabled"]) or row["closed"]
                or row["resolved"]
            )
            self._timeline.setdefault(
                row["condition_id"], []
            ).append((float(row["effective_from"]), blocking))
        for row in conn.execute(
            "SELECT m.condition_id, v.effective_from, v.resolution_time "
            "FROM contract_versions v "
            "JOIN markets m ON m.market_id = v.market_id"
        ):
            self._versions.setdefault(
                row["condition_id"], []
            ).append(float(row["effective_from"]))
            if row["resolution_time"] is not None:
                current = self._resolution.get(
                    row["condition_id"], float("inf")
                )
                self._resolution[row["condition_id"]] = min(
                    current, float(row["resolution_time"])
                )
        for values in self._timeline.values():
            values.sort()
        for values in self._versions.values():
            values.sort()

    def open_through(
        self, condition_id: str, start: float, end: float
    ) -> bool:
        from bisect import bisect_left
        from bisect import bisect_right as _br

        timeline = self._timeline.get(condition_id, [])
        # the status in force at the window START must be tradeable ...
        index = _br([t for t, _ in timeline], start) - 1
        if index >= 0 and timeline[index][1]:
            return False           # already blocked when the window opens
        # ... and no blocking status may take effect inside the window
        for effective_from, blocking in timeline[index + 1:]:
            if effective_from > end:
                break
            if blocking:
                return False
        versions = self._versions.get(condition_id, [])
        if versions and (
            bisect_left(versions, start + 1e-9)
            != _br(versions, end)
        ):
            return False           # the contract changed mid-window
        resolution = self._resolution.get(condition_id)
        if resolution is not None and end >= resolution:
            return False
        return True

    def time_to_resolution(
        self, condition_id: str, ts: float
    ) -> float | None:
        resolution = self._resolution.get(condition_id)
        if resolution is None:
            return None
        return max(0.0, resolution - ts)


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
        found = self.close_asof_with_time(condition_id, ts, max_staleness)
        return found[1] if found else None

    def close_asof_with_time(
        self, condition_id: str, ts: float,
        max_staleness: float | None = None,
    ) -> tuple[float, float] | None:
        times = self._times.get(condition_id)
        if not times:
            return None
        index = bisect_right(times, ts) - 1
        if index < 0:
            return None
        if (max_staleness is not None
                and ts - times[index] > max_staleness):
            return None
        return times[index], self._values[condition_id][index]

    def close_near_target(
        self, condition_id: str, target: float,
        *, after: float, tolerance: float | None = None,
    ) -> tuple[float, float] | None:
        """The close nearest ``target`` that is (a) strictly after
        ``after`` (the base timestamp — so a stale series can never
        reuse the base close and manufacture a zero future return) and
        (b) within ``tolerance`` of the target (default: one bar).
        Returns (timestamp, value) or None — missing is missing."""
        tolerance = (
            tolerance if tolerance is not None else self.bin_seconds
        )
        times = self._times.get(condition_id)
        if not times:
            return None
        best: tuple[float, float] | None = None
        index = bisect_right(times, target + tolerance) - 1
        while index >= 0 and times[index] >= target - tolerance:
            if times[index] > after:
                candidate = (times[index],
                             self._values[condition_id][index])
                if (best is None or abs(candidate[0] - target)
                        < abs(best[0] - target)):
                    best = candidate
            index -= 1
        return best


# ---------------------------------------------------------------------------
# OLS with fixed effects and clustered standard errors
# ---------------------------------------------------------------------------
def _cluster_cov(
    X: np.ndarray, residuals: np.ndarray, labels: np.ndarray,
    XtX_inv: np.ndarray,
) -> np.ndarray:
    n, k = X.shape
    groups = np.unique(labels)
    meat = np.zeros((k, k))
    for group in groups:
        mask = labels == group
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    g = len(groups)
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return correction * XtX_inv @ meat @ XtX_inv


def ols_clustered(
    y: np.ndarray,
    X: np.ndarray,
    clusters: dict[str, np.ndarray],
) -> dict:
    """OLS with CR1 cluster-robust SEs per clustering, plus CGM
    two-way clustering (V_a + V_b - V_intersection) when exactly the
    'market' and 'utc_day' clusterings are both supplied, and
    cluster-count diagnostics."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta
    out = {"beta": beta, "n": int(n), "se": {}, "cluster_counts": {}}
    covs = {}
    for name, labels in clusters.items():
        cov = _cluster_cov(X, residuals, labels, XtX_inv)
        covs[name] = cov
        out["se"][name] = np.sqrt(np.clip(np.diag(cov), 0, None))
        out["cluster_counts"][name] = int(len(np.unique(labels)))
    if "market" in clusters and "utc_day" in clusters:
        intersection = (
            clusters["market"].astype(np.int64) * 1_000_003
            + clusters["utc_day"].astype(np.int64)
        )
        cov_two_way = (
            covs["market"] + covs["utc_day"]
            - _cluster_cov(X, residuals, intersection, XtX_inv)
        )
        out["se"]["two_way"] = np.sqrt(
            np.clip(np.diag(cov_two_way), 0, None)
        )
        out["cluster_counts"]["two_way_min"] = min(
            out["cluster_counts"]["market"],
            out["cluster_counts"]["utc_day"],
        )
    return out


def moving_block_bootstrap(
    y: np.ndarray, X: np.ndarray, day_labels: np.ndarray,
    *, coef_index: int, block_days: int, n_boot: int = 200,
    seed: int = 7,
) -> dict:
    """Moving-DATE-block bootstrap for one coefficient: contiguous
    day blocks at least as long as the horizon are resampled with
    replacement, so overlapping-outcome dependence within a block is
    preserved."""
    rng = np.random.default_rng(seed)
    days = np.unique(day_labels)
    if len(days) < 2 * block_days:
        return {"skipped": f"only {len(days)} days for "
                           f"{block_days}-day blocks"}
    day_index = {day: np.flatnonzero(day_labels == day)
                 for day in days}
    starts = days[: max(1, len(days) - block_days + 1)]
    n_blocks = max(1, int(math.ceil(len(days) / block_days)))
    draws = []
    for _ in range(n_boot):
        rows: list[np.ndarray] = []
        for start in rng.choice(starts, size=n_blocks, replace=True):
            for day in range(int(start), int(start) + block_days):
                if day in day_index:
                    rows.append(day_index[day])
        take = np.concatenate(rows)
        Xb, yb = X[take], y[take]
        beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ yb)
        draws.append(float(beta[coef_index]))
    draws_arr = np.asarray(draws)
    return {
        "se": float(draws_arr.std(ddof=1)),
        "ci_2_5": float(np.percentile(draws_arr, 2.5)),
        "ci_97_5": float(np.percentile(draws_arr, 97.5)),
        "n_boot": int(n_boot),
        "block_days": int(block_days),
    }


def wild_cluster_bootstrap_p(
    y: np.ndarray, X: np.ndarray, labels: np.ndarray,
    *, coef_index: int, n_boot: int = 499, seed: int = 11,
) -> float:
    """Rademacher wild-cluster bootstrap p-value for H0: beta_j = 0
    (null-imposed residuals, per-cluster sign flips) — the small-
    cluster-count inference of Cameron-Gelbach-Miller."""
    rng = np.random.default_rng(seed)
    restricted = np.delete(X, coef_index, axis=1)
    beta_r = np.linalg.pinv(restricted.T @ restricted) @ (
        restricted.T @ y
    )
    u0 = y - restricted @ beta_r
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    se = np.sqrt(_cluster_cov(
        X, y - X @ beta_hat, labels, XtX_inv
    )[coef_index, coef_index])
    if se <= 0:
        return float("nan")
    t_obs = abs(beta_hat[coef_index] / se)
    groups = np.unique(labels)
    exceed = 0
    for _ in range(n_boot):
        flips = rng.choice([-1.0, 1.0], size=len(groups))
        u_star = u0 * flips[np.searchsorted(groups, labels)]
        y_star = restricted @ beta_r + u_star
        beta_star = XtX_inv @ (X.T @ y_star)
        se_star = np.sqrt(_cluster_cov(
            X, y_star - X @ beta_star, labels, XtX_inv
        )[coef_index, coef_index])
        if se_star > 0 and abs(
            beta_star[coef_index] / se_star
        ) >= t_obs:
            exceed += 1
    return (exceed + 1) / (n_boot + 1)


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
    cluster_counts: dict[str, int]
    censored: int = 0
    stale_endpoint_dropped: int = 0
    inference_note: str | None = None
    block_bootstrap: dict | None = None
    wild_cluster_p_news: float | None = None

    @property
    def inference_admissible(self) -> bool:
        return self.cluster_counts.get("market", 0)             >= MIN_MARKET_CLUSTERS

    def as_dict(self) -> dict:
        t_news = None
        if self.inference_admissible:
            t_news = {
                name: (self.beta_news / se if se > 0 else None)
                for name, se in self.se_news_by_cluster.items()
            }
        return {
            "horizon_seconds": self.horizon_seconds,
            "spec": self.spec,
            "n": self.n,
            "beta_news": self.beta_news,
            "beta_nonnews": self.beta_nonnews,
            "se_news": self.se_news_by_cluster,
            "se_nonnews": self.se_nonnews_by_cluster,
            "cluster_counts": self.cluster_counts,
            "censored_observations": self.censored,
            "stale_endpoint_dropped": self.stale_endpoint_dropped,
            "t_news": t_news,
            "inference_note": self.inference_note,
            "block_bootstrap_news": self.block_bootstrap,
            "wild_cluster_p_news": self.wild_cluster_p_news,
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
    censor = MarketCensor(conn)
    results = []
    for horizon in horizons:
        rows, y = [], []
        condition_labels, day_labels = [], []
        censored = stale_dropped = 0
        for index, record in enumerate(records):
            t_end = record.bin_start + config.bin_seconds
            base = closes.close_asof_with_time(
                record.condition_id, t_end
            )
            if base is None:
                continue
            base_ts, base_value = base
            if not censor.open_through(
                record.condition_id, t_end, t_end + horizon
            ):
                censored += 1     # resolution is not continuation
                continue
            future = closes.close_near_target(
                record.condition_id, t_end + horizon, after=base_ts
            )
            if future is None:
                stale_dropped += 1  # missing is missing, never zero
                continue
            ttr = censor.time_to_resolution(record.condition_id, t_end)
            rows.append([
                record.r_news, record.r_nonnews, record.close,
                abs(record.close),                # probability region
                record.spread_mean or 0.0,
                math.log1p(record.turnover),
                _trailing_vol(records, index),
                math.log1p(ttr) if ttr is not None else 0.0,
                0.0 if ttr is not None else 1.0,  # ttr missing flag
            ])
            y.append(future[1] - base_value)
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
        market_labels = np.unique(entities, return_inverse=True)[1]
        days_arr = np.asarray(day_labels)
        fit = ols_clustered(
            y_arr, X,
            clusters={"market": market_labels, "utc_day": days_arr},
        )
        n_markets = fit["cluster_counts"]["market"]
        note = None
        wild_p = None
        if n_markets < MIN_MARKET_CLUSTERS:
            note = (
                f"only {n_markets} market clusters (<"
                f"{MIN_MARKET_CLUSTERS}): t-statistics refused; wild-"
                "cluster bootstrap p reported instead"
            )
            wild_p = wild_cluster_bootstrap_p(
                y_arr, X, market_labels, coef_index=1
            )
        block = moving_block_bootstrap(
            y_arr, X, days_arr, coef_index=1,
            block_days=max(1, int(math.ceil(horizon / 86400.0))),
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
            cluster_counts=fit["cluster_counts"],
            censored=censored,
            stale_endpoint_dropped=stale_dropped,
            inference_note=note,
            block_bootstrap=block,
            wild_cluster_p_news=wild_p,
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
    censor = MarketCensor(conn)
    events = []
    all_arrivals = _news_arrivals(conn, config, spec)
    for condition_id, arrivals in all_arrivals.items():
        ordered = sorted(arrivals)
        for position, (tau, claim_id) in enumerate(ordered):
            pre_found = closes.close_asof_with_time(
                condition_id, tau, max_staleness=2 * config.bin_seconds
            )
            pre = pre_found[1] if pre_found else None
            # endpoints must be genuinely POST-news, fresh, and the
            # market must stay open/unchanged through the window
            h0_found = closes.close_near_target(
                condition_id, tau + h0, after=tau,
            ) if pre_found else None
            h_found = closes.close_near_target(
                condition_id, tau + drift_horizon,
                after=(h0_found[0] if h0_found else tau),
            ) if h0_found else None
            open_through = censor.open_through(
                condition_id, tau, tau + drift_horizon
            )
            at_h0 = h0_found[1] if h0_found else None
            at_h = h_found[1] if h_found else None
            coverage_complete = (
                None not in (pre, at_h0, at_h) and open_through
            )
            intervening = any(
                tau < other_tau <= tau + drift_horizon
                for other_tau, _ in ordered[position + 1:]
            )
            record = {
                "condition_id": condition_id, "claim_id": claim_id,
                "news_time": tau, "coverage_complete": coverage_complete,
                "market_open_through_window": open_through,
                "intervening_news": intervening,
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
