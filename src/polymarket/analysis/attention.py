"""Attention / distraction mechanism — 'Pervasive Underreaction'
section on investor distraction, using proxies computable from
Polymarket data alone.

Per interval, the distraction proxies are:

* ``cross_market_claim_count``: claims arriving anywhere in the
  trailing 24 hours (total news volume competing for attention);
* ``unrelated_family_count``: event families active in the trailing 24
  hours excluding families with a relevance judgment for THIS market;
* ``weekend``: UTC Saturday/Sunday flag (low-attention timing);
* ``event_mode_prevalence`` (optional, needs a mode run): the fraction
  of assigned markets whose current bin is in event mode.

The mechanism test interacts the news return with a standardized
distraction index in the drift regression:

    R_future = a + bN r_news + bI (r_news x distraction_z)
                 + bD distraction_z + bNN r_nonnews + controls

The paper's mechanism predicts bI > 0: drift after news is stronger
when attention is scarcer.  The analyst-revision mechanism remains
UNTESTED (no external expectations series).
"""

from __future__ import annotations

import datetime
import math
import sqlite3

import numpy as np

from polymarket.analysis.news_returns import (
    DecompositionConfig,
    IntervalRecord,
    build_interval_records,
)
from polymarket.analysis.underreaction import (
    CloseSeries,
    _within_demean,
    ols_clustered,
)

TRAILING_WINDOW = 24 * 3600.0


def _claim_times(conn: sqlite3.Connection) -> list[float]:
    return [
        row[0] for row in conn.execute(
            "SELECT first_available_at FROM news_claims "
            "ORDER BY first_available_at"
        )
    ]


def _family_times(
    conn: sqlite3.Connection,
) -> list[tuple[float, str]]:
    return conn.execute(
        "SELECT earliest_available_at, event_family_id FROM "
        "event_families ORDER BY earliest_available_at"
    ).fetchall()


def _own_families(conn: sqlite3.Connection) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT DISTINCT m.condition_id, r.event_family_id "
        "FROM relevance_judgments r "
        "JOIN markets m ON m.market_id = r.market_id"
    ):
        out.setdefault(row[0], set()).add(row[1])
    return out


def _is_weekend(ts: float) -> bool:
    return datetime.datetime.fromtimestamp(
        ts, tz=datetime.timezone.utc
    ).weekday() >= 5


def compute_distraction(
    conn: sqlite3.Connection,
    records: list[IntervalRecord],
    mode_run_id: str | None = None,
) -> list[dict]:
    """One proxy row per interval record, order-aligned."""
    from bisect import bisect_left, bisect_right

    claim_times = _claim_times(conn)
    family_rows = _family_times(conn)
    family_times = [row[0] for row in family_rows]
    own = _own_families(conn)
    labels = {}
    if mode_run_id is not None:
        for row in conn.execute(
            "SELECT condition_id, bin_start, mode_label_online FROM "
            "liquidity_mode_assignments WHERE mode_run_id = ?",
            (mode_run_id,),
        ):
            labels[(row[0], row[1])] = row[2]
    out = []
    for record in records:
        t = record.bin_start
        low = bisect_left(claim_times, t - TRAILING_WINDOW)
        high = bisect_right(claim_times, t)
        window_families = {
            family_rows[i][1]
            for i in range(
                bisect_left(family_times, t - TRAILING_WINDOW),
                bisect_right(family_times, t),
            )
        }
        unrelated = window_families - own.get(record.condition_id, set())
        prevalence = None
        if labels:
            bin_labels = [
                label for (condition, bin_start), label in labels.items()
                if bin_start == t
            ]
            if bin_labels:
                prevalence = (
                    sum(1 for label in bin_labels if label == "event")
                    / len(bin_labels)
                )
        out.append({
            "cross_market_claim_count": high - low,
            "unrelated_family_count": len(unrelated),
            "weekend": _is_weekend(t),
            "event_mode_prevalence": prevalence,
        })
    return out


def distraction_interaction_regression(
    conn: sqlite3.Connection,
    config: DecompositionConfig = DecompositionConfig(),
    spec: str = "all_relevant",
    horizon: float = 24 * 3600.0,
    mode_run_id: str | None = None,
) -> dict | None:
    records = build_interval_records(conn, config, spec)
    proxies = compute_distraction(conn, records, mode_run_id)
    closes = CloseSeries(conn, config.bin_seconds)
    rows, y = [], []
    condition_labels, day_labels = [], []
    raw_distraction = [
        p["cross_market_claim_count"] + p["unrelated_family_count"]
        + (5 if p["weekend"] else 0)
        for p in proxies
    ]
    if not raw_distraction:
        return None
    mean = sum(raw_distraction) / len(raw_distraction)
    var = sum((x - mean) ** 2 for x in raw_distraction) \
        / max(len(raw_distraction) - 1, 1)
    scale = math.sqrt(var) if var > 0 else 1.0
    for record, raw in zip(records, raw_distraction):
        t_end = record.bin_start + config.bin_seconds
        base = closes.close_asof(record.condition_id, t_end)
        future = closes.close_asof(
            record.condition_id, t_end + horizon, max_staleness=horizon
        )
        if base is None or future is None:
            continue
        z = (raw - mean) / scale
        rows.append([
            record.r_news, record.r_news * z, z, record.r_nonnews,
            record.close, math.log1p(record.turnover),
        ])
        y.append(future - base)
        condition_labels.append(record.condition_id)
        day_labels.append(int(record.bin_start // 86400))
    if len(rows) < 20:
        return None
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
    return {
        "horizon_seconds": horizon,
        "spec": spec,
        "n": fit["n"],
        "beta_news": float(fit["beta"][1]),
        "beta_news_x_distraction": float(fit["beta"][2]),
        "beta_distraction": float(fit["beta"][3]),
        "se_interaction": {
            name: float(se[2]) for name, se in fit["se"].items()
        },
        "prediction": "beta_news_x_distraction > 0 under the paper's "
                      "distraction mechanism",
    }
