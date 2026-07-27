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


def _own_family_events(
    conn: sqlite3.Connection, relevant_classes: tuple[str, ...],
    availability_policy: str = "retrospective_source",
) -> dict[str, list[tuple[float, str]]]:
    """condition -> [(available_at, family)] for RELEVANTLY judged
    families, sorted by when the classification became available under
    the policy:

    * ``online_scored``: when the scorer actually RAN
      (``scored_at``) — a backdated LLM rescore did not exist
      historically, so online proxies must not use it earlier;
    * ``retrospective_source`` (default for the paper analysis): when
      the underlying text was available (``source_effective_at``) —
      for labelled retrospective runs with frozen scorers.

    Legacy rows fall back to ``computed_at`` for whichever field is
    NULL."""
    if availability_policy == "online_scored":
        time_expr = "COALESCE(r.scored_at, r.computed_at)"
    elif availability_policy == "retrospective_source":
        time_expr = "COALESCE(r.source_effective_at, r.computed_at)"
    else:
        raise ValueError(
            f"unknown availability policy: {availability_policy}"
        )
    placeholders = ",".join("?" for _ in relevant_classes)
    out: dict[str, list[tuple[float, str]]] = {}
    for row in conn.execute(
        f"SELECT m.condition_id, r.event_family_id, "
        f"MIN({time_expr}) AS available_at "
        f"FROM relevance_judgments r "
        f"JOIN markets m ON m.market_id = r.market_id "
        f"WHERE r.rel_class IN ({placeholders}) "
        f"GROUP BY m.condition_id, r.event_family_id",
        relevant_classes,
    ):
        out.setdefault(row["condition_id"], []).append(
            (float(row["available_at"]), row["event_family_id"])
        )
    for values in out.values():
        values.sort()
    return out


def _is_weekend(ts: float) -> bool:
    return datetime.datetime.fromtimestamp(
        ts, tz=datetime.timezone.utc
    ).weekday() >= 5


def compute_distraction(
    conn: sqlite3.Connection,
    records: list[IntervalRecord],
    mode_run_id: str | None = None,
    availability_policy: str = "retrospective_source",
) -> list[dict]:
    """One proxy row per interval record, order-aligned."""
    from bisect import bisect_left, bisect_right

    from polymarket.analysis.news_returns import RELEVANT_CLASSES

    claim_times = _claim_times(conn)
    family_rows = _family_times(conn)
    family_times = [row[0] for row in family_rows]
    own_events = _own_family_events(
        conn, RELEVANT_CLASSES, availability_policy
    )
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
        own_list = own_events.get(record.condition_id, [])
        own_asof = {
            family for available_at, family in own_list
            if available_at <= t
        }
        unrelated = window_families - own_asof
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


PROXY_NAMES = ("cross_market_claim_count", "unrelated_family_count",
               "weekend", "event_mode_prevalence")


def distraction_interaction_regressions(
    conn: sqlite3.Connection,
    config: DecompositionConfig = DecompositionConfig(),
    spec: str = "all_relevant",
    horizon: float = 24 * 3600.0,
    mode_run_id: str | None = None,
    availability_policy: str = "retrospective_source",
) -> dict | None:
    """ONE regression per standardized proxy (no composite index): the
    interaction r_news x proxy_z is reported for each mechanism
    separately, sharing the censored, fresh-endpoint sample with the
    main drift regressions."""
    from polymarket.analysis.underreaction import MarketCensor

    records = build_interval_records(conn, config, spec)
    proxies = compute_distraction(
        conn, records, mode_run_id, availability_policy
    )
    closes = CloseSeries(conn, config.bin_seconds)
    censor = MarketCensor(conn)
    results: dict[str, dict] = {}
    for proxy_name in PROXY_NAMES:
        raw = [
            (1.0 if p[proxy_name] else 0.0)
            if proxy_name == "weekend" else p[proxy_name]
            for p in proxies
        ]
        if any(value is None for value in raw):
            results[proxy_name] = {"skipped": "proxy unavailable"}
            continue
        mean = sum(raw) / len(raw) if raw else 0.0
        var = sum((x - mean) ** 2 for x in raw) / max(len(raw) - 1, 1)
        scale = math.sqrt(var) if var > 0 else 1.0
        rows, y = [], []
        condition_labels, day_labels = [], []
        for record, value in zip(records, raw):
            t_end = record.bin_start + config.bin_seconds
            base = closes.close_asof_with_time(
                record.condition_id, t_end
            )
            if base is None:
                continue
            if not censor.open_through(
                record.condition_id, t_end, t_end + horizon
            ):
                continue
            future = closes.close_near_target(
                record.condition_id, t_end + horizon, after=base[0]
            )
            if future is None:
                continue
            z = (value - mean) / scale
            rows.append([
                record.r_news, record.r_news * z, z,
                record.r_nonnews, record.close,
                math.log1p(record.turnover),
            ])
            y.append(future[1] - base[1])
            condition_labels.append(record.condition_id)
            day_labels.append(int(record.bin_start // 86400))
        if len(rows) < 20:
            results[proxy_name] = {"skipped": "insufficient sample"}
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
        results[proxy_name] = {
            "horizon_seconds": horizon,
            "n": fit["n"],
            "beta_news": float(fit["beta"][1]),
            "beta_news_x_proxy": float(fit["beta"][2]),
            "beta_proxy": float(fit["beta"][3]),
            "se_interaction": {
                name: float(se[2]) for name, se in fit["se"].items()
            },
            "cluster_counts": fit["cluster_counts"],
        }
    if not results:
        return None
    return {
        "spec": spec,
        "availability_policy": availability_policy,
        "prediction": "beta_news_x_proxy > 0 under the paper's "
                      "distraction mechanism (proxies are ANALOGUES "
                      "of the paper's attention measures, not direct "
                      "reproductions)",
        "proxies": results,
    }

