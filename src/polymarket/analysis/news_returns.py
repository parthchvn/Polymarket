"""News / non-news return decomposition — 'Pervasive Underreaction',
adapted to prediction markets.

The paper decomposes each stock's high-frequency returns into intervals
that contain firm news and intervals that do not, then shows the news
component CONTINUES (positive future-drift loading) while the non-news
component does not.  Here:

* intervals are non-overlapping 15-minute liquidity bars
  (``bin_seconds=900``) of book-mid LOG-ODDS closes — returns are
  ``r_j = logit_close_j − logit_close_{j−1}`` over contiguous complete
  bars (a gap breaks the pair rather than being bridged);
* an interval is news-driven under one of two specifications:
  - ``all_relevant``: any claim with a relevance judgment in the
    configured relevant classes arrived inside the interval;
  - ``screened_impactful``: any claim whose liquidity impact screen
    detected a calm→event transition (basis-aware) arrived inside it;
* the decomposition is retrospective/descriptive (paper replication);
  the online-availability discipline for DRC features lives in the
  screens and PR 10, not here.

Aggregation: interval-level records are the primitive; fixed-UTC-day
sums are provided for the paper-style daily view.  Event-level
analysis (initial response vs later drift) lives in ``underreaction``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

RELEVANT_CLASSES = ("supports_positive", "supports_negative", "direct")


@dataclass(frozen=True)
class DecompositionConfig:
    bin_seconds: float = 900.0
    relevant_classes: tuple[str, ...] = RELEVANT_CLASSES
    screen_basis: str = "retrospective_smoothed"
    mode_run_id: str | None = None


@dataclass
class IntervalRecord:
    condition_id: str
    bin_start: float
    ret: float
    close: float
    spread_mean: float | None
    turnover: float
    is_news: bool
    news_claims: list[str] = field(default_factory=list)

    @property
    def r_news(self) -> float:
        return self.ret if self.is_news else 0.0

    @property
    def r_nonnews(self) -> float:
        return self.ret if not self.is_news else 0.0


def _close_series(
    conn: sqlite3.Connection, config: DecompositionConfig
) -> dict[str, list[sqlite3.Row]]:
    rows = conn.execute(
        "SELECT condition_id, bin_start, logit_close, spread_mean, "
        "turnover_notional FROM liquidity_bars WHERE bin_seconds = ? "
        "AND coverage_complete = 1 AND logit_close IS NOT NULL "
        "ORDER BY condition_id, bin_start",
        (config.bin_seconds,),
    ).fetchall()
    out: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        out.setdefault(row["condition_id"], []).append(row)
    return out


def _news_arrivals(
    conn: sqlite3.Connection, config: DecompositionConfig, spec: str
) -> dict[str, list[tuple[float, str]]]:
    """condition -> [(arrival_time, claim_id)] under the spec."""
    if spec == "all_relevant":
        placeholders = ",".join("?" for _ in config.relevant_classes)
        rows = conn.execute(
            f"""
            SELECT DISTINCT m.condition_id,
                   c.first_available_at AS ts, c.claim_id
            FROM relevance_judgments r
            JOIN news_claims c ON c.claim_id = r.claim_id
            JOIN markets m ON m.market_id = r.market_id
            WHERE r.rel_class IN ({placeholders})
            """,
            config.relevant_classes,
        ).fetchall()
    elif spec == "screened_impactful":
        if config.mode_run_id is None:
            return {}
        rows = conn.execute(
            """
            SELECT DISTINCT condition_id, news_time AS ts, claim_id
            FROM news_impact_screens
            WHERE mode_run_id = ? AND assignment_basis = ?
              AND screen_status = 'screened' AND transition_detected = 1
            """,
            (config.mode_run_id, config.screen_basis),
        ).fetchall()
    else:
        raise ValueError(f"unknown decomposition spec: {spec}")
    out: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        out.setdefault(row["condition_id"], []).append(
            (float(row["ts"]), row["claim_id"])
        )
    return out


def build_interval_records(
    conn: sqlite3.Connection,
    config: DecompositionConfig = DecompositionConfig(),
    spec: str = "all_relevant",
) -> list[IntervalRecord]:
    """Interval returns with news classification.  Returns are between
    CONTIGUOUS complete bars only."""
    arrivals = _news_arrivals(conn, config, spec)
    records: list[IntervalRecord] = []
    for condition_id, rows in _close_series(conn, config).items():
        news_times = arrivals.get(condition_id, [])
        for previous, current in zip(rows, rows[1:]):
            if (current["bin_start"] - previous["bin_start"]
                    != config.bin_seconds):
                continue  # gap: no bridged return
            start = current["bin_start"]
            end = start + config.bin_seconds
            claims = [
                claim for ts, claim in news_times if start <= ts < end
            ]
            records.append(IntervalRecord(
                condition_id=condition_id,
                bin_start=start,
                ret=current["logit_close"] - previous["logit_close"],
                close=current["logit_close"],
                spread_mean=current["spread_mean"],
                turnover=current["turnover_notional"] or 0.0,
                is_news=bool(claims),
                news_claims=claims,
            ))
    return records


def daily_aggregation(
    records: list[IntervalRecord],
) -> list[dict]:
    """Paper-style fixed-UTC-day news / non-news return sums."""
    days: dict[tuple[str, int], dict] = {}
    for record in records:
        day = int(record.bin_start // 86400)
        bucket = days.setdefault(
            (record.condition_id, day),
            {"condition_id": record.condition_id, "utc_day": day,
             "r_news": 0.0, "r_nonnews": 0.0,
             "news_intervals": 0, "intervals": 0},
        )
        bucket["r_news"] += record.r_news
        bucket["r_nonnews"] += record.r_nonnews
        bucket["news_intervals"] += int(record.is_news)
        bucket["intervals"] += 1
    return [days[key] for key in sorted(days)]
