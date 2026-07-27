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

RELEVANT_CLASSES = ("supports_positive", "supports_negative")


@dataclass(frozen=True)
class DecompositionConfig:
    """The pinned news-sample contract, recorded with every analysis.

    A claim qualifies as news for a market only when ALL hold:

    * its LATEST relevance judgment per (claim, market) — under the
      approved method/model when pinned, otherwise across methods —
      has ``rel_class`` in ``relevant_classes`` with ``rel_score >=
      min_rel_score``;
    * that judgment's contract version equals the version ACTIVE at
      the claim's arrival (semantics judged against the contract the
      trader saw);
    * the claim entered its family with ``edge_type = 'new'``
      (novelty: duplicates and confirmations are excluded by default —
      change ``novel_edge_types`` to broaden explicitly).
    """

    bin_seconds: float = 900.0
    relevant_classes: tuple[str, ...] = RELEVANT_CLASSES
    min_rel_score: float = 0.5
    relevance_method: str | None = None       # pin e.g. 'ollama_llm'
    relevance_model_version: str | None = None
    novel_edge_types: tuple[str, ...] = ("new",)
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


def _qualified_relevant_claims(
    conn: sqlite3.Connection, config: DecompositionConfig
) -> dict[str, list[tuple[float, str]]]:
    """condition -> [(arrival, claim)] under the pinned contract."""
    method_clause, args = "", []
    if config.relevance_method:
        method_clause += " AND r.method = ?"
        args.append(config.relevance_method)
    if config.relevance_model_version:
        method_clause += " AND r.model_version = ?"
        args.append(config.relevance_model_version)
    edge_placeholders = ",".join("?" for _ in config.novel_edge_types)
    rows = conn.execute(
        f"""
        SELECT m.condition_id, c.first_available_at AS ts, c.claim_id,
               r.rel_class, r.rel_score, r.contract_version_seq,
               r.computed_at, r.market_id
        FROM relevance_judgments r
        JOIN news_claims c ON c.claim_id = r.claim_id
        JOIN claim_edges e ON e.claim_id = c.claim_id
        JOIN markets m ON m.market_id = r.market_id
        WHERE e.edge_type IN ({edge_placeholders}){method_clause}
        ORDER BY c.claim_id, r.market_id, r.computed_at
        """,
        (*config.novel_edge_types, *args),
    ).fetchall()
    latest: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:                       # last write wins = latest
        latest[(row["claim_id"], row["market_id"])] = row
    active_version: dict[tuple[str, float], int | None] = {}

    def version_at(market_id: str, ts: float) -> int | None:
        key = (market_id, ts)
        if key not in active_version:
            found = conn.execute(
                "SELECT version_seq FROM contract_versions WHERE "
                "market_id = ? AND effective_from <= ? "
                "ORDER BY effective_from DESC LIMIT 1",
                (market_id, ts),
            ).fetchone()
            active_version[key] = found[0] if found else None
        return active_version[key]

    out: dict[str, list[tuple[float, str]]] = {}
    for row in latest.values():
        if row["rel_class"] not in config.relevant_classes:
            continue
        if (row["rel_score"] or 0.0) < config.min_rel_score:
            continue
        if row["contract_version_seq"] != version_at(
            row["market_id"], float(row["ts"])
        ):
            continue                       # judged against a stale text
        out.setdefault(row["condition_id"], []).append(
            (float(row["ts"]), row["claim_id"])
        )
    return out


def _news_arrivals(
    conn: sqlite3.Connection, config: DecompositionConfig, spec: str
) -> dict[str, list[tuple[float, str]]]:
    """condition -> [(arrival_time, claim_id)] under the spec."""
    relevant = _qualified_relevant_claims(conn, config)
    if spec == "all_relevant":
        return relevant
    if spec == "screened_impactful":
        # impact AND semantic relevance: a background claim near a
        # liquidity transition is not news for this market
        if config.mode_run_id is None:
            return {}
        impactful = {
            (row["condition_id"], row["claim_id"])
            for row in conn.execute(
                """
                SELECT condition_id, claim_id FROM news_impact_screens
                WHERE mode_run_id = ? AND assignment_basis = ?
                  AND screen_status = 'screened'
                  AND transition_detected = 1
                """,
                (config.mode_run_id, config.screen_basis),
            )
        }
        return {
            condition: [
                (ts, claim) for ts, claim in claims
                if (condition, claim) in impactful
            ]
            for condition, claims in relevant.items()
        }
    raise ValueError(f"unknown decomposition spec: {spec}")


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
