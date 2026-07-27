"""News impact screen — eq. 4.1 of the screening paper.

A news event arriving inside five-minute bin t is IMPACTFUL iff a
calm -> event mode transition happens at the bin boundary immediately
before its arrival bin (t-1 -> t) or immediately after it (t -> t+1).
The market's own liquidity reaction decides impact; semantic relevance
and direction stay with the (LLM) relevance scorers — the two are
never conflated.

The screening unit is the CLAIM (article-level arrival) x market, as
in the paper's per-release screening; event-family aggregation is a
downstream choice, and the family id is carried on every row.

Assignment basis and availability honesty:

* ``online_filtered`` (default): modes come from the forward-only
  decoder, so the mode of bin t used observations only through t.
  ``screen_available_at = end of bin t+1`` is then a TRUE claim, and
  these screens may feed live DRC.
* ``retrospective_smoothed``: modes from the full-sequence DP —
  strictly better mode estimates, but the assignment at t can depend
  on later bars, so these screens are for paper replication and
  offline analysis only, never for online availability claims.

Model availability: an ``online_filtered`` screen additionally
requires that the fitted model EXISTED before the news — the run's
``fit_cutoff`` is its ``model_effective_from``, and claims arriving
before it get ``screen_status = 'model_unavailable'`` under the online
basis (a model trained partly on bars after an event must not claim to
have screened that event online).  Retrospective screens are exempt by
definition.

Boundary coverage is three-valued: IMPACTFUL when any observed
boundary is calm->event; NOT impactful only when BOTH boundaries were
observed and neither transitions; ``partial_coverage`` when only one
boundary was observable and it did not transition (the unobserved
boundary could have carried the transition); ``insufficient_coverage``
when neither boundary was observable.  Absence of coverage is never
conflated with absence of impact.
"""

from __future__ import annotations

import sqlite3
import time

from polymarket.analysis.liquidity_modes import load_jump_model_run


def _news_targets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """(claim, condition) pairs to screen — one row per news release
    per market, as in the paper.  Later confirmations, corrections and
    materially new claims in the same family are screened
    independently; the family id rides along for aggregation."""
    return conn.execute(
        """
        SELECT DISTINCT c.claim_id,
               c.first_available_at AS news_time,
               e.event_family_id,
               m.condition_id
        FROM news_claims c
        JOIN claim_edges e ON e.claim_id = c.claim_id
        JOIN relevance_judgments r
          ON r.event_family_id = e.event_family_id
        JOIN markets m ON m.market_id = r.market_id
        ORDER BY c.first_available_at, c.claim_id
        """
    ).fetchall()


def screen_news_impact(
    conn: sqlite3.Connection,
    mode_run_id: str,
    assignment_basis: str = "online_filtered",
) -> dict:
    """Screen every (claim, market) pair under a fitted mode run.
    Idempotent: rows keyed by (run, claim, condition, basis)."""
    if assignment_basis not in (
        "online_filtered", "retrospective_smoothed"
    ):
        raise ValueError(f"unknown assignment basis: {assignment_basis}")
    label_column = (
        "mode_label_online" if assignment_basis == "online_filtered"
        else "mode_label"
    )
    run = load_jump_model_run(conn, mode_run_id)
    bin_seconds = float(run["bin_seconds"])
    labels: dict[tuple[str, float], str] = {
        (row["condition_id"], row["bin_start"]): row[label_column]
        for row in conn.execute(
            f"SELECT condition_id, bin_start, {label_column} "
            f"FROM liquidity_mode_assignments WHERE mode_run_id = ?",
            (mode_run_id,),
        )
    }
    counters = {
        "screened": 0, "impactful": 0, "insufficient_coverage": 0,
        "partial_coverage": 0, "model_unavailable": 0,
        "mode_run_id": mode_run_id,
        "assignment_basis": assignment_basis,
    }
    model_effective_from = float(run["fit_cutoff"])
    now = time.time()
    for target in _news_targets(conn):
        news_time = float(target["news_time"])
        condition_id = target["condition_id"]
        arrival_bin = (news_time // bin_seconds) * bin_seconds
        pre = labels.get((condition_id, arrival_bin - bin_seconds))
        arrival = labels.get((condition_id, arrival_bin))
        post = labels.get((condition_id, arrival_bin + bin_seconds))
        # eq 4.1: calm->event at the boundary just before the arrival
        # bin, or at the boundary just after it
        before_jump = pre == "calm" and arrival == "event"
        after_jump = arrival == "calm" and post == "event"
        evaluable_before = pre is not None and arrival is not None
        evaluable_after = arrival is not None and post is not None
        observed = (
            (1 if evaluable_before else 0)
            + (1 if evaluable_after else 0)
        )
        if (assignment_basis == "online_filtered"
                and news_time < model_effective_from):
            # the model did not exist before this news
            status, transition = "model_unavailable", 0
            counters["model_unavailable"] += 1
        elif observed == 0:
            status, transition = "insufficient_coverage", 0
            counters["insufficient_coverage"] += 1
        elif (evaluable_before and before_jump) or (
                evaluable_after and after_jump):
            status, transition = "screened", 1
            counters["screened"] += 1
            counters["impactful"] += 1
        elif observed == 2:
            status, transition = "screened", 0   # both clear: reliable
            counters["screened"] += 1
        else:
            # one boundary observed, no transition on it: the missing
            # boundary could have carried the jump
            status, transition = "partial_coverage", 0
            counters["partial_coverage"] += 1
        conn.execute(
            """
            INSERT OR REPLACE INTO news_impact_screens
                (mode_run_id, claim_id, event_family_id, condition_id,
                 assignment_basis, news_time, arrival_bin_start,
                 pre_mode_label, arrival_mode_label, post_mode_label,
                 transition_detected, impact_score, screen_status,
                 screen_available_at, model_effective_from,
                 screen_model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mode_run_id, target["claim_id"],
                target["event_family_id"], condition_id,
                assignment_basis, news_time, arrival_bin, pre, arrival,
                post, transition, float(transition), status,
                arrival_bin + 2 * bin_seconds,  # end of bin t+1
                model_effective_from, run["model_version"], now,
            ),
        )
    conn.commit()
    return counters


def impactful_news_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    cutoff: float,
    mode_run_id: str,
    assignment_basis: str = "online_filtered",
) -> list[sqlite3.Row]:
    """Screens usable at a decision time: strictly available before the
    cutoff, screened (not insufficient), for this market.  Only
    ``online_filtered`` screens make a true availability claim; asking
    for retrospective screens here is for labelled offline analysis."""
    return conn.execute(
        """
        SELECT * FROM news_impact_screens
        WHERE mode_run_id = ? AND condition_id = ?
          AND assignment_basis = ?
          AND screen_status = 'screened'
          AND screen_available_at < ?
        ORDER BY news_time
        """,
        (mode_run_id, condition_id, assignment_basis, cutoff),
    ).fetchall()
