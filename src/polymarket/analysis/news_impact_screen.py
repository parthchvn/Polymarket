"""News impact screen — eq. 4.1 of the screening paper.

A news event arriving inside five-minute bin t is IMPACTFUL iff a
calm -> event mode transition happens at the bin boundary immediately
before its arrival bin (t-1 -> t) or immediately after it (t -> t+1).
The market's own liquidity reaction decides impact; semantic relevance
and direction stay with the (LLM) relevance scorers — the two are
never conflated.

Strict availability: the screen needs the mode of bin t+1, which is
only computable once that bin closes, so

    screen_available_at = arrival_bin_end + bin_seconds

(the end of bin t+1).  A wallet decision may condition on the screen
only when ``screen_available_at < decision_time`` — enforced by the
DRC integration, recorded here.

Families whose surrounding bins lack complete assignments get
``screen_status = 'insufficient_coverage'`` instead of a silent skip:
absence of coverage is recorded, never conflated with absence of
impact.
"""

from __future__ import annotations

import sqlite3
import time

from polymarket.analysis.liquidity_modes import load_jump_model_run


def _news_targets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """(event family, condition) pairs to screen: every family holding
    any relevance judgment for the market, with the family's earliest
    availability as the news time."""
    return conn.execute(
        """
        SELECT DISTINCT f.event_family_id,
               f.earliest_available_at AS news_time,
               m.condition_id
        FROM event_families f
        JOIN relevance_judgments r
          ON r.event_family_id = f.event_family_id
        JOIN markets m ON m.market_id = r.market_id
        ORDER BY f.earliest_available_at, f.event_family_id
        """
    ).fetchall()


def screen_news_impact(
    conn: sqlite3.Connection,
    mode_run_id: str,
) -> dict:
    """Screen every (family, market) pair under a fitted mode run.
    Idempotent: rows are keyed by (mode_run_id, family, condition)."""
    run = load_jump_model_run(conn, mode_run_id)
    bin_seconds = float(run["bin_seconds"])
    labels: dict[tuple[str, float], str] = {
        (row["condition_id"], row["bin_start"]): row["mode_label"]
        for row in conn.execute(
            "SELECT condition_id, bin_start, mode_label "
            "FROM liquidity_mode_assignments WHERE mode_run_id = ?",
            (mode_run_id,),
        )
    }
    counters = {
        "screened": 0, "impactful": 0, "insufficient_coverage": 0,
        "mode_run_id": mode_run_id,
    }
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
        if not evaluable_before and not evaluable_after:
            status, transition = "insufficient_coverage", 0
            counters["insufficient_coverage"] += 1
        else:
            status = "screened"
            transition = int(bool(
                (evaluable_before and before_jump)
                or (evaluable_after and after_jump)
            ))
            counters["screened"] += 1
            counters["impactful"] += transition
        conn.execute(
            """
            INSERT OR REPLACE INTO news_impact_screens
                (mode_run_id, event_family_id, condition_id, news_time,
                 arrival_bin_start, pre_mode_label, arrival_mode_label,
                 post_mode_label, transition_detected, impact_score,
                 screen_status, screen_available_at,
                 screen_model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mode_run_id, target["event_family_id"], condition_id,
                news_time, arrival_bin, pre, arrival, post, transition,
                float(transition), status,
                arrival_bin + 2 * bin_seconds,  # end of bin t+1
                run["model_version"], now,
            ),
        )
    conn.commit()
    return counters


def impactful_news_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    cutoff: float,
    mode_run_id: str,
) -> list[sqlite3.Row]:
    """Screens usable at a decision time: strictly available before the
    cutoff, screened (not insufficient), for this market."""
    return conn.execute(
        """
        SELECT * FROM news_impact_screens
        WHERE mode_run_id = ? AND condition_id = ?
          AND screen_status = 'screened'
          AND screen_available_at < ?
        ORDER BY news_time
        """,
        (mode_run_id, condition_id, cutoff),
    ).fetchall()
