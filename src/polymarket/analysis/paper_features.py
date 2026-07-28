"""Paper-derived decision-context state — strict as-of accessors.

Everything here answers one question per decision: what, from the two
paper pipelines, was KNOWABLE strictly before the decision time?

* liquidity mode: the ONLINE (forward-filtered) label of the latest
  bin that had fully CLOSED before t — an assignment for bin b is
  available only from ``bin_start + bin_seconds`` onward;
* impact screens: online-basis screens with
  ``screen_available_at < t`` only (the accessor in
  ``news_impact_screen`` already enforces basis + availability + the
  model-deployment gate recorded per row);
* initial market response so far: for the latest available impactful
  claim, the log-odds move from the last pre-news close to the last
  close observed BEFORE t — both endpoints are past observations, so
  this belongs to C (realised drift AFTER the decision never does; it
  lives exclusively in the ex-post outcome layer O);
* attention loads: trailing cross-market claim volume and unrelated
  active families under the ONLINE availability policy (a relevance
  classification whose scorer had not yet run cannot shape the set);
* event-mode prevalence: the fraction of assigned markets whose
  latest closed bin before t is in event mode.

Absence is recorded as absence: when no mode run is supplied or a
signal has no coverage, the corresponding ``*_missing`` flags are 1
and values are 0 (the zero-preserving missing convention used across
the feature layer).
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_right


def _mode_run(conn: sqlite3.Connection, mode_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM liquidity_mode_runs WHERE mode_run_id = ?",
        (mode_run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown mode run: {mode_run_id}")
    return row


def liquidity_mode_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    mode_run_id: str,
) -> dict | None:
    """ONLINE label of the latest bin fully closed before t."""
    run = _mode_run(conn, mode_run_id)
    bin_seconds = float(run["bin_seconds"])
    row = conn.execute(
        """
        SELECT bin_start, mode_label_online FROM liquidity_mode_assignments
        WHERE mode_run_id = ? AND condition_id = ?
          AND bin_start + ? <= ?
        ORDER BY bin_start DESC LIMIT 1
        """,
        (mode_run_id, condition_id, bin_seconds, t),
    ).fetchone()
    if row is None:
        return None
    closed_at = float(row["bin_start"]) + bin_seconds
    return {
        "mode_label_online": row["mode_label_online"],
        "bin_start": float(row["bin_start"]),
        "closed_at": closed_at,
        "age_seconds": t - closed_at,
        "mode_run_id": mode_run_id,
    }


def event_mode_prevalence_asof(
    conn: sqlite3.Connection,
    t: float,
    mode_run_id: str,
) -> float | None:
    """Fraction of assigned markets whose latest closed bin before t is
    event mode — a market-wide 'how much is happening' proxy."""
    run = _mode_run(conn, mode_run_id)
    bin_seconds = float(run["bin_seconds"])
    rows = conn.execute(
        """
        SELECT a.condition_id, a.mode_label_online
        FROM liquidity_mode_assignments a
        JOIN (
            SELECT condition_id, MAX(bin_start) AS latest
            FROM liquidity_mode_assignments
            WHERE mode_run_id = ? AND bin_start + ? <= ?
            GROUP BY condition_id
        ) last ON last.condition_id = a.condition_id
             AND last.latest = a.bin_start
        WHERE a.mode_run_id = ?
        """,
        (mode_run_id, bin_seconds, t, mode_run_id),
    ).fetchall()
    if not rows:
        return None
    return sum(
        1 for r in rows if r["mode_label_online"] == "event"
    ) / len(rows)


def impact_screens_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    mode_run_id: str,
) -> list[dict]:
    """Online-basis impactful screens available strictly before t."""
    from polymarket.analysis.news_impact_screen import impactful_news_asof

    return [dict(r) for r in impactful_news_asof(
        conn, condition_id, t, mode_run_id,
        assignment_basis="online_filtered",
    )]


def screens_evaluated_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    mode_run_id: str,
) -> int:
    """How many online screens (impactful OR clear) were available
    before t — distinguishes 'no impact found' from 'nothing was
    screened yet'."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM news_impact_screens
        WHERE mode_run_id = ? AND condition_id = ?
          AND assignment_basis = 'online_filtered'
          AND screen_status = 'screened'
          AND screen_available_at < ?
        """,
        (mode_run_id, condition_id, t),
    ).fetchone()
    return int(row[0])


def initial_response_so_far(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    screen: dict,
    bin_seconds: float,
) -> float | None:
    """Log-odds move from the last pre-news close to the last close
    observed BEFORE t, for one impactful screen.  Both endpoints are
    strictly-past observations: this is C-legal.  None when either
    endpoint lacks coverage — missing is missing."""
    rows = conn.execute(
        """
        SELECT bin_start, logit_close FROM liquidity_bars
        WHERE condition_id = ? AND bin_seconds = ?
          AND coverage_complete = 1 AND logit_close IS NOT NULL
        ORDER BY bin_start
        """,
        (condition_id, bin_seconds),
    ).fetchall()
    if not rows:
        return None
    ends = [float(r["bin_start"]) + bin_seconds for r in rows]
    closes = [float(r["logit_close"]) for r in rows]
    pre_index = bisect_right(ends, float(screen["news_time"])) - 1
    now_index = bisect_right(ends, t) - 1
    if pre_index < 0 or now_index <= pre_index:
        return None
    return closes[now_index] - closes[pre_index]


def attention_loads_asof(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    *,
    window_seconds: float = 24 * 3600.0,
) -> dict:
    """Trailing claim volume and unrelated active families under the
    ONLINE availability policy: only relevance classifications whose
    scorer had actually run before t define the 'own family' set."""
    from polymarket.analysis.attention import _own_family_events
    from polymarket.analysis.news_returns import RELEVANT_CLASSES

    claim_count = conn.execute(
        "SELECT COUNT(*) FROM news_claims "
        "WHERE first_available_at > ? AND first_available_at <= ?",
        (t - window_seconds, t),
    ).fetchone()[0]
    families = conn.execute(
        "SELECT event_family_id, earliest_available_at FROM "
        "event_families WHERE earliest_available_at > ? "
        "AND earliest_available_at <= ?",
        (t - window_seconds, t),
    ).fetchall()
    own_events = _own_family_events(
        conn, RELEVANT_CLASSES, availability_policy="online_scored"
    ).get(condition_id, [])
    own_times = [available_at for available_at, _ in own_events]
    own_asof = {
        family for i, (_, family) in enumerate(own_events)
        if own_times[i] <= t
    }
    unrelated = {
        r["event_family_id"] for r in families
    } - own_asof
    return {
        "claim_count_24h": int(claim_count),
        "unrelated_family_count_24h": len(unrelated),
    }


def build_paper_state(
    conn: sqlite3.Connection,
    condition_id: str,
    t: float,
    mode_run_id: str | None,
) -> dict:
    """The full paper-derived state for one decision, strictly as-of t.
    ``mode_run_id=None`` returns explicit absence (features fall back
    to missing flags)."""
    state: dict = {
        "mode_run_id": mode_run_id,
        "liquidity_mode": None,
        "event_mode_prevalence": None,
        "impact_screens": [],
        "screens_evaluated": 0,
        "initial_response_so_far": None,
        "attention": attention_loads_asof(conn, condition_id, t),
    }
    if mode_run_id is None:
        return state
    run = _mode_run(conn, mode_run_id)
    state["liquidity_mode"] = liquidity_mode_asof(
        conn, condition_id, t, mode_run_id
    )
    state["event_mode_prevalence"] = event_mode_prevalence_asof(
        conn, t, mode_run_id
    )
    screens = impact_screens_asof(conn, condition_id, t, mode_run_id)
    state["impact_screens"] = screens
    state["screens_evaluated"] = screens_evaluated_asof(
        conn, condition_id, t, mode_run_id
    )
    if screens:
        latest = max(screens, key=lambda s: s["news_time"])
        state["initial_response_so_far"] = initial_response_so_far(
            conn, condition_id, t, latest, float(run["bin_seconds"])
        )
        state["latest_impactful_news_time"] = latest["news_time"]
    return state
