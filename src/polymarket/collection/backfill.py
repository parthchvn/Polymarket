"""Bounded-window backfill over half-open intervals ``[start, end)``.

Windows are tracked in ``backfill_windows`` and only marked ``complete``
after validation.  Failed or truncated windows become ``incomplete`` and a
collector gap is recorded so downstream analysis can see coverage holes.

Historical completeness is never claimed unless demonstrated: the live
API's historical time-window parameter semantics are a working assumption
(see docs/RESEARCH_ASSUMPTIONS.md).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

from polymarket.collection.pagination import PaginationOutcome
from polymarket.collection.raw_store import record_gap

# window_fetch(window_start, window_end) -> PaginationOutcome
WindowFetch = Callable[[float, float], PaginationOutcome]


@dataclass
class BackfillReport:
    windows_attempted: int = 0
    windows_complete: int = 0
    windows_incomplete: int = 0
    windows_failed: int = 0
    record_count: int = 0


def _upsert_window(
    conn: sqlite3.Connection,
    collector: str,
    object_id: str,
    window_start: float,
    window_end: float,
    **updates: Any,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO backfill_windows
            (collector, object_id, window_start, window_end, status)
        VALUES (?, ?, ?, ?, 'pending')
        """,
        (collector, object_id, window_start, window_end),
    )
    if updates:
        sets = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"""
            UPDATE backfill_windows SET {sets}
            WHERE collector = ? AND object_id = ?
              AND window_start = ? AND window_end = ?
            """,
            (*updates.values(), collector, object_id, window_start, window_end),
        )
    conn.commit()


def pending_windows(
    conn: sqlite3.Connection, collector: str, object_id: str
) -> list[tuple[float, float]]:
    rows = conn.execute(
        """
        SELECT window_start, window_end FROM backfill_windows
        WHERE collector = ? AND object_id = ?
          AND status IN ('pending', 'incomplete', 'failed', 'running')
        ORDER BY window_end DESC
        """,
        (collector, object_id),
    ).fetchall()
    return [(r["window_start"], r["window_end"]) for r in rows]


def plan_windows(
    start_time: float,
    end_time: float,
    window_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[tuple[float, float]]:
    """Plan half-open windows moving backward from ``end_time``."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap_seconds >= window_seconds:
        raise ValueError("overlap must be smaller than window")
    windows: list[tuple[float, float]] = []
    upper = end_time
    while upper > start_time:
        lower = max(start_time, upper - window_seconds)
        windows.append((lower, upper))
        if lower <= start_time:
            break
        upper = lower + overlap_seconds
    return windows


def run_backfill(
    conn: sqlite3.Connection,
    *,
    collector: str,
    object_id: str,
    windows: list[tuple[float, float]],
    fetch_window: WindowFetch,
    extract_ts: Callable[[Any], float | None],
) -> BackfillReport:
    """Execute backfill windows, storing raw responses via the fetch
    callable and validating observed timestamps against window bounds."""
    report = BackfillReport()
    for window_start, window_end in windows:
        report.windows_attempted += 1
        _upsert_window(
            conn, collector, object_id, window_start, window_end,
            status="running", started_at=time.time(),
        )
        outcome = fetch_window(window_start, window_end)
        timestamps = [
            ts
            for page in outcome.pages
            for record in page.records
            if (ts := extract_ts(record)) is not None
        ]
        observed_min = min(timestamps) if timestamps else None
        observed_max = max(timestamps) if timestamps else None
        out_of_window = any(
            ts < window_start or ts >= window_end for ts in timestamps
        )
        if outcome.status == "complete" and not out_of_window:
            status = "complete"
            report.windows_complete += 1
        elif outcome.status == "failed":
            status = "failed"
            report.windows_failed += 1
        else:
            status = "incomplete"
            report.windows_incomplete += 1
        note = outcome.note
        if out_of_window:
            status = "incomplete"
            note = (note or "") + " observed timestamps outside half-open window"
        if status != "complete":
            record_gap(
                conn,
                collector=collector,
                surface="backfill",
                object_id=object_id,
                gap_start=window_start,
                gap_end=window_end,
                reason=note or status,
            )
        _upsert_window(
            conn, collector, object_id, window_start, window_end,
            status=status,
            completed_at=time.time() if status == "complete" else None,
            page_count=len(outcome.pages),
            record_count=outcome.record_count,
            observed_min_ts=observed_min,
            observed_max_ts=observed_max,
            note=note,
        )
        report.record_count += outcome.record_count
    return report
