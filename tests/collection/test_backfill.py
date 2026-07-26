"""29.4 backfill tests."""

from polymarket.collection.backfill import (
    pending_windows,
    plan_windows,
    run_backfill,
)
from polymarket.collection.pagination import PageResult, PaginationOutcome


def _outcome(records, status="complete", note=None):
    out = PaginationOutcome(status=status, note=note)
    out.pages = [PageResult(1, records, {})]
    out.record_count = len(records)
    return out


def test_half_open_windows_and_overlap():
    windows = plan_windows(0.0, 100.0, 40.0, overlap_seconds=10.0)
    assert windows[0] == (60.0, 100.0)
    # backward with overlap: next upper = previous lower + overlap
    assert windows[1][1] == 70.0
    assert windows[-1][0] == 0.0


def test_complete_window(db):
    report = run_backfill(
        db, collector="trades", object_id="c1",
        windows=[(0.0, 10.0)],
        fetch_window=lambda s, e: _outcome([{"ts": 5.0}]),
        extract_ts=lambda r: r["ts"],
    )
    assert report.windows_complete == 1
    row = db.execute("SELECT * FROM backfill_windows").fetchone()
    assert row["status"] == "complete"
    assert row["record_count"] == 1
    assert row["observed_min_ts"] == row["observed_max_ts"] == 5.0


def test_timestamp_outside_half_open_window_marks_incomplete(db):
    report = run_backfill(
        db, collector="trades", object_id="c1",
        windows=[(0.0, 10.0)],
        fetch_window=lambda s, e: _outcome([{"ts": 10.0}]),  # == end: outside
        extract_ts=lambda r: r["ts"],
    )
    assert report.windows_incomplete == 1
    gap = db.execute("SELECT * FROM collector_gaps").fetchone()
    assert gap is not None and gap["reason"]


def test_failed_window_records_gap_and_is_resumable(db):
    run_backfill(
        db, collector="trades", object_id="c1",
        windows=[(0.0, 10.0)],
        fetch_window=lambda s, e: _outcome([], status="failed", note="boom"),
        extract_ts=lambda r: None,
    )
    assert pending_windows(db, "trades", "c1") == [(0.0, 10.0)]
    # safe restart: rerun same window successfully
    run_backfill(
        db, collector="trades", object_id="c1",
        windows=[(0.0, 10.0)],
        fetch_window=lambda s, e: _outcome([{"ts": 3.0}]),
        extract_ts=lambda r: r["ts"],
    )
    assert pending_windows(db, "trades", "c1") == []
