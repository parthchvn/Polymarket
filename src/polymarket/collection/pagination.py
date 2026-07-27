"""Generic pagination engine.

Supports offset-based and cursor-based pagination, with safeguards:
maximum page count, repeated-page detection, record counting, and safe
interruption/resume through an externally supplied start state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from polymarket.collection.canonical import sha256_bytes

# fetch(page_params) -> (raw_response_id, records or None)
FetchFn = Callable[[dict[str, Any]], tuple[int, list[Any] | None]]


@dataclass
class PageResult:
    raw_response_id: int
    records: list[Any]
    params: dict[str, Any]


@dataclass
class PaginationOutcome:
    pages: list[PageResult] = field(default_factory=list)
    record_count: int = 0
    status: str = "complete"  # complete | incomplete | failed
    note: str | None = None
    next_state: dict[str, Any] | None = None  # for resume


def paginate_offset(
    fetch: FetchFn,
    *,
    base_params: dict[str, Any] | None = None,
    limit: int = 100,
    max_pages: int = 1000,
    start_offset: int = 0,
    stop_predicate: Callable[[list[Any]], bool] | None = None,
) -> PaginationOutcome:
    """Offset-based pagination with repeated-page detection.

    ``stop_predicate(records)`` is evaluated AFTER a page is stored:
    returning True ends pagination with the page included, so
    newest-first feeds can stop once a page reaches already-collected
    territory without losing the overlap records."""
    outcome = PaginationOutcome()
    seen_hashes: set[str] = set()
    offset = start_offset
    for _ in range(max_pages):
        params = dict(base_params or {})
        params.update({"limit": limit, "offset": offset})
        raw_id, records = fetch(params)
        if records is None:
            outcome.status = "failed"
            outcome.note = f"fetch failed at offset {offset}"
            outcome.next_state = {"offset": offset}
            return outcome
        page_hash = sha256_bytes(
            json.dumps(records, sort_keys=True, default=str).encode("utf-8")
        )
        if records and page_hash in seen_hashes:
            outcome.status = "incomplete"
            outcome.note = f"repeated page detected at offset {offset}"
            outcome.next_state = {"offset": offset}
            return outcome
        seen_hashes.add(page_hash)
        outcome.pages.append(PageResult(raw_id, records, params))
        outcome.record_count += len(records)
        if len(records) < limit:
            return outcome
        if stop_predicate is not None and stop_predicate(records):
            outcome.note = "stopped by predicate (cursor reached)"
            return outcome
        offset += limit
    outcome.status = "incomplete"
    outcome.note = f"max_pages={max_pages} reached"
    outcome.next_state = {"offset": offset}
    return outcome


def paginate_cursor(
    fetch: FetchFn,
    *,
    base_params: dict[str, Any] | None = None,
    cursor_param: str = "next_cursor",
    extract_cursor: Callable[[list[Any]], str | None] | None = None,
    max_pages: int = 1000,
    start_cursor: str | None = None,
) -> PaginationOutcome:
    """Cursor-based pagination.  ``extract_cursor`` derives the next cursor
    from the returned records (API specific); ``None`` ends pagination."""
    outcome = PaginationOutcome()
    cursor = start_cursor
    seen_cursors: set[str] = set()
    for _ in range(max_pages):
        params = dict(base_params or {})
        if cursor is not None:
            params[cursor_param] = cursor
        raw_id, records = fetch(params)
        if records is None:
            outcome.status = "failed"
            outcome.note = "fetch failed"
            outcome.next_state = {"cursor": cursor}
            return outcome
        outcome.pages.append(PageResult(raw_id, records, params))
        outcome.record_count += len(records)
        next_cursor = extract_cursor(records) if extract_cursor else None
        if next_cursor is None:
            return outcome
        if next_cursor in seen_cursors:
            outcome.status = "incomplete"
            outcome.note = f"repeated cursor {next_cursor!r}"
            outcome.next_state = {"cursor": next_cursor}
            return outcome
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    outcome.status = "incomplete"
    outcome.note = f"max_pages={max_pages} reached"
    outcome.next_state = {"cursor": cursor}
    return outcome
