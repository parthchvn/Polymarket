"""Immutable raw response store.

Raw observations are append-only.  Every HTTP observation — including a
repeated identical payload and including failures — is a distinct row.
Raw rows are never updated or deleted during normal operation.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any, Mapping

from polymarket.collection.canonical import canonical_json, sha256_bytes


def start_collector_run(
    conn: sqlite3.Connection,
    collector: str,
    configuration: Mapping[str, Any] | None = None,
    note: str | None = None,
) -> str:
    run_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            """
            INSERT INTO collector_runs
                (collector_run_id, collector, started_at, status,
                 configuration_json, note)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (run_id, collector, time.time(), canonical_json(dict(configuration or {})), note),
        )
    return run_id


def finish_collector_run(
    conn: sqlite3.Connection, run_id: str, status: str, note: str | None = None
) -> None:
    if status not in {"succeeded", "failed", "partial"}:
        raise ValueError(f"invalid terminal status: {status}")
    with conn:
        conn.execute(
            """
            UPDATE collector_runs
            SET finished_at = ?, status = ?,
                note = COALESCE(?, note)
            WHERE collector_run_id = ?
            """,
            (time.time(), status, note, run_id),
        )


def insert_raw_response(
    conn: sqlite3.Connection,
    *,
    collector_run_id: str,
    collector: str,
    base_url: str,
    endpoint: str,
    params: Mapping[str, Any] | None,
    requested_at: float,
    received_at: float,
    http_status: int | None,
    headers: Mapping[str, str] | None,
    payload: bytes,
    error_text: str | None = None,
) -> int:
    """Store one raw observation and return raw_response_id.

    ``payload`` is the exact received bytes.  For requests that failed
    before any bytes were received, callers pass a deterministic UTF-8
    encoded error JSON payload so the failure itself remains observable.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be exact bytes")
    canonical_params = canonical_json(dict(params or {}))
    header_map = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    cur = conn.execute(
        """
        INSERT INTO raw_responses
            (collector_run_id, collector, base_url, endpoint,
             canonical_params_json, requested_at, received_at,
             http_status, response_headers_json, content_hash,
             payload, error_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            collector_run_id,
            collector,
            base_url,
            endpoint,
            canonical_params,
            requested_at,
            received_at,
            http_status,
            canonical_json(header_map),
            sha256_bytes(payload),
            payload,
            error_text,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def record_gap(
    conn: sqlite3.Connection,
    *,
    collector: str,
    surface: str,
    object_id: str | None,
    gap_start: float,
    gap_end: float | None,
    reason: str,
) -> None:
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO collector_gaps
                (collector, surface, object_id, gap_start, gap_end,
                 reason, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (collector, surface, object_id or "", gap_start, gap_end, reason, time.time()),
        )
