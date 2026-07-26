"""29.2 raw-store tests."""

import json

import pytest

from polymarket.collection.canonical import canonical_json, sha256_bytes
from polymarket.collection.raw_store import (
    insert_raw_response,
    start_collector_run,
)


def _insert(db, run_id, payload: bytes, params=None, status=200, error=None):
    return insert_raw_response(
        db, collector_run_id=run_id, collector="test", base_url="http://x",
        endpoint="e", params=params or {}, requested_at=1.0, received_at=2.0,
        http_status=status, headers={"X-Test": "yes"}, payload=payload,
        error_text=error,
    )


def test_exact_bytes_preserved(db):
    run_id = start_collector_run(db, "test")
    payload = b'{"a": 1, "weird  spacing": true}\n'
    raw_id = _insert(db, run_id, payload)
    row = db.execute(
        "SELECT payload, content_hash FROM raw_responses WHERE raw_response_id=?",
        (raw_id,),
    ).fetchone()
    assert bytes(row["payload"]) == payload
    assert row["content_hash"] == sha256_bytes(payload)


def test_hash_equality_and_difference(db):
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert sha256_bytes(b"abc") != sha256_bytes(b"abd")


def test_repeated_observations_are_separate_rows(db):
    run_id = start_collector_run(db, "test")
    payload = b"[1,2,3]"
    id1 = _insert(db, run_id, payload)
    id2 = _insert(db, run_id, payload)
    assert id1 != id2
    rows = db.execute(
        "SELECT content_hash FROM raw_responses ORDER BY raw_response_id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["content_hash"] == rows[1]["content_hash"]


def test_canonical_param_order_deterministic(db):
    run_id = start_collector_run(db, "test")
    id1 = _insert(db, run_id, b"x", params={"b": 2, "a": 1})
    id2 = _insert(db, run_id, b"x", params={"a": 1, "b": 2})
    rows = db.execute(
        "SELECT canonical_params_json FROM raw_responses "
        "WHERE raw_response_id IN (?, ?)", (id1, id2),
    ).fetchall()
    assert rows[0][0] == rows[1][0] == canonical_json({"a": 1, "b": 2})


def test_failed_responses_retained_with_headers(db):
    run_id = start_collector_run(db, "test")
    raw_id = _insert(db, run_id, b'{"error":"boom"}', status=None,
                     error="ConnectError")
    row = db.execute(
        "SELECT * FROM raw_responses WHERE raw_response_id=?", (raw_id,)
    ).fetchone()
    assert row["http_status"] is None
    assert row["error_text"] == "ConnectError"
    assert json.loads(row["response_headers_json"]) == {"x-test": "yes"}


def test_payload_must_be_bytes(db):
    run_id = start_collector_run(db, "test")
    with pytest.raises(TypeError):
        _insert(db, run_id, "not-bytes")  # type: ignore[arg-type]
