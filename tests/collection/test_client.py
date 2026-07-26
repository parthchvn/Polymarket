"""HTTP client tests with mocked transports."""

import httpx

from polymarket.collection.client import ObservingClient
from polymarket.collection.raw_store import start_collector_run


def _client(db, handler, **kw):
    run_id = start_collector_run(db, "test")
    return ObservingClient(
        db, "test", run_id,
        transport=httpx.MockTransport(handler), sleep=lambda _t: None, **kw,
    )


def test_success_records_raw(db):
    def handler(request):
        return httpx.Response(200, json=[{"ok": True}])

    with _client(db, handler) as client:
        raw_id, response = client.get("http://api", "trades", {"a": 1})
    assert response is not None and response.status_code == 200
    row = db.execute(
        "SELECT * FROM raw_responses WHERE raw_response_id=?", (raw_id,)
    ).fetchone()
    assert row["http_status"] == 200
    assert row["endpoint"] == "trades"


def test_retry_on_500_records_every_attempt(db):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500)
        return httpx.Response(200, json=[])

    with _client(db, handler) as client:
        raw_id, response = client.get("http://api", "trades")
    assert response.status_code == 200
    assert db.execute("SELECT COUNT(*) FROM raw_responses").fetchone()[0] == 3


def test_transport_error_stores_deterministic_error_payload(db):
    def handler(request):
        raise httpx.ConnectError("nope")

    with _client(db, handler, max_retries=1) as client:
        raw_id, response = client.get("http://api", "trades")
    assert response is None
    row = db.execute(
        "SELECT * FROM raw_responses WHERE raw_response_id=?", (raw_id,)
    ).fetchone()
    assert row["http_status"] is None
    assert b"ConnectError" in bytes(row["payload"])
