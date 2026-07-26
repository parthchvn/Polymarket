"""Shared helpers for normalization tests."""

from __future__ import annotations

import json

from polymarket.collection.raw_store import insert_raw_response, start_collector_run
from polymarket.contracts.types import NormalizationResult


def insert_payload(db, collector, endpoint, records, params=None,
                   received_at=100.0):
    run_id = start_collector_run(db, collector)
    return insert_raw_response(
        db, collector_run_id=run_id, collector=collector, base_url="s://x",
        endpoint=endpoint, params=params or {}, requested_at=received_at - 1,
        received_at=received_at, http_status=200, headers={},
        payload=json.dumps(records).encode(),
    )


def raw_row(db, raw_id):
    return db.execute(
        "SELECT * FROM raw_responses WHERE raw_response_id=?", (raw_id,)
    ).fetchone()


def result_for(raw_id):
    return NormalizationResult(raw_response_id=raw_id, collector="t", endpoint="t")


MARKET = {
    "id": "m1", "conditionId": "c1",
    "question": "Will X happen?", "category": "test",
    "rules": "r1", "resolutionSource": "s", "createdAt": 0.0,
    "tradingEnabled": True, "closed": False, "resolved": False,
    "tokens": [
        {"token_id": "yes1", "outcome": "Yes", "sign": 1},
        {"token_id": "no1", "outcome": "No", "sign": -1},
    ],
}


def setup_market(db, record=None, received_at=10.0):
    from polymarket.normalization.markets import normalize_market_records

    raw_id = insert_payload(db, "markets", "markets", [record or MARKET],
                            received_at=received_at)
    result = result_for(raw_id)
    normalize_market_records(db, raw_row(db, raw_id), [record or MARKET], result)
    return raw_id, result
