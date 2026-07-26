"""Write synthetic raw-like responses into the raw store.

Synthetic data uses the SAME raw tables and the SAME normalizer as real
data — there is no separate simplified trades table and no second schema.
"""

from __future__ import annotations

import json
import sqlite3

from polymarket.collection.raw_store import (
    finish_collector_run,
    insert_raw_response,
    record_gap,
    start_collector_run,
)
from polymarket.synthetic import scenarios as sc


def _insert(
    conn: sqlite3.Connection,
    run_id: str,
    collector: str,
    endpoint: str,
    params: dict,
    records: list[dict],
    observed_at: float,
) -> int:
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return insert_raw_response(
        conn,
        collector_run_id=run_id,
        collector=collector,
        base_url="synthetic://polymarket",
        endpoint=endpoint,
        params=params,
        requested_at=observed_at - 0.5,
        received_at=observed_at,
        http_status=200,
        headers={"content-type": "application/json", "x-synthetic": "1"},
        payload=payload,
        error_text=None,
    )


def generate_raw_world(conn: sqlite3.Connection) -> dict[str, list[int]]:
    """Insert every synthetic raw response.  Deterministic content and
    deterministic observation times (wall-clock only in run bookkeeping)."""
    ids: dict[str, list[int]] = {}
    run_id = start_collector_run(
        conn, "synthetic", {"scenario": "v1"}, note="deterministic synthetic world"
    )

    ids["markets"] = [
        _insert(conn, run_id, "markets", "markets", {},
                sc.markets_payload_v1(), sc.BASE),
        _insert(conn, run_id, "markets", "markets", {},
                sc.markets_payload_v2(), sc.BASE + 50 * sc.HOUR),
    ]
    ids["trades_taker"] = [
        _insert(conn, run_id, "trades_taker", "trades",
                {"takerOnly": "true"}, sc.taker_trades_payload(),
                sc.BASE + 61 * sc.HOUR),
    ]
    ids["trades_expanded"] = [
        _insert(conn, run_id, "trades_expanded", "trades",
                {"takerOnly": "false"}, sc.expanded_trades_payload(),
                sc.BASE + 61 * sc.HOUR),
    ]
    ids["activity"] = [
        _insert(conn, run_id, "activity", "activity", {},
                sc.activity_payload(), sc.BASE + 61 * sc.HOUR),
    ]
    ids["positions"] = [
        _insert(conn, run_id, "positions", "positions", {},
                sc.positions_payload(), sc.BASE + 70 * sc.HOUR),
    ]
    ids["books"] = [
        _insert(conn, run_id, "books", "book", {"asset": "multi"},
                records, observed_at)
        for observed_at, records in sc.books_payloads()
    ]
    ids["news"] = [
        _insert(conn, run_id, f"news:{sc.NEWS_SOURCE}", "news_feed", {},
                records, observed_at)
        for observed_at, records in sc.news_payloads()
    ]
    record_gap(conn, **sc.GAP)
    finish_collector_run(conn, run_id, "succeeded")
    return ids
