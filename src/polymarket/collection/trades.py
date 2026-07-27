"""Trade collectors.

Two distinct views (working assumptions from prior live probes; see
docs/TRADE_SEMANTICS.md):

* ``takerOnly=true``  — approximately one taker-side record per
  transaction; leading source for canonical executions.
* ``takerOnly=false`` — expanded wallet-side / counterparty records
  revealing maker activity; source for actor trade legs.

The two views are not interchangeable.
"""

from __future__ import annotations

import json
from typing import Any

from polymarket.collection.client import BASE_URLS, ObservingClient
from polymarket.collection.pagination import PaginationOutcome, paginate_offset

TRADES_ENDPOINT = "trades"


def _fetch_factory(client: ObservingClient, endpoint: str, base_key: str = "data"):
    def fetch(params: dict[str, Any]) -> tuple[int, list[Any] | None]:
        raw_id, response = client.get(BASE_URLS[base_key], endpoint, params)
        if response is None or response.status_code >= 400:
            return raw_id, None
        try:
            body = json.loads(response.content)
        except json.JSONDecodeError:
            return raw_id, None
        records = body if isinstance(body, list) else body.get("data", [])
        return raw_id, records

    return fetch


def collect_trades(
    client: ObservingClient,
    *,
    condition_id: str,
    taker_only: bool,
    start_ts: float | None = None,
    end_ts: float | None = None,
    limit: int = 100,
    max_pages: int = 1000,
) -> PaginationOutcome:
    params: dict[str, Any] = {
        "market": condition_id,
        "takerOnly": "true" if taker_only else "false",
    }
    # VALIDATED against production (2026-07): the data-api /trades
    # endpoint IGNORES every server-side time-filter param (after,
    # startTs, from, fromTimestamp).  Results are newest-first, so
    # incremental collection uses a client-side early stop: paginate
    # until a page reaches already-collected territory (start_ts), and
    # keep that page for overlap — dedup happens at normalization via
    # record fingerprints.
    stop_predicate = None
    if start_ts is not None:
        def stop_predicate(records: list[Any]) -> bool:
            timestamps = [
                float(r.get("timestamp", 0)) for r in records
                if isinstance(r, dict)
            ]
            return bool(timestamps) and min(timestamps) <= start_ts
    if end_ts is not None:
        params["before"] = int(end_ts)  # still unvalidated; explicit only
    return paginate_offset(
        _fetch_factory(client, TRADES_ENDPOINT),
        base_params=params,
        limit=limit,
        max_pages=max_pages,
        stop_predicate=stop_predicate,
    )
