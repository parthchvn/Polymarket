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
    # Historical time-window parameter semantics are unvalidated against
    # the live API; pass through only when explicitly requested.
    if start_ts is not None:
        params["after"] = int(start_ts)
    if end_ts is not None:
        params["before"] = int(end_ts)
    return paginate_offset(
        _fetch_factory(client, TRADES_ENDPOINT),
        base_params=params,
        limit=limit,
        max_pages=max_pages,
    )
