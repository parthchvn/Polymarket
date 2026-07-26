"""Market / contract metadata and status collectors (gamma API)."""

from __future__ import annotations

from typing import Any

from polymarket.collection.client import BASE_URLS, ObservingClient
from polymarket.collection.pagination import PaginationOutcome, paginate_offset
from polymarket.collection.trades import _fetch_factory

MARKETS_ENDPOINT = "markets"


def collect_markets(
    client: ObservingClient,
    *,
    condition_ids: list[str] | None = None,
    limit: int = 100,
    max_pages: int = 100,
) -> PaginationOutcome:
    params: dict[str, Any] = {}
    if condition_ids:
        params["condition_ids"] = ",".join(condition_ids)
    return paginate_offset(
        _fetch_factory(client, MARKETS_ENDPOINT, base_key="gamma"),
        base_params=params,
        limit=limit,
        max_pages=max_pages,
    )


def collect_market_status(
    client: ObservingClient, *, condition_id: str
) -> tuple[int, Any]:
    """Single-shot forward status snapshot for one market."""
    raw_id, response = client.get(
        BASE_URLS["gamma"], MARKETS_ENDPOINT, {"condition_ids": condition_id}
    )
    return raw_id, response
