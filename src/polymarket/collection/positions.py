"""Position snapshot collector (data API)."""

from __future__ import annotations

from typing import Any

from polymarket.collection.client import ObservingClient
from polymarket.collection.pagination import PaginationOutcome, paginate_offset
from polymarket.collection.trades import _fetch_factory

POSITIONS_ENDPOINT = "positions"


def collect_positions(
    client: ObservingClient,
    *,
    wallet: str,
    limit: int = 100,
    max_pages: int = 100,
) -> PaginationOutcome:
    params: dict[str, Any] = {"user": wallet}
    return paginate_offset(
        _fetch_factory(client, POSITIONS_ENDPOINT),
        base_params=params,
        limit=limit,
        max_pages=max_pages,
    )
