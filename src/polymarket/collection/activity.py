"""Wallet activity collector (trades, splits, merges, redeems, ...)."""

from __future__ import annotations

from typing import Any

from polymarket.collection.client import ObservingClient
from polymarket.collection.pagination import PaginationOutcome, paginate_offset
from polymarket.collection.trades import _fetch_factory

ACTIVITY_ENDPOINT = "activity"


def collect_activity(
    client: ObservingClient,
    *,
    wallet: str,
    limit: int = 100,
    max_pages: int = 1000,
) -> PaginationOutcome:
    params: dict[str, Any] = {"user": wallet}
    return paginate_offset(
        _fetch_factory(client, ACTIVITY_ENDPOINT),
        base_params=params,
        limit=limit,
        max_pages=max_pages,
    )
