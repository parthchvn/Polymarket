"""Order-book snapshot collector (CLOB API).

Historical books are generally unavailable; forward collection matters.
"""

from __future__ import annotations

from polymarket.collection.client import BASE_URLS, ObservingClient

BOOK_ENDPOINT = "book"


def collect_book(client: ObservingClient, *, asset: str) -> tuple[int, object]:
    return client.get(BASE_URLS["clob"], BOOK_ENDPOINT, {"token_id": asset})
