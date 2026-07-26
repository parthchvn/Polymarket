"""Configurable news feed collector.

First implementation targets narrow, timestamp-reliable sources.  Each
feed is configured with a base URL and endpoint; response format is
normalized downstream.  ``first_observed_at`` is set from the collector's
receive time, never trusted from the publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polymarket.collection.client import ObservingClient


@dataclass(frozen=True)
class NewsFeedConfig:
    source_id: str
    base_url: str
    endpoint: str
    params: dict[str, Any] | None = None


def collect_news_feed(
    client: ObservingClient, feed: NewsFeedConfig
) -> tuple[int, object]:
    return client.get(feed.base_url, feed.endpoint, feed.params or {})
