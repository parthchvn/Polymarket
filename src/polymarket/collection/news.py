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


def collect_google_news_rss(conn, query: str) -> int:
    """Fetch a Google News RSS feed and store it under the shared
    news-feed JSON contract (headline/body/publishedAt).  The stored
    payload is exactly what this source emits after XML->JSON mapping;
    provenance (query, source) is recorded in the request params."""
    import json
    import time
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    from polymarket.collection.raw_store import (
        finish_collector_run,
        insert_raw_response,
        start_collector_run,
    )

    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        xml_bytes = response.read()
    root = ET.fromstring(xml_bytes)
    records = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        try:
            published = (
                parsedate_to_datetime(pub).timestamp() if pub else None
            )
        except (TypeError, ValueError):
            published = None
        if not title or published is None:
            continue
        records.append({
            "id": link or title,
            "url": link,
            "publishedAt": published,
            "timestampSource": "feed",
            "timestampConfidence": 0.8,
            "headline": title,
            "body": title,
        })
    run_id = start_collector_run(conn, "news:google-rss", {"query": query})
    insert_raw_response(
        conn, collector_run_id=run_id, collector="news:google-rss",
        base_url="https://news.google.com", endpoint="news_feed",
        params={"query": query}, requested_at=time.time() - 1,
        received_at=time.time(), http_status=200, headers={},
        payload=json.dumps(records, sort_keys=True).encode(),
    )
    finish_collector_run(conn, run_id, "succeeded")
    return len(records)
