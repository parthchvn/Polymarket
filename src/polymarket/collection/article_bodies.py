"""Article body download — turning headline-only RSS rows into real
article text.

The Google News RSS collector stores the headline as the body (the
feed carries nothing else), so body-level LLM extraction had nothing
to read.  This module downloads the actual articles:

1. Google News links are not HTTP redirects — they are encoded
   article ids resolved via ``googlenewsdecoder`` to the publisher
   URL; direct publisher URLs pass through unchanged.
2. The publisher page is fetched with browser headers and the main
   text extracted with ``trafilatura``.
3. Provenance: every attempt (success or failure) is recorded as a
   raw response under the ``news:article-body`` collector, keyed by
   article id — failed articles are not retried forever, and the raw
   HTML is preserved for re-extraction.
4. On success the article row gains its real body and an honest
   ``download_completed_at``: downstream claim extraction uses
   max(first_observed_at, download_completed_at) as availability, so
   a claim from a late-downloaded body never pretends to have been
   readable before the text was actually in hand.

Some publishers block automated fetches (503s); those are recorded
and skipped — partial coverage with provenance beats fake coverage.
"""

from __future__ import annotations

import sqlite3
import time
import urllib.request

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 "
        "Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

MIN_EXTRACTED_CHARS = 400


def resolve_and_fetch(url: str, *, timeout: float = 25.0) -> tuple[str, str, str]:
    """(final_url, html, extracted_text) for one article URL; raises
    on any failure (caller records it)."""
    import trafilatura

    final_url = url
    if "news.google.com" in url:
        from googlenewsdecoder import gnewsdecoder

        decoded = gnewsdecoder(url, interval=1)
        if not decoded.get("status"):
            raise ValueError(
                f"google news decode failed: {decoded.get('message')}"
            )
        final_url = decoded["decoded_url"]
    request = urllib.request.Request(final_url, headers=FETCH_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", "ignore")
    text = trafilatura.extract(html) or ""
    if len(text) < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"extracted only {len(text)} chars from {final_url}"
        )
    return final_url, html, text


def _attempted(conn: sqlite3.Connection, article_id: str) -> bool:
    from polymarket.collection.canonical import canonical_json

    row = conn.execute(
        "SELECT 1 FROM raw_responses WHERE collector = "
        "'news:article-body' AND canonical_params_json = ? LIMIT 1",
        (canonical_json({"article_id": article_id}),),
    ).fetchone()
    return row is not None


def download_article_bodies(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    timeout: float = 25.0,
    fetcher=resolve_and_fetch,
    order: str = "newest",
) -> dict:
    """Download real bodies for articles still carrying the headline
    placeholder.  Newest first by default (fresh news feeds the online
    screens).  Every attempt is recorded; failures are skipped and not
    retried."""
    from polymarket.collection.raw_store import (
        finish_collector_run,
        insert_raw_response,
        start_collector_run,
    )

    direction = "DESC" if order == "newest" else "ASC"
    rows = conn.execute(
        f"""
        SELECT article_id, source_url, headline, first_observed_at
        FROM news_articles
        WHERE source_url LIKE 'http%'
          AND (body IS NULL OR body = headline
               OR LENGTH(body) < {MIN_EXTRACTED_CHARS})
        ORDER BY first_observed_at {direction}
        """
    ).fetchall()
    run_id = start_collector_run(
        conn, "news:article-body", {"limit": limit}
    )
    downloaded = 0
    failed = 0
    skipped_attempted = 0
    failures: list[str] = []
    for row in rows:
        if limit is not None and downloaded + failed >= limit:
            break
        article_id = row["article_id"]
        if _attempted(conn, article_id):
            skipped_attempted += 1
            continue
        requested_at = time.time()
        try:
            final_url, html, text = fetcher(
                row["source_url"], timeout=timeout
            )
        except Exception as exc:                # noqa: BLE001
            failed += 1
            failures.append(f"{article_id}: {exc}")
            insert_raw_response(
                conn, collector_run_id=run_id,
                collector="news:article-body",
                base_url=row["source_url"], endpoint="article_body",
                params={"article_id": article_id},
                requested_at=requested_at, received_at=time.time(),
                http_status=None, headers={},
                payload=b"", error_text=str(exc)[:500],
            )
            conn.commit()
            continue
        insert_raw_response(
            conn, collector_run_id=run_id,
            collector="news:article-body",
            base_url=final_url, endpoint="article_body",
            params={"article_id": article_id},
            requested_at=requested_at, received_at=time.time(),
            http_status=200, headers={},
            payload=html.encode(),
        )
        conn.execute(
            "UPDATE news_articles SET body = ?, "
            "download_completed_at = ? WHERE article_id = ?",
            (text, time.time(), article_id),
        )
        downloaded += 1
        conn.commit()
    finish_collector_run(
        conn, run_id, "succeeded" if not failures else "partial"
    )
    conn.commit()
    return {
        "articles_placeholder": len(rows),
        "downloaded": downloaded,
        "failed": failed,
        "skipped_previous_attempts": skipped_attempted,
        "failed_examples": failures[:5],
    }
