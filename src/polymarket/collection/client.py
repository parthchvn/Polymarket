"""Reusable observing HTTP client.

Every request produces exactly one raw_responses row per final attempt
outcome: success, HTTP error, or transport failure.  The raw observation
is never discarded.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Mapping

import httpx

from polymarket.collection.canonical import canonical_json
from polymarket.collection.raw_store import insert_raw_response

BASE_URLS = {
    "data": "https://data-api.polymarket.com",
    "gamma": "https://gamma-api.polymarket.com",
    "clob": "https://clob.polymarket.com",
}


class ObservingClient:
    """Records every observation to the raw store and returns its id."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        collector: str,
        collector_run_id: str,
        *,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        self._conn = conn
        self._collector = collector
        self._run_id = collector_run_id
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ObservingClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def get(
        self,
        base_url: str,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[int, httpx.Response | None]:
        """GET with retry.  Returns (raw_response_id, response-or-None).

        Retries on transport errors and 5xx with exponential backoff.
        The final attempt (successful or not) is stored.  Intermediate
        failed attempts are also stored so the observation history is
        complete.
        """
        url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        last_raw_id = -1
        for attempt in range(self._max_retries + 1):
            requested_at = time.time()
            try:
                response = self._client.get(url, params=dict(params or {}))
                received_at = time.time()
                last_raw_id = insert_raw_response(
                    self._conn,
                    collector_run_id=self._run_id,
                    collector=self._collector,
                    base_url=base_url,
                    endpoint=endpoint,
                    params=params,
                    requested_at=requested_at,
                    received_at=received_at,
                    http_status=response.status_code,
                    headers=dict(response.headers),
                    payload=response.content,
                    error_text=(
                        None if response.status_code < 400
                        else f"http {response.status_code}"
                    ),
                )
                if response.status_code < 500:
                    return last_raw_id, response
            except httpx.HTTPError as exc:
                received_at = time.time()
                error_payload = canonical_json(
                    {"error": type(exc).__name__, "detail": str(exc)}
                ).encode("utf-8")
                last_raw_id = insert_raw_response(
                    self._conn,
                    collector_run_id=self._run_id,
                    collector=self._collector,
                    base_url=base_url,
                    endpoint=endpoint,
                    params=params,
                    requested_at=requested_at,
                    received_at=received_at,
                    http_status=None,
                    headers={},
                    payload=error_payload,
                    error_text=str(exc),
                )
            if attempt < self._max_retries:
                self._sleep(self._backoff_base * (2**attempt))
        return last_raw_id, None
