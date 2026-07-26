"""Central normalizer: the ONE normalization path.

Collectors write only raw responses.  This class reads raw responses and
writes normalized tables.  Real and synthetic raw responses pass through
the same dispatch and the same parsers into the same schema.
"""

from __future__ import annotations

import json
import sqlite3

from polymarket.contracts.types import NormalizationResult
from polymarket.normalization.books import normalize_books
from polymarket.normalization.markets import normalize_market_records
from polymarket.normalization.news import (
    ClaimExtractor,
    RelevanceScorer,
    normalize_news,
)
from polymarket.normalization.positions import (
    normalize_activity,
    normalize_position_snapshots,
)
from polymarket.normalization.trades import (
    normalize_expanded_trades,
    normalize_taker_trades,
)


class Normalizer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        claim_extractor: ClaimExtractor | None = None,
        relevance_scorer: RelevanceScorer | None = None,
    ) -> None:
        self._conn = conn
        self._claim_extractor = claim_extractor
        self._relevance_scorer = relevance_scorer

    # ------------------------------------------------------------------
    def normalize_raw_response(self, raw_response_id: int) -> NormalizationResult:
        raw_row = self._conn.execute(
            "SELECT * FROM raw_responses WHERE raw_response_id = ?",
            (raw_response_id,),
        ).fetchone()
        if raw_row is None:
            raise KeyError(f"raw_response_id {raw_response_id} not found")
        result = NormalizationResult(
            raw_response_id=raw_response_id,
            collector=raw_row["collector"],
            endpoint=raw_row["endpoint"],
        )
        status = raw_row["http_status"]
        if status is None or status >= 400:
            result.errors.append(f"skipping failed response (status={status})")
            return result
        try:
            body = json.loads(bytes(raw_row["payload"]))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            result.errors.append(f"undecodable payload: {exc}")
            return result
        if isinstance(body, list):
            records = body
        elif isinstance(body, dict):
            # {"data": [...]} envelopes unwrap; bare object payloads
            # (e.g. a single CLOB order book) normalize as one record
            records = body.get("data", body)
        else:
            records = []
        if not isinstance(records, list):
            records = [records]

        collector = str(raw_row["collector"])
        endpoint = str(raw_row["endpoint"])
        params = json.loads(raw_row["canonical_params_json"] or "{}")

        if endpoint == "trades" or collector.startswith("trades"):
            taker_only = (
                str(params.get("takerOnly", "")).lower() == "true"
                or collector == "trades_taker"
            )
            if taker_only:
                normalize_taker_trades(self._conn, raw_row, records, result)
            else:
                normalize_expanded_trades(self._conn, raw_row, records, result)
        elif endpoint == "markets" or collector in {"markets", "market_status"}:
            normalize_market_records(self._conn, raw_row, records, result)
        elif endpoint == "activity" or collector == "activity":
            normalize_activity(self._conn, raw_row, records, result)
        elif endpoint == "positions" or collector == "positions":
            normalize_position_snapshots(self._conn, raw_row, records, result)
        elif endpoint == "book" or collector == "books":
            normalize_books(self._conn, raw_row, records, result)
        elif collector.startswith("news") or endpoint.startswith("news"):
            normalize_news(
                self._conn,
                raw_row,
                records,
                result,
                extractor=self._claim_extractor,
                scorer=self._relevance_scorer,
            )
        else:
            result.errors.append(
                f"no parser for collector={collector!r} endpoint={endpoint!r}"
            )
        return result

    # ------------------------------------------------------------------
    def normalize_all(
        self, *, collector_order: tuple[str, ...] = ("markets",)
    ) -> list[NormalizationResult]:
        """Normalize every successful raw response.

        Market metadata is normalized first so that outcome-token mappings
        and resolution evidence exist before trades, positions and news are
        parsed.
        """
        rows = self._conn.execute(
            """
            SELECT raw_response_id, collector, endpoint FROM raw_responses
            WHERE http_status IS NOT NULL AND http_status < 400
            ORDER BY raw_response_id
            """
        ).fetchall()

        def priority(row: sqlite3.Row) -> tuple[int, int]:
            collector = str(row["collector"])
            for i, prefix in enumerate(collector_order):
                if collector.startswith(prefix) or row["endpoint"] == prefix:
                    return (i, row["raw_response_id"])
            return (len(collector_order), row["raw_response_id"])

        results = []
        for row in sorted(rows, key=priority):
            results.append(self.normalize_raw_response(row["raw_response_id"]))
        return results
