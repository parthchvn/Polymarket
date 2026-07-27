"""Normalize market metadata payloads.

Expected record shape (see docs/DATA_CONTRACT.md):

.. code-block:: json

    {
      "id": "mkt-1",
      "conditionId": "0xabc",
      "question": "...",
      "category": "...",
      "rules": "...",
      "resolutionSource": "...",
      "resolutionTime": 1700000000,
      "createdAt": ..., "closedAt": ..., "resolvedAt": ...,
      "tradingEnabled": true, "closed": false, "resolved": false,
      "winningAsset": null, "isCombo": false,
      "tokens": [
        {"token_id": "t-yes", "outcome": "Yes", "sign": 1},
        {"token_id": "t-no", "outcome": "No", "sign": -1}
      ]
    }

Outcomes are NOT assumed to be literally YES/NO.  Sign resolution order:
explicit ``sign`` field (confidence ``explicit``), Yes/No-style labels
(``high``), positional first/second (``assumed``).
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
from polymarket.contracts.types import NormalizationResult
from polymarket.normalization.ids import namespace_id, raw_record_hash, sha256_bytes

_POSITIVE_LABELS = {"yes", "true", "will", "positive"}
_NEGATIVE_LABELS = {"no", "false", "wont", "won't", "negative"}


def _token_sign(token: dict[str, Any], index: int) -> tuple[int | None, str]:
    if "sign" in token and token["sign"] in (-1, 1):
        return int(token["sign"]), "explicit"
    label = str(token.get("outcome", "")).strip().lower()
    if label in _POSITIVE_LABELS:
        return 1, "high"
    if label in _NEGATIVE_LABELS:
        return -1, "high"
    if index == 0:
        return 1, "assumed"
    if index == 1:
        return -1, "assumed"
    return None, "unknown"


def _iso_epoch(value: Any) -> float | None:
    """ISO-8601 timestamps (production gamma) to epoch seconds."""
    if value is None or isinstance(value, (int, float)):
        return value
    try:
        from datetime import datetime

        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError):
        return None


def _adapt_production_market(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt a production gamma record to the parser's canonical shape.

    Production gamma serves ``clobTokenIds`` and ``outcomes`` as JSON
    strings, ``description`` instead of ``rules``, ``acceptingOrders``
    instead of ``tradingEnabled`` and ISO-8601 timestamps.  Synthetic /
    canonical records pass through untouched.  The RAW payload is never
    modified — adaptation happens at parse time only.
    """
    if "tokens" in record or "clobTokenIds" not in record:
        return record
    adapted = dict(record)
    try:
        token_ids = json.loads(record.get("clobTokenIds") or "[]")
        outcomes = json.loads(record.get("outcomes") or "[]")
    except (TypeError, ValueError):
        token_ids, outcomes = [], []
    adapted["tokens"] = [
        {
            "token_id": token_id,
            "outcome": outcomes[i] if i < len(outcomes) else None,
        }
        for i, token_id in enumerate(token_ids)
    ]
    if "rules" not in adapted:
        adapted["rules"] = record.get("description")
    if "tradingEnabled" not in adapted:
        adapted["tradingEnabled"] = bool(
            record.get("acceptingOrders", record.get("active", True))
        )
    if "resolved" not in adapted:
        adapted["resolved"] = (
            str(record.get("umaResolutionStatus") or "").lower() == "resolved"
        )
    if "resolutionTime" not in adapted:
        adapted["resolutionTime"] = _iso_epoch(record.get("endDate"))
    adapted["createdAt"] = _iso_epoch(record.get("createdAt"))
    adapted["closedAt"] = _iso_epoch(record.get("closedAt"))
    return adapted


def normalize_market_records(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    now = time.time()
    observed_at = float(raw_row["received_at"])
    raw_id = int(raw_row["raw_response_id"])
    for index, record in enumerate(records):
        rec_hash = raw_record_hash(record)
        record = _adapt_production_market(record)
        condition_id = record.get("conditionId")
        market_id = record.get("id") or condition_id
        if not condition_id or not market_id:
            result.unresolved.append(
                {"table": "markets", "index": index, "reason": "missing ids"}
            )
            continue
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO markets
                (market_id, condition_id, category, question, created_at,
                 closed_at, resolved_at, is_combo,
                 raw_response_id, raw_record_index, raw_record_hash,
                 parser_version, schema_version, normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id, condition_id, record.get("category"),
                record.get("question"), record.get("createdAt"),
                record.get("closedAt"), record.get("resolvedAt"),
                1 if record.get("isCombo") else 0,
                raw_id, index, rec_hash,
                PARSER_VERSION, SCHEMA_VERSION, now,
            ),
        )
        (result.add_inserted if cur.rowcount else result.add_ignored)("markets")

        # ---- outcome tokens -------------------------------------------------
        for t_index, token in enumerate(record.get("tokens", [])):
            asset = token.get("token_id") or token.get("asset")
            if not asset:
                continue
            sign, confidence = _token_sign(token, t_index)
            if sign is None:
                result.unresolved.append(
                    {"table": "outcome_tokens", "asset": asset,
                     "reason": "unresolvable outcome sign"}
                )
                continue
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO outcome_tokens
                    (condition_id, asset, outcome_label, outcome_sign,
                     mapping_effective_from, mapping_confidence,
                     raw_response_id, raw_record_index, raw_record_hash,
                     parser_version, schema_version, normalized_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    condition_id, asset, token.get("outcome"), sign,
                    record.get("createdAt") or 0.0, confidence,
                    raw_id, index, rec_hash,
                    PARSER_VERSION, SCHEMA_VERSION, now,
                ),
            )
            (result.add_inserted if cur.rowcount else result.add_ignored)(
                "outcome_tokens"
            )

        # ---- contract versions ---------------------------------------------
        contract_payload = {
            "question": record.get("question"),
            "rules_text": record.get("rules"),
            "resolution_source": record.get("resolutionSource"),
            "resolution_time": record.get("resolutionTime"),
        }
        content_hash = sha256_bytes(
            namespace_id("contract", *contract_payload.values()).encode()
        )
        latest = conn.execute(
            """
            SELECT version_seq, content_hash FROM contract_versions
            WHERE market_id = ? ORDER BY version_seq DESC LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if latest is None or latest["content_hash"] != content_hash:
            version_seq = 1 if latest is None else latest["version_seq"] + 1
            conn.execute(
                """
                INSERT INTO contract_versions
                    (market_id, version_seq, effective_from, first_observed_at,
                     question, rules_text, resolution_source, resolution_time,
                     content_hash, raw_response_id,
                     parser_version, schema_version, normalized_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, version_seq, observed_at, observed_at,
                    contract_payload["question"], contract_payload["rules_text"],
                    contract_payload["resolution_source"],
                    contract_payload["resolution_time"],
                    content_hash, raw_id,
                    PARSER_VERSION, SCHEMA_VERSION, now,
                ),
            )
            result.add_inserted("contract_versions")
        else:
            result.add_ignored("contract_versions")

        # ---- market status versions ----------------------------------------
        status_tuple = (
            1 if record.get("tradingEnabled", True) else 0,
            1 if record.get("closed") else 0,
            1 if record.get("resolved") else 0,
            record.get("winningAsset"),
        )
        latest_status = conn.execute(
            """
            SELECT trading_enabled, closed, resolved, winning_asset
            FROM market_status_versions WHERE market_id = ?
            ORDER BY effective_from DESC LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        changed = latest_status is None or tuple(latest_status) != status_tuple
        if changed:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO market_status_versions
                    (market_id, effective_from, first_observed_at,
                     trading_enabled, closed, resolved, winning_asset,
                     raw_response_id, parser_version, schema_version,
                     normalized_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (market_id, observed_at, observed_at, *status_tuple,
                 raw_id, PARSER_VERSION, SCHEMA_VERSION, now),
            )
            (result.add_inserted if cur.rowcount else result.add_ignored)(
                "market_status_versions"
            )
        else:
            result.add_ignored("market_status_versions")
    conn.commit()


def derive_market_state_from_executions(
    conn: sqlite3.Connection,
    condition_id: str,
    bucket_seconds: float = 60.0,
) -> int:
    """Derive market_state rows from canonical executions only.

    Actor-leg volume is never mixed into canonical market volume because
    expanded counterparty legs can double-count executions.
    """
    now = time.time()
    rows = conn.execute(
        """
        SELECT ts, positive_price, size, notional
        FROM canonical_executions WHERE condition_id = ?
        ORDER BY ts
        """,
        (condition_id,),
    ).fetchall()
    buckets: dict[float, dict[str, float]] = {}
    for row in rows:
        bucket = (row["ts"] // bucket_seconds) * bucket_seconds
        b = buckets.setdefault(bucket, {"volume": 0.0, "last_price": None})
        b["volume"] += row["size"]
        b["last_price"] = row["positive_price"]
    inserted = 0
    for bucket_ts, b in sorted(buckets.items()):
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO market_state
                (condition_id, ts, positive_price, volume, spread, depth,
                 imbalance, state_source, coverage_complete, raw_response_id,
                 parser_version, schema_version, normalized_at)
            VALUES (?, ?, ?, ?, NULL, NULL, NULL, 'executions', 1, NULL,
                    ?, ?, ?)
            """,
            (condition_id, bucket_ts + bucket_seconds, b["last_price"],
             b["volume"], PARSER_VERSION, SCHEMA_VERSION, now),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def derive_market_state_from_books(
    conn: sqlite3.Connection,
    condition_id: str,
) -> int:
    """Derive mid-price market_state rows from order-book snapshots
    (state_source='book_mid').

    Execution-derived state oscillates with taker flow (buys print near
    the ask, sells near the bid), which can make flow look mechanically
    contrarian.  Book-mid state is flow-independent: the positive
    outcome token's (best_bid + best_ask) / 2 at each snapshot, with the
    snapshot's spread, depth and imbalance carried through.  Rows
    coexist with execution-derived state under the
    (condition_id, ts, state_source) primary key; readers take the
    latest state before a cutoff regardless of source.
    """
    rows = conn.execute(
        """
        SELECT b.observed_at, b.best_bid, b.best_ask, b.spread,
               b.bid_depth, b.ask_depth, b.imbalance
        FROM order_book_snapshots b
        JOIN outcome_tokens o ON o.asset = b.asset
        WHERE o.condition_id = ? AND o.outcome_sign = 1
          AND b.best_bid IS NOT NULL AND b.best_ask IS NOT NULL
        ORDER BY b.observed_at
        """,
        (condition_id,),
    ).fetchall()
    written = 0
    now = time.time()
    for observed_at, bid, ask, spread, bid_depth, ask_depth, imbalance in rows:
        mid = (bid + ask) / 2.0
        depth = (bid_depth or 0.0) + (ask_depth or 0.0)
        conn.execute(
            """
            INSERT OR REPLACE INTO market_state
                (condition_id, ts, positive_price, volume, spread, depth,
                 imbalance, state_source, coverage_complete,
                 raw_response_id, parser_version, schema_version,
                 normalized_at)
            VALUES (?, ?, ?, NULL, ?, ?, ?, 'book_mid', 1, NULL, ?, ?, ?)
            """,
            (
                condition_id, observed_at, mid, spread, depth, imbalance,
                PARSER_VERSION, SCHEMA_VERSION, now,
            ),
        )
        written += 1
    conn.commit()
    return written
