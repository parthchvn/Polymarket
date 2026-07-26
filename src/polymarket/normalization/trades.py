"""Normalize trade payloads.

Expected record shape (both views; see docs/TRADE_SEMANTICS.md):

.. code-block:: json

    {
      "id": "optional-source-record-id",
      "transactionHash": "0x...",
      "logIndex": 3,
      "occurrence": 0,
      "conditionId": "0xabc",
      "asset": "t-yes",
      "proxyWallet": "0xwallet",
      "side": "BUY",
      "size": 10.0,
      "price": 0.62,
      "timestamp": 1700000100
    }

``takerOnly=true`` records become canonical executions (market-wide
volume, price paths).  ``takerOnly=false`` records become actor trade
legs (actor-level analysis).  The views are not interchangeable and actor
legs must never define market-wide volume.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
from polymarket.contracts.types import NormalizationResult
from polymarket.normalization.ids import (
    actor_leg_id,
    candidate_fingerprint,
    execution_id,
    raw_record_hash,
)

PRICE_TOLERANCE = 1e-9


def outcome_sign_for(
    conn: sqlite3.Connection, condition_id: str, asset: str, ts: float
) -> tuple[int | None, str | None]:
    """Latest outcome-sign mapping effective at or before ``ts``."""
    row = conn.execute(
        """
        SELECT outcome_sign, outcome_label FROM outcome_tokens
        WHERE condition_id = ? AND asset = ? AND mapping_effective_from <= ?
        ORDER BY mapping_effective_from DESC LIMIT 1
        """,
        (condition_id, asset, ts),
    ).fetchone()
    if row is None:
        return None, None
    return int(row["outcome_sign"]), row["outcome_label"]


def to_positive(
    price: float, side: str, sign: int
) -> tuple[float, str]:
    """Convert an observed (price, side) on any outcome token to the
    positive-proposition convention."""
    if sign == 1:
        return price, side
    positive_price = 1.0 - price
    positive_side = "SELL" if side == "BUY" else "BUY"
    return positive_price, positive_side


def _valid_price(price: float) -> bool:
    return -PRICE_TOLERANCE <= price <= 1.0 + PRICE_TOLERANCE


def normalize_taker_trades(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    """takerOnly=true records -> canonical_executions."""
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    for index, record in enumerate(records):
        rec_hash = raw_record_hash(record)
        try:
            condition_id = record["conditionId"]
            asset = record["asset"]
            side = str(record["side"]).upper()
            size = float(record["size"])
            price = float(record["price"])
            ts = float(record["timestamp"])
            tx = record["transactionHash"]
        except (KeyError, TypeError, ValueError) as exc:
            result.unresolved.append(
                {"table": "canonical_executions", "index": index,
                 "reason": f"malformed record: {exc}"}
            )
            continue
        if not _valid_price(price):
            result.unresolved.append(
                {"table": "canonical_executions", "index": index,
                 "reason": f"price outside [0,1] tolerance: {price}"}
            )
            continue
        sign, _label = outcome_sign_for(conn, condition_id, asset, ts)
        if sign is None:
            result.unresolved.append(
                {"table": "canonical_executions", "index": index,
                 "reason": "no outcome-sign mapping", "asset": asset}
            )
            continue
        positive_price, positive_side = to_positive(price, side, sign)
        reconciliation_status = "direct" if sign == 1 else "complemented"
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO canonical_executions
                (execution_id, source_record_id, transaction_hash,
                 transaction_log_index, transaction_occurrence, condition_id,
                 positive_price, positive_side, size, notional, ts,
                 taker_wallet, raw_response_id, raw_record_index,
                 raw_record_hash, reconciliation_status,
                 parser_version, schema_version, normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id(raw_id, index), record.get("id"), tx,
                record.get("logIndex"), record.get("occurrence"), condition_id,
                positive_price, positive_side, size, size * positive_price, ts,
                record.get("proxyWallet"), raw_id, index, rec_hash,
                reconciliation_status, PARSER_VERSION, SCHEMA_VERSION, now,
            ),
        )
        (result.add_inserted if cur.rowcount else result.add_ignored)(
            "canonical_executions"
        )
    conn.commit()


def normalize_expanded_trades(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    """takerOnly=false records -> actor_trade_legs.

    Roles start as ``unknown`` and are set by the reconciliation pass.
    Repeated legitimate source records survive: the primary key is built
    from raw provenance, and a shared transaction hash alone never
    deduplicates.
    """
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    for index, record in enumerate(records):
        rec_hash = raw_record_hash(record)
        try:
            condition_id = record["conditionId"]
            asset = record["asset"]
            wallet = record["proxyWallet"]
            side = str(record["side"]).upper()
            size = float(record["size"])
            price = float(record["price"])
            ts = float(record["timestamp"])
            tx = record["transactionHash"]
        except (KeyError, TypeError, ValueError) as exc:
            result.unresolved.append(
                {"table": "actor_trade_legs", "index": index,
                 "reason": f"malformed record: {exc}"}
            )
            continue
        if not _valid_price(price):
            result.unresolved.append(
                {"table": "actor_trade_legs", "index": index,
                 "reason": f"price outside [0,1] tolerance: {price}"}
            )
            continue
        sign, label = outcome_sign_for(conn, condition_id, asset, ts)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO actor_trade_legs
                (actor_leg_id, source_record_id, candidate_fingerprint,
                 transaction_hash, transaction_log_index,
                 transaction_occurrence, proxy_wallet, condition_id, asset,
                 outcome_label, outcome_sign, side, size, price, ts,
                 liquidity_role, role_confidence,
                 raw_response_id, raw_record_index, raw_record_hash,
                 parser_version, schema_version, normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'unknown', 'unreconciled', ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_leg_id(raw_id, index), record.get("id"),
                candidate_fingerprint(tx, wallet, asset, side, size, price),
                tx, record.get("logIndex"), record.get("occurrence"),
                wallet, condition_id, asset, label, sign, side, size, price,
                ts, raw_id, index, rec_hash,
                PARSER_VERSION, SCHEMA_VERSION, now,
            ),
        )
        (result.add_inserted if cur.rowcount else result.add_ignored)(
            "actor_trade_legs"
        )
    conn.commit()
