"""Normalize wallet activity into position events, and position payloads
into snapshots.

Expected activity record shape:

.. code-block:: json

    {
      "type": "TRADE" | "SPLIT" | "MERGE" | "REDEEM" | "CONVERT" | ...,
      "proxyWallet": "0xwallet",
      "conditionId": "0xabc",
      "asset": "t-yes",          // TRADE / REDEEM
      "side": "BUY",             // TRADE only
      "size": 10.0,
      "price": 0.62,             // TRADE only
      "timestamp": 1700000100,
      "transactionHash": "0x..."
    }

Semantics (binary markets):
* TRADE:  token += side_sign * size, collateral -= side_sign * size * price
* SPLIT:  +q on both outcome tokens, collateral -= q
* MERGE:  -q on both outcome tokens, collateral += q
* REDEEM: -q winning token, collateral += q  (requires resolution evidence)
* anything else: recorded with ``accounting_confidence = 'unresolved'`` —
  unknown semantics are never guessed.

Missing data is not zero: unresolved events carry NULL token/collateral
changes, not substantive zeros.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
from polymarket.contracts.types import NormalizationResult
from polymarket.normalization.ids import position_event_id, raw_record_hash

KNOWN_TYPES = {"TRADE", "SPLIT", "MERGE", "REDEEM"}


def _outcome_assets(
    conn: sqlite3.Connection, condition_id: str, ts: float
) -> dict[int, str]:
    rows = conn.execute(
        """
        SELECT asset, outcome_sign FROM outcome_tokens
        WHERE condition_id = ? AND mapping_effective_from <= ?
        ORDER BY mapping_effective_from
        """,
        (condition_id, ts),
    ).fetchall()
    return {int(r["outcome_sign"]): r["asset"] for r in rows}


def _winning_asset_asof(
    conn: sqlite3.Connection, condition_id: str, ts: float
) -> tuple[str | None, int | None]:
    """Winning asset from resolved market status effective before ts."""
    row = conn.execute(
        """
        SELECT msv.winning_asset,
               (SELECT COUNT(*) FROM market_status_versions m2
                 WHERE m2.market_id = msv.market_id
                   AND m2.effective_from <= msv.effective_from) AS version_n
        FROM market_status_versions msv
        JOIN markets mk ON mk.market_id = msv.market_id
        WHERE mk.condition_id = ? AND msv.resolved = 1
          AND msv.effective_from <= ?
        ORDER BY msv.effective_from DESC LIMIT 1
        """,
        (condition_id, ts),
    ).fetchone()
    if row is None or row["winning_asset"] is None:
        return None, None
    return row["winning_asset"], int(row["version_n"])


def _insert_event(
    conn: sqlite3.Connection,
    result: NormalizationResult,
    **fields: Any,
) -> None:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO position_events
            (position_event_id, wallet, condition_id, asset, ts, event_type,
             signed_token_change, collateral_change, transaction_hash,
             transaction_log_index, accounting_confidence, resolution_version,
             is_combo, raw_response_id, raw_record_index, raw_record_hash,
             parser_version, schema_version, normalized_at)
        VALUES (:position_event_id, :wallet, :condition_id, :asset, :ts,
                :event_type, :signed_token_change, :collateral_change,
                :transaction_hash, :transaction_log_index,
                :accounting_confidence, :resolution_version, :is_combo,
                :raw_response_id, :raw_record_index, :raw_record_hash,
                :parser_version, :schema_version, :normalized_at)
        """,
        fields,
    )
    (result.add_inserted if cur.rowcount else result.add_ignored)(
        "position_events"
    )


def normalize_activity(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    for index, record in enumerate(records):
        rec_hash = raw_record_hash(record)
        base = dict(
            wallet=record.get("proxyWallet"),
            condition_id=record.get("conditionId"),
            ts=record.get("timestamp"),
            transaction_hash=record.get("transactionHash"),
            transaction_log_index=record.get("logIndex"),
            resolution_version=None,
            is_combo=1 if record.get("isCombo") else 0,
            raw_response_id=raw_id,
            raw_record_index=index,
            raw_record_hash=rec_hash,
            parser_version=PARSER_VERSION,
            schema_version=SCHEMA_VERSION,
            normalized_at=now,
        )
        if base["wallet"] is None or base["condition_id"] is None or base["ts"] is None:
            result.unresolved.append(
                {"table": "position_events", "index": index,
                 "reason": "missing wallet/condition/timestamp"}
            )
            continue
        base["ts"] = float(base["ts"])
        event_type = str(record.get("type", "")).upper()
        size = record.get("size")
        size = float(size) if size is not None else None

        if event_type == "TRADE" and size is not None:
            side = str(record.get("side", "")).upper()
            price = record.get("price")
            side_sign = 1.0 if side == "BUY" else -1.0
            signed_token_change = side_sign * size
            collateral_change = (
                -side_sign * size * float(price) if price is not None else None
            )
            _insert_event(
                conn, result,
                position_event_id=position_event_id(raw_id, index),
                event_type="TRADE", asset=record.get("asset"),
                signed_token_change=signed_token_change,
                collateral_change=collateral_change,
                accounting_confidence="exact" if price is not None else "inferred",
                **base,
            )
        elif event_type in {"SPLIT", "MERGE"} and size is not None:
            assets = _outcome_assets(conn, base["condition_id"], base["ts"])
            token_sign = 1.0 if event_type == "SPLIT" else -1.0
            if 1 in assets and -1 in assets:
                for sub_index, sign in enumerate((1, -1)):
                    _insert_event(
                        conn, result,
                        position_event_id=position_event_id(raw_id, index, sub_index),
                        event_type=event_type, asset=assets[sign],
                        signed_token_change=token_sign * size,
                        # attach the collateral leg once, on the first sub-event
                        collateral_change=(-token_sign * size if sub_index == 0 else 0.0),
                        accounting_confidence="exact",
                        **base,
                    )
            else:
                _insert_event(
                    conn, result,
                    position_event_id=position_event_id(raw_id, index),
                    event_type=event_type, asset=None,
                    signed_token_change=None, collateral_change=None,
                    accounting_confidence="unresolved",
                    **base,
                )
                result.unresolved.append(
                    {"table": "position_events", "index": index,
                     "reason": f"{event_type} without complete outcome mapping"}
                )
        elif event_type == "REDEEM" and size is not None:
            winning_asset, resolution_version = _winning_asset_asof(
                conn, base["condition_id"], base["ts"]
            )
            asset = record.get("asset") or winning_asset
            if winning_asset is not None and asset == winning_asset:
                base["resolution_version"] = resolution_version
                _insert_event(
                    conn, result,
                    position_event_id=position_event_id(raw_id, index),
                    event_type="REDEEM", asset=winning_asset,
                    signed_token_change=-size, collateral_change=size,
                    accounting_confidence="exact",
                    **base,
                )
            else:
                _insert_event(
                    conn, result,
                    position_event_id=position_event_id(raw_id, index),
                    event_type="REDEEM", asset=asset,
                    signed_token_change=None, collateral_change=None,
                    accounting_confidence="unresolved",
                    **base,
                )
                result.unresolved.append(
                    {"table": "position_events", "index": index,
                     "reason": "REDEEM without resolution evidence"}
                )
        else:
            # CONVERT, transfers, unknown types: do not guess semantics.
            _insert_event(
                conn, result,
                position_event_id=position_event_id(raw_id, index),
                event_type=event_type or "UNKNOWN", asset=record.get("asset"),
                signed_token_change=None, collateral_change=None,
                accounting_confidence="unresolved",
                **base,
            )
            result.unresolved.append(
                {"table": "position_events", "index": index,
                 "reason": f"unresolved event type {event_type!r}"}
            )
    conn.commit()


def normalize_position_snapshots(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    observed_at = float(raw_row["received_at"])
    for index, record in enumerate(records):
        wallet = record.get("proxyWallet")
        asset = record.get("asset")
        size = record.get("size")
        if wallet is None or asset is None or size is None:
            result.unresolved.append(
                {"table": "position_snapshots", "index": index,
                 "reason": "missing wallet/asset/size"}
            )
            continue
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO position_snapshots
                (wallet, asset, observed_at, reported_size, source,
                 raw_response_id, parser_version, schema_version,
                 normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (wallet, asset, observed_at, float(size), "data-api",
             raw_id, PARSER_VERSION, SCHEMA_VERSION, now),
        )
        (result.add_inserted if cur.rowcount else result.add_ignored)(
            "position_snapshots"
        )
    conn.commit()
