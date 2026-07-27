"""Normalize order-book payloads into order_book_snapshots.

Expected record shape:

.. code-block:: json

    {
      "asset": "t-yes",
      "bids": [{"price": 0.61, "size": 100.0}, ...],
      "asks": [{"price": 0.63, "size": 80.0}, ...]
    }

Missing sides remain NULL — missing data is not zero.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
from polymarket.contracts.types import NormalizationResult


def _levels(side: Any) -> list[tuple[float, float]]:
    if not isinstance(side, list):
        return []
    out = []
    for level in side:
        try:
            out.append((float(level["price"]), float(level["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def normalize_books(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
) -> None:
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    observed_at = float(raw_row["received_at"])
    for index, record in enumerate(records):
        asset = (
            record.get("asset") or record.get("token_id")
            or record.get("asset_id")  # production CLOB key
        )
        if asset is None:
            result.unresolved.append(
                {"table": "order_book_snapshots", "index": index,
                 "reason": "missing asset"}
            )
            continue
        bids = _levels(record.get("bids"))
        asks = _levels(record.get("asks"))
        best_bid = max((p for p, _ in bids), default=None)
        best_ask = min((p for p, _ in asks), default=None)
        spread = (
            best_ask - best_bid
            if best_bid is not None and best_ask is not None
            else None
        )
        bid_depth = sum(s for _, s in bids) if bids else None
        ask_depth = sum(s for _, s in asks) if asks else None
        imbalance = None
        if bid_depth is not None and ask_depth is not None and bid_depth + ask_depth > 0:
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        # the screening paper's book-size variable is the volume at the
        # BEST bid/ask, not summed depth — store both, never conflated
        best_bid_size = next(
            (size for price, size in bids if price == best_bid), None
        )
        best_ask_size = next(
            (size for price, size in asks if price == best_ask), None
        )
        try:
            tick_size = (
                float(record["tick_size"]) if "tick_size" in record else None
            )
        except (TypeError, ValueError):
            tick_size = None
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO order_book_snapshots
                (asset, observed_at, best_bid, best_ask, spread, bid_depth,
                 ask_depth, imbalance, best_bid_size, best_ask_size,
                 tick_size, raw_response_id, parser_version,
                 schema_version, normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (asset, observed_at, best_bid, best_ask, spread, bid_depth,
             ask_depth, imbalance, best_bid_size, best_ask_size, tick_size,
             raw_id, PARSER_VERSION, SCHEMA_VERSION, now),
        )
        (result.add_inserted if cur.rowcount else result.add_ignored)(
            "order_book_snapshots"
        )
    conn.commit()
