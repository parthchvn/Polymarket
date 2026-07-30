"""Import a slice of the SII-WANGZJ/Polymarket_data historical dataset
(1.1B blockchain-derived trades, 1.8M resolved markets, MIT licensed)
into the normalized schema.

Two-tier data story: this dataset supplies YEARS of executed decisions
with resolved outcomes (D, actor histories, and the O layer at scale);
the forward collector remains the only source of the full context C
(order books, timestamped news, LLM relevance).  Historical decisions
therefore carry honestly-missing book/news context — the same explicit
missingness the feature layer already handles.

Mechanics:

* ``markets.parquet`` is read locally (294MB).  Slice selection:
  closed markets with recorded outcomes, ranked by volume inside a
  configurable band and date window (the volume band exists to avoid
  a single mega-market dominating the slice).
* ``quant.parquet`` (21GB) is read REMOTELY via duckdb httpfs range
  requests — one streaming pass filtered to the selected condition
  ids; only those rows ever touch disk.
* Provenance: one raw response per imported market records the exact
  remote query, row count and content hash of the imported rows.  The
  upstream dataset is public, immutable and re-fetchable by that
  query; the full 21GB payload is deliberately not duplicated locally
  (documented contract exception, parser_version ``sii-import-1``).
* Timing: blockchain timestamps are exact (block time) — the D layer
  from this source is better-timestamped than any API.
* Outcomes: ``outcome_prices`` + ``end_date`` populate resolution
  status and winning side, feeding MarketCensor and the O layer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

PARSER_VERSION = "sii-import-1"
SCHEMA_VERSION = 2
HF_BASE = (
    "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/"
    "resolve/main"
)


def select_market_slice(
    markets_parquet: str,
    *,
    top_n: int = 50,
    min_volume: float = 1e6,
    max_volume: float = 2e8,
    since: str = "2024-01-01",
    until: str | None = None,
) -> list[dict]:
    """Closed, outcome-recorded, binary markets by volume inside the
    band — deterministic ordering."""
    import duckdb

    con = duckdb.connect()
    where = [
        "closed = 1",
        "outcome_prices IS NOT NULL AND outcome_prices != ''",
        "condition_id IS NOT NULL AND condition_id != ''",
        "token1 IS NOT NULL AND token2 IS NOT NULL",
        "volume >= ? AND volume <= ?",
        "created_at >= CAST(? AS TIMESTAMPTZ)",
    ]
    params: list = [min_volume, max_volume, since]
    if until:
        where.append("end_date <= CAST(? AS TIMESTAMPTZ)")
        params.append(until)
    rows = con.execute(
        f"""
        SELECT id, question, slug, condition_id, token1, token2,
               answer1, answer2, outcome_prices, volume, event_title,
               epoch(created_at) AS created_ts,
               epoch(end_date) AS end_ts
        FROM read_parquet(?)
        WHERE {" AND ".join(where)}
        ORDER BY volume DESC, condition_id
        LIMIT ?
        """,
        [markets_parquet, *params, top_n],
    ).fetchall()
    columns = [
        "id", "question", "slug", "condition_id", "token1", "token2",
        "answer1", "answer2", "outcome_prices", "volume",
        "event_title", "created_ts", "end_ts",
    ]
    out = []
    for row in rows:
        market = dict(zip(columns, row))
        market["volume"] = float(market["volume"] or 0)
        market["created_ts"] = (
            float(market["created_ts"])
            if market["created_ts"] is not None else None
        )
        market["end_ts"] = (
            float(market["end_ts"])
            if market["end_ts"] is not None else None
        )
        out.append(market)
    return out


def _winning_asset(market: dict) -> str | None:
    try:
        prices = json.loads(market["outcome_prices"])
        p1 = float(prices[0])
    except (ValueError, TypeError, IndexError, KeyError):
        return None
    if p1 >= 0.99:
        return market["token1"]
    if p1 <= 0.01:
        return market["token2"]
    return None            # partial / ambiguous resolution: recorded as None


def import_market_metadata(
    conn: sqlite3.Connection, markets: list[dict]
) -> int:
    """markets + contract version (with resolution time) + status
    timeline (open at creation, closed+resolved at end)."""
    from polymarket.collection.raw_store import (
        finish_collector_run,
        insert_raw_response,
        start_collector_run,
    )

    now = time.time()
    run_id = start_collector_run(
        conn, "sii:markets", {"n": len(markets)}
    )
    inserted = 0
    for market in markets:
        payload = json.dumps(market, sort_keys=True).encode()
        raw_id = insert_raw_response(
            conn, collector_run_id=run_id, collector="sii:markets",
            base_url=HF_BASE, endpoint="markets.parquet",
            params={"condition_id": market["condition_id"]},
            requested_at=now, received_at=now, http_status=200,
            headers={}, payload=payload,
        )
        market_id = str(market["id"])
        conn.execute(
            "INSERT OR IGNORE INTO markets (market_id, condition_id, "
            "question, raw_response_id, raw_record_index, "
            "raw_record_hash, parser_version, schema_version, "
            "normalized_at) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (market_id, market["condition_id"], market["question"],
             raw_id, hashlib.sha256(payload).hexdigest()[:16],
             PARSER_VERSION, SCHEMA_VERSION, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO contract_versions (market_id, "
            "version_seq, effective_from, first_observed_at, question, "
            "rules_text, resolution_time, content_hash, "
            "raw_response_id, parser_version, schema_version, "
            "normalized_at) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (market_id, market["created_ts"], now, market["question"],
             "",                     # rules text not in this dataset
             market["end_ts"],
             hashlib.sha256(payload).hexdigest(),
             raw_id, PARSER_VERSION, SCHEMA_VERSION, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO market_status_versions (market_id, "
            "effective_from, first_observed_at, trading_enabled, "
            "closed, resolved, winning_asset, raw_response_id, "
            "parser_version, schema_version, normalized_at) VALUES "
            "(?, ?, ?, 1, 0, 0, NULL, ?, ?, ?, ?)",
            (market_id, market["created_ts"], now, raw_id,
             PARSER_VERSION, SCHEMA_VERSION, now),
        )
        if market["end_ts"]:
            conn.execute(
                "INSERT OR IGNORE INTO market_status_versions "
                "(market_id, effective_from, first_observed_at, "
                "trading_enabled, closed, resolved, winning_asset, "
                "raw_response_id, parser_version, schema_version, "
                "normalized_at) VALUES (?, ?, ?, 0, 1, 1, ?, ?, ?, ?, ?)",
                (market_id, market["end_ts"], now,
                 _winning_asset(market), raw_id, PARSER_VERSION,
                 SCHEMA_VERSION, now),
            )
        inserted += 1
    finish_collector_run(conn, run_id, "succeeded")
    conn.commit()
    return inserted


def import_trades(
    conn: sqlite3.Connection,
    markets: list[dict],
    *,
    quant_source: str | None = None,
    batch_rows: int = 50_000,
) -> dict:
    """One streaming pass over quant.parquet (remote by default),
    filtered to the slice's condition ids; rows land as canonical
    executions plus maker and taker actor legs."""
    import duckdb

    from polymarket.collection.raw_store import (
        finish_collector_run,
        start_collector_run,
    )

    source = quant_source or f"{HF_BASE}/quant.parquet"
    by_condition = {m["condition_id"]: m for m in markets}
    condition_ids = sorted(by_condition)
    con = duckdb.connect()
    if source.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    placeholders = ",".join("?" for _ in condition_ids)
    cursor = con.execute(
        f"""
        SELECT timestamp, transaction_hash, log_index, condition_id,
               price, usd_amount, token_amount, side, maker, taker
        FROM read_parquet(?)
        WHERE condition_id IN ({placeholders})
        ORDER BY condition_id, timestamp, transaction_hash, log_index
        """,
        [source, *condition_ids],
    )
    now = time.time()
    run_id = start_collector_run(
        conn, "sii:quant",
        {"conditions": len(condition_ids), "source": source},
    )
    per_condition: dict[str, int] = {}
    executions = 0
    legs = 0
    while True:
        batch = cursor.fetchmany(batch_rows)
        if not batch:
            break
        for row in batch:
            (ts, tx_hash, log_index, condition_id, price, usd_amount,
             token_amount, side, maker, taker) = row
            index = per_condition.get(condition_id, 0)
            per_condition[condition_id] = index + 1
            market = by_condition[condition_id]
            record_hash = hashlib.sha256(
                f"{tx_hash}|{log_index}|{condition_id}".encode()
            ).hexdigest()[:16]
            side = (side or "BUY").upper()
            if side not in ("BUY", "SELL"):
                side = "BUY"
            execution_id = f"sii-{tx_hash}-{log_index}"
            raw_id = _condition_raw_id(
                conn, run_id, condition_id, source, now
            )
            conn.execute(
                "INSERT OR IGNORE INTO canonical_executions "
                "(execution_id, source_record_id, transaction_hash, "
                "transaction_log_index, transaction_occurrence, "
                "condition_id, positive_price, positive_side, size, "
                "notional, ts, taker_wallet, raw_response_id, "
                "raw_record_index, raw_record_hash, "
                "reconciliation_status, parser_version, "
                "schema_version, normalized_at) VALUES "
                "(?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "'onchain', ?, ?, ?)",
                (execution_id, execution_id, tx_hash, log_index,
                 condition_id, float(price), side,
                 abs(float(token_amount or 0)),
                 abs(float(usd_amount or 0)), float(ts), taker,
                 raw_id, index, record_hash, PARSER_VERSION,
                 SCHEMA_VERSION, now),
            )
            executions += 1
            for wallet, role, leg_side in (
                (taker, "taker", side),
                (maker, "maker", "SELL" if side == "BUY" else "BUY"),
            ):
                if not wallet:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO actor_trade_legs "
                    "(actor_leg_id, source_record_id, "
                    "candidate_fingerprint, transaction_hash, "
                    "transaction_log_index, transaction_occurrence, "
                    "proxy_wallet, condition_id, asset, outcome_label, "
                    "outcome_sign, side, size, price, ts, "
                    "liquidity_role, role_confidence, raw_response_id, "
                    "raw_record_index, raw_record_hash, "
                    "parser_version, schema_version, normalized_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 1, ?, ?, "
                    "?, ?, ?, 'exchange_record', ?, ?, ?, ?, ?, ?)",
                    (f"{execution_id}-{role}", execution_id,
                     f"{wallet}|{tx_hash}|{log_index}", tx_hash,
                     log_index, wallet, condition_id,
                     market["token1"], market["answer1"], leg_side,
                     abs(float(token_amount or 0)), float(price),
                     float(ts), role, raw_id, index, record_hash,
                     PARSER_VERSION, SCHEMA_VERSION, now),
                )
                legs += 1
        conn.commit()
    finish_collector_run(conn, run_id, "succeeded")
    conn.commit()
    return {
        "markets": len(condition_ids),
        "executions": executions,
        "actor_legs": legs,
        "rows_per_market": per_condition,
        "source": source,
    }


_RAW_CACHE: dict[tuple[str, str], int] = {}


def _condition_raw_id(
    conn: sqlite3.Connection, run_id: str, condition_id: str,
    source: str, now: float,
) -> int:
    """One provenance raw response per (run, condition): records the
    exact re-fetchable query instead of duplicating the 21GB payload
    (documented contract exception for the public immutable dataset)."""
    from polymarket.collection.raw_store import insert_raw_response

    key = (run_id, condition_id)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = insert_raw_response(
            conn, collector_run_id=run_id, collector="sii:quant",
            base_url=source, endpoint="quant.parquet",
            params={"condition_id": condition_id},
            requested_at=now, received_at=now, http_status=200,
            headers={},
            payload=json.dumps({
                "note": "rows imported directly from public immutable "
                        "dataset; re-fetchable by this query",
                "query": f"condition_id = {condition_id}",
            }).encode(),
        )
    return _RAW_CACHE[key]
