"""SII historical import: slice selection, metadata + resolution
mapping, trade/leg normalization, provenance, idempotency — against
tiny local parquet fixtures (the real source is remote)."""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from polymarket.collection.sii_import import (  # noqa: E402
    import_market_metadata,
    import_trades,
    select_market_slice,
)
from polymarket.contracts.schema import init_db  # noqa: E402


@pytest.fixture()
def fixtures(tmp_path):
    con = duckdb.connect()
    markets_path = str(tmp_path / "markets.parquet")
    quant_path = str(tmp_path / "quant.parquet")
    con.execute(f"""
        COPY (SELECT * FROM (VALUES
          ('m1', 'Will X happen?', 'x', '0xc1', 't1a', 't1b',
           'Yes', 'No', '["1", "0"]', 5000000.0, 'Ev',
           TIMESTAMPTZ '2024-06-01', TIMESTAMPTZ '2024-08-01',
           TIMESTAMPTZ '2024-08-01', 0, 1, 0),
          ('m2', 'Will Y happen?', 'y', '0xc2', 't2a', 't2b',
           'Yes', 'No', '["0", "1"]', 2000000.0, 'Ev',
           TIMESTAMPTZ '2024-06-01', TIMESTAMPTZ '2024-09-01',
           TIMESTAMPTZ '2024-09-01', 0, 1, 0),
          ('m3', 'Tiny market', 'z', '0xc3', 't3a', 't3b',
           'Yes', 'No', '["1", "0"]', 1000.0, 'Ev',
           TIMESTAMPTZ '2024-06-01', TIMESTAMPTZ '2024-07-01',
           TIMESTAMPTZ '2024-07-01', 0, 1, 0)
        ) AS t(id, question, slug, condition_id, token1, token2,
               answer1, answer2, outcome_prices, volume, event_title,
               created_at, end_date, updated_at, neg_risk, closed,
               active)) TO '{markets_path}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (SELECT * FROM (VALUES
          (1722000000, 'tx1', 1, 'm1', '0xc1', 'e1', 0.6, 60.0, 100.0,
           'BUY', '0xmaker1', '0xtaker1'),
          (1722000060, 'tx2', 2, 'm1', '0xc1', 'e1', 0.62, 31.0, 50.0,
           'SELL', '0xmaker2', '0xtaker1'),
          (1722000120, 'tx3', 1, 'm2', '0xc2', 'e2', 0.4, 40.0, 100.0,
           'BUY', '0xmaker1', '0xtaker2')
        ) AS t(timestamp, transaction_hash, log_index, market_id,
               condition_id, event_id, price, usd_amount,
               token_amount, side, maker, taker))
        TO '{quant_path}' (FORMAT PARQUET)
    """)
    return markets_path, quant_path


def test_slice_selection_filters_and_orders(fixtures):
    markets_path, _ = fixtures
    markets = select_market_slice(
        markets_path, top_n=10, min_volume=1e6,
        max_volume=1e8, since="2024-01-01",
    )
    assert [m["id"] for m in markets] == ["m1", "m2"]  # volume desc
    assert all(m["end_ts"] is not None for m in markets)


def test_full_import_with_resolution_and_provenance(fixtures, tmp_path):
    markets_path, quant_path = fixtures
    conn = init_db(str(tmp_path / "sii.sqlite"), description="sii")
    markets = select_market_slice(
        markets_path, top_n=10, min_volume=1e6, max_volume=1e8,
    )
    assert import_market_metadata(conn, markets) == 2
    report = import_trades(conn, markets, quant_source=quant_path)
    assert report["executions"] == 3
    assert report["actor_legs"] == 6            # maker + taker each

    # resolution mapped: m1 resolves YES -> winning token1
    row = conn.execute(
        "SELECT winning_asset, resolved FROM market_status_versions "
        "WHERE market_id = 'm1' AND closed = 1"
    ).fetchone()
    assert row["winning_asset"] == "t1a" and row["resolved"] == 1
    # m2 resolves NO -> token2
    assert conn.execute(
        "SELECT winning_asset FROM market_status_versions "
        "WHERE market_id = 'm2' AND closed = 1"
    ).fetchone()[0] == "t2b"
    # contract carries the resolution time for MarketCensor / O
    assert conn.execute(
        "SELECT resolution_time FROM contract_versions "
        "WHERE market_id = 'm1'"
    ).fetchone()[0] is not None

    # legs: maker side is the flip of taker side
    sides = dict(conn.execute(
        "SELECT liquidity_role, side FROM actor_trade_legs "
        "WHERE transaction_hash = 'tx1'"
    ).fetchall())
    assert sides == {"taker": "BUY", "maker": "SELL"}

    # provenance: one re-fetchable query per (run, condition)
    payloads = [
        json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM raw_responses "
            "WHERE collector = 'sii:quant'"
        )
    ]
    assert len(payloads) == 2
    assert all("re-fetchable" in p["note"] for p in payloads)

    # idempotency: re-import inserts nothing new
    again = import_trades(conn, markets, quant_source=quant_path)
    assert again["executions"] == 3             # scanned, OR IGNOREd
    total = conn.execute(
        "SELECT COUNT(*) FROM canonical_executions"
    ).fetchone()[0]
    assert total == 3

    # the episode builder runs on imported legs
    from polymarket.analysis.decisions import build_decision_episodes
    from polymarket.analysis.reader import SQLiteNormalizedReader

    episodes = build_decision_episodes(
        SQLiteNormalizedReader(conn), end_time=1_800_000_000.0
    )
    assert len(episodes) >= 2
    assert {e.actor_id for e in episodes} >= {"0xtaker1", "0xtaker2"}
