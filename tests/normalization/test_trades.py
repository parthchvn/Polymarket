"""29.5 trade normalization tests."""

from polymarket.normalization.normalizer import Normalizer
from polymarket.normalization.reconciliation import reconcile_roles
from polymarket.normalization.trades import (
    normalize_expanded_trades,
    normalize_taker_trades,
)
from tests.normalization.helpers import (
    insert_payload,
    raw_row,
    result_for,
    setup_market,
)


def _trade(wallet="w1", asset="yes1", side="BUY", price=0.6, tx="0x1",
           size=5.0, ts=100.0, **extra):
    return {"transactionHash": tx, "conditionId": "c1", "asset": asset,
            "proxyWallet": wallet, "side": side, "size": size,
            "price": price, "timestamp": ts, **extra}


def test_taker_rows_create_canonical_executions(db):
    setup_market(db)
    records = [_trade(logIndex=1)]
    raw_id = insert_payload(db, "trades_taker", "trades", records,
                            params={"takerOnly": "true"})
    result = result_for(raw_id)
    normalize_taker_trades(db, raw_row(db, raw_id), records, result)
    assert result.inserted["canonical_executions"] == 1
    row = db.execute("SELECT * FROM canonical_executions").fetchone()
    assert row["positive_price"] == 0.6
    assert row["positive_side"] == "BUY"
    assert row["reconciliation_status"] == "direct"


def test_negative_token_price_converts(db):
    setup_market(db)
    records = [_trade(asset="no1", side="BUY", price=0.3)]
    raw_id = insert_payload(db, "trades_taker", "trades", records)
    normalize_taker_trades(db, raw_row(db, raw_id), records, result_for(raw_id))
    row = db.execute("SELECT * FROM canonical_executions").fetchone()
    assert abs(row["positive_price"] - 0.7) < 1e-12
    assert row["positive_side"] == "SELL"
    assert row["reconciliation_status"] == "complemented"


def test_price_outside_tolerance_rejected(db):
    setup_market(db)
    records = [_trade(price=1.5)]
    raw_id = insert_payload(db, "trades_taker", "trades", records)
    result = result_for(raw_id)
    normalize_taker_trades(db, raw_row(db, raw_id), records, result)
    assert db.execute("SELECT COUNT(*) FROM canonical_executions").fetchone()[0] == 0
    assert any("tolerance" in u["reason"] for u in result.unresolved)


def test_expanded_rows_preserve_counterparties(db):
    setup_market(db)
    records = [
        _trade(wallet="taker", side="BUY"),
        _trade(wallet="maker", side="SELL"),
    ]
    raw_id = insert_payload(db, "trades_expanded", "trades", records,
                            params={"takerOnly": "false"})
    normalize_expanded_trades(db, raw_row(db, raw_id), records, result_for(raw_id))
    wallets = {r["proxy_wallet"] for r in db.execute("SELECT * FROM actor_trade_legs")}
    assert wallets == {"taker", "maker"}


def test_repeated_legitimate_rows_survive_and_tx_hash_does_not_dedupe(db):
    setup_market(db)
    # identical tuples in the SAME transaction: legitimate repetition
    records = [_trade(), _trade()]
    raw_id = insert_payload(db, "trades_expanded", "trades", records)
    normalize_expanded_trades(db, raw_row(db, raw_id), records, result_for(raw_id))
    rows = db.execute("SELECT * FROM actor_trade_legs").fetchall()
    assert len(rows) == 2
    assert rows[0]["candidate_fingerprint"] == rows[1]["candidate_fingerprint"]
    assert rows[0]["actor_leg_id"] != rows[1]["actor_leg_id"]


def test_renormalization_is_idempotent(db):
    setup_market(db)
    records = [_trade()]
    raw_id = insert_payload(db, "trades_expanded", "trades", records,
                            params={"takerOnly": "false"})
    normalizer = Normalizer(db)
    normalizer.normalize_raw_response(raw_id)
    second = normalizer.normalize_raw_response(raw_id)
    assert db.execute("SELECT COUNT(*) FROM actor_trade_legs").fetchone()[0] == 1
    assert second.ignored.get("actor_trade_legs") == 1


def test_actor_legs_do_not_define_market_volume(db, synthetic_db_path):
    """Market state volume must come from canonical executions only."""
    from polymarket.contracts.schema import connect

    conn = connect(synthetic_db_path)
    state_volume = conn.execute(
        "SELECT SUM(volume) FROM market_state WHERE state_source='executions'"
    ).fetchone()[0]
    execution_volume = conn.execute(
        "SELECT SUM(size) FROM canonical_executions"
    ).fetchone()[0]
    leg_volume = conn.execute(
        "SELECT SUM(size) FROM actor_trade_legs"
    ).fetchone()[0]
    assert abs(state_volume - execution_volume) < 1e-9
    assert abs(state_volume - leg_volume) > 1e-9  # legs would double-count


def test_ambiguous_role_becomes_unknown(db):
    setup_market(db)
    ts = 100.0
    taker_records = [
        _trade(wallet="wa", tx="0xamb", ts=ts),
        _trade(wallet="wb", tx="0xamb", ts=ts),
    ]
    raw_id = insert_payload(db, "trades_taker", "trades", taker_records)
    normalize_taker_trades(db, raw_row(db, raw_id), taker_records, result_for(raw_id))
    expanded = [
        _trade(wallet="wa", tx="0xamb", ts=ts),
        _trade(wallet="wb", tx="0xamb", ts=ts),
    ]
    raw_id2 = insert_payload(db, "trades_expanded", "trades", expanded)
    normalize_expanded_trades(db, raw_row(db, raw_id2), expanded, result_for(raw_id2))
    diag = reconcile_roles(db)
    roles = {r["liquidity_role"] for r in db.execute("SELECT * FROM actor_trade_legs")}
    assert roles == {"unknown"}
    assert diag.unknown_remaining == 2
    assert "0xamb" in diag.ambiguous_transactions


def test_taker_and_maker_assignment(db):
    setup_market(db)
    taker_records = [_trade(wallet="taker", side="BUY", logIndex=1)]
    raw_id = insert_payload(db, "trades_taker", "trades", taker_records)
    normalize_taker_trades(db, raw_row(db, raw_id), taker_records, result_for(raw_id))
    expanded = [
        _trade(wallet="taker", side="BUY", logIndex=1),
        _trade(wallet="maker", side="SELL", logIndex=1),
    ]
    raw_id2 = insert_payload(db, "trades_expanded", "trades", expanded)
    normalize_expanded_trades(db, raw_row(db, raw_id2), expanded, result_for(raw_id2))
    reconcile_roles(db)
    roles = {
        r["proxy_wallet"]: r["liquidity_role"]
        for r in db.execute("SELECT * FROM actor_trade_legs")
    }
    assert roles == {"taker": "taker", "maker": "maker"}
