"""29.6 position tests."""

from polymarket.analysis.positions import reconcile_wallet_positions
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.normalization.positions import (
    normalize_activity,
    normalize_position_snapshots,
)
from tests.normalization.helpers import (
    insert_payload,
    raw_row,
    result_for,
    setup_market,
)


def _activity(db, events, received_at=200.0):
    raw_id = insert_payload(db, "activity", "activity", events,
                            received_at=received_at)
    result = result_for(raw_id)
    normalize_activity(db, raw_row(db, raw_id), events, result)
    return result


def _event(type_, wallet="w1", size=10.0, ts=100.0, **extra):
    return {"type": type_, "proxyWallet": wallet, "conditionId": "c1",
            "size": size, "timestamp": ts, "transactionHash": "0x1", **extra}


def test_buy_and_sell_token_changes(db):
    setup_market(db)
    _activity(db, [
        _event("TRADE", asset="yes1", side="BUY", price=0.6),
        _event("TRADE", asset="yes1", side="SELL", price=0.7, size=4.0, ts=110.0),
    ])
    rows = db.execute("SELECT * FROM position_events ORDER BY ts").fetchall()
    assert rows[0]["signed_token_change"] == 10.0
    assert rows[0]["collateral_change"] == -6.0
    assert rows[1]["signed_token_change"] == -4.0
    assert abs(rows[1]["collateral_change"] - 2.8) < 1e-12


def test_split_and_merge(db):
    setup_market(db)
    _activity(db, [
        _event("SPLIT", size=10.0, ts=100.0),
        _event("MERGE", size=4.0, ts=110.0),
    ])
    reader = SQLiteNormalizedReader(db)
    position = reader.position_asof("w1", "c1", 200.0)
    assert position["balances"] == {"yes1": 6.0, "no1": 6.0}
    assert abs(position["collateral_change"] - (-6.0)) < 1e-12
    assert position["complete"]


def test_redeem_requires_resolution_evidence(db):
    market = dict(
        __import__("tests.normalization.helpers", fromlist=["MARKET"]).MARKET
    )
    setup_market(db, market)
    # redeem BEFORE any resolved status observed -> unresolved
    result = _activity(db, [_event("REDEEM", asset="yes1", size=5.0, ts=100.0)])
    row = db.execute("SELECT * FROM position_events").fetchone()
    assert row["accounting_confidence"] == "unresolved"
    assert row["signed_token_change"] is None
    assert result.unresolved

    # now observe a resolved status, then a later redeem resolves exactly
    resolved = dict(market)
    resolved.update({"resolved": True, "closed": True, "winningAsset": "yes1"})
    setup_market(db, resolved, received_at=150.0)
    _activity(db, [_event("REDEEM", asset="yes1", size=5.0, ts=160.0,
                          transactionHash="0x2")])
    rows = db.execute(
        "SELECT * FROM position_events WHERE ts=160.0"
    ).fetchall()
    assert rows[0]["accounting_confidence"] == "exact"
    assert rows[0]["signed_token_change"] == -5.0
    assert rows[0]["collateral_change"] == 5.0
    assert rows[0]["resolution_version"] is not None


def test_unresolved_conversion_not_guessed(db):
    setup_market(db)
    result = _activity(db, [_event("CONVERT", size=3.0)])
    row = db.execute("SELECT * FROM position_events").fetchone()
    assert row["accounting_confidence"] == "unresolved"
    assert row["signed_token_change"] is None
    assert row["collateral_change"] is None
    assert result.unresolved


def test_union_based_audit_with_tolerances(db):
    setup_market(db)
    _activity(db, [
        _event("TRADE", asset="yes1", side="BUY", price=0.6, size=10.0),
    ])
    snapshots = [
        {"proxyWallet": "w1", "asset": "yes1", "size": 10.0 + 5e-7},
        {"proxyWallet": "w1", "asset": "ghost", "size": 2.0},  # platform only
    ]
    raw_id = insert_payload(db, "positions", "positions", snapshots,
                            received_at=300.0)
    normalize_position_snapshots(db, raw_row(db, raw_id), snapshots,
                                 result_for(raw_id))
    reader = SQLiteNormalizedReader(db)
    audit = reconcile_wallet_positions(reader, "w1", 400.0)
    assert audit.platform_asset_count == 2
    assert audit.reconstructed_nonzero_asset_count == 1
    assert audit.union_asset_count == 2
    assert audit.match_count == 1          # yes1 within abs+rel tolerance
    assert audit.missing_reconstructed_count == 1  # ghost not reconstructed
    assert audit.max_absolute_error >= 2.0


def test_incomplete_histories_remain_flagged(db):
    setup_market(db)
    _activity(db, [
        _event("TRADE", asset="yes1", side="BUY", price=0.5, size=5.0),
        _event("CONVERT", size=1.0, ts=120.0),
    ])
    reader = SQLiteNormalizedReader(db)
    position = reader.position_asof("w1", "c1", 200.0)
    assert not position["complete"]
    assert position["unresolved_event_count"] == 1
    audit = reconcile_wallet_positions(reader, "w1", 200.0)
    assert audit.unresolved_event_count == 1
