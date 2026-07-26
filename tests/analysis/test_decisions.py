"""29.8 decision tests."""

from polymarket.analysis.decisions import (
    build_decision_episodes,
    proposition_change,
)
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.normalization.trades import normalize_expanded_trades
from tests.normalization.helpers import (
    insert_payload,
    raw_row,
    result_for,
    setup_market,
)


def _legs(db, specs):
    records = [
        {"transactionHash": f"0x{i}", "conditionId": "c1", "asset": asset,
         "proxyWallet": wallet, "side": side, "size": size, "price": 0.5,
         "timestamp": ts}
        for i, (wallet, asset, side, size, ts) in enumerate(specs)
    ]
    raw_id = insert_payload(db, "trades_expanded", "trades", records)
    normalize_expanded_trades(db, raw_row(db, raw_id), records, result_for(raw_id))
    return raw_id


def _set_all_roles(db, role):
    db.execute("UPDATE actor_trade_legs SET liquidity_role=?", (role,))
    db.commit()


def test_proposition_change_signs():
    assert proposition_change("BUY", 1, 5.0) == 5.0
    assert proposition_change("SELL", 1, 5.0) == -5.0
    assert proposition_change("BUY", -1, 5.0) == -5.0   # buying NO = negative
    assert proposition_change("SELL", -1, 5.0) == 5.0


def test_maker_excluded_taker_included(db):
    setup_market(db)
    _legs(db, [("w1", "yes1", "BUY", 5.0, 100.0),
               ("w2", "yes1", "SELL", 5.0, 100.0)])
    db.execute(
        "UPDATE actor_trade_legs SET liquidity_role='taker' "
        "WHERE proxy_wallet='w1'"
    )
    db.execute(
        "UPDATE actor_trade_legs SET liquidity_role='maker' "
        "WHERE proxy_wallet='w2'"
    )
    db.commit()
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(db), end_time=1000.0
    )
    assert [e.actor_id for e in episodes] == ["w1"]


def test_mixed_activity_returns_none_direction(db):
    setup_market(db)
    _legs(db, [("w1", "yes1", "BUY", 5.0, 100.0),
               ("w1", "yes1", "SELL", 5.0, 200.0)])
    _set_all_roles(db, "taker")
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(db), end_time=1000.0, interval_seconds=3600.0
    )
    assert len(episodes) == 1
    assert episodes[0].direction is None
    assert episodes[0].mixed_activity_ratio == 0.0
    assert episodes[0].gross_quantity == 10.0


def test_direction_positive_and_negative(db):
    setup_market(db)
    _legs(db, [("w1", "yes1", "BUY", 5.0, 100.0),
               ("w2", "no1", "BUY", 5.0, 100.0)])
    _set_all_roles(db, "taker")
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(db), end_time=1000.0
    )
    directions = {e.actor_id: e.direction for e in episodes}
    assert directions == {"w1": "positive", "w2": "negative"}


def test_interval_bucketing(db):
    setup_market(db)
    _legs(db, [("w1", "yes1", "BUY", 1.0, 100.0),
               ("w1", "yes1", "BUY", 1.0, 200.0),
               ("w1", "yes1", "BUY", 1.0, 100.0 + 3600.0)])  # new episode
    _set_all_roles(db, "taker")
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(db), end_time=10000.0, interval_seconds=3600.0
    )
    assert len(episodes) == 2
    assert episodes[0].coverage["leg_count"] == 2
    assert episodes[1].coverage["leg_count"] == 1


def test_pre_decision_position_excludes_anchor_time_events(db):
    setup_market(db)
    from polymarket.normalization.positions import normalize_activity

    events = [
        {"type": "TRADE", "proxyWallet": "w1", "conditionId": "c1",
         "asset": "yes1", "side": "BUY", "size": 3.0, "price": 0.5,
         "timestamp": 50.0, "transactionHash": "0xa"},
        {"type": "TRADE", "proxyWallet": "w1", "conditionId": "c1",
         "asset": "yes1", "side": "BUY", "size": 7.0, "price": 0.5,
         "timestamp": 100.0, "transactionHash": "0xb"},  # at anchor
    ]
    raw_id = insert_payload(db, "activity", "activity", events)
    normalize_activity(db, raw_row(db, raw_id), events, result_for(raw_id))
    _legs(db, [("w1", "yes1", "BUY", 7.0, 100.0)])
    _set_all_roles(db, "taker")
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(db), end_time=1000.0
    )
    assert episodes[0].anchor_time == 100.0
    assert episodes[0].pre_decision_position["balances"] == {"yes1": 3.0}
