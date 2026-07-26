"""29.7 temporal tests: strict < cutoff everywhere."""

import pytest

from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.temporal import (
    TemporalContaminationError,
    assert_no_future_information,
)
from polymarket.normalization.news import normalize_news
from polymarket.normalization.trades import normalize_expanded_trades
from tests.normalization.helpers import (
    insert_payload,
    raw_row,
    result_for,
    setup_market,
)

CUTOFF = 100.0


def test_rows_exactly_at_cutoff_excluded_for_trades(db):
    setup_market(db)
    records = [
        {"transactionHash": "0x1", "conditionId": "c1", "asset": "yes1",
         "proxyWallet": "w1", "side": "BUY", "size": 1.0, "price": 0.5,
         "timestamp": ts}
        for ts in (99.0, 100.0, 101.0)
    ]
    raw_id = insert_payload(db, "trades_expanded", "trades", records)
    normalize_expanded_trades(db, raw_row(db, raw_id), records, result_for(raw_id))
    reader = SQLiteNormalizedReader(db)
    rows = reader.actor_trade_legs_before(CUTOFF)
    assert [r["ts"] for r in rows] == [99.0]


def test_contract_and_status_exactly_at_cutoff_excluded(db):
    setup_market(db, received_at=CUTOFF)  # first_observed_at == cutoff
    reader = SQLiteNormalizedReader(db)
    assert reader.contract_asof("m1", CUTOFF) is None
    assert reader.market_status_asof("m1", CUTOFF) is None
    assert reader.contract_asof("m1", CUTOFF + 0.001) is not None


def test_article_and_relevance_exactly_at_cutoff_excluded(db):
    setup_market(db, received_at=10.0)
    articles = [{"id": "a1", "headline": "Will X happen soon", "body": "X."}]
    raw_id = insert_payload(db, "news:wire", "news_feed", articles,
                            received_at=CUTOFF)
    normalize_news(db, raw_row(db, raw_id), articles, result_for(raw_id))
    reader = SQLiteNormalizedReader(db)
    assert reader.articles_asof(CUTOFF) == []
    assert reader.relevance_asof("m1", CUTOFF) == []
    assert reader.event_families_asof(CUTOFF) == []
    assert len(reader.articles_asof(CUTOFF + 0.001)) == 1


def test_position_events_at_cutoff_excluded(db):
    setup_market(db)
    from polymarket.normalization.positions import normalize_activity

    events = [
        {"type": "TRADE", "proxyWallet": "w1", "conditionId": "c1",
         "asset": "yes1", "side": "BUY", "size": 5.0, "price": 0.5,
         "timestamp": ts, "transactionHash": "0x1"}
        for ts in (99.0, 100.0)
    ]
    raw_id = insert_payload(db, "activity", "activity", events)
    normalize_activity(db, raw_row(db, raw_id), events, result_for(raw_id))
    reader = SQLiteNormalizedReader(db)
    position = reader.position_asof("w1", "c1", CUTOFF)
    assert position["balances"]["yes1"] == 5.0  # only the 99.0 event


def test_future_assertion_catches_contamination():
    context = {"market_state": [{"ts": 99.0}, {"ts": 100.0}]}
    with pytest.raises(TemporalContaminationError):
        assert_no_future_information(context, 100.0)
    assert_no_future_information({"market_state": [{"ts": 99.0}]}, 100.0)


def test_assertion_recurses_nested_structures():
    context = {"nested": {"deep": [{"first_observed_at": 200.0}]}}
    with pytest.raises(TemporalContaminationError):
        assert_no_future_information(context, 150.0)
