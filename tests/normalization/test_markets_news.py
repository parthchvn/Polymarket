"""Market metadata and news ledger normalization tests."""

from polymarket.normalization.markets import normalize_market_records
from polymarket.normalization.news import (
    RuleBasedClaimExtractor,
    RuleBasedRelevanceScorer,
)
from tests.normalization.helpers import (
    MARKET,
    insert_payload,
    raw_row,
    result_for,
    setup_market,
)


def test_contract_change_creates_new_version_never_rewrites(db):
    setup_market(db, received_at=10.0)
    changed = dict(MARKET)
    changed["rules"] = "r2-changed"
    raw_id = insert_payload(db, "markets", "markets", [changed], received_at=20.0)
    normalize_market_records(db, raw_row(db, raw_id), [changed], result_for(raw_id))
    rows = db.execute(
        "SELECT version_seq, rules_text FROM contract_versions ORDER BY version_seq"
    ).fetchall()
    assert [(r["version_seq"], r["rules_text"]) for r in rows] == [
        (1, "r1"), (2, "r2-changed"),
    ]


def test_unchanged_market_does_not_create_versions(db):
    setup_market(db, received_at=10.0)
    setup_market(db, received_at=20.0)
    assert db.execute("SELECT COUNT(*) FROM contract_versions").fetchone()[0] == 1
    assert db.execute(
        "SELECT COUNT(*) FROM market_status_versions"
    ).fetchone()[0] == 1


def test_outcome_sign_fallbacks(db):
    record = dict(MARKET)
    record["tokens"] = [
        {"token_id": "a1", "outcome": "Alpha"},
        {"token_id": "b1", "outcome": "Beta"},
    ]
    setup_market(db, record)
    rows = {
        r["asset"]: (r["outcome_sign"], r["mapping_confidence"])
        for r in db.execute("SELECT * FROM outcome_tokens")
    }
    assert rows == {"a1": (1, "assumed"), "b1": (-1, "assumed")}


def test_claim_extractor_deterministic():
    extractor = RuleBasedClaimExtractor()
    a = extractor.extract("Alice Carter wins debate", "She led all night.")
    b = extractor.extract("Alice Carter wins debate", "She led all night.")
    assert a == b
    assert "Alice" in a[0]["entities"]


def test_relevance_scorer_directions():
    scorer = RuleBasedRelevanceScorer()
    positive = scorer.score(
        "Alice Carter wins the election debate",
        "Will Alice Carter win the election?", None,
    )
    negative = scorer.score(
        "Alice Carter loses ground in the election",
        "Will Alice Carter win the election?", None,
    )
    irrelevant = scorer.score(
        "Bakery celebrates anniversary",
        "Will Alice Carter win the election?", None,
    )
    assert positive["rel_class"] == "supports_positive"
    assert negative["rel_class"] == "supports_negative"
    assert irrelevant["rel_class"] == "irrelevant"


def test_first_observed_at_governs_availability(db):
    """Publication time may precede collector availability; availability
    uses first_observed_at."""
    setup_market(db)
    from polymarket.normalization.news import normalize_news

    articles = [{"id": "a1", "publishedAt": 50.0,
                 "headline": "Will X happen", "body": "X."}]
    raw_id = insert_payload(db, "news:wire", "news_feed", articles,
                            received_at=500.0)
    normalize_news(db, raw_row(db, raw_id), articles, result_for(raw_id))
    row = db.execute("SELECT * FROM news_articles").fetchone()
    assert row["source_published_at"] == 50.0
    assert row["first_observed_at"] == 500.0
    from polymarket.analysis.reader import SQLiteNormalizedReader

    reader = SQLiteNormalizedReader(db)
    assert reader.articles_asof(400.0) == []       # not yet observed
    assert len(reader.articles_asof(600.0)) == 1
