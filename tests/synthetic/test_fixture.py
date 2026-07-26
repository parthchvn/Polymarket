"""Synthetic fixture determinism and required-case coverage."""

from polymarket.contracts.schema import connect
from polymarket.synthetic.fixtures import build_synthetic_fixture


def test_fixture_deterministic_content(tmp_path):
    a = build_synthetic_fixture(str(tmp_path / "a.sqlite"), overwrite=True)
    b = build_synthetic_fixture(str(tmp_path / "b.sqlite"), overwrite=True)
    for table, key in [
        ("canonical_executions", "execution_id"),
        ("actor_trade_legs", "actor_leg_id"),
        ("position_events", "position_event_id"),
        ("news_articles", "article_id"),
        ("relevance_judgments", "event_family_id"),
    ]:
        rows_a = [r[0] for r in a.execute(
            f"SELECT {key} FROM {table} ORDER BY {key}")]
        rows_b = [r[0] for r in b.execute(
            f"SELECT {key} FROM {table} ORDER BY {key}")]
        assert rows_a == rows_b, f"{table} not deterministic"


def test_fixture_covers_required_cases(synthetic_db_path):
    conn = connect(synthetic_db_path)

    def count(sql, *args):
        return conn.execute(sql, args).fetchone()[0]

    assert count("SELECT COUNT(*) FROM markets") >= 2
    assert count("SELECT COUNT(DISTINCT outcome_sign) FROM outcome_tokens") == 2
    assert count(
        "SELECT COUNT(DISTINCT proxy_wallet) FROM actor_trade_legs "
        "WHERE liquidity_role='taker'"
    ) >= 3
    assert count(
        "SELECT COUNT(*) FROM actor_trade_legs WHERE liquidity_role='maker'"
    ) > 0
    assert count("SELECT COUNT(*) FROM canonical_executions") > 0
    for event_type in ("SPLIT", "MERGE", "REDEEM"):
        assert count(
            "SELECT COUNT(*) FROM position_events WHERE event_type=?",
            event_type,
        ) > 0, event_type
    assert count("SELECT COUNT(*) FROM position_snapshots") > 0
    assert count("SELECT COUNT(*) FROM market_status_versions") >= 2
    assert count("SELECT COUNT(*) FROM order_book_snapshots") > 0
    assert count(
        "SELECT COUNT(*) FROM contract_versions WHERE market_id='mkt-election'"
    ) == 2
    assert count("SELECT COUNT(*) FROM news_articles") >= 3
    assert count("SELECT COUNT(*) FROM event_families") >= 2
    assert count(
        "SELECT COUNT(*) FROM relevance_judgments "
        "WHERE rel_class IN ('supports_positive','supports_negative')"
    ) >= 2
    assert count(
        "SELECT COUNT(*) FROM relevance_judgments WHERE rel_class='irrelevant'"
    ) > 0
    # deliberately ambiguous maker/taker case
    assert count(
        "SELECT COUNT(*) FROM actor_trade_legs WHERE liquidity_role='unknown'"
    ) == 2
    # collector gap
    assert count(
        "SELECT COUNT(*) FROM collector_gaps WHERE resolved_at IS NULL"
    ) == 1
    # incomplete position history
    assert count(
        "SELECT COUNT(*) FROM position_events "
        "WHERE accounting_confidence='unresolved'"
    ) == 1
    # everything came through the single normalization path
    assert count("SELECT COUNT(*) FROM raw_responses") > 0
