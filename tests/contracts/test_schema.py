"""29.1 contract tests."""

from polymarket.contracts.schema import (
    LINEAGE_COLUMNS,
    PARSER_VERSION,
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    table_names,
)


def test_schema_executes_and_tables_exist(db):
    names = table_names(db)
    for table in REQUIRED_TABLES:
        assert table in names, f"missing table {table}"


def test_schema_version_recorded(db):
    row = db.execute(
        "SELECT schema_version, parser_version FROM schema_metadata"
    ).fetchone()
    assert row["schema_version"] == SCHEMA_VERSION == 1
    assert row["parser_version"] == PARSER_VERSION


def test_lineage_columns_exist(db):
    for table in ("markets", "actor_trade_legs", "canonical_executions",
                  "position_events", "order_book_snapshots", "news_articles"):
        cols = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
        for col in LINEAGE_COLUMNS:
            assert col in cols, f"{table} missing lineage column {col}"


def test_foreign_keys_enabled(db):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_synthetic_uses_same_table_names(db, synthetic_db_path):
    from polymarket.contracts.schema import connect

    synthetic = connect(synthetic_db_path)
    assert set(REQUIRED_TABLES) <= table_names(synthetic)
    assert set(REQUIRED_TABLES) <= table_names(db)
    # no real_/synthetic_ table split anywhere
    assert not any(
        n.startswith(("real_", "synthetic_")) for n in table_names(synthetic)
    )
