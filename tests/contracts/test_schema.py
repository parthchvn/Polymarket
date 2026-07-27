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
    assert row["schema_version"] == SCHEMA_VERSION == 2
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


def test_v1_database_migrates_to_v2_and_audit_reports_it(tmp_path):
    """Migration from an actual schema-v1 fixture: physical upgrade AND
    metadata/audit truthfulness."""
    import sqlite3

    from polymarket.analysis.reporting import audit_database
    from polymarket.contracts.schema import (
        DDL,
        PARSER_VERSION,
        REQUIRED_TABLES,
        ensure_paper_schema,
    )

    db = str(tmp_path / "v1.sqlite")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # build a faithful v1 fixture: current DDL minus v2 additions
    for statement in DDL:
        if "liquidity_bars" in statement:
            continue
        text = statement
        if "order_book_snapshots" in text:
            for column in ("best_bid_size REAL,", "best_ask_size REAL,",
                           "tick_size REAL,"):
                text = text.replace(column, "")
        if "relevance_judgments" in text:
            text = (
                text
                .replace("relevance_judgment_id TEXT PRIMARY KEY,", "")
                .replace("claim_id TEXT,", "")
                .replace("source_effective_at REAL,", "")
                .replace("scored_at REAL,", "")
                .replace(
                    "evidence_json TEXT",
                    "evidence_json TEXT,\n"
                    "        PRIMARY KEY (event_family_id, market_id, "
                    "contract_version_seq, computed_at)",
                )
            )
        conn.executescript(text)
    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, "
        "parser_version, description) VALUES (1, 0, ?, 'v1 fixture')",
        (PARSER_VERSION,),
    )
    conn.commit()

    applied = ensure_paper_schema(conn)
    assert any("liquidity_bars" in a for a in applied)
    assert any("schema_metadata:v2" in a for a in applied)
    assert ensure_paper_schema(conn) == []          # idempotent

    report = audit_database(conn)
    assert report["schema_version"] == 2            # audit tells the truth
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert set(REQUIRED_TABLES) <= tables
