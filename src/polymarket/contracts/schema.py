"""Shared normalized SQLite schema.

One database holds raw observations and normalized tables.  Real and
synthetic data write through the same schema and the same normalization
path.  Timestamps are UTC epoch seconds stored as REAL.  Raw payloads are
exact bytes stored as BLOB.
"""

from __future__ import annotations

import sqlite3
import time

SCHEMA_VERSION = 1
PARSER_VERSION = "1.0.0"

PRAGMAS = [
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode = WAL;",
    "PRAGMA busy_timeout = 5000;",
]

DDL: list[str] = [
    # ------------------------------------------------------------------
    # metadata
    """
    CREATE TABLE IF NOT EXISTS schema_metadata (
        schema_version INTEGER PRIMARY KEY,
        applied_at REAL NOT NULL,
        parser_version TEXT NOT NULL,
        description TEXT
    );
    """,
    # ------------------------------------------------------------------
    # operational / raw layer
    """
    CREATE TABLE IF NOT EXISTS collector_runs (
        collector_run_id TEXT PRIMARY KEY,
        collector TEXT NOT NULL,
        started_at REAL NOT NULL,
        finished_at REAL,
        status TEXT NOT NULL CHECK (
            status IN ('running', 'succeeded', 'failed', 'partial')
        ),
        configuration_json TEXT NOT NULL,
        note TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_responses (
        raw_response_id INTEGER PRIMARY KEY AUTOINCREMENT,
        collector_run_id TEXT NOT NULL,
        collector TEXT NOT NULL,
        base_url TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        canonical_params_json TEXT NOT NULL,
        requested_at REAL NOT NULL,
        received_at REAL NOT NULL,
        http_status INTEGER,
        response_headers_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        payload BLOB NOT NULL,
        error_text TEXT,
        FOREIGN KEY (collector_run_id)
            REFERENCES collector_runs(collector_run_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backfill_windows (
        collector TEXT NOT NULL,
        object_id TEXT NOT NULL,
        window_start REAL NOT NULL,
        window_end REAL NOT NULL,
        started_at REAL,
        completed_at REAL,
        page_count INTEGER NOT NULL DEFAULT 0,
        record_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'running', 'complete', 'incomplete', 'failed')
        ),
        observed_min_ts REAL,
        observed_max_ts REAL,
        note TEXT,
        PRIMARY KEY (collector, object_id, window_start, window_end)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collector_gaps (
        collector TEXT NOT NULL,
        surface TEXT NOT NULL,
        object_id TEXT,
        gap_start REAL NOT NULL,
        gap_end REAL,
        reason TEXT NOT NULL,
        detected_at REAL NOT NULL,
        resolved_at REAL,
        PRIMARY KEY (collector, surface, object_id, gap_start)
    );
    """,
    # ------------------------------------------------------------------
    # normalized market layer
    """
    CREATE TABLE IF NOT EXISTS markets (
        market_id TEXT PRIMARY KEY,
        condition_id TEXT NOT NULL UNIQUE,
        category TEXT,
        question TEXT,
        created_at REAL,
        closed_at REAL,
        resolved_at REAL,
        is_combo INTEGER NOT NULL DEFAULT 0,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS outcome_tokens (
        condition_id TEXT NOT NULL,
        asset TEXT NOT NULL,
        outcome_label TEXT,
        outcome_sign INTEGER NOT NULL CHECK (outcome_sign IN (-1, 1)),
        mapping_effective_from REAL NOT NULL,
        mapping_confidence TEXT NOT NULL,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (condition_id, asset, mapping_effective_from)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS contract_versions (
        market_id TEXT NOT NULL,
        version_seq INTEGER NOT NULL,
        effective_from REAL NOT NULL,
        first_observed_at REAL NOT NULL,
        question TEXT,
        rules_text TEXT,
        resolution_source TEXT,
        resolution_time REAL,
        content_hash TEXT NOT NULL,
        raw_response_id INTEGER NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (market_id, version_seq)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_status_versions (
        market_id TEXT NOT NULL,
        effective_from REAL NOT NULL,
        first_observed_at REAL NOT NULL,
        trading_enabled INTEGER NOT NULL,
        closed INTEGER NOT NULL,
        resolved INTEGER NOT NULL,
        winning_asset TEXT,
        raw_response_id INTEGER NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (market_id, effective_from)
    );
    """,
    # ------------------------------------------------------------------
    # trades
    """
    CREATE TABLE IF NOT EXISTS actor_trade_legs (
        actor_leg_id TEXT PRIMARY KEY,
        source_record_id TEXT,
        candidate_fingerprint TEXT NOT NULL,
        transaction_hash TEXT NOT NULL,
        transaction_log_index INTEGER,
        transaction_occurrence INTEGER,
        proxy_wallet TEXT NOT NULL,
        condition_id TEXT NOT NULL,
        asset TEXT NOT NULL,
        outcome_label TEXT,
        outcome_sign INTEGER CHECK (outcome_sign IN (-1, 1)),
        side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
        size REAL NOT NULL,
        price REAL NOT NULL,
        ts REAL NOT NULL,
        liquidity_role TEXT NOT NULL CHECK (
            liquidity_role IN ('taker', 'maker', 'unknown')
        ),
        role_confidence TEXT NOT NULL,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_executions (
        execution_id TEXT PRIMARY KEY,
        source_record_id TEXT,
        transaction_hash TEXT NOT NULL,
        transaction_log_index INTEGER,
        transaction_occurrence INTEGER,
        condition_id TEXT NOT NULL,
        positive_price REAL NOT NULL,
        positive_side TEXT CHECK (positive_side IN ('BUY', 'SELL')),
        size REAL NOT NULL,
        notional REAL NOT NULL,
        ts REAL NOT NULL,
        taker_wallet TEXT,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        reconciliation_status TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL
    );
    """,
    # ------------------------------------------------------------------
    # positions
    """
    CREATE TABLE IF NOT EXISTS position_events (
        position_event_id TEXT PRIMARY KEY,
        wallet TEXT NOT NULL,
        condition_id TEXT NOT NULL,
        asset TEXT,
        ts REAL NOT NULL,
        event_type TEXT NOT NULL,
        signed_token_change REAL,
        collateral_change REAL,
        transaction_hash TEXT,
        transaction_log_index INTEGER,
        accounting_confidence TEXT NOT NULL CHECK (
            accounting_confidence IN ('exact', 'inferred', 'unresolved')
        ),
        resolution_version INTEGER,
        is_combo INTEGER NOT NULL DEFAULT 0,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS position_snapshots (
        wallet TEXT NOT NULL,
        asset TEXT NOT NULL,
        observed_at REAL NOT NULL,
        reported_size REAL NOT NULL,
        source TEXT NOT NULL,
        raw_response_id INTEGER NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (wallet, asset, observed_at)
    );
    """,
    # ------------------------------------------------------------------
    # market state
    """
    CREATE TABLE IF NOT EXISTS order_book_snapshots (
        asset TEXT NOT NULL,
        observed_at REAL NOT NULL,
        best_bid REAL,
        best_ask REAL,
        spread REAL,
        bid_depth REAL,
        ask_depth REAL,
        imbalance REAL,
        raw_response_id INTEGER NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (asset, observed_at)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_state (
        condition_id TEXT NOT NULL,
        ts REAL NOT NULL,
        positive_price REAL,
        volume REAL,
        spread REAL,
        depth REAL,
        imbalance REAL,
        state_source TEXT NOT NULL,
        coverage_complete INTEGER NOT NULL,
        raw_response_id INTEGER,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL,
        PRIMARY KEY (condition_id, ts, state_source)
    );
    """,
    # ------------------------------------------------------------------
    # news
    """
    CREATE TABLE IF NOT EXISTS news_articles (
        article_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        source_url TEXT,
        source_published_at REAL,
        first_observed_at REAL NOT NULL,
        download_completed_at REAL NOT NULL,
        timestamp_source TEXT,
        timestamp_confidence REAL,
        headline TEXT,
        body TEXT,
        content_hash TEXT NOT NULL,
        previous_article_id TEXT,
        raw_response_id INTEGER NOT NULL,
        raw_record_index INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        normalized_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS event_families (
        event_family_id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        earliest_available_at REAL NOT NULL,
        created_by TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_claims (
        claim_id TEXT PRIMARY KEY,
        article_id TEXT NOT NULL,
        claim_text TEXT NOT NULL,
        entities_json TEXT,
        quantities_json TEXT,
        supporting_span TEXT,
        first_available_at REAL NOT NULL,
        extractor_version TEXT NOT NULL,
        confidence REAL,
        FOREIGN KEY (article_id) REFERENCES news_articles(article_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS claim_edges (
        edge_id TEXT PRIMARY KEY,
        claim_id TEXT NOT NULL,
        event_family_id TEXT NOT NULL,
        edge_type TEXT NOT NULL CHECK (
            edge_type IN (
                'new', 'duplicate', 'confirmation',
                'correction', 'contradiction', 'supersession'
            )
        ),
        effective_from REAL NOT NULL,
        evidence TEXT,
        confidence REAL,
        FOREIGN KEY (claim_id) REFERENCES news_claims(claim_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS relevance_judgments (
        event_family_id TEXT NOT NULL,
        market_id TEXT NOT NULL,
        contract_version_seq INTEGER NOT NULL,
        computed_at REAL NOT NULL,
        rel_class TEXT NOT NULL CHECK (
            rel_class IN (
                'supports_positive', 'supports_negative', 'indirect',
                'background', 'irrelevant', 'ambiguous'
            )
        ),
        rel_score REAL NOT NULL,
        direction REAL NOT NULL,
        novelty REAL,
        surprise REAL,
        method TEXT NOT NULL,
        model_version TEXT,
        evidence_json TEXT,
        PRIMARY KEY (event_family_id, market_id, contract_version_seq, computed_at)
    );
    """,
]

REQUIRED_TABLES = [
    "schema_metadata",
    "collector_runs",
    "raw_responses",
    "backfill_windows",
    "collector_gaps",
    "markets",
    "outcome_tokens",
    "contract_versions",
    "market_status_versions",
    "actor_trade_legs",
    "canonical_executions",
    "position_events",
    "position_snapshots",
    "order_book_snapshots",
    "market_state",
    "news_articles",
    "event_families",
    "news_claims",
    "claim_edges",
    "relevance_judgments",
]

LINEAGE_COLUMNS = ("raw_response_id", "parser_version", "schema_version", "normalized_at")


def connect(path: str) -> sqlite3.Connection:
    """Open a connection with required pragmas applied."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for pragma in PRAGMAS:
        conn.execute(pragma)
    return conn


def init_db(path: str, description: str = "initial schema") -> sqlite3.Connection:
    """Create all tables and record schema metadata.  Idempotent."""
    conn = connect(path)
    with conn:
        for stmt in DDL:
            conn.execute(stmt)
        conn.execute(
            """
            INSERT OR IGNORE INTO schema_metadata
                (schema_version, applied_at, parser_version, description)
            VALUES (?, ?, ?, ?)
            """,
            (SCHEMA_VERSION, time.time(), PARSER_VERSION, description),
        )
    return conn


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}
