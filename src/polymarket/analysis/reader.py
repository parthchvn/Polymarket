"""Strict as-of reader over the normalized schema.

EVERY method uses strict inequality ``available_at < cutoff``.  Rows
whose availability timestamp equals the cutoff are excluded.  All
temporal SQL is centralized here; downstream analysis must not issue
unrestricted raw SQL.
"""

from __future__ import annotations

import sqlite3

from polymarket.contracts.schema import connect


class SQLiteNormalizedReader:
    def __init__(self, conn_or_path: sqlite3.Connection | str) -> None:
        if isinstance(conn_or_path, str):
            self._conn = connect(conn_or_path)
        else:
            self._conn = conn_or_path

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ------------------------------------------------------------------
    # trades
    def actor_trade_legs_before(
        self,
        cutoff: float,
        actor: str | None = None,
        condition_id: str | None = None,
        liquidity_role: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM actor_trade_legs WHERE ts < ?"
        args: list = [cutoff]
        if actor is not None:
            sql += " AND proxy_wallet = ?"
            args.append(actor)
        if condition_id is not None:
            sql += " AND condition_id = ?"
            args.append(condition_id)
        if liquidity_role is not None:
            sql += " AND liquidity_role = ?"
            args.append(liquidity_role)
        sql += " ORDER BY ts"
        return self._conn.execute(sql, args).fetchall()

    def canonical_executions_before(
        self, cutoff: float, condition_id: str | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM canonical_executions WHERE ts < ?"
        args: list = [cutoff]
        if condition_id is not None:
            sql += " AND condition_id = ?"
            args.append(condition_id)
        sql += " ORDER BY ts"
        return self._conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------
    # positions
    def position_events_before(
        self,
        cutoff: float,
        wallet: str | None = None,
        condition_id: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM position_events WHERE ts < ?"
        args: list = [cutoff]
        if wallet is not None:
            sql += " AND wallet = ?"
            args.append(wallet)
        if condition_id is not None:
            sql += " AND condition_id = ?"
            args.append(condition_id)
        sql += " ORDER BY ts"
        return self._conn.execute(sql, args).fetchall()

    def position_asof(
        self, wallet: str, condition_id: str, cutoff: float
    ) -> dict[str, object]:
        """Reconstructed token balances strictly before the cutoff, with
        explicit accounting-quality diagnostics."""
        events = self.position_events_before(
            cutoff, wallet=wallet, condition_id=condition_id
        )
        balances: dict[str, float] = {}
        collateral = 0.0
        unresolved = 0
        for event in events:
            if event["accounting_confidence"] == "unresolved":
                unresolved += 1
                continue
            asset = event["asset"]
            if asset is not None and event["signed_token_change"] is not None:
                balances[asset] = (
                    balances.get(asset, 0.0) + event["signed_token_change"]
                )
            if event["collateral_change"] is not None:
                collateral += event["collateral_change"]
        return {
            "balances": balances,
            "collateral_change": collateral,
            "event_count": len(events),
            "unresolved_event_count": unresolved,
            "complete": unresolved == 0,
        }

    def position_snapshots_before(
        self, cutoff: float, wallet: str | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM position_snapshots WHERE observed_at < ?"
        args: list = [cutoff]
        if wallet is not None:
            sql += " AND wallet = ?"
            args.append(wallet)
        sql += " ORDER BY observed_at"
        return self._conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------
    # market metadata
    def market(self, market_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM markets WHERE market_id = ?", (market_id,)
        ).fetchone()

    def market_by_condition(self, condition_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM markets WHERE condition_id = ?", (condition_id,)
        ).fetchone()

    def market_status_asof(
        self, market_id: str, cutoff: float
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM market_status_versions
            WHERE market_id = ? AND first_observed_at < ?
            ORDER BY effective_from DESC LIMIT 1
            """,
            (market_id, cutoff),
        ).fetchone()

    def contract_asof(self, market_id: str, cutoff: float) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM contract_versions
            WHERE market_id = ? AND first_observed_at < ?
            ORDER BY version_seq DESC LIMIT 1
            """,
            (market_id, cutoff),
        ).fetchone()

    def outcome_tokens_asof(
        self, condition_id: str, cutoff: float
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM outcome_tokens
            WHERE condition_id = ? AND mapping_effective_from < ?
            ORDER BY mapping_effective_from
            """,
            (condition_id, cutoff),
        ).fetchall()

    # ------------------------------------------------------------------
    # market state
    def market_series_before(
        self,
        condition_id: str,
        cutoff: float,
        lookback: float,
        policy: str = "book_preferred",
    ) -> tuple[list[sqlite3.Row], str]:
        """Canonical market price series with an EXPLICIT source policy.

        Once execution-derived and book-mid state coexist, mixing them
        into one series conflates midquotes with trade prints (bid-ask
        bounce) and duplicates timestamps.  Policies:

        * ``book_only``     — book-mid rows only (paper analyses);
        * ``book_preferred``— book-mid when any exists in the window,
          otherwise execution-derived, with the fallback reported;
        * ``execution_only``— execution-derived rows only.

        Returns (rows, source_used) where source_used is 'book_mid',
        'derived' or 'none'.
        """
        def rows_for(source: str) -> list[sqlite3.Row]:
            return self._conn.execute(
                """
                SELECT * FROM market_state
                WHERE condition_id = ? AND ts < ? AND ts >= ?
                  AND state_source = ?
                ORDER BY ts
                """,
                (condition_id, cutoff, cutoff - lookback, source),
            ).fetchall()

        if policy == "book_only":
            rows = rows_for("book_mid")
            return rows, "book_mid" if rows else "none"
        if policy == "execution_only":
            rows = rows_for("executions")
            return rows, "executions" if rows else "none"
        if policy != "book_preferred":
            raise ValueError(f"unknown market-series policy: {policy}")
        rows = rows_for("book_mid")
        if rows:
            return rows, "book_mid"
        rows = rows_for("executions")
        return rows, "executions" if rows else "none"

    def market_state_before(
        self, condition_id: str, cutoff: float, lookback: float
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM market_state
            WHERE condition_id = ? AND ts < ? AND ts >= ?
            ORDER BY ts
            """,
            (condition_id, cutoff, cutoff - lookback),
        ).fetchall()

    def order_book_before(
        self, asset: str, cutoff: float
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM order_book_snapshots
            WHERE asset = ? AND observed_at < ?
            ORDER BY observed_at DESC LIMIT 1
            """,
            (asset, cutoff),
        ).fetchone()

    # ------------------------------------------------------------------
    # news
    def articles_asof(self, cutoff: float) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM news_articles
            WHERE first_observed_at < ?
            ORDER BY first_observed_at
            """,
            (cutoff,),
        ).fetchall()

    def event_families_asof(self, cutoff: float) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM event_families
            WHERE earliest_available_at < ?
            ORDER BY earliest_available_at
            """,
            (cutoff,),
        ).fetchall()

    def relevance_snapshot_asof(
        self,
        market_id: str,
        contract_version_seq: int,
        cutoff: float,
        method: str | None = None,
        model_version: str | None = None,
        allow_version_fallback: bool = True,
    ) -> tuple[list[sqlite3.Row], bool]:
        """Exactly one judgment per event family, strictly before cutoff.

        Primary rule: only judgments computed against the contract version
        active at the decision are eligible (obsolete contract-version
        judgments are excluded); the LATEST computed_at per family wins,
        so repeated recomputations never multiply news evidence; method /
        model_version restrict to approved scorers when configured.

        Fallback: batch normalization stamps judgments with the newest
        contract version known at normalization time, so a pipeline that
        does not recompute judgments per contract version may have zero
        judgments for the active version.  When ``allow_version_fallback``
        is set and the primary rule matches nothing, the latest judgment
        per family across versions (still strictly before the cutoff) is
        returned and the second element of the result is True so callers
        can flag the mismatch.  See docs/RESEARCH_ASSUMPTIONS.md.

        Returns (rows, used_version_fallback).
        """

        def query(version_clause: str, version_args: list) -> list[sqlite3.Row]:
            extra = ""
            extra_args: list = []
            if method is not None:
                extra += " AND method = ?"
                extra_args.append(method)
            if model_version is not None:
                extra += " AND model_version = ?"
                extra_args.append(model_version)
            inner_extra = extra.replace("method", "r2.method").replace(
                "model_version", "r2.model_version"
            )
            sql = f"""
                SELECT r.* FROM relevance_judgments r
                WHERE r.market_id = ? AND r.computed_at < ?
                  {version_clause.replace('contract_version_seq',
                                          'r.contract_version_seq')}
                  {extra.replace('method', 'r.method').replace(
                      'model_version', 'r.model_version')}
                  AND r.computed_at = (
                      SELECT MAX(r2.computed_at) FROM relevance_judgments r2
                      WHERE r2.event_family_id = r.event_family_id
                        AND r2.market_id = r.market_id AND r2.computed_at < ?
                        {version_clause.replace('contract_version_seq',
                                                'r2.contract_version_seq')}
                        {inner_extra}
                  )
                ORDER BY r.event_family_id
            """
            args = (
                [market_id, cutoff] + version_args + extra_args
                + [cutoff] + version_args + extra_args
            )
            return self._conn.execute(sql, args).fetchall()

        rows = query("AND contract_version_seq = ?", [contract_version_seq])
        if rows or not allow_version_fallback:
            return rows, False
        return query("", []), True

    def relevance_asof(self, market_id: str, cutoff: float) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM relevance_judgments
            WHERE market_id = ? AND computed_at < ?
            ORDER BY computed_at
            """,
            (market_id, cutoff),
        ).fetchall()

    def claims_asof(self, cutoff: float) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM news_claims
            WHERE first_available_at < ?
            ORDER BY first_available_at
            """,
            (cutoff,),
        ).fetchall()

    # ------------------------------------------------------------------
    # coverage
    def blocking_gaps(
        self,
        object_id: str | None,
        window_start: float,
        window_end: float,
    ) -> list[sqlite3.Row]:
        """Unresolved collector gaps overlapping [window_start, window_end)."""
        return self._conn.execute(
            """
            SELECT * FROM collector_gaps
            WHERE resolved_at IS NULL
              AND (object_id = ? OR object_id = '' OR ? IS NULL)
              AND gap_start < ?
              AND (gap_end IS NULL OR gap_end > ?)
            """,
            (object_id, object_id, window_end, window_start),
        ).fetchall()
