"""Versioned relevance rescoring for stored news claims.

Re-scores existing claims against the exact contract semantics with a
chosen scorer (rule-based or Ollama LLM), writing NEW versioned
judgments — existing judgments are never rewritten or deleted, so every
run's judgments remain auditable side by side.

Temporal semantics (matching batch normalization): a judgment's
``computed_at`` is anchored to the article's ``first_observed_at`` —
relevance is a property of (article, contract) available from the
moment the article was observed — plus a one-second method offset so
the rescored judgment (a) has a distinct primary key and (b) becomes
the LATEST judgment per family, which the as-of relevance snapshot
prefers.  Provenance (method, model version, wall-clock rescore time)
is recorded on every row.

Resumable: (family, market, version, method, model_version)
combinations that already have a judgment are skipped, so an
interrupted long LLM run continues where it stopped.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from polymarket.collection.canonical import canonical_json

_METHOD_OFFSET_SECONDS = 1.0


def _targets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every (claim, family, market, contract version active at the
    article's first observation) pair, oldest articles first."""
    return conn.execute(
        """
        SELECT c.claim_id, c.claim_text, e.event_family_id,
               a.first_observed_at,
               m.market_id, v.version_seq, v.question, v.rules_text
        FROM news_claims c
        JOIN news_articles a ON a.article_id = c.article_id
        JOIN claim_edges e ON e.claim_id = c.claim_id
        JOIN markets m
        JOIN contract_versions v ON v.market_id = m.market_id
         AND v.first_observed_at = (
             SELECT MAX(v2.first_observed_at) FROM contract_versions v2
             WHERE v2.market_id = m.market_id
               AND v2.first_observed_at <= a.first_observed_at
         )
        ORDER BY a.first_observed_at, c.claim_id, m.market_id
        """
    ).fetchall()


def rescore_news(
    conn: sqlite3.Connection,
    scorer: Any,
    *,
    method: str,
    limit: int | None = None,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Rescore stored claims with ``scorer`` (``.score(claim_text,
    question, rules_text) -> dict`` with rel_class/rel_score/direction,
    ``.version`` attribute).  Returns counters.  Never rewrites."""
    model_version = getattr(scorer, "version", "unversioned")
    existing = {
        tuple(row) for row in conn.execute(
            """
            SELECT event_family_id, market_id, contract_version_seq
            FROM relevance_judgments
            WHERE method = ? AND model_version = ?
            """,
            (method, model_version),
        )
    }
    counters: dict[str, Any] = {
        "scored": 0, "skipped_existing": 0, "errors": 0,
        "by_class": {}, "method": method, "model_version": model_version,
    }
    now = time.time()
    for row in _targets(conn):
        key = (row["event_family_id"], row["market_id"], row["version_seq"])
        if key in existing:
            counters["skipped_existing"] += 1
            continue
        existing.add(key)  # one judgment per family/market/version/run
        try:
            scored = scorer.score(
                row["claim_text"], row["question"], row["rules_text"]
            )
        except Exception as exc:  # noqa: BLE001 - resumable long runs
            counters["errors"] += 1
            counters.setdefault("error_samples", [])
            if len(counters["error_samples"]) < 5:
                counters["error_samples"].append(str(exc)[:160])
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO relevance_judgments
                (event_family_id, market_id, contract_version_seq,
                 computed_at, rel_class, rel_score, direction, novelty,
                 surprise, method, model_version, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                row["event_family_id"], row["market_id"],
                row["version_seq"],
                float(row["first_observed_at"]) + _METHOD_OFFSET_SECONDS,
                scored["rel_class"], scored["rel_score"],
                scored["direction"], method, model_version,
                canonical_json({
                    "rescored_at": now,
                    "claim_id": row["claim_id"],
                    **(scored.get("evidence") or {}),
                }),
            ),
        )
        counters["scored"] += 1
        counters["by_class"][scored["rel_class"]] = (
            counters["by_class"].get(scored["rel_class"], 0) + 1
        )
        if limit is not None and counters["scored"] >= limit:
            break
        if counters["scored"] % progress_every == 0:
            conn.commit()
            print(
                f"  rescored {counters['scored']} "
                f"(skipped {counters['skipped_existing']}, "
                f"errors {counters['errors']})",
                flush=True,
            )
    conn.commit()
    return counters


def make_scorer(method: str, model: str | None = None) -> Any:
    if method == "rule":
        from polymarket.normalization.news import RuleBasedRelevanceScorer

        return RuleBasedRelevanceScorer()
    if method == "ollama":
        from polymarket.normalization.llm_news import OllamaRelevanceScorer

        return OllamaRelevanceScorer(model=model or "qwen3:8b")
    raise ValueError(f"unknown rescore method: {method}")
