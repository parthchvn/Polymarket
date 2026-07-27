"""Normalize news payloads into a temporally versioned news ledger.

Expected record shape:

.. code-block:: json

    {
      "id": "src-article-1",
      "url": "https://...",
      "publishedAt": 1700000000,
      "timestampSource": "feed",
      "timestampConfidence": 0.9,
      "headline": "...",
      "body": "...",
      "previousId": null
    }

``first_observed_at`` comes from the collector's receive time, never from
the publisher: publication time is not first collector availability.
Updates containing new information create a new article record linked to
the previous record — old records are never rewritten.

Claim extraction and relevance scoring are defined as protocols with
deterministic rule-based default implementations, so the standard test
suite requires no external LLM API.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Protocol

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
from polymarket.contracts.types import NormalizationResult
from polymarket.contracts.versions import EXTRACTOR_VERSION, RELEVANCE_MODEL_VERSION
from polymarket.normalization.ids import (
    article_id as make_article_id,
)
from polymarket.normalization.ids import (
    canonical_json,
    namespace_id,
    raw_record_hash,
    sha256_bytes,
)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is",
    "are", "was", "were", "will", "by", "at", "with", "as", "that", "this",
}
_POSITIVE_CUES = {
    "wins", "win", "won", "leads", "lead", "surges", "confirmed",
    "approves", "approved", "passes", "passed", "yes", "rises", "ahead",
}
_NEGATIVE_CUES = {
    "loses", "lose", "lost", "trails", "trail", "collapses", "denied",
    "rejects", "rejected", "fails", "failed", "no", "falls", "behind",
}


def _tokens(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


class ClaimExtractor(Protocol):
    def extract(self, headline: str, body: str) -> list[dict[str, Any]]: ...


class RelevanceScorer(Protocol):
    def score(
        self, claim_text: str, question: str, rules_text: str | None
    ) -> dict[str, Any]: ...


class RuleBasedClaimExtractor:
    """One claim per article: headline plus first sentence of the body."""

    version = EXTRACTOR_VERSION

    def extract(self, headline: str, body: str) -> list[dict[str, Any]]:
        first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0] if body else ""
        claim_text = (headline or "").strip()
        if first_sentence and first_sentence.lower() != claim_text.lower():
            claim_text = f"{claim_text}. {first_sentence}".strip(". ")
        if not claim_text:
            return []
        entities = sorted(
            {
                w
                for w in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", f"{headline} {body}")
                if w.lower() not in _STOPWORDS
            }
        )
        quantities = re.findall(r"\b\d+(?:\.\d+)?%?\b", f"{headline} {body}")
        return [
            {
                "claim_text": claim_text,
                "entities": entities,
                "quantities": quantities,
                "supporting_span": headline,
                "confidence": 0.5,
            }
        ]


class RuleBasedRelevanceScorer:
    """Keyword-overlap relevance with cue-word direction."""

    version = RELEVANCE_MODEL_VERSION

    def score(
        self, claim_text: str, question: str, rules_text: str | None
    ) -> dict[str, Any]:
        claim_tokens = set(_tokens(claim_text))
        question_tokens = set(_tokens(f"{question or ''} {rules_text or ''}"))
        if not claim_tokens or not question_tokens:
            return {"rel_class": "irrelevant", "rel_score": 0.0, "direction": 0.0}
        overlap = claim_tokens & question_tokens
        rel_score = len(overlap) / max(len(question_tokens), 1)
        raw_words = set(re.findall(r"[a-z']+", claim_text.lower()))
        direction = 0.0
        if raw_words & _POSITIVE_CUES:
            direction += 1.0
        if raw_words & _NEGATIVE_CUES:
            direction -= 1.0
        if rel_score >= 0.3 and direction > 0:
            rel_class = "supports_positive"
        elif rel_score >= 0.3 and direction < 0:
            rel_class = "supports_negative"
        elif rel_score >= 0.3:
            rel_class = "ambiguous"
        elif rel_score >= 0.1:
            rel_class = "indirect"
        elif rel_score > 0:
            rel_class = "background"
        else:
            rel_class = "irrelevant"
        return {
            "rel_class": rel_class,
            "rel_score": round(rel_score, 6),
            "direction": direction,
            "evidence": {"overlap": sorted(overlap)},
        }


def _event_family_key(entities: list[str], claim_text: str) -> str:
    if entities:
        return "|".join(sorted(e.lower() for e in entities)[:4])
    return "|".join(sorted(set(_tokens(claim_text)))[:4]) or "misc"


def normalize_news(
    conn: sqlite3.Connection,
    raw_row: sqlite3.Row,
    records: list[dict[str, Any]],
    result: NormalizationResult,
    *,
    extractor: ClaimExtractor | None = None,
    scorer: RelevanceScorer | None = None,
    source_id: str | None = None,
) -> None:
    extractor = extractor or RuleBasedClaimExtractor()
    scorer = scorer or RuleBasedRelevanceScorer()
    now = time.time()
    raw_id = int(raw_row["raw_response_id"])
    first_observed_at = float(raw_row["received_at"])
    src = source_id or str(raw_row["collector"]).removeprefix("news:")
    markets = conn.execute(
        """
        SELECT m.market_id, m.question,
               cv.version_seq, cv.rules_text
        FROM markets m
        LEFT JOIN contract_versions cv ON cv.market_id = m.market_id
            AND cv.version_seq = (
                SELECT MAX(version_seq) FROM contract_versions
                WHERE market_id = m.market_id
            )
        """
    ).fetchall()

    for index, record in enumerate(records):
        rec_hash = raw_record_hash(record)
        headline = record.get("headline") or ""
        body = record.get("body") or ""
        content_hash = sha256_bytes(
            canonical_json({"headline": headline, "body": body}).encode()
        )
        art_id = make_article_id(src, content_hash, raw_id)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO news_articles
                (article_id, source_id, source_url, source_published_at,
                 first_observed_at, download_completed_at, timestamp_source,
                 timestamp_confidence, headline, body, content_hash,
                 previous_article_id, raw_response_id, raw_record_index,
                 raw_record_hash, parser_version, schema_version,
                 normalized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                art_id, src, record.get("url"), record.get("publishedAt"),
                first_observed_at, first_observed_at,
                record.get("timestampSource"),
                record.get("timestampConfidence"), headline, body,
                content_hash, record.get("previousId"),
                raw_id, index, rec_hash,
                PARSER_VERSION, SCHEMA_VERSION, now,
            ),
        )
        if not cur.rowcount:
            result.add_ignored("news_articles")
            continue
        result.add_inserted("news_articles")

        for claim in extractor.extract(headline, body):
            claim_id = namespace_id("claim", art_id, claim["claim_text"])
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO news_claims
                    (claim_id, article_id, claim_text, entities_json,
                     quantities_json, supporting_span, first_available_at,
                     extractor_version, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_id, art_id, claim["claim_text"],
                    canonical_json(claim.get("entities", [])),
                    canonical_json(claim.get("quantities", [])),
                    claim.get("supporting_span"), first_observed_at,
                    getattr(extractor, "version", EXTRACTOR_VERSION),
                    claim.get("confidence"),
                ),
            )
            if not cur.rowcount:
                result.add_ignored("news_claims")
                continue
            result.add_inserted("news_claims")

            # ---- event family --------------------------------------------
            family_key = _event_family_key(
                claim.get("entities", []), claim["claim_text"]
            )
            family_id = namespace_id("event_family", family_key)
            existing = conn.execute(
                "SELECT earliest_available_at FROM event_families "
                "WHERE event_family_id = ?",
                (family_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO event_families
                        (event_family_id, label, earliest_available_at,
                         created_by, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (family_id, family_key, first_observed_at,
                     "rule-based", now),
                )
                result.add_inserted("event_families")
                edge_type = "new"
            else:
                duplicate = conn.execute(
                    """
                    SELECT 1 FROM news_articles a
                    JOIN news_claims c ON c.article_id = a.article_id
                    JOIN claim_edges e ON e.claim_id = c.claim_id
                    WHERE e.event_family_id = ? AND a.content_hash = ?
                      AND c.claim_id != ?
                    LIMIT 1
                    """,
                    (family_id, content_hash, claim_id),
                ).fetchone()
                edge_type = "duplicate" if duplicate else "confirmation"
            conn.execute(
                """
                INSERT OR IGNORE INTO claim_edges
                    (edge_id, claim_id, event_family_id, edge_type,
                     effective_from, evidence, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace_id("edge", claim_id, family_id), claim_id,
                    family_id, edge_type, first_observed_at, family_key, 0.5,
                ),
            )
            result.add_inserted("claim_edges")

            # ---- relevance judgments -------------------------------------
            for market in markets:
                if market["version_seq"] is None:
                    continue
                scored = scorer.score(
                    claim["claim_text"], market["question"], market["rules_text"]
                )
                novelty = 1.0 if edge_type == "new" else 0.3
                model_version = getattr(
                    scorer, "version", RELEVANCE_MODEL_VERSION
                )
                judgment_id = namespace_id(
                    "relevance", claim_id, family_id,
                    market["market_id"], market["version_seq"],
                    "rule_keyword_overlap", model_version,
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO relevance_judgments
                        (relevance_judgment_id, claim_id, event_family_id,
                         market_id, contract_version_seq,
                         source_effective_at, scored_at, computed_at,
                         rel_class, rel_score, direction, novelty,
                         surprise, method, model_version, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        judgment_id, claim_id, family_id,
                        market["market_id"], market["version_seq"],
                        first_observed_at, now, first_observed_at,
                        scored["rel_class"], scored["rel_score"],
                        scored["direction"], novelty,
                        record.get("surprise"), "rule_keyword_overlap",
                        model_version,
                        canonical_json(scored.get("evidence", {})),
                    ),
                )
                result.add_inserted("relevance_judgments")
    conn.commit()
