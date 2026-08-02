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


class LimitedClaimExtractor:
    """Budget wrapper for expensive (LLM) claim extractors.

    * Articles that already carry claims from the SAME extractor
      version are skipped without spending tokens, so repeated
      ``normalize --news-llm`` runs are cheap and resumable.
    * With ``limit=N``, at most N new articles are extracted this run;
      the rest return no claims and are picked up by the next run —
      the bounded-test workflow (``--llm-limit 10``).

    The wrapper is keyed by content hash of (headline, body), matching
    the article identity used by normalization.
    """

    def __init__(self, inner, conn, *, limit: int | None = None):
        self._inner = inner
        self._conn = conn
        self._limit = limit
        self.version = inner.version
        self.extracted = 0
        self.skipped_existing = 0
        self.deferred = 0

    def _already_extracted(self, headline: str, body: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM news_claims c
            JOIN news_articles a ON a.article_id = c.article_id
            WHERE a.headline = ? AND c.extractor_version = ?
            LIMIT 1
            """,
            (headline, self.version),
        ).fetchone()
        return row is not None

    def extract(self, headline: str, body: str) -> list[dict[str, Any]]:
        if self._already_extracted(headline, body):
            self.skipped_existing += 1
            return []
        if self._limit is not None and self.extracted >= self._limit:
            self.deferred += 1
            return []
        claims = self._inner.extract(headline, body)
        self.extracted += 1
        return claims


class LimitedRelevanceScorer:
    """Budget wrapper for expensive (LLM) relevance scorers: at most
    ``limit`` NEW scores per run; over-budget claims are skipped this
    run and picked up by the next (rescore-news and batch skip-existing
    make this resumable).  Separate from the claim-extraction budget so
    extraction and scoring can be bounded independently."""

    def __init__(self, inner, *, limit: int | None = None):
        self._inner = inner
        self._limit = limit
        self.version = inner.version
        self.method = getattr(inner, "method", "rule_keyword_overlap")
        self.scored = 0
        self.deferred = 0

    def score(self, claim_text, question, rules_text):
        if self._limit is not None and self.scored >= self._limit:
            self.deferred += 1
            return None                    # caller skips: no judgment
        result = self._inner.score(claim_text, question, rules_text)
        self.scored += 1
        return result


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
    method = "rule_keyword_overlap"
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


def _article_hash(conn: sqlite3.Connection, art_id: str) -> str:
    row = conn.execute(
        "SELECT content_hash FROM news_articles WHERE article_id = ?",
        (art_id,),
    ).fetchone()
    return row["content_hash"] if row else ""


def ingest_claims_for_article(
    conn: sqlite3.Connection,
    result: NormalizationResult,
    *,
    art_id: str,
    headline: str,
    body: str,
    first_observed_at: float,
    now: float,
    markets: list,
    extractor,
    scorer,
    surprise=None,
    content_hash: str | None = None,
) -> None:
    """Claim extraction + event-family linkage + relevance scoring for
    ONE article.  Shared by inline normalization and the LLM backfill:
    claim ``first_available_at`` is the article's observation time
    (text availability), while relevance rows carry their own
    ``scored_at`` — a backfilled extraction never pretends to have
    existed earlier than its text, and never hides when it was
    scored."""
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
                (family_id, content_hash or _article_hash(
                    conn, art_id
                ), claim_id),
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
        scoreable = [m for m in markets if m["version_seq"] is not None]
        if hasattr(scorer, "score_many"):
            # batched: one LLM call scores this claim against every
            # market (claims x markets calls collapse to claims)
            batch = scorer.score_many(
                claim["claim_text"],
                [{"question": m["question"],
                  "rules_text": m["rules_text"]} for m in scoreable],
            )
            scored_pairs = list(zip(scoreable, batch))
        else:
            scored_pairs = [
                (m, scorer.score(
                    claim["claim_text"], m["question"], m["rules_text"]
                ))
                for m in scoreable
            ]
        for market, scored in scored_pairs:
            if scored is None:
                continue    # deferred by budget, or unscored in batch
            novelty = 1.0 if edge_type == "new" else 0.3
            model_version = getattr(
                scorer, "version", RELEVANCE_MODEL_VERSION
            )
            scorer_method = getattr(
                scorer, "method", "rule_keyword_overlap"
            )
            judgment_id = namespace_id(
                "relevance", claim_id, family_id,
                market["market_id"], market["version_seq"],
                scorer_method, model_version,
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
                    surprise, scorer_method,
                    model_version,
                    canonical_json(scored.get("evidence", {})),
                ),
            )
            result.add_inserted("relevance_judgments")


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

        ingest_claims_for_article(
            conn, result,
            art_id=art_id, headline=headline, body=body,
            first_observed_at=first_observed_at, now=now,
            markets=markets, extractor=extractor, scorer=scorer,
            surprise=record.get("surprise"),
            content_hash=content_hash,
        )
    conn.commit()


def market_contract_rows(conn: sqlite3.Connection) -> list:
    return conn.execute(
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


RELEVANT_PRIORITY_CLASSES = (
    "supports_positive", "supports_negative", "indirect",
)


def backfill_llm_claims(
    conn: sqlite3.Connection,
    extractor,
    scorer,
    *,
    limit: int | None = None,
    order: str = "newest",
    min_body_chars: int = 400,
    min_rel_score: float = 0.03,
    relevance_filter: bool = True,
) -> dict:
    """Run an (LLM) claim extractor over EXISTING articles that have no
    claims from this extractor version — normalization only extracts
    on first article insert, so body-level LLM extraction needs this
    backfill for articles first normalized by the rule extractor.
    Resumable: already-covered articles are skipped; ``limit`` bounds
    NEW extractions per run.

    At minutes-per-article LLM cost against a growing feed, the queue
    is PRIORITIZED, not exhaustive: newest first (fresh news feeds the
    online screens; stale generic RSS does not), only articles with a
    real body (headline-only RSS gains nothing from body extraction),
    and only articles whose cheap rule-scored claims already look
    plausibly relevant to some market (``min_rel_score`` on any
    judgment, or any judgment in the relevant classes).  Each filter
    can be relaxed via arguments."""
    import time as _time

    version = getattr(extractor, "version", None)
    conditions = [
        """NOT EXISTS (
            SELECT 1 FROM news_claims c
            WHERE c.article_id = a.article_id
              AND c.extractor_version = ?
        )""",
        "LENGTH(COALESCE(a.body, '')) >= ?",
    ]
    params: list = [version, min_body_chars]
    if relevance_filter:
        placeholders = ",".join("?" for _ in RELEVANT_PRIORITY_CLASSES)
        conditions.append(
            f"""EXISTS (
            SELECT 1 FROM news_claims c
            JOIN relevance_judgments r ON r.claim_id = c.claim_id
            WHERE c.article_id = a.article_id
              AND (r.rel_score >= ?
                   OR r.rel_class IN ({placeholders}))
        )"""
        )
        params.append(min_rel_score)
        params.extend(RELEVANT_PRIORITY_CLASSES)
    direction = "DESC" if order == "newest" else "ASC"
    rows = conn.execute(
        f"""
        SELECT a.article_id, a.headline, a.body, a.first_observed_at,
               a.download_completed_at
        FROM news_articles a
        WHERE {" AND ".join(conditions)}
        ORDER BY a.first_observed_at {direction}
        """,
        params,
    ).fetchall()
    result = NormalizationResult(
        raw_response_id=-1, collector="backfill", endpoint="backfill"
    )
    markets = market_contract_rows(conn)
    now = _time.time()
    processed = 0
    failures: list[str] = []
    for row in rows:
        if limit is not None and processed >= limit:
            break
        try:
            ingest_claims_for_article(
                conn, result,
                art_id=row["article_id"],
                headline=row["headline"] or "",
                body=row["body"] or "",
                # availability honesty: a claim from a
                # late-downloaded body was readable only once the text
                # was in hand
                first_observed_at=max(
                    float(row["first_observed_at"]),
                    float(row["download_completed_at"] or 0.0),
                ),
                now=now, markets=markets,
                extractor=extractor, scorer=scorer,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:                # noqa: BLE001
            # a poison article (degenerate generation, parse failure)
            # must not block the queue: record it, roll back its
            # partial writes, move on
            conn.rollback()
            failures.append(f"{row['article_id']}: {exc}")
            continue
        processed += 1
        # commit PER ARTICLE: LLM extraction can take minutes per
        # article, so progress must be durable and interruption must
        # be free (Ctrl-C loses at most the article in flight)
        conn.commit()
    conn.commit()
    return {
        "articles_pending": len(rows),
        "articles_processed": processed,
        "articles_failed": len(failures),
        "failed_examples": failures[:5],
        "inserted": dict(result.inserted),
        "extractor_version": version,
        "queue_policy": {
            "order": order,
            "min_body_chars": min_body_chars,
            "relevance_filter": relevance_filter,
            "min_rel_score": min_rel_score if relevance_filter
            else None,
        },
    }
