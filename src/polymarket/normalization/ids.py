"""Deterministic identifiers for normalized rows.

IDs are constructed so that re-running the same parser is idempotent
(same raw row -> same ID -> INSERT OR IGNORE), while legitimate repeated
source records remain distinct because raw provenance (raw_response_id,
record index) is included whenever multiplicity matters.
"""

from __future__ import annotations

from polymarket.collection.canonical import (
    canonical_json,
    namespace_id,
    raw_record_hash,
    sha256_bytes,
)

__all__ = [
    "canonical_json",
    "namespace_id",
    "raw_record_hash",
    "sha256_bytes",
    "actor_leg_id",
    "execution_id",
    "position_event_id",
    "article_id",
    "candidate_fingerprint",
]


def candidate_fingerprint(
    transaction_hash: str,
    proxy_wallet: str,
    asset: str,
    side: str,
    size: float,
    price: float,
) -> str:
    """Diagnostic fingerprint.  NOT proven globally unique — never use it
    alone as a primary key."""
    return namespace_id(
        "fingerprint", transaction_hash, proxy_wallet, asset, side, size, price
    )


def actor_leg_id(raw_response_id: int, raw_record_index: int) -> str:
    return namespace_id("actor_leg", raw_response_id, raw_record_index)


def execution_id(raw_response_id: int, raw_record_index: int) -> str:
    return namespace_id("execution", raw_response_id, raw_record_index)


def position_event_id(
    raw_response_id: int, raw_record_index: int, sub_index: int = 0
) -> str:
    return namespace_id("position_event", raw_response_id, raw_record_index, sub_index)


def article_id(source_id: str, content_hash: str, raw_response_id: int) -> str:
    return namespace_id("article", source_id, content_hash, raw_response_id)
