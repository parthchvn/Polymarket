"""Canonical serialization, hashing and deterministic identifiers.

Never use Python's built-in ``hash()`` for persistent identity: it is
salted per-process.  All persistent identity here is derived from SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest over exact bytes."""
    return hashlib.sha256(data).hexdigest()


def raw_record_hash(record: Any) -> str:
    """Hash of one canonicalized record inside a raw payload."""
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def namespace_id(namespace: str, *parts: Any) -> str:
    """Deterministic ID from a namespace and ordered parts.

    Re-running the same parser over the same raw data yields the same ID
    (idempotent inserts).  Where legitimate repetition must survive, the
    caller includes raw provenance (raw_response_id, record index) in the
    parts so distinct observations remain distinct.
    """
    payload = canonical_json([namespace, *[str(p) for p in parts]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
