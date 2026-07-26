"""Shared value types used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

UNKNOWN = "unknown"
UNRESOLVED = "unresolved"
INCOMPLETE = "incomplete"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RawRecordRef:
    """Provenance pointer into a raw response."""

    raw_response_id: int
    raw_record_index: int
    raw_record_hash: str


@dataclass
class NormalizationResult:
    """Outcome of normalizing one raw response."""

    raw_response_id: int
    collector: str
    endpoint: str
    inserted: dict[str, int] = field(default_factory=dict)
    ignored: dict[str, int] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_inserted(self, table: str, n: int = 1) -> None:
        self.inserted[table] = self.inserted.get(table, 0) + n

    def add_ignored(self, table: str, n: int = 1) -> None:
        self.ignored[table] = self.ignored.get(table, 0) + n
