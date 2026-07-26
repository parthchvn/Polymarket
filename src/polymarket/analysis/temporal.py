"""Temporal validity enforcement.

The single scientific rule: for a decision at time ``t``, every piece of
context must satisfy ``available_at < t`` (strict).  This module provides
the runtime assertion used by replay.
"""

from __future__ import annotations

from typing import Any, Iterable


class TemporalContaminationError(AssertionError):
    """Raised when context contains information at or after decision time."""


_TS_KEYS = {
    "ts", "observed_at", "first_observed_at", "first_available_at",
    "effective_from", "computed_at", "earliest_available_at",
    "mapping_effective_from",
}


def _iter_timestamps(obj: Any, path: str = "") -> Iterable[tuple[str, float]]:
    if hasattr(obj, "keys"):  # sqlite3.Row or dict
        try:
            items = [(k, obj[k]) for k in obj.keys()]
        except Exception:
            items = []
        for key, value in items:
            if key in _TS_KEYS and isinstance(value, (int, float)):
                yield f"{path}.{key}" if path else key, float(value)
            elif isinstance(value, (dict, list, tuple)):
                yield from _iter_timestamps(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _iter_timestamps(item, f"{path}[{i}]")


def assert_no_future_information(context: Any, decision_time: float) -> None:
    """Fail if any recognized availability timestamp is >= decision_time."""
    violations = [
        (where, ts)
        for where, ts in _iter_timestamps(context)
        if ts >= decision_time
    ]
    if violations:
        detail = ", ".join(f"{w}={ts}" for w, ts in violations[:10])
        raise TemporalContaminationError(
            f"context contains information at or after decision time "
            f"{decision_time}: {detail}"
        )
