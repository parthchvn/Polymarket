"""Protocol for strict as-of readers.

Every method that accepts a cutoff MUST use strict inequality:
``available_at < cutoff``.  Rows exactly at the cutoff are excluded.
"""

from __future__ import annotations

from typing import Any, Protocol


class AsOfReader(Protocol):
    def actor_trade_legs_before(
        self,
        cutoff: float,
        actor: str | None = None,
        condition_id: str | None = None,
        liquidity_role: str | None = None,
    ) -> list[Any]: ...

    def canonical_executions_before(
        self, cutoff: float, condition_id: str | None = None
    ) -> list[Any]: ...

    def position_events_before(
        self,
        cutoff: float,
        wallet: str | None = None,
        condition_id: str | None = None,
    ) -> list[Any]: ...

    def position_asof(
        self, wallet: str, condition_id: str, cutoff: float
    ) -> dict[str, float]: ...

    def market_status_asof(self, market_id: str, cutoff: float) -> Any: ...

    def contract_asof(self, market_id: str, cutoff: float) -> Any: ...

    def market_state_before(
        self, condition_id: str, cutoff: float, lookback: float
    ) -> list[Any]: ...

    def articles_asof(self, cutoff: float) -> list[Any]: ...

    def event_families_asof(self, cutoff: float) -> list[Any]: ...

    def relevance_asof(self, market_id: str, cutoff: float) -> list[Any]: ...
