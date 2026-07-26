"""Opportunity / risk-set validity for actor-market decisions.

A valid opportunity requires the market to exist, be open and tradable
as-of the decision time, required coverage to exist, and no unresolved
blocking gap overlapping the context window.  Missing contexts are
represented explicitly — never treated as valid zero-valued observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polymarket.analysis.reader import SQLiteNormalizedReader


@dataclass
class OpportunityCheck:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


def check_opportunity(
    reader: SQLiteNormalizedReader,
    *,
    actor: str,
    condition_id: str,
    decision_time: float,
    context_lookback: float = 86400.0,
    exclude_combo: bool = True,
) -> OpportunityCheck:
    reasons: list[str] = []
    coverage: dict = {}

    market = reader.market_by_condition(condition_id)
    if market is None:
        return OpportunityCheck(False, ["market does not exist"], coverage)
    if exclude_combo and market["is_combo"]:
        reasons.append("combo market excluded")

    status = reader.market_status_asof(market["market_id"], decision_time)
    coverage["has_status"] = status is not None
    if status is None:
        reasons.append("no market status as-of decision time")
    else:
        if not status["trading_enabled"]:
            reasons.append("trading not enabled")
        if status["closed"]:
            reasons.append("market closed")
        if status["resolved"]:
            reasons.append("market resolved")

    contract = reader.contract_asof(market["market_id"], decision_time)
    coverage["has_contract_version"] = contract is not None
    if contract is None:
        reasons.append("no contract version as-of decision time")

    state = reader.market_state_before(
        condition_id, decision_time, context_lookback
    )
    coverage["market_state_rows"] = len(state)
    if not state:
        reasons.append("no market-state coverage in context window")

    history = reader.actor_trade_legs_before(decision_time, actor=actor)
    coverage["actor_history_rows"] = len(history)

    position = reader.position_asof(actor, condition_id, decision_time)
    coverage["position_history_complete"] = position["complete"]
    coverage["unresolved_position_events"] = position["unresolved_event_count"]

    gaps = reader.blocking_gaps(
        condition_id, decision_time - context_lookback, decision_time
    )
    coverage["blocking_gap_count"] = len(gaps)
    if gaps:
        reasons.append("unresolved collector gap overlaps context window")

    return OpportunityCheck(valid=not reasons, reasons=reasons, coverage=coverage)
