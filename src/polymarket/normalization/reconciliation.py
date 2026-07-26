"""Maker/taker reconciliation between expanded actor legs and taker-only
canonical executions.

Matching uses the strongest available identifiers in order: source record
ID, transaction hash, log index, transaction-local occurrence, condition
ID, asset (via positive conversion), size, price, timestamp tolerance.

Rules:
* an expanded leg matching a taker-only record is ``taker``;
* other confidently matched legs in the same execution are ``maker``;
* ambiguous matches stay ``unknown`` — roles are never silently forced.

Note: ``liquidity_role`` on actor_trade_legs is derived reconciliation
state, not a raw observation, so updating it does not violate raw
immutability.  Raw responses themselves are never touched.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from polymarket.normalization.trades import to_positive

TS_TOLERANCE = 2.0
SIZE_TOLERANCE = 1e-9
PRICE_TOLERANCE = 1e-9


@dataclass
class ReconciliationDiagnostics:
    legs_considered: int = 0
    taker_assigned: int = 0
    maker_assigned: int = 0
    unknown_remaining: int = 0
    ambiguous_transactions: list[str] = field(default_factory=list)


def _leg_matches_execution(leg: sqlite3.Row, execution: sqlite3.Row) -> bool:
    if leg["transaction_hash"] != execution["transaction_hash"]:
        return False
    if (
        leg["source_record_id"]
        and execution["source_record_id"]
        and leg["source_record_id"] == execution["source_record_id"]
    ):
        return True
    if (
        leg["transaction_log_index"] is not None
        and execution["transaction_log_index"] is not None
        and leg["transaction_log_index"] != execution["transaction_log_index"]
    ):
        return False
    if (
        leg["transaction_occurrence"] is not None
        and execution["transaction_occurrence"] is not None
        and leg["transaction_occurrence"] != execution["transaction_occurrence"]
    ):
        return False
    if leg["condition_id"] != execution["condition_id"]:
        return False
    if leg["outcome_sign"] is None:
        return False
    leg_pos_price, leg_pos_side = to_positive(
        leg["price"], leg["side"], leg["outcome_sign"]
    )
    if abs(leg["size"] - execution["size"]) > SIZE_TOLERANCE:
        return False
    if abs(leg_pos_price - execution["positive_price"]) > PRICE_TOLERANCE:
        return False
    if abs(leg["ts"] - execution["ts"]) > TS_TOLERANCE:
        return False
    return True


def reconcile_roles(conn: sqlite3.Connection) -> ReconciliationDiagnostics:
    diag = ReconciliationDiagnostics()
    legs = conn.execute(
        "SELECT * FROM actor_trade_legs ORDER BY transaction_hash, ts"
    ).fetchall()
    diag.legs_considered = len(legs)
    executions_by_tx: dict[str, list[sqlite3.Row]] = {}
    for execution in conn.execute("SELECT * FROM canonical_executions"):
        executions_by_tx.setdefault(execution["transaction_hash"], []).append(
            execution
        )

    for leg in legs:
        executions = executions_by_tx.get(leg["transaction_hash"], [])
        matches = [e for e in executions if _leg_matches_execution(leg, e)]
        if not executions:
            role, confidence = "unknown", "no_canonical_execution"
        elif not matches:
            # confidently part of a known execution's transaction but not
            # the taker-side record itself
            wallet_is_taker = any(
                e["taker_wallet"] == leg["proxy_wallet"] for e in executions
            )
            if wallet_is_taker:
                role, confidence = "unknown", "ambiguous_taker_wallet"
                diag.ambiguous_transactions.append(leg["transaction_hash"])
            else:
                role, confidence = "maker", "matched_transaction"
        elif len(matches) == 1:
            match = matches[0]
            if (
                match["taker_wallet"] is not None
                and match["taker_wallet"] != leg["proxy_wallet"]
            ):
                role, confidence = "maker", "matched_execution_other_taker"
            else:
                role, confidence = "taker", "matched_execution"
        else:
            role, confidence = "unknown", "multiple_candidate_executions"
            diag.ambiguous_transactions.append(leg["transaction_hash"])
        conn.execute(
            """
            UPDATE actor_trade_legs
            SET liquidity_role = ?, role_confidence = ?
            WHERE actor_leg_id = ?
            """,
            (role, confidence, leg["actor_leg_id"]),
        )
        if role == "taker":
            diag.taker_assigned += 1
        elif role == "maker":
            diag.maker_assigned += 1
        else:
            diag.unknown_remaining += 1
    conn.commit()
    return diag
