"""Ex-post outcome layer O — realised post-decision drift, kept
STRICTLY out of P(R|D,C).

Outcomes are attached to DRC records in a separate pass AFTER contexts,
features, attributions, posteriors and counterfactuals are complete;
nothing in this module is importable by the feature or reasoning
layers (enforced by test).  Each record's ``O`` block states this
contract on itself.

Measurement discipline matches the underreaction analysis: endpoints
via ``close_near_target`` (strictly after the decision, within one bar
of the target — stale series yield missing, never zero), and horizon
windows censored unless the market is positively known open through
the endpoint.
"""

from __future__ import annotations

import sqlite3

OUTCOME_HORIZONS = (3600.0, 6 * 3600.0, 24 * 3600.0)

OUTCOME_CONTRACT = (
    "ex-post outcomes computed after the decision; NEVER inputs to "
    "features, attribution, posteriors or counterfactuals (O is not "
    "part of C)"
)


def compute_outcomes(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    decision_time: float,
    direction: str | None,
    bin_seconds: float = 900.0,
    horizons: tuple[float, ...] = OUTCOME_HORIZONS,
) -> dict:
    """Realised log-odds drift after one decision, per horizon, with
    censoring and honest missingness."""
    from polymarket.analysis.underreaction import (
        CloseSeries,
        MarketCensor,
    )

    closes = CloseSeries(conn, bin_seconds)
    censor = MarketCensor(conn)
    base = closes.close_asof_with_time(
        condition_id, decision_time, max_staleness=2 * bin_seconds
    )
    sign = (
        1.0 if direction == "positive"
        else -1.0 if direction == "negative" else None
    )
    out: dict = {"contract": OUTCOME_CONTRACT, "horizons": {}}
    for horizon in horizons:
        key = f"{int(horizon)}s"
        if base is None:
            out["horizons"][key] = {"status": "no_base_close"}
            continue
        if not censor.open_through(
            condition_id, decision_time, decision_time + horizon
        ):
            out["horizons"][key] = {"status": "censored"}
            continue
        future = closes.close_near_target(
            condition_id, decision_time + horizon, after=base[0]
        )
        if future is None:
            out["horizons"][key] = {"status": "stale_endpoint"}
            continue
        drift = future[1] - base[1]
        entry: dict = {"status": "observed", "realized_drift": drift}
        if sign is not None:
            entry["same_direction_continuation"] = bool(
                sign * drift > 0
            )
        out["horizons"][key] = entry
    observed = [
        h for h in out["horizons"].values() if h["status"] == "observed"
    ]
    if observed and sign is not None:
        continuations = [
            h["same_direction_continuation"] for h in observed
        ]
        out["ex_post_continuation_rate"] = (
            sum(continuations) / len(continuations)
        )
    return out


def attach_outcomes(
    records: list[dict],
    conn: sqlite3.Connection,
    *,
    bin_seconds: float = 900.0,
) -> int:
    """Attach an ``O`` block to each DRC record in place.  Runs as a
    final export pass — features and posteriors are already frozen in
    the records by the time this executes."""
    attached = 0
    for record in records:
        decision = record.get("D", {})
        condition_id = decision.get("condition_id")
        decision_time = decision.get("decision_time")
        if condition_id is None or decision_time is None:
            continue
        record["O"] = compute_outcomes(
            conn,
            condition_id=condition_id,
            decision_time=float(decision_time),
            direction=decision.get("direction"),
            bin_seconds=bin_seconds,
        )
        attached += 1
    return attached
