"""Occurrence reasoning target: at-risk interval -> trade / no-trade.

Direction reasoning explains WHICH WAY an observed clean taker trade
went; it never explains why the wallet traded at all.  This builder
creates an ANALYTICAL opportunity grid so a separate occurrence head can
model P(trade | at risk, C).  Size is deliberately not modelled.

An at-risk interval exists only when, at the interval start (strictly
pre-interval information):

* the market was open and trading was enabled;
* coverage was certified: a market_state observation exists within the
  lookback and no blocking collector gap overlaps the interval;
* the actor was engaged: non-dust position, or a qualifying taker
  execution within the recency window.

Occurrence opportunities carry ``direction=None`` pseudo-episodes: they
can NEVER enter the direction model (which requires clean single-
direction taker trades), and clean-trade episodes with mixed direction
stay out of the direction model exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from polymarket.analysis.decisions import DecisionEpisode
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.collection.canonical import namespace_id

DUST_POSITION = 1.0
RECENT_EXECUTION_WINDOW = 7 * 86400.0

TARGET_DIRECTION = "direction"
TARGET_OCCURRENCE = "occurrence"


@dataclass
class AtRiskOpportunity:
    actor_id: str
    condition_id: str
    market_id: str | None
    interval_start: float
    interval_end: float
    traded: bool
    episode: DecisionEpisode  # pseudo-episode (direction=None) for features


def _actor_engaged(
    reader: SQLiteNormalizedReader,
    actor_id: str,
    condition_id: str,
    t: float,
    taker_times: list[float],
) -> bool:
    position = reader.position_asof(actor_id, condition_id, t)
    balances = position.get("balances", {}) if position else {}
    if any(abs(b) > DUST_POSITION for b in balances.values()):
        return True
    return any(
        t - RECENT_EXECUTION_WINDOW <= leg_time < t for leg_time in taker_times
    )


def build_at_risk_opportunities(
    reader: SQLiteNormalizedReader,
    *,
    end_time: float,
    interval_seconds: float = 6 * 3600.0,
    market_state_lookback: float = 24 * 3600.0,
) -> list[AtRiskOpportunity]:
    """Build the at-risk grid for every (actor, condition) pair in which
    the actor has any taker execution.  Deterministic."""
    conn = reader._conn
    # the at-risk pair universe is the UNION of every surface through
    # which an actor can hold risk before an interval: taker executions,
    # position events (trades, splits, merges, transfers, redemptions)
    # and reported position snapshots (mapped to conditions through the
    # outcome-token mapping)
    pairs = conn.execute(
        """
        SELECT DISTINCT proxy_wallet, condition_id FROM (
            SELECT proxy_wallet, condition_id FROM actor_trade_legs
            WHERE liquidity_role = 'taker'
            UNION
            SELECT wallet AS proxy_wallet, condition_id
            FROM position_events
            UNION
            SELECT s.wallet AS proxy_wallet, o.condition_id
            FROM position_snapshots s
            JOIN outcome_tokens o ON o.asset = s.asset
        )
        ORDER BY proxy_wallet, condition_id
        """
    ).fetchall()
    opportunities: list[AtRiskOpportunity] = []
    for actor_id, condition_id in pairs:
        market_row = conn.execute(
            "SELECT market_id FROM markets WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        market_id = market_row[0] if market_row else None
        taker_times = [
            row[0] for row in conn.execute(
                """
                SELECT ts FROM actor_trade_legs
                WHERE proxy_wallet = ? AND condition_id = ?
                  AND liquidity_role = 'taker'
                ORDER BY ts
                """,
                (actor_id, condition_id),
            ).fetchall()
        ]
        first_state = conn.execute(
            "SELECT MIN(ts) FROM market_state WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()[0]
        if first_state is None:
            continue
        t = first_state + market_state_lookback
        while t + interval_seconds <= end_time:
            interval = (t, t + interval_seconds)
            t += interval_seconds
            if market_id is not None:
                status = reader.market_status_asof(market_id, interval[0])
                if status is None or not status["trading_enabled"] or (
                    status["closed"] or status["resolved"]
                ):
                    continue
            state = reader.market_state_before(
                condition_id, interval[0], market_state_lookback
            )
            if not state:
                continue  # coverage not certified
            if reader.blocking_gaps(condition_id, interval[0], interval[1]):
                continue
            if not _actor_engaged(
                reader, actor_id, condition_id, interval[0], taker_times
            ):
                continue
            traded = any(
                interval[0] <= leg_time < interval[1]
                for leg_time in taker_times
            )
            episode = DecisionEpisode(
                decision_id=namespace_id(
                    "opportunity", actor_id, condition_id,
                    f"{interval[0]:.0f}",
                ),
                actor_id=actor_id,
                market_id=market_id,
                condition_id=condition_id,
                anchor_time=interval[0],
                interval_start=interval[0],
                interval_end=interval[1],
                positive_quantity_change=0.0,
                gross_quantity=0.0,
                direction=None,  # NEVER eligible for the direction model
                mixed_activity_ratio=None,
                coverage={"target": TARGET_OCCURRENCE},
            )
            opportunities.append(
                AtRiskOpportunity(
                    actor_id=actor_id,
                    condition_id=condition_id,
                    market_id=market_id,
                    interval_start=interval[0],
                    interval_end=interval[1],
                    traded=traded,
                    episode=episode,
                )
            )
    return opportunities


def occurrence_labels(
    opportunities: list[AtRiskOpportunity],
) -> list[float]:
    """+1 traded / -1 no-trade labels for the occurrence head."""
    return [1.0 if o.traded else -1.0 for o in opportunities]
