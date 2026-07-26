"""Decision-episode construction.

The primary decision estimand uses audited taker activity only
(``liquidity_role = 'taker'``).  Maker legs are excluded from the primary
outcome; unknown-role legs are excluded and surfaced in coverage flags.

Direction is proposition-aware and never forced for highly mixed
activity (``direction = None`` below the mixed threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.collection.canonical import namespace_id


def proposition_change(side: str, outcome_sign: int, quantity: float) -> float:
    side_sign = 1.0 if side.upper() == "BUY" else -1.0
    return float(quantity) * side_sign * float(outcome_sign)


@dataclass
class DecisionEpisode:
    decision_id: str
    actor_id: str
    market_id: str | None
    condition_id: str
    anchor_time: float
    interval_start: float
    interval_end: float
    positive_quantity_change: float
    gross_quantity: float
    direction: str | None
    mixed_activity_ratio: float | None
    pre_decision_position: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    source_leg_ids: list[str] = field(default_factory=list)


def build_decision_episodes(
    reader: SQLiteNormalizedReader,
    *,
    end_time: float,
    interval_seconds: float = 3600.0,
    mixed_threshold: float = 0.2,
) -> list[DecisionEpisode]:
    """Group taker legs (strictly before ``end_time``) into episodes.

    Legs for one (actor, condition) are grouped greedily: an episode is
    anchored at its first leg and absorbs subsequent legs within
    ``interval_seconds``.
    """
    legs = reader.actor_trade_legs_before(end_time, liquidity_role="taker")
    grouped: dict[tuple[str, str], list] = {}
    for leg in legs:
        grouped.setdefault(
            (leg["proxy_wallet"], leg["condition_id"]), []
        ).append(leg)

    episodes: list[DecisionEpisode] = []
    for (actor, condition_id), actor_legs in sorted(grouped.items()):
        actor_legs.sort(key=lambda r: (r["ts"], r["actor_leg_id"]))
        bucket: list = []
        for leg in actor_legs:
            if bucket and leg["ts"] - bucket[0]["ts"] >= interval_seconds:
                episodes.append(
                    _episode_from_legs(
                        reader, actor, condition_id, bucket,
                        interval_seconds, mixed_threshold,
                    )
                )
                bucket = []
            bucket.append(leg)
        if bucket:
            episodes.append(
                _episode_from_legs(
                    reader, actor, condition_id, bucket,
                    interval_seconds, mixed_threshold,
                )
            )
    episodes.sort(key=lambda e: (e.anchor_time, e.decision_id))
    return episodes


def _episode_from_legs(
    reader: SQLiteNormalizedReader,
    actor: str,
    condition_id: str,
    legs: list,
    interval_seconds: float,
    mixed_threshold: float,
) -> DecisionEpisode:
    anchor = legs[0]["ts"]
    changes: list[float] = []
    skipped_unmapped = 0
    for leg in legs:
        if leg["outcome_sign"] is None:
            skipped_unmapped += 1
            continue
        changes.append(
            proposition_change(leg["side"], leg["outcome_sign"], leg["size"])
        )
    net = sum(changes)
    gross = sum(abs(c) for c in changes)
    if gross == 0:
        direction = None
        mixed_ratio = None
    else:
        mixed_ratio = abs(net) / gross
        if mixed_ratio < mixed_threshold:
            direction = None
        else:
            direction = "positive" if net > 0 else "negative"

    market = reader.market_by_condition(condition_id)
    # pre-decision position: strictly before the anchor time — events at
    # the anchor itself are excluded.
    pre_position = reader.position_asof(actor, condition_id, anchor)
    return DecisionEpisode(
        decision_id=namespace_id(
            "decision", actor, condition_id, anchor, len(legs)
        ),
        actor_id=actor,
        market_id=market["market_id"] if market else None,
        condition_id=condition_id,
        anchor_time=anchor,
        interval_start=anchor,
        interval_end=anchor + interval_seconds,
        positive_quantity_change=net,
        gross_quantity=gross,
        direction=direction,
        mixed_activity_ratio=mixed_ratio,
        pre_decision_position=pre_position,
        coverage={
            "leg_count": len(legs),
            "unmapped_outcome_legs": skipped_unmapped,
            "position_history_complete": pre_position["complete"],
        },
        source_leg_ids=[leg["actor_leg_id"] for leg in legs],
    )
