"""Candidate news attribution for decisions.

Results are CANDIDATE attributions only.  Temporal proximity is not
causal evidence; this module never asserts causal attribution, and the
output is labelled accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polymarket.analysis.context import DecisionContext
from polymarket.analysis.decisions import DecisionEpisode


@dataclass
class AttributionCandidate:
    event_family_id: str
    attribution_score: float
    components: dict[str, float]
    supporting_judgments: int


@dataclass
class AttributionResult:
    decision_id: str
    label: str = "candidate attribution"
    top_candidate: AttributionCandidate | None = None
    alternatives: list[AttributionCandidate] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


def attribute_decision(
    context: DecisionContext,
    episode: DecisionEpisode,
    *,
    lookback: float = 86400.0,
) -> AttributionResult:
    t = context.decision_time
    result = AttributionResult(decision_id=episode.decision_id)

    families = {
        fam["event_family_id"]: fam
        for fam in context.event_families
        if fam["earliest_available_at"] < t
        and fam["earliest_available_at"] >= t - lookback
    }
    if not families:
        result.notes.append("no candidate event families in lookback window")
        return result

    decision_direction = (
        1.0 if episode.direction == "positive"
        else -1.0 if episode.direction == "negative"
        else 0.0
    )

    # was the market already moving before the earliest candidate event?
    prices = [
        (s["ts"], s["positive_price"])
        for s in context.market_state
        if s["positive_price"] is not None
    ]

    candidates: list[AttributionCandidate] = []
    for family_id, family in families.items():
        judgments = [
            r for r in context.relevance
            if r["event_family_id"] == family_id
            and r["rel_class"] != "irrelevant"
        ]
        if not judgments:
            continue
        top = max(judgments, key=lambda r: r["rel_score"])
        relevance = top["rel_score"]
        alignment = (
            1.0 if decision_direction and top["direction"] == decision_direction
            else 0.5 if top["direction"] == 0 or decision_direction == 0
            else 0.0
        )
        age = t - family["earliest_available_at"]
        proximity = max(0.0, 1.0 - age / lookback)
        novelty = max((r["novelty"] or 0.0) for r in judgments)
        surprise = max((r["surprise"] or 0.0) for r in judgments)
        sources = len(judgments)

        pre_event = [p for ts, p in prices if ts < family["earliest_available_at"]]
        post_event = [p for ts, p in prices if ts >= family["earliest_available_at"]]
        pre_move = abs(pre_event[-1] - pre_event[0]) if len(pre_event) >= 2 else 0.0
        post_move = abs(post_event[-1] - post_event[0]) if len(post_event) >= 2 else 0.0
        movement_after = 1.0 if post_move > pre_move else 0.3

        components = {
            "relevance": relevance,
            "direction_alignment": alignment,
            "temporal_proximity": proximity,
            "novelty": novelty,
            "surprise": surprise,
            "source_diversity": min(sources / 3.0, 1.0),
            "movement_after_event": movement_after,
        }
        score = (
            0.35 * relevance
            + 0.2 * alignment
            + 0.15 * proximity
            + 0.1 * novelty
            + 0.05 * surprise
            + 0.05 * components["source_diversity"]
            + 0.1 * movement_after
        )
        candidates.append(
            AttributionCandidate(
                event_family_id=family_id,
                attribution_score=round(score, 6),
                components=components,
                supporting_judgments=sources,
            )
        )

    if not candidates:
        result.notes.append("no relevant judgments for candidate families")
        return result
    candidates.sort(key=lambda c: -c.attribution_score)
    result.top_candidate = candidates[0]
    result.alternatives = candidates[1:]
    if len(candidates) > 1:
        margin = candidates[0].attribution_score - candidates[1].attribution_score
        result.confidence = min(1.0, max(0.0, margin * 2 + 0.3))
    else:
        result.confidence = min(1.0, candidates[0].attribution_score)
    result.notes.append(
        "temporal proximity alone does not establish causality"
    )
    return result
