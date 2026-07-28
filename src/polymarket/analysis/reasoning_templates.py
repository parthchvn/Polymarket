"""Structured reasoning-template ontology.

Templates are OBSERVATIONALLY HONEST hypothesis names: they describe the
behavioural pattern a decision is most consistent with under the fitted
model, never the actor's private mental state and never a definitive
causal mechanism.  In particular, delayed alignment with older news is
named PERSISTENT_NEWS_ADJUSTMENT — rationale text may say "consistent
with delayed adjustment or underreaction", but the stored template makes
no underreaction claim.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_DIRECTION = "direction"
TARGET_OCCURRENCE = "occurrence"


@dataclass(frozen=True)
class ReasoningTemplate:
    name: str
    description: str
    applicable_targets: tuple[str, ...]
    expected_channels: tuple[str, ...]
    required_evidence: tuple[str, ...]
    required_counterfactuals: tuple[str, ...]
    incompatible_conditions: tuple[str, ...]


TEMPLATES: dict[str, ReasoningTemplate] = {
    template.name: template
    for template in (
        ReasoningTemplate(
            name="FRESH_NEWS_RESPONSE",
            description=(
                "Decision aligns with the direction of recently available "
                "relevant news; the fresh-news channel is material and "
                "removing it reduces the probability of the observed action."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=("fresh_news",),
            required_evidence=("recent_relevant_news",),
            required_counterfactuals=("remove_fresh_news",),
            incompatible_conditions=("news_decay_missing",),
        ),
        ReasoningTemplate(
            name="PERSISTENT_NEWS_ADJUSTMENT",
            description=(
                "Decision aligns with older relevant news whose decayed "
                "signal remains material after controlling for market "
                "trend; consistent with delayed adjustment (no causal "
                "underreaction claim is stored)."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=("persistent_news",),
            required_evidence=("aged_relevant_news",),
            required_counterfactuals=("remove_persistent_news",),
            # tightened (PR C): when impact screens ran for this market
            # and found NO impactful news, the market's own liquidity
            # reaction contradicts a persistent-adjustment story
            incompatible_conditions=(
                "news_decay_missing", "impact_screen_contradiction",
            ),
        ),
        ReasoningTemplate(
            name="MARKET_MOMENTUM",
            description=(
                "Decision aligns with the recent market-price movement; the "
                "market-trend channel dominates and no news channel is "
                "required."
            ),
            applicable_targets=(TARGET_DIRECTION,),
            expected_channels=("market_trend",),
            required_evidence=("recent_price_trend",),
            required_counterfactuals=("flatten_market_trend",),
            incompatible_conditions=(),
        ),
        ReasoningTemplate(
            name="CONTRARIAN_REVERSAL",
            description=(
                "Decision opposes the recent market trend; trend information "
                "is material and the fitted decision is consistent with "
                "reversal rather than continuation."
            ),
            applicable_targets=(TARGET_DIRECTION,),
            expected_channels=("market_trend",),
            required_evidence=("recent_price_trend",),
            required_counterfactuals=("flatten_market_trend",),
            incompatible_conditions=(),
        ),
        ReasoningTemplate(
            name="INVENTORY_REBALANCING",
            description=(
                "Position channel is material and the observed trade "
                "reduces absolute net proposition exposure."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=("position",),
            required_evidence=("existing_exposure", "exposure_reduction"),
            required_counterfactuals=("neutralise_position",),
            incompatible_conditions=("no_pre_position",),
        ),
        ReasoningTemplate(
            name="POSITION_BUILDING",
            description=(
                "Position channel is material and the decision increases "
                "exposure in the existing direction."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=("position",),
            required_evidence=("existing_exposure", "exposure_increase"),
            required_counterfactuals=("neutralise_position",),
            incompatible_conditions=("no_pre_position",),
        ),
        ReasoningTemplate(
            name="LIQUIDITY_TIMING",
            description=(
                "Liquidity channel is material: spread, depth or imbalance "
                "primarily explains why the decision occurred at that time. "
                "An execution-timing rationale, not necessarily a "
                "directional belief."
            ),
            applicable_targets=(TARGET_OCCURRENCE, TARGET_DIRECTION),
            expected_channels=("liquidity",),
            required_evidence=("order_book_state",),
            required_counterfactuals=("replace_liquidity_with_training_reference",),
            incompatible_conditions=("no_order_book_coverage",),
        ),
        ReasoningTemplate(
            name="ACTOR_PRIOR",
            description=(
                "Actor-history/base channels dominate and no situational "
                "channel has sufficient evidence."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=("actor", "base"),
            required_evidence=("actor_history",),
            required_counterfactuals=("remove_actor_history",),
            incompatible_conditions=(),
        ),
        ReasoningTemplate(
            name="MIXED_OR_UNRESOLVED",
            description=(
                "Insufficient separation, weak evidence, failed "
                "counterfactuals or incompatible templates; no single "
                "behavioural hypothesis is selected."
            ),
            applicable_targets=(TARGET_DIRECTION, TARGET_OCCURRENCE),
            expected_channels=(),
            required_evidence=(),
            required_counterfactuals=(),
            incompatible_conditions=(),
        ),
    )
}

TEMPLATE_NAMES = tuple(TEMPLATES)


def templates_for_target(target: str) -> tuple[str, ...]:
    return tuple(
        name for name, template in TEMPLATES.items()
        if target in template.applicable_targets
    )
