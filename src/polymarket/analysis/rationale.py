"""Deterministic rationale text rendered ONLY from the structured record.

No LLM chooses the template or introduces evidence; an LLM may later
paraphrase this text but never selects content.  Rules: mention only
evidence present in the structured record, mention alternatives when
posterior mass is material, state uncertainty, never assert "the trader
believed" as a fact, and never generate a confident rationale for
ambiguous or failed records.
"""

from __future__ import annotations

from typing import Any

_ALTERNATIVE_MASS = 0.2

_TEMPLATE_PHRASES = {
    "FRESH_NEWS_RESPONSE": "a response to recently available relevant news",
    "PERSISTENT_NEWS_ADJUSTMENT": (
        "a persistent-news adjustment (consistent with delayed adjustment "
        "or underreaction, though no causal claim is made)"
    ),
    "MARKET_MOMENTUM": "trading with the recent market-price movement",
    "CONTRARIAN_REVERSAL": "trading against the recent market trend",
    "INVENTORY_REBALANCING": (
        "inventory rebalancing that reduces net proposition exposure"
    ),
    "POSITION_BUILDING": (
        "building additional exposure in the existing direction"
    ),
    "LIQUIDITY_TIMING": (
        "execution timing around favourable order-book liquidity rather "
        "than necessarily a directional view"
    ),
    "ACTOR_PRIOR": "this actor's habitual trading propensity",
    "MIXED_OR_UNRESOLVED": "no single behavioural hypothesis",
}

_STATUS_TEXT = {
    "ambiguous": (
        "No single reasoning hypothesis met the acceptance thresholds; "
        "the posterior over templates is retained without a primary "
        "selection."
    ),
    "insufficient_context": (
        "The pre-decision context was not coverage-complete, so no "
        "reasoning hypothesis is asserted."
    ),
    "counterfactual_failure": (
        "The leading reasoning hypothesis failed its required "
        "counterfactual check, so no primary template is asserted."
    ),
    "attribution_template_disagreement": (
        "The predictive driver attribution and the template posterior "
        "disagree; the disagreement is surfaced rather than resolved "
        "into a narrative."
    ),
}

_DISCLAIMER = (
    "This is a behavioural inference, not a claim about the trader's "
    "private thoughts."
)


def _evidence_sentences(record: dict[str, Any]) -> list[str]:
    sentences: list[str] = []
    reasoning = record["R"]
    news = record["C"].get("news_evidence") or []
    primary = reasoning.get("primary_template")
    if primary in ("FRESH_NEWS_RESPONSE", "PERSISTENT_NEWS_ADJUSTMENT") and news:
        youngest = min(news, key=lambda e: e["age_hours"])
        direction = (
            "positive" if youngest.get("direction", 0) > 0
            else "negative" if youngest.get("direction", 0) < 0 else "neutral"
        )
        sentences.append(
            f"Relevant {direction} news had remained available for "
            f"{youngest['age_hours']:.0f} hours before the decision."
        )
    attribution = reasoning.get("driver_attribution") or {}
    channel = attribution.get("primary_channel")
    if channel:
        sentences.append(
            f"The {channel.replace('_', '-')} channel carried the largest "
            "stable predictive contribution, and refitting without it "
            "reduced the model's probability of the observed action."
        )
    counterfactuals = reasoning.get("counterfactuals") or {}
    deltas = counterfactuals.get("deltas") or {}
    if primary:
        from polymarket.analysis.reasoning_templates import TEMPLATES

        for name in TEMPLATES[primary].required_counterfactuals:
            delta = deltas.get(name)
            if delta is not None and delta > 0:
                sentences.append(
                    f"Under the fixed model, the '{name}' intervention "
                    "reduced the probability of the observed action."
                )
    return sentences


def _alternatives_sentence(record: dict[str, Any]) -> str | None:
    reasoning = record["R"]
    primary = reasoning.get("primary_template")
    posterior = reasoning.get("template_posterior") or {}
    alternatives = [
        _TEMPLATE_PHRASES.get(name, name)
        for name, mass in sorted(posterior.items(), key=lambda kv: -kv[1])
        if name != primary and mass >= _ALTERNATIVE_MASS
    ]
    if not alternatives:
        return None
    return (
        "Alternative explanations remained plausible: "
        + "; ".join(alternatives) + "."
    )


def render_rationale(record: dict[str, Any]) -> str:
    """Render deterministic rationale text from a structured DRC record."""
    reasoning = record["R"]
    status = reasoning["status"]
    if status != "accepted":
        parts = [_STATUS_TEXT.get(status, _STATUS_TEXT["ambiguous"])]
        alternatives = _alternatives_sentence(record)
        if status == "ambiguous" and alternatives:
            parts.append(alternatives)
        parts.append(_DISCLAIMER)
        return " ".join(parts)

    primary = reasoning["primary_template"]
    direction = record["D"].get("direction")
    subject = (
        f"The {direction}-direction trade" if direction
        else "The decision to trade at this time"
    )
    top_p = reasoning["template_posterior"].get(primary, 0.0)
    parts = [
        f"{subject} is most consistent with "
        f"{_TEMPLATE_PHRASES.get(primary, primary)} "
        f"(posterior probability {top_p:.2f})."
    ]
    parts.extend(_evidence_sentences(record))
    alternatives = _alternatives_sentence(record)
    if alternatives:
        parts.append(alternatives)
    confidence = reasoning.get("reasoning_confidence", 0.0)
    parts.append(
        f"Overall reasoning confidence is {confidence:.2f}; residual "
        "uncertainty remains across the retained posterior."
    )
    parts.append(_DISCLAIMER)
    return " ".join(parts)
