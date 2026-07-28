"""Local Ollama implementations for news extraction and relevance scoring."""

from __future__ import annotations

import re
from typing import Literal

try:  # optional dependency: only needed for --method ollama
    from ollama import chat
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    def chat(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "the 'ollama' package is not installed; "
            "pip install ollama to use the LLM relevance scorer"
        )
from pydantic import BaseModel, Field


class Quantity(BaseModel):
    value: float | None = None
    raw_value: str
    unit: str | None = None


class ExtractedClaim(BaseModel):
    claim_text: str
    supporting_span: str
    entities: list[str] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    event_type: str | None = None
    effective_time_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimExtractionResult(BaseModel):
    claims: list[ExtractedClaim]


RelevanceClass = Literal[
    "supports_positive",
    "supports_negative",
    "indirect",
    "background",
    "irrelevant",
    "ambiguous",
]


class RelevanceResult(BaseModel):
    rel_class: RelevanceClass
    rel_score: float = Field(ge=0.0, le=1.0)
    direction: float = Field(ge=-1.0, le=1.0)
    directness: Literal[
        "direct",
        "indirect",
        "background",
        "irrelevant",
        "ambiguous",
    ]
    reasoning_summary: str
    supporting_rule_span: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


_QUANTITY_PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:basis points?|bps)\b", re.IGNORECASE),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:percentage points?|percent)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?%", re.IGNORECASE),
    re.compile(
        r"[$£€]\s?\d+(?:,\d{3})*(?:\.\d+)?"
        r"(?:\s*(?:million|billion|trillion))?",
        re.IGNORECASE,
    ),
)

_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9&.'-]*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9&.'-]*|of|the|and)){1,5}\b"
)


def _quantity_unit(raw_value: str) -> str | None:
    lowered = raw_value.lower()

    if "basis point" in lowered or "bps" in lowered:
        return "basis points"
    if "percentage point" in lowered:
        return "percentage points"
    if "%" in lowered or "percent" in lowered:
        return "percent"
    if raw_value.startswith("$"):
        return "USD"
    if raw_value.startswith("£"):
        return "GBP"
    if raw_value.startswith("€"):
        return "EUR"

    return None


def _quantity_value(raw_value: str) -> float | None:
    match = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", raw_value)

    if match is None:
        return None

    return float(match.group(0).replace(",", ""))


def _source_quantities(text: str) -> list[dict]:
    found: list[dict] = []
    seen: set[str] = set()

    for pattern in _QUANTITY_PATTERNS:
        for match in pattern.finditer(text):
            raw_value = match.group(0).strip()
            key = raw_value.lower()

            if key in seen:
                continue

            seen.add(key)
            found.append(
                {
                    "value": _quantity_value(raw_value),
                    "raw_value": raw_value,
                    "unit": _quantity_unit(raw_value),
                }
            )

    return found


def _source_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()

    for match in _ENTITY_PATTERN.finditer(text):
        entity = match.group(0).strip()

        if entity.startswith("The "):
            entity = entity[4:]

        key = entity.lower()

        if key in seen:
            continue

        seen.add(key)
        entities.append(entity)

    return entities


MAX_EXTRACTION_RESPONSE_CHARS = 200_000
MAX_CLAIMS_PER_ARTICLE = 12


def cap_extracted_claims(claims: list) -> list:
    """Schema-constrained decoding at temperature 0 can degenerate
    into enormous repetitive claim arrays; a real article rarely
    yields more than a dozen atomic claims worth scoring (each claim
    also costs one relevance call per market downstream).  Keep the
    highest-confidence claims up to the cap, deduplicated by text."""
    seen: set[str] = set()
    unique = []
    for claim in claims:
        key = claim.claim_text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(claim)
    unique.sort(key=lambda c: -float(c.confidence or 0))
    return unique[:MAX_CLAIMS_PER_ARTICLE]


class OllamaClaimExtractor:
    """Extract atomic factual claims using a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        *,
        max_body_chars: int = 6000,
    ) -> None:
        self.model = model
        self.max_body_chars = max_body_chars
        self.version = f"ollama-{model}-claims-v1"

    def extract(
        self,
        headline: str,
        body: str,
    ) -> list[dict]:
        body_excerpt = (body or "")[: self.max_body_chars]

        prompt = f"""
You are a precise news information-extraction system.

Extract every atomic factual claim explicitly stated in either the headline
or article body.

Rules:
1. Treat the headline as part of the source.
2. Do not use outside knowledge.
3. Extract only explicitly stated facts.
4. Preserve uncertainty words such as "may", "expects", "reportedly",
   "according to", and "allegedly".
5. Extract organisations, people, locations, dates, times, quantities,
   percentages, currencies, vote counts, and units.
6. Each claim should contain one independently verifiable assertion.
7. When the headline and body describe the same event but provide
   complementary details, merge them into one claim.
8. Do not create separate claims solely because one detail occurs in the
   headline and another occurs in the body.
9. supporting_span must preserve the exact source wording.
10. When two source spans are required, join them using " || ".
11. event_type should be a concise snake_case category.
12. effective_time_text must preserve explicitly mentioned timing.
13. Confidence measures extraction certainty, not whether the claim is true.
14. Use confidence 1.0 only when every field is completely unambiguous.
15. Return only the required structured output.

HEADLINE:
{headline}

ARTICLE BODY:
{body_excerpt}
"""

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=ClaimExtractionResult.model_json_schema(),
            think=False,
            keep_alive="30m",
            options={
                "temperature": 0,
                "num_ctx": 8192,    # prompt + schema must actually fit
            },
        )

        content = response.message.content or ""
        if len(content) > MAX_EXTRACTION_RESPONSE_CHARS:
            raise ValueError(
                f"degenerate extraction response "
                f"({len(content)} chars); skipping article"
            )
        parsed = ClaimExtractionResult.model_validate_json(content)

        output: list[dict] = []
        source_text = f"{headline}\n{body_excerpt}"
        source_entity_list = list(_source_entities(source_text))

        for claim in cap_extracted_claims(list(parsed.claims)):
            claim_text = claim.claim_text.strip()
            supporting_span = claim.supporting_span.strip()
            claim_source = f"{claim_text}\n{supporting_span}"

            entities = [
                entity.strip()
                for entity in claim.entities
                if entity.strip()
            ]

            entity_keys = {entity.lower() for entity in entities}

            for entity in source_entity_list:
                if (
                    entity.lower() in claim_source.lower()
                    and entity.lower() not in entity_keys
                ):
                    entities.append(entity)
                    entity_keys.add(entity.lower())

            quantities = [
                quantity.model_dump()
                for quantity in claim.quantities
            ]

            quantity_keys = {
                quantity["raw_value"].lower()
                for quantity in quantities
            }

            for quantity in _source_quantities(claim_source):
                key = quantity["raw_value"].lower()

                if key not in quantity_keys:
                    quantities.append(quantity)
                    quantity_keys.add(key)

            output.append(
                {
                    "claim_text": claim_text,
                    "supporting_span": supporting_span,
                    "entities": entities,
                    "quantities": quantities,
                    "event_type": (
                        claim.event_type.strip()
                        if claim.event_type
                        else None
                    ),
                    "effective_time_text": (
                        claim.effective_time_text.strip()
                        if claim.effective_time_text
                        and claim.effective_time_text.strip()
                        else None
                    ),
                    # Avoid meaningless absolute certainty.
                    "confidence": min(float(claim.confidence), 0.99),
                }
            )

        return output


class OllamaRelevanceScorer:
    """Compare an extracted claim with exact market resolution semantics."""

    method = "ollama_llm"

    def __init__(self, model: str = "qwen3:8b") -> None:
        self.model = model
        self.version = f"ollama-{model}-relevance-v2"  # v2: indirect/background/irrelevant guidance

    def score(
        self,
        claim_text: str,
        question: str,
        rules_text: str | None,
    ) -> dict:
        prompt = f"""
You evaluate whether a news claim is relevant to a binary prediction market.

Definitions:
- supports_positive: the claim provides evidence that the market's positive
  proposition is true or more likely to resolve positive.
- supports_negative: the claim provides evidence that the positive proposition
  is false or more likely to resolve negative.
- indirect: related information that may affect the proposition but does not
  directly satisfy or contradict the resolution condition.
- background: topically related but weakly decision-relevant.
- irrelevant: unrelated to the market's resolution.
- ambiguous: relation or direction cannot be determined reliably.

Rules:
1. Use only the supplied claim, market question, and market rules.
2. Do not use outside knowledge.
3. Interpret relevance using the exact resolution rules.
4. rel_score must be between 0 and 1.
5. direction must be between -1 and 1:
   positive supports the positive proposition;
   negative supports the negative proposition.
6. Do not confuse news importance with contract relevance.
7. Use indirect when a claim meaningfully informs the likelihood of the
   proposition through intentions, capabilities, military readiness,
   resource constraints, diplomacy, escalation, negotiations, or planning,
   even when it does not itself satisfy the resolution condition.
8. Use background for topically related information that provides little
   directional evidence about the proposition.
9. Use irrelevant only when there is no plausible informational or causal
   connection to the proposition.
10. Do not classify a claim as irrelevant merely because it does not directly
    satisfy the resolution condition.
11. supporting_rule_span should quote exact wording from the question or rules.
12. Return only the required structured output.

CLAIM:
{claim_text}

MARKET QUESTION:
{question}

MARKET RULES:
{rules_text or ""}
"""

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=RelevanceResult.model_json_schema(),
            think=False,
            keep_alive="30m",
            options={
                "temperature": 0,
                "num_ctx": 4096,
            },
        )

        parsed = RelevanceResult.model_validate_json(
            response.message.content
        )

        return {
            "rel_class": parsed.rel_class,
            "rel_score": float(parsed.rel_score),
            "direction": float(parsed.direction),
            "evidence": {
                "directness": parsed.directness,
                "reasoning_summary": parsed.reasoning_summary,
                "supporting_rule_span": parsed.supporting_rule_span,
                "confidence": min(float(parsed.confidence), 0.99),
            },
        }
