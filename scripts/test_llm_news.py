"""Manual local test for Ollama news extraction and relevance."""

from __future__ import annotations

import json

from polymarket.normalization.llm_news import (
    OllamaClaimExtractor,
    OllamaRelevanceScorer,
)


def main() -> None:
    extractor = OllamaClaimExtractor("qwen3:8b")
    scorer = OllamaRelevanceScorer("qwen3:8b")

    headline = "Federal Reserve cuts rates by 25 basis points"
    body = (
        "The Federal Reserve lowered its benchmark interest rate on Wednesday "
        "after officials observed cooling inflation."
    )

    question = "Will the Federal Reserve cut interest rates before September?"
    rules = (
        "This market resolves Yes if the Federal Reserve announces a reduction "
        "in its benchmark target interest rate before September."
    )

    claims = extractor.extract(headline, body)

    print("\nEXTRACTED CLAIMS")
    print(json.dumps(claims, indent=2))

    print("\nMARKET RELEVANCE")

    for claim in claims:
        relevance = scorer.score(
            claim["claim_text"],
            question,
            rules,
        )

        print(
            json.dumps(
                {
                    "claim": claim["claim_text"],
                    "relevance": relevance,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
