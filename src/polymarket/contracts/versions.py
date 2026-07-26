"""Version constants re-exported for convenience."""

from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION

EXTRACTOR_VERSION = "rule-1.0.0"
RELEVANCE_MODEL_VERSION = "rule-1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "PARSER_VERSION",
    "EXTRACTOR_VERSION",
    "RELEVANCE_MODEL_VERSION",
]
