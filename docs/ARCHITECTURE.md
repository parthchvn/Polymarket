# Architecture

```text
Polymarket APIs and selected news sources
        ↓  collection/          (collectors write ONLY raw responses)
immutable raw responses with exact provenance
        ↓  normalization/       (the ONE normalization path)
shared normalized SQLite schema (contracts/schema.py)
        ↓  analysis/reader.py   (strict as-of readers, always < cutoff)
decision-episode construction   (analysis/decisions.py)
        ↓
market, actor, position and news features (analysis/features.py)
        ↓
nested models, attribution and placebos (analysis/models.py, placebos.py)
        ↓
reports and machine-readable outputs (analysis/reporting.py)
```

## Raw collection

`collection/client.py` wraps httpx.  Every request attempt — success,
HTTP error or transport failure — becomes an immutable `raw_responses`
row with exact bytes, SHA-256 content hash, canonical parameters and
canonical headers.  Repeated identical payloads remain separate rows.
`pagination.py` and `backfill.py` provide the generic paging/backfill
engines with repeated-page detection, max-page safeguards, half-open
windows, resumable state and gap recording.

## Normalization

`normalization/normalizer.py` is the single entry point.  It loads a raw
response, identifies collector/endpoint, decodes the payload and
dispatches to endpoint-specific parsers, which attach full lineage
(`raw_response_id`, `raw_record_index`, `raw_record_hash`,
`parser_version`, `schema_version`, `normalized_at`) and insert
idempotently via deterministic IDs.  `reconciliation.py` performs the
maker/taker role pass across the two trade views.

## Analysis

`analysis/reader.py` centralizes all temporal SQL with strict `<`
inequalities.  `decisions.py` builds proposition-aware taker episodes,
`context.py` assembles strictly-before contexts (with a runtime
no-future assertion), `features.py` computes group-separated features,
`models.py` fits the nested M0–M3 suite under blocked expanding-window
evaluation, `placebos.py` and `uncertainty.py` provide the negative
controls and bootstrap intervals, and `replay.py`/`reporting.py`
orchestrate runs and write outputs.

## Synthetic regression

`synthetic/` generates a deterministic raw-like world and pushes it
through the SAME normalizer into the SAME schema — never a parallel
simplified table — producing `fixtures/synthetic_normalized.sqlite` for
the end-to-end suite.
