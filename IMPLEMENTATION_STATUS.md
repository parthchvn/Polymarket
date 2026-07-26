# Implementation status

## Completed (tested)

- Shared schema v1 with all required raw + normalized tables, lineage
  columns, pragmas, schema_metadata (contract tests).
- Immutable raw store: exact bytes, SHA-256, canonical params/headers,
  failures retained, repeated payloads as separate rows.
- ObservingClient with retry/backoff recording every attempt (mocked
  HTTP tests); collectors for trades (both views), activity, markets,
  status, books, positions, configurable news feeds.
- Pagination engine (offset + cursor, repeated-page detection, max-page
  guard, resume) and half-open backfill with validation, gap recording
  and safe restart.
- Single Normalizer path: markets, outcome tokens (explicit/label/
  positional sign resolution), contract versions, status versions,
  taker executions (positive-price conversion, tolerance checks),
  expanded actor legs, maker/taker reconciliation with unknown roles,
  books, position events (TRADE/SPLIT/MERGE/REDEEM, unresolved
  otherwise), snapshots, news ledger (articles/claims/families/edges/
  relevance) with deterministic rule-based extractor and scorer.
- Strict as-of reader (`< cutoff` everywhere) + runtime no-future
  assertion; opportunity checks; union-based position reconciliation
  with abs+rel tolerance.
- Proposition-aware taker decision episodes with mixed-activity None
  direction; feature groups with missingness indicators; nested M0–M3
  logistic suite under blocked expanding-window evaluation with
  embargo; training-only standardization; Brier/log-loss/accuracy/
  calibration; five placebos with recorded seeds; actor/market cluster
  and moving-block bootstraps; candidate news attribution.
- Deterministic synthetic world through the same normalizer; committed
  fixture; 75-test suite incl. end-to-end and CLI; full docs.

## Unresolved / open items

- Live API validation of every collector (field names, pagination,
  `after`/`before` semantics) — everything is fixture/mock-tested only.
- CLI `collect` wires only the trades surface; other surfaces are
  library-level.
- Cursor pagination is implemented generically but not wired to a live
  endpoint.
- Optional linear model for signed quantity not implemented (direction
  logistic suite only).
- Terminal-holdout bookkeeping is by convention (last fold untouched
  during development) rather than an enforced mechanism.
- All items in docs/RESEARCH_ASSUMPTIONS.md.
