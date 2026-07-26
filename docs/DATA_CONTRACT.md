# Data contract

One SQLite database holds raw and normalized tables.  Timestamps are UTC
epoch seconds (REAL).  Raw payloads are exact bytes (BLOB).

## Lineage fields (on every normalized observational row)

| field | meaning |
|---|---|
| `raw_response_id` | raw_responses row the record was parsed from |
| `raw_record_index` | position of the record inside that payload |
| `raw_record_hash` | SHA-256 of the canonicalized individual record |
| `parser_version` | parser version that produced the row |
| `schema_version` | schema version at normalization time |
| `normalized_at` | wall-clock normalization time (not an availability time) |

## Operational / raw tables

- **schema_metadata** — schema_version (PK), applied_at, parser_version, description.
- **collector_runs** — collector_run_id (PK), collector, started_at, finished_at, status (running/succeeded/failed/partial), configuration_json, note.  Operational status may be updated; observational data may not.
- **raw_responses** — append-only observations: collector_run_id, collector, base_url, endpoint, canonical_params_json, requested_at, received_at, http_status (NULL for transport failures), response_headers_json (canonical, lower-cased keys), content_hash (SHA-256 of exact bytes), payload (exact bytes), error_text.  Repeated identical payloads are separate rows; failures are retained.
- **backfill_windows** — (collector, object_id, window_start, window_end) PK; half-open [start, end); status pending/running/complete/incomplete/failed; page_count, record_count, observed_min/max_ts, note.
- **collector_gaps** — (collector, surface, object_id, gap_start) PK; gap_end, reason, detected_at, resolved_at.  Queryable coverage holes.

## Normalized market tables

- **markets** — market_id (PK), condition_id (UNIQUE), category, question, created_at, closed_at, resolved_at, is_combo, + lineage (with raw_record_index/hash).
- **outcome_tokens** — (condition_id, asset, mapping_effective_from) PK; outcome_label, outcome_sign (+1 positive / -1 complementary), mapping_confidence (explicit/high/assumed), + lineage.  Outcomes are not assumed to be literal YES/NO.
- **contract_versions** — (market_id, version_seq) PK; effective_from, first_observed_at, question, rules_text, resolution_source, resolution_time, content_hash, + lineage.  A changed contract creates a NEW row; old versions are never rewritten.
- **market_status_versions** — (market_id, effective_from) PK; first_observed_at, trading_enabled, closed, resolved, winning_asset, + lineage.

## Trade tables

- **actor_trade_legs** — actor_leg_id (PK, from raw provenance), source_record_id, candidate_fingerprint (diagnostic only), transaction_hash, transaction_log_index, transaction_occurrence, proxy_wallet, condition_id, asset, outcome_label, outcome_sign (nullable when unmapped), side, size, price, ts, liquidity_role (taker/maker/unknown), role_confidence, + lineage.
- **canonical_executions** — execution_id (PK, from raw provenance), source_record_id, transaction identifiers, condition_id, positive_price, positive_side, size, notional, ts, taker_wallet, reconciliation_status (direct/complemented), + lineage.  Sole basis for market-wide volume and price paths.

## Position tables

- **position_events** — position_event_id (PK), wallet, condition_id, asset (nullable), ts, event_type (TRADE/SPLIT/MERGE/REDEEM/other), signed_token_change and collateral_change (NULL when unresolved — missing is not zero), transaction identifiers, accounting_confidence (exact/inferred/unresolved), resolution_version (for REDEEM), is_combo, + lineage.
- **position_snapshots** — (wallet, asset, observed_at) PK; reported_size, source, + lineage.

## Market state

- **order_book_snapshots** — (asset, observed_at) PK; best_bid, best_ask, spread, bid_depth, ask_depth, imbalance (all NULLable), + lineage.
- **market_state** — (condition_id, ts, state_source) PK; positive_price, volume, spread, depth, imbalance, coverage_complete, + lineage.  `ts` for execution-derived rows is the bucket END time.  state_source labels executions/books/synthetic explicitly.

## News ledger

- **news_articles** — article_id (PK), source_id, source_url, source_published_at, first_observed_at (collector receive time — governs availability), download_completed_at, timestamp_source, timestamp_confidence, headline, body, content_hash, previous_article_id (links updates to prior records), + lineage.
- **event_families** — event_family_id (PK), label, earliest_available_at, created_by, created_at.
- **news_claims** — claim_id (PK), article_id (FK), claim_text, entities_json, quantities_json, supporting_span, first_available_at, extractor_version, confidence.
- **claim_edges** — edge_id (PK), claim_id (FK), event_family_id, edge_type (new/duplicate/confirmation/correction/contradiction/supersession), effective_from, evidence, confidence.
- **relevance_judgments** — (event_family_id, market_id, contract_version_seq, computed_at) PK; rel_class, rel_score, direction, novelty, surprise, method, model_version, evidence_json.

## Expected payload record shapes

Parsers expect the record shapes documented in the module docstrings of
`normalization/markets.py`, `normalization/trades.py`,
`normalization/positions.py`, `normalization/books.py` and
`normalization/news.py`.  Live Polymarket field names may differ; the
collector-facing mapping is a working assumption to validate against the
live API (see RESEARCH_ASSUMPTIONS.md).
