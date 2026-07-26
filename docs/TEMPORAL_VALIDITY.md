# Temporal validity

The single scientific rule: for a decision reconstructed at time `t`,
every piece of context must satisfy

```text
available_at < t
```

Strict inequality, never `<=`.  Rows exactly at the cutoff are excluded.

## What "available_at" means per surface

| surface                  | availability column        |
|--------------------------|----------------------------|
| trades / executions      | `ts`                       |
| position events          | `ts`                       |
| position snapshots       | `observed_at`              |
| order books              | `observed_at`              |
| market state             | `ts` (bucket END for execution-derived state) |
| contract versions        | `first_observed_at`        |
| market status versions   | `first_observed_at`        |
| outcome-token mappings   | `mapping_effective_from`   |
| news articles            | `first_observed_at` (collector time, NOT publication time) |
| news claims              | `first_available_at`       |
| event families           | `earliest_available_at`    |
| relevance judgments      | `computed_at`              |

## Enforcement

1. All reads go through `analysis/reader.py::SQLiteNormalizedReader`,
   which uses `< cutoff` in every method.  Downstream analysis must not
   issue unrestricted raw SQL.
2. `analysis/temporal.py::assert_no_future_information(context, t)` runs
   on every assembled replay context and raises
   `TemporalContaminationError` if any recognized availability timestamp
   is `>= t`.
3. Leakage tests (`tests/analysis/test_temporal.py`) verify rows exactly
   at the cutoff are excluded on every surface, and the future-lead
   placebo acts as an ongoing leakage-sensitivity diagnostic.

## News availability

Publication time is not first collector availability.  All news
attachment uses `first_observed_at < decision_time`.  Ingestion lag
(`first_observed_at - source_published_at`) is retained and audited.

## Time-decayed news features

The permanent semantic relevance score in `relevance_judgments` is never
modified.  At decision time, a SEPARATE dynamic weight is recomputed per
decision from the age of each judgment:

```python
decay = 2.0 ** (-(decision_time - computed_at) / half_life_seconds)
```

Rules:

* eligibility is defensive as well as reader-enforced: rows with
  `computed_at >= decision_time` contribute exactly zero, even if they
  somehow reach the feature code;
* four half-lives (6h, 24h, 72h, 168h) are exposed side by side so the
  model can learn the persistence of news value;
* decayed news is capped at a 28-day maximum age;
* the raw 24-hour features remain unchanged for backwards compatibility
  (`news_recent_missing` carries their old missingness semantics, while
  `news_missing` now equals `news_decay_missing` — "no news information
  is available to any news feature");
* rows are aggregated per `event_family_id` (max positive and max
  negative within a family, sum across families) so duplicate articles
  do not multiply the signal, while positive and negative evidence stay
  separately visible.

## Reasoning-layer temporal guarantees

* The relevance surface used by contexts is `relevance_snapshot_asof`:
  exactly one judgment per event family, judged against the contract
  version active at the decision, `computed_at` strictly before the
  decision (a judgment at exactly the decision time is excluded), latest
  recomputation wins — repeated recomputations never multiply evidence.
* Posterior inputs are strictly pre-decision: Layer 1 deltas and
  contributions, decision direction, pre-decision position/exposure,
  news evidence with age and alignment, pre-decision trend and
  liquidity, actor history, and coverage indicators.  Post-decision
  price movement, market outcomes, future wallet trades, later news and
  resolution results are never inputs (tested by inserting post-decision
  rows and asserting identical reasoning inputs).
* Synthetic validation splits are by WORLD SEED; no episode from one
  world appears on both sides, and thresholds are never tuned on the
  held-out test worlds.
