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

## Reasoning Layer 1: predictive driver attribution

`analysis/reasoning.py` produces one record per labeled decision
answering "which feature channels moved the model's probability of the
observed action?".  This is PREDICTIVE ATTRIBUTION, not mechanism
inference: a large news contribution alone cannot distinguish immediate
reaction, delayed underreaction, continuation of a news-caused price
move, or correlation with an unobserved signal.  Mechanism inference
(template posteriors such as FRESH_NEWS_REACTION vs
DELAYED_NEWS_UNDERREACTION) is a later layer; the `reasoning_judgments`
schema reserves its fields (`primary_template`,
`template_posterior_json`) as NULL.

Channels: base, actor, market_trend, liquidity, position, and —
deliberately separated — `fresh_news` (raw 24h components + 6h decay
kernel) vs `persistent_news` (24h/72h/168h decay kernels), so a fresh
reaction and delayed adjustment remain distinguishable.

Method per chronological fold (same embargoed expanding-window
discipline as the nested suite): exact standardized logit contributions
per channel from the full model, plus refit group-ablation deltas
`log P(D|C) - log P(D|C without channel)` which double as counterfactual
results.  Statuses: accepted / ambiguous / insufficient_context /
counterfactual_failure (plus attribution_template_disagreement, reserved
for Layer 2+3 agreement checks).

Leakage rule: every model scoring a decision is trained only on strictly
earlier decisions, and all evidence fields (including
`aligned_move_since_news`) are computed from the strict pre-decision
context — post-trade price or liquidity responses never appear in a
decision's own record.  They may later serve as offline labels for an
impact classifier, but not as context.

## Reasoning reconstruction layer (Layer 2)

The reasoning layer produces, for each decision episode, a structured
`(D, C, R)` record: the observed wallet decision `D`, the strict
pre-decision context `C`, and a calibrated posterior `R` over structured
reasoning hypotheses most consistent with `D` and `C` under the fitted
model.  **`R` is a behavioural inference — "this behaviour is most
consistent with reasoning template X" — never a claim about the trader's
private thoughts.**

Components (all in `src/polymarket/analysis/`):

* `reasoning.py` — Layer 1 predictive driver attribution, hardened with
  optimiser diagnostics (`FitDiagnostics`), configurable acceptance
  rules (`AttributionConfig`: ablation-delta, margin, block-resample
  stability, permutation-null and fold-mean informativeness gates, and a
  minimum-training-rows gate: an underdetermined fit is never trusted),
  separated prediction vs attribution confidence, and an exact logit
  decomposition (`sigmoid(intercept + sum(channel contributions))`
  reconstructs the predicted probability).
* `reasoning_templates.py` — the frozen template ontology (nine
  observationally honest hypotheses; delayed news alignment is
  `PERSISTENT_NEWS_ADJUSTMENT`, never a causal "underreaction" label).
* `reasoning_posterior.py` — deterministic class-balanced multinomial
  posterior over templates with training-only standardization, L2
  regularization, and scalar temperature calibration fitted on held-out
  synthetic VALIDATION worlds (never on test worlds).  Primary templates
  are withheld (`None`) when gating fails (top probability, margin,
  entropy, Layer 1 not accepted, incomplete coverage).
* `reasoning_counterfactuals.py` — fixed-model context interventions
  (`remove_fresh_news`, `flatten_market_trend`, `neutralise_position`,
  ...), deliberately distinct from refit ablation, with missingness
  semantics preserved.  Each template declares required counterfactuals;
  a failed requirement demotes the record to `counterfactual_failure`.
* `reasoning_targets.py` — the occurrence target: an at-risk interval
  grid (market open, coverage certified, actor engaged) feeding a
  separate `P(trade | at-risk, C)` head.  Occurrence pseudo-episodes are
  `direction=None` and structurally cannot enter the direction model.
* `drc.py` / `rationale.py` — structured DRC record assembly,
  attribution-template agreement scoring (disagreement is surfaced as
  `attribution_template_disagreement`, never smoothed into a narrative),
  idempotent persistence into `reasoning_judgments` (the `confidence`
  column stores final reasoning confidence), and a deterministic
  rationale renderer (no LLM chooses templates or evidence).
* `versioning.py` — SHA-256 identities: `feature_version`,
  `reasoning_method_version`, `template_ontology_version`,
  `synthetic_generator_version`.  `PARSER_VERSION` is never reused.
* `reasoning_validation.py` — the synthetic validation harness:
  ground-truth worlds (`synthetic/reasoning_worlds.py`) with known
  mechanisms by construction, split by world seed, Layer 1 trained on
  pooled cross-world rows (leave-one-world-out for training worlds),
  and the six report files (`reasoning_metrics.json`,
  `reasoning_confusion_matrix.csv`, `reasoning_calibration.json`,
  `reasoning_failures.json`, `reasoning_manifest.json`,
  `dr_validation_records.jsonl`).
