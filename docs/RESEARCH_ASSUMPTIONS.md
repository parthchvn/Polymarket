# Research assumptions (unverified or partially verified)

None of the following may be silently assumed resolved.

1. **Trade identity.** The candidate actor-leg fingerprint
   (transactionHash, proxyWallet, asset, side, size, price) is NOT
   proven globally unique.  It is stored as a diagnostic only.
2. **Canonical fill completeness.** `takerOnly=true` as "one taker-side
   record per transaction" is a working assumption from prior probes; it
   must be continuously audited against expanded records.
3. **Backfill semantics.** Historical time-window parameters
   (`after`/`before`) are UNVALIDATED against the live API.  Windows are
   marked complete only after observed-timestamp validation.
4. **Position completeness.** Wallet activity may omit transfers or
   other balance-changing events; reconstruction vs snapshot audits
   quantify (not eliminate) this risk.  CONVERT and unknown event types
   remain `unresolved`.
5. **Contract history.** Historical rule text may not be retrievable
   retrospectively; forward contract snapshots are the reliable source.
6. **Order-book history.** Historical books are generally unavailable;
   forward collection matters.  Book gaps are recorded, not imputed.
7. **News availability.** Publication time is not first collector
   availability; only `first_observed_at` governs temporal validity.
   Rule-based claim extraction / relevance scoring are deliberately
   simple deterministic baselines, not validated NLP.
8. **Actor identity.** A proxy wallet must not be described as one
   real-world person; wallets may be shared, automated or split.
9. **Causality.** News attribution is CANDIDATE attribution.  Temporal
   proximity plus relevance is not causal identification.
10. **Model validation.** Retrospective performance can suffer
    repeated-test overfitting; the terminal holdout discipline and
    placebo suite mitigate but do not eliminate this.
11. **Live payload field names.** All collectors were exercised against
    mocked/synthetic payloads only in this implementation; live response
    shapes (field names, pagination behaviour, status semantics) remain
    to be validated with `pytest -m live` style probes.
12. **News decay parameters.** The half-lives (6h/24h/72h/168h), the
    28-day maximum age, and the event-family max-positive/max-negative
    aggregation are MODELLING CHOICES, not established causal
    parameters.  Every run records them in config.json and
    feature_manifest.json; no hidden decay parameters are permitted.
13. **Driver attribution is not mechanism or causation.** Layer-1
    records identify which feature channels carried predictive weight
    for the observed action under the fitted model.  They do not recover
    the actor's actual reasoning, do not separate immediate reaction
    from delayed underreaction (that requires the template layer), and
    are not causal claims.  With few decisions, per-decision ablation
    deltas are noisy; statuses (ambiguous / counterfactual_failure) must
    be respected downstream.

## Reasoning-layer assumptions

* **`R` is inferred behaviourally.**  A reasoning judgment asserts only
  that the observed decision is most consistent with a template under
  the fitted model.  It is not the actor's private mental state, and
  rationale text always carries that disclaimer.
* **Relevance-snapshot version fallback.**  Batch normalization stamps
  relevance judgments with the newest contract version known at
  normalization time, so a pipeline that does not recompute judgments
  per contract version can have zero judgments for the version active
  at a decision.  `relevance_snapshot_asof` then (by default) falls
  back to the latest judgment per family across versions — still
  strictly pre-decision — and flags the fallback in context coverage
  (`relevance_version_fallback`).  Live forward operation, where news
  normalizes as it arrives, does not hit this path.
* **Acceptance is conservative by design.**  Underdetermined fits
  (training rows below `min_train_rows`), attributions that fail the
  family-wise permutation-null or fold-mean informativeness gates, weak
  margins, unstable resamples, or missing evidence all demote a record
  to `ambiguous` / `insufficient_context` rather than accepting a
  narrative.  The posterior is retained either way; only the primary
  template is withheld.
* **Template recovery is validated on synthetic mechanisms.**  Recovery
  quality is measured on worlds where the mechanism is true BY
  CONSTRUCTION; real-data performance depends on decision density and
  collection coverage and must be re-validated against the live
  pipeline before any claim is made.
