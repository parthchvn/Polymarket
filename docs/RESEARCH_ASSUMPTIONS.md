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
