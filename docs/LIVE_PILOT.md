# Live collection pilot — first real-data DRC run

Date: 2026-07-26.  Read-only collection; no keys, no orders, no trading.

## Setup

Three active, news-sensitive markets collected via
`scripts/run_live_pilot.py`:

* "Will the U.S. invade Iran before 2027?"
* "Strait of Hormuz traffic returns to normal by August 31?"
* "Will the Fed decrease interest rates by 50+ bps after the July 2026
  meeting?"

Surfaces: market metadata/status (gamma), 12,000 taker + 12,000
expanded trade records (data-api), current order books for all six
outcome tokens (CLOB), activity history for the 30 most active taker
wallets (28,241 position events), and three Google News RSS feeds (302
articles, 279 event families).  Reasoning model: the synthetic-trained
artifact from `reasoning_validation` (feature hash verified at load).

## Pipeline results

* Normalization: 0 errors.  1,575 unresolved activity events reference
  markets outside the collected set (expected; wallets trade elsewhere).
* Role reconciliation on real fills: **5,191 taker / 6,795 maker / 14
  unknown (0.1%)**.
* Derived market state: 854 hourly buckets.  849 relevance judgments
  computed against the real contract questions.
* Analysis: **2,657 labeled real decisions -> 2,657 persisted DRC
  records** (run id `live-pilot-1`), deterministic and idempotent.

Status distribution: 225 accepted, 1,600 ambiguous, 797
insufficient_context, 34 counterfactual_failure, 1
attribution_template_disagreement.  Every status in the taxonomy fired
on real data, and the conservative gates behaved as designed: early
chronological folds below `min_train_rows` refuse attribution.

## Findings

**1. A population-level contrarian pattern.**  224 of 225 accepted
records are CONTRARIAN_REVERSAL, spread across **204 distinct wallets**
(median ablation delta 0.35, median agreement 1.0).  Real taker flow in
these markets systematically trades against the recent derived price
move, and removing trend information from the fitted model materially
reduces the probability of the observed actions.

**Caveat — microstructure alternative.**  `market_state` is derived
from taker executions themselves.  Buys print near the ask and sells
near the bid, so the derived price oscillates with order flow and the
next trade mechanically "opposes" the last move (bid-ask bounce).  Some
or much of the contrarian signal may be this artifact rather than
deliberate mean-reversion.  Discriminating them requires mid-price
state from continuous order-book collection — the forward-collection
mode this pipeline was designed for.  Until then, CONTRARIAN_REVERSAL
on this dataset should be read as "trades against the last prints",
which is itself a real behavioural regularity, not as confirmed
belief-driven reversal.

**2. Rule-based news relevance is too conservative for real
headlines.**  All 849 judgments were background (455), irrelevant (392)
or indirect (2); zero decisions carried news evidence, so the fresh /
persistent news channels never activated.  Real headlines rarely clear
the token-overlap and cue requirements tuned on synthetic articles.
The Ollama LLM relevance path exists for exactly this and is the next
normalization improvement (its test coverage is already on the
backlog).

**3. Production payload compatibility required four fixes**, all found
by this pilot and now committed: gamma markets adapter (clobTokenIds /
outcomes JSON strings, ISO timestamps, acceptingOrders), bare-dict
payload dispatch (CLOB books), the `asset_id` book key, and repeated
`condition_ids` query params (a comma-joined value silently matches
nothing).  PARSER_VERSION bumped to 1.1.0.

## What this proves and what it does not

Proven: the full DRC pipeline — collection, immutable raw storage,
normalization, strict as-of contexts, Layer-1 gates, template
posterior, counterfactuals, agreement, persistence, rationale — runs on
live Polymarket data end to end and produces defensible, hedged,
versioned reasoning records at real scale.

Not yet proven: template diversity on real data (news channels need the
LLM relevance path; book-history channels need forward collection), and
occurrence reasoning at production pair counts (needs batching).

## Next steps

1. Forward collection loop (books + trades every N minutes for days) to
   get real mid-price state and liquidity history.
2. Ollama LLM news relevance on the collected articles, with tests.
3. Occurrence-target batching for production pair universes.
