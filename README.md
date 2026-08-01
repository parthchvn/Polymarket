# Polymarket decision-reasoning pipeline

A research pipeline that reconstructs how identifiable Polymarket
actors make real decisions — and learns structured
Decision-Reasoning-Context (DRC) records from them.  Every decision is
paired with a STRICT pre-decision context (only information available
before the decision, `available_at < t`, never `<=`), a behavioral
reasoning inference with explicit uncertainty, and an ex-post outcome
layer that is structurally quarantined from inference.

**This is a research pipeline, not a trading bot.**  Read-only with
respect to Polymarket: no order placement, no wallet signing, no
private keys.

## What it produces

```text
D  the decision      wallet, market, direction, size, exact timing
C  strict context    market state, order books, positions, actor
                     history, screened news, liquidity modes —
                     everything knowable BEFORE the decision, with
                     explicit missingness (absence is recorded as
                     absence, never silently filled)
R  reasoning         template posteriors + latent behavioral
                     primitives, accepted only through gates; weak
                     evidence yields MIXED/ambiguous, thin coverage
                     yields insufficient_context — refusal is a
                     first-class answer
O  outcomes          realised post-decision drift and resolution,
                     attached as a FINAL export pass; a structural
                     test proves no outcome ever enters C or R
```

## Architecture

```text
forward collector (books 60s, trades/news 300s, wallet activity 1h)
  + historical import (SII blockchain dataset: 1.8M resolved markets)
  → immutable raw responses (exact bytes + provenance, gaps recorded)
  → single normalization path (real and synthetic share one schema)
  → article-body download + LLM claim extraction + relevance v2
  → liquidity bars → jump-model modes → prequential impact screens
  → strict as-of decision contexts (paper-derived features included)
  → DRC records with counterfactuals, calibration, honest abstention
  → latent reasoning scaffold (rank-K bottleneck; held-out actors
    AND held-out time; clusters unnamed until stability holds)
  → human annotation loop (LLM-drafted labels, human-reviewed,
    Cohen's kappa; consensus-only gold labels)
  → one-command pipeline → acceptance_report.json (four gates)
```

## The acceptance gates

A reasoning claim is accepted only when it earns it:

1. **Predictive value** — latent R beats the C-only baseline AND the
   base-rate null on held-out actors and a held-out final time slice.
2. **Human agreement** — model output vs unanimous human consensus
   labels on strict pre-decision records.
3. **Stability** — latent clusters reproduce across seeds
   (rotation-invariant partition matching).
4. **Honest abstention** — the DRC status distribution; abstention
   must fall as coverage grows, tracked across runs.

Refusals state exactly what data would flip them.

## Quickstart

```bash
pip install -e ".[llm]"           # + duckdb for the historical import
python -m pytest -q               # the full suite

# continuous collection (own terminal; owns forward.sqlite exclusively)
python -m polymarket.cli collect-loop --db runs/forward.sqlite \
  --condition-id 0x... --news-query "..." --book-every 60

# everything else, one command (on a snapshot/work database):
python -m polymarket.cli reasoning-pipeline --db runs/work.sqlite \
  --output runs/daily --reasoning-model runs/model/reasoning_model.json
cat runs/daily/acceptance_report.json

# historical scale (blockchain dataset, resolved outcomes):
python -m polymarket.cli import-sii --db runs/history.sqlite \
  --markets-parquet runs/sii/markets.parquet \
  --quant-source runs/sii/quant.parquet --top-n 50
```

Operational pattern: the collector owns its database exclusively;
analysis and LLM work run on a snapshot-merged work database
(`docs/OPERATIONS.md`).

## Method notes

The liquidity-mode and news-impact machinery implements the jump-model
screening of arXiv 2304.05115 with a genuinely online decoder and
prequential deployment (each window's news is screened by the PREVIOUS
cycle's model — no lookahead, with the applied threshold recorded per
row).  The underreaction analysis follows the drift-regression design
of the pervasive-underreaction literature with market censoring,
fresh-endpoint discipline, two-way clustered errors and per-dimension
refusal.  `docs/PAPER_METHODS.md` records every adaptation and every
deliberate deviation.

## Documentation

`docs/ARCHITECTURE.md` · `docs/DATA_CONTRACT.md` ·
`docs/TEMPORAL_VALIDITY.md` · `docs/TRADE_SEMANTICS.md` ·
`docs/OPERATIONS.md` · `docs/PAPER_METHODS.md` ·
`docs/RESEARCH_ASSUMPTIONS.md`
