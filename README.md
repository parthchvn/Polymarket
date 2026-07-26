# Polymarket research pipeline

A research pipeline that studies how identifiable Polymarket actors make
decisions in response to market state, their positions, their history,
newly available news, and contract rules.  It reconstructs decision
episodes using ONLY information available strictly before each decision,
then tests whether relevant news adds predictive value beyond actor,
market and position context.

**This is a research pipeline, not a trading bot.**  It is read-only
with respect to Polymarket: no order placement, no wallet signing, no
private keys.

## Architecture

```text
Polymarket APIs and news sources
  → immutable raw responses (exact bytes + provenance)
  → single normalization path
  → shared normalized SQLite schema (real and synthetic identical)
  → strict as-of readers (available_at < t, never <=)
  → decision episodes → features → nested models M0–M3
  → placebos, bootstraps, candidate news attribution
  → predictions.csv / metrics.json / config.json / feature_manifest.json
```

See `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`,
`docs/TEMPORAL_VALIDITY.md`, `docs/TRADE_SEMANTICS.md`,
`docs/OPERATIONS.md`, `docs/RESEARCH_ASSUMPTIONS.md`.

## Installation

```bash
cd ~/Polymarket
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick start (no network needed)

```bash
python -m polymarket.cli init-db --db /tmp/polymarket-empty.sqlite
python -m polymarket.cli build-synthetic --db /tmp/polymarket-synthetic.sqlite --overwrite
python -m polymarket.cli audit --db /tmp/polymarket-synthetic.sqlite
python -m polymarket.cli run-analysis --db /tmp/polymarket-synthetic.sqlite \
  --output /tmp/polymarket-analysis
```

## CLI

| command | purpose |
|---|---|
| `init-db --db PATH` | create the schema |
| `collect --db PATH --surface trades --condition-id ID` | live collection (both trade views) |
| `normalize --db PATH` | normalize all stored raw responses + reconcile roles |
| `build-synthetic --db PATH --overwrite` | deterministic synthetic fixture |
| `audit --db PATH [--json]` | database audit report |
| `run-analysis --db PATH --output DIR` | replay, models, placebos, outputs |

## Tests

```bash
ruff check src tests
pytest -q            # no live network required
pytest -m live       # (reserved) live API probes only
```

## The temporal rule

Every piece of decision context satisfies `available_at < t` (strict).
News availability is `first_observed_at` (collector time), never
publication time.  A runtime assertion rejects contaminated contexts.
See `docs/TEMPORAL_VALIDITY.md`.

## Time-decayed news features

Semantic relevance is permanent (`relevance_judgments` is never
rewritten); the decision-specific news signal decays with age using
half-life weighting over four horizons (6h, 24h, 72h, 168h, capped at 28
days), so yesterday's news can still inform today's decision at reduced
weight.  Evidence is aggregated per event family (duplicates take the
max, independent events add) and positive/negative components remain
separate.  No news at or after the decision time ever contributes.  All
decay parameters are recorded in `config.json` and
`feature_manifest.json`; the half-lives are modelling choices, not
established causal parameters.

## Data safety

- Raw observations are immutable and append-only.
- Never commit `.env`, API keys, wallet secrets, or production
  databases (`.gitignore` enforces the common cases).
- The committed `fixtures/synthetic_normalized.sqlite` is fully
  synthetic and safe.

## Limitations

- All collectors are fixture/mock-tested; live API payload shapes and
  historical parameter semantics are unvalidated (see
  `docs/RESEARCH_ASSUMPTIONS.md`).
- News attribution output is candidate attribution, never causal.
- Historical completeness is never claimed unless demonstrated by
  backfill validation.
