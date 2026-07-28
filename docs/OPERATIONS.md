# Operations

## Environment

```bash
cd ~/Polymarket
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional credentials (none required for the read-only pipeline) go in
environment variables, never in the repository.  `.env` is gitignored.

## Forward collection

```bash
python -m polymarket.cli init-db --db data/polymarket.sqlite
python -m polymarket.cli collect --db data/polymarket.sqlite \
  --surface trades --condition-id <CONDITION_ID>
python -m polymarket.cli normalize --db data/polymarket.sqlite
```

Collection stores raw only; normalization is a separate, idempotent
step.  Order books and contract snapshots have little historical depth —
forward collection matters most for them.

## Restart and backfill

Backfill windows are tracked in `backfill_windows`.  Only validated
windows are `complete`; anything else is `incomplete`/`failed` and shows
up in `pending_windows`, so restarts are safe.  Failed windows record
`collector_gaps` rows that downstream analysis treats as blocking.

## Gaps

`collector_gaps` is queryable by surface/object/time.  The opportunity
checker rejects decision windows that overlap unresolved gaps.

## Audit

```bash
python -m polymarket.cli audit --db data/polymarket.sqlite [--json]
```

## Backups

The database is a single SQLite file in WAL mode.  Back up by copying
the `.sqlite` file after `PRAGMA wal_checkpoint(TRUNCATE)` or while no
writer is active.  Never commit production databases.


## Forward collection

Continuous incremental collection with PER-SURFACE cadences (seconds):

    python -m polymarket.cli collect-loop --db runs/forward.sqlite \
        --condition-id 0x... --condition-id 0x... \
        --book-every 60 --trade-every 300 --news-every 300 \
        --activity-every 3600 --duration-hours 72 --news-query "..."

The screening paper's five-minute liquidity bars need many within-bin
book observations (average spread, average best-level size, within-bin
volatility), so books default to 60s while trades/news poll at 300s
and activity at 3600s.  The loop ticks at the fastest cadence and each
surface runs only when due, so paper-grade book sampling never repeats
trade calls.  The data-api /trades endpoint ignores server-side time
filters (validated), so trade increments use newest-first pagination
with a client-side early stop at the last-collected timestamp; the
overlap page is deduplicated at normalization by record fingerprint.
Failures are isolated per surface and recorded as BOUNDED collector
gaps for exactly the affected window; restart downtime is detected
from the newest stored response and recorded the same way.  Ctrl-C
stops after the current cycle; restarts resume incrementally.  Upgrade
an existing database first with `python -m polymarket.cli migrate
--db ...` (idempotent: best-level book sizes, tick size,
liquidity_bars table).

## Paper data contract

`liquidity_bars` (built by
`polymarket.analysis.liquidity_bars.build_liquidity_bars`) carries the
screening paper's four variables per five-minute bin, adapted to
bounded prices: mean spread (raw and in ticks), executed notional
turnover (canonical executions only), realized log-odds variance with
logit OHLC, and mean BEST-level book size kept separate from mean
total depth.  Bars form a REGULAR grid — empty
intervals exist explicitly as incomplete rather than vanishing from
the temporal sequence (dropping them would bias the jump model's
transitions).  `coverage_complete` requires no blocking gap, at least
`min_book_observations` (default 4) book observations, and a
computable realized variance (seeded with the last midquote before the
bin so the first within-bin return is kept); expected counts and the
achieved coverage fraction are stored per bar.  Build bars with:

    python -m polymarket.cli build-liquidity-bars --db runs/forward.sqlite

which is incremental by default (--rebuild recomputes) and prints
per-market coverage statistics.

The canonical market series (`reader.market_series_before`) takes an
explicit source policy — `book_only` for paper analyses,
`book_preferred` (flagged execution fallback) for general reasoning —
so midquotes and trade prints are never silently mixed; the
`mkt_state_from_executions` feature exposes the fallback to the model.

## News relevance rescoring

    python -m polymarket.cli rescore-news --db runs/forward.sqlite \
        --method ollama --model qwen3:8b        # or --method rule

Writes NEW versioned judgments against exact contract semantics.
Every judgment carries a deterministic id over (claim, family, market,
contract version, method, model version): different scorers never
collide, later claims in an already-scored family still get scored,
and runs are resumable (existing ids are skipped).  Rows keep
`source_effective_at` (text availability) and `scored_at` (when the
scorer ran) apart; the as-of ordering key stays anchored to text
availability +1s so rescores supersede batch judgments in snapshots,
while live-online analyses that must not pretend an LLM result
predated its computation can order by `scored_at`.  The ollama method needs `pip install ollama` and a
running Ollama server.


## Concurrent access

Connections use WAL with a 30s busy timeout: a collector loop and a
short analysis command can share one database.  Long writers (LLM
normalization holds a transaction per raw response, potentially for
minutes) must NOT run against the collector's database.  Overnight
pattern: the collector owns ``forward.sqlite`` exclusively; a work
database accumulates via snapshot-and-merge —

    python-level: sqlite3 backup of forward.sqlite to snap.sqlite,
    then on work.sqlite: ATTACH snap; INSERT OR IGNORE into
    collector_runs, collector_gaps, raw_responses; DETACH.

The raw layer is append-only with stable primary keys, so the merge
is idempotent; all normalization/LLM/analysis then runs on
work.sqlite with zero contention.
