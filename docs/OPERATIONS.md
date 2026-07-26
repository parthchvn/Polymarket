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
