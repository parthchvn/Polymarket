"""Live Polymarket collection pilot: real data through the full DRC
pipeline.

Collects markets, taker + expanded trade history, current order books,
wallet activity for the most active takers, and (optionally) a public
news feed for chosen markets; then normalizes, derives market state,
reconciles roles, audits, and runs the full reasoning analysis with a
trained model artifact.

Read-only throughout: no keys, no orders, no trading.

Usage:
    python scripts/run_live_pilot.py --db runs/pilot/pilot.sqlite \
        --output runs/pilot/out \
        --reasoning-model runs/validation/reasoning_model.json \
        --condition-id 0x... --condition-id 0x... \
        [--max-trade-pages 40] [--activity-wallets 30] [--news-query "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from polymarket.collection.activity import collect_activity
from polymarket.collection.books import collect_book
from polymarket.collection.client import ObservingClient
from polymarket.collection.markets import collect_markets
from polymarket.collection.news import collect_google_news_rss
from polymarket.collection.raw_store import (
    finish_collector_run,
    start_collector_run,
)
from polymarket.collection.trades import collect_trades
from polymarket.contracts.schema import connect, init_db
from polymarket.normalization.markets import (
    derive_market_state_from_books,
    derive_market_state_from_executions,
)
from polymarket.normalization.normalizer import Normalizer
from polymarket.normalization.reconciliation import reconcile_roles


def _collector_run(conn, collector, params):
    return start_collector_run(conn, collector, params)


def collect_surface(conn, collector, params, fn, **kwargs):
    run_id = _collector_run(conn, collector, params)
    try:
        with ObservingClient(conn, collector, run_id) as client:
            outcome = fn(client, **kwargs)
        status = "succeeded"
        note = getattr(outcome, "note", None)
        count = getattr(outcome, "record_count", None)
    except Exception as exc:  # noqa: BLE001 - pilot must report, not die
        finish_collector_run(conn, run_id, "failed", note=str(exc))
        print(f"  {collector}: FAILED ({exc})")
        return None
    finish_collector_run(conn, run_id, status, note=note)
    if count is not None:
        print(f"  {collector}: {count} records")
    return outcome


def top_taker_wallets(conn, condition_ids, limit):
    placeholders = ",".join("?" for _ in condition_ids)
    rows = conn.execute(
        f"""
        SELECT proxy_wallet, COUNT(*) AS n FROM actor_trade_legs
        WHERE condition_id IN ({placeholders})
        GROUP BY proxy_wallet ORDER BY n DESC LIMIT ?
        """,
        (*condition_ids, limit),
    ).fetchall()
    return [row[0] for row in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reasoning-model", required=True)
    parser.add_argument("--condition-id", action="append", required=True)
    parser.add_argument("--max-trade-pages", type=int, default=40)
    parser.add_argument("--activity-wallets", type=int, default=30)
    parser.add_argument("--news-query", action="append", default=[])
    parser.add_argument("--run-id", default="live-pilot")
    parser.add_argument("--skip-collection", action="store_true",
                        help="reuse an already-collected database")
    args = parser.parse_args(argv)

    if os.path.exists(args.db):
        conn = connect(args.db)
    else:
        os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
        conn = init_db(args.db, description="live collection pilot")

    if not args.skip_collection:
        print("collecting market metadata + status ...")
        collect_surface(
            conn, "markets", {"condition_ids": args.condition_id},
            collect_markets, condition_ids=args.condition_id,
        )
        for condition_id in args.condition_id:
            print(f"collecting trades for {condition_id[:18]}… ")
            for taker_only in (True, False):
                collector = (
                    "trades_taker" if taker_only else "trades_expanded"
                )
                collect_surface(
                    conn, collector,
                    {"condition_id": condition_id, "takerOnly": taker_only},
                    collect_trades, condition_id=condition_id,
                    taker_only=taker_only, max_pages=args.max_trade_pages,
                )
        # normalize what we have so far so token ids and wallets exist
        print("normalizing (pass 1: markets + trades) ...")
        Normalizer(conn).normalize_all()
        assets = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT asset FROM outcome_tokens "
                "WHERE condition_id IN ({})".format(
                    ",".join("?" for _ in args.condition_id)
                ),
                args.condition_id,
            )
        ]
        print(f"collecting order books for {len(assets)} assets ...")
        for asset in assets:
            run_id = _collector_run(conn, "books", {"asset": asset})
            try:
                with ObservingClient(conn, "books", run_id) as client:
                    collect_book(client, asset=asset)
                finish_collector_run(conn, run_id, "succeeded")
            except Exception as exc:  # noqa: BLE001
                finish_collector_run(conn, run_id, "failed", note=str(exc))
                print(f"  book {asset[:16]}…: FAILED ({exc})")
        wallets = top_taker_wallets(
            conn, args.condition_id, args.activity_wallets
        )
        print(f"collecting activity for {len(wallets)} most active wallets ...")
        for wallet in wallets:
            collect_surface(
                conn, "activity", {"wallet": wallet},
                collect_activity, wallet=wallet, max_pages=10,
            )
        for query in args.news_query:
            count = collect_google_news_rss(conn, query)
            print(f"  news:google-rss [{query!r}]: {count} articles")

    print("normalizing (final pass) ...")
    results = Normalizer(conn).normalize_all()
    errors = [e for r in results for e in r.errors]
    if errors:
        print(f"  normalization errors: {len(errors)} (continuing)")
        for error in errors[:5]:
            print(f"    {error}")
    reconcile_roles(conn)
    for condition_id in args.condition_id:
        derive_market_state_from_executions(
            conn, condition_id, bucket_seconds=3600.0
        )
        book_states = derive_market_state_from_books(conn, condition_id)
        if book_states:
            print(f"  book-mid states for {condition_id[:14]}…: {book_states}")
    conn.commit()

    from polymarket.analysis.reporting import audit_database

    audit = audit_database(conn)
    print("audit:", json.dumps(
        {k: audit[k] for k in sorted(audit) if isinstance(audit[k], (int, float))},
        indent=None,
    )[:500])
    conn.close()

    print("running full reasoning analysis ...")
    from polymarket.cli import main as cli_main

    return cli_main([
        "run-analysis", "--db", args.db, "--output", args.output,
        "--reasoning-model", args.reasoning_model,
        "--reasoning-target", "direction",
        "--run-id", args.run_id,
    ])


if __name__ == "__main__":
    sys.exit(main())
