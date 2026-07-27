"""Command-line interface.

Commands: init-db, collect, normalize, build-synthetic, audit,
run-analysis.  Expected user errors exit nonzero without stack traces.
No secrets are read from the command line or logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from polymarket.contracts.schema import connect, init_db


def _require_db(path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"error: database not found: {path}")


def cmd_init_db(args: argparse.Namespace) -> int:
    directory = os.path.dirname(args.db)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = init_db(args.db)
    conn.close()
    print(f"initialized schema in {args.db}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    _require_db(args.db)
    from polymarket.collection.client import ObservingClient
    from polymarket.collection.raw_store import (
        finish_collector_run,
        start_collector_run,
    )
    from polymarket.collection.trades import collect_trades

    conn = connect(args.db)
    if args.surface != "trades":
        raise SystemExit(
            f"error: surface {args.surface!r} not wired for CLI collection yet; "
            "supported: trades"
        )
    if not args.condition_id:
        raise SystemExit("error: --condition-id is required for trades")
    for taker_only in (True, False):
        collector = "trades_taker" if taker_only else "trades_expanded"
        run_id = start_collector_run(
            conn, collector,
            {"condition_id": args.condition_id, "takerOnly": taker_only},
        )
        with ObservingClient(conn, collector, run_id) as client:
            outcome = collect_trades(
                client, condition_id=args.condition_id, taker_only=taker_only
            )
        status = "succeeded" if outcome.status == "complete" else "partial"
        finish_collector_run(conn, run_id, status, note=outcome.note)
        print(
            f"{collector}: {outcome.record_count} records over "
            f"{len(outcome.pages)} pages ({outcome.status})"
        )
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    _require_db(args.db)
    from polymarket.normalization.normalizer import Normalizer
    from polymarket.normalization.reconciliation import reconcile_roles

    claim_extractor = None
    relevance_scorer = None

    if args.news_llm:
        try:
            from polymarket.normalization.llm_news import (
                OllamaClaimExtractor,
                OllamaRelevanceScorer,
            )
        except ImportError as exc:
            raise SystemExit(
                'error: install LLM dependencies with '
                'python -m pip install -e ".[llm]"'
            ) from exc

        claim_extractor = OllamaClaimExtractor(args.llm_model)
        relevance_scorer = OllamaRelevanceScorer(args.llm_model)

    conn = connect(args.db)
    results = Normalizer(
        conn,
        claim_extractor=claim_extractor,
        relevance_scorer=relevance_scorer,
    ).normalize_all()
    inserted: dict[str, int] = {}
    unresolved = 0
    for result in results:
        for table, n in result.inserted.items():
            inserted[table] = inserted.get(table, 0) + n
        unresolved += len(result.unresolved)
    diag = reconcile_roles(conn)
    print(f"normalized {len(results)} raw responses")
    for table, n in sorted(inserted.items()):
        print(f"  {table}: {n} inserted")
    print(f"  unresolved records: {unresolved}")
    print(
        f"  roles: {diag.taker_assigned} taker, {diag.maker_assigned} maker, "
        f"{diag.unknown_remaining} unknown"
    )
    return 0


def cmd_build_synthetic(args: argparse.Namespace) -> int:
    from polymarket.synthetic.fixtures import build_synthetic_fixture

    try:
        conn = build_synthetic_fixture(args.db, overwrite=args.overwrite)
    except FileExistsError as exc:
        raise SystemExit(f"error: {exc}") from exc
    n = conn.execute("SELECT COUNT(*) FROM raw_responses").fetchone()[0]
    print(f"built synthetic fixture at {args.db} ({n} raw responses)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    _require_db(args.db)
    from polymarket.analysis.reporting import audit_database

    conn = connect(args.db)
    report = audit_database(conn)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"schema version: {report['schema_version']}")
        print(f"parser version: {report['parser_version']}")
        print("table row counts:")
        for table, count in report["table_row_counts"].items():
            print(f"  {table}: {count}")
        print(f"raw responses by collector: {report['raw_responses_by_collector']}")
        print(f"http status distribution: {report['http_status_distribution']}")
        print(f"collector gaps: {report['collector_gaps']}")
        print(f"backfill windows: {report['backfill']}")
        print(f"unknown-role legs: {report['unknown_role_legs']}")
        print(f"unresolved position events: {report['unresolved_position_events']}")
        print(
            "markets missing outcome mappings: "
            f"{report['markets_missing_outcome_mappings']}"
        )
        print(f"contract version counts: {report['contract_version_counts']}")
        print(f"news ingestion lag: {report['news_ingestion_lag']}")
    return 0


def cmd_run_analysis(args: argparse.Namespace) -> int:
    _require_db(args.db)
    from polymarket.analysis.reader import SQLiteNormalizedReader
    from polymarket.analysis.replay import run_replay
    from polymarket.analysis.reporting import audit_database, write_run_outputs

    reasoning_model = None
    if not args.no_reasoning:
        if not args.reasoning_model:
            raise SystemExit(
                "error: --reasoning-model PATH is required (or pass "
                "--no-reasoning for Layer-1-only output); train one with "
                "python -m polymarket.analysis.reasoning_validation OUT_DIR"
            )
        from polymarket.analysis.reasoning_artifact import (
            ArtifactVersionMismatch,
            load_reasoning_model,
        )

        try:
            reasoning_model, _artifact = load_reasoning_model(
                args.reasoning_model
            )
        except FileNotFoundError:
            raise SystemExit(
                f"error: reasoning model not found: {args.reasoning_model}"
            ) from None
        except ArtifactVersionMismatch as exc:
            raise SystemExit(f"error: {exc}") from None

    reader = SQLiteNormalizedReader(args.db)
    run = run_replay(
        reader,
        end_time=args.end_time,
        interval_seconds=args.interval,
        embargo_seconds=args.embargo,
        seed=args.seed,
        run_id=args.run_id,
        reasoning_model=reasoning_model,
        reasoning_target=args.reasoning_target,
    )
    if run.evaluation is None:
        raise SystemExit(
            "error: analysis produced no evaluation "
            f"({'; '.join(run.notes) or 'insufficient labeled decisions'})"
        )
    from polymarket.analysis.versioning import feature_version_hash

    if reasoning_model is not None:
        from polymarket.analysis.drc import persist_reasoning_records

        persisted = persist_reasoning_records(
            reader.conn,
            run.drc_records + run.occurrence_drc_records,
            reasoning_run_id=run.run_id,
        )
    else:
        # Layer-1-only fallback: still never PARSER_VERSION
        from polymarket.analysis.reasoning import persist_driver_attributions

        persisted = persist_driver_attributions(
            reader.conn, run.driver_attributions,
            feature_version=feature_version_hash(),
        )
    paths = write_run_outputs(run, args.output)
    if reasoning_model is not None:
        from polymarket.analysis.reasoning_reconstruction import (
            write_reasoning_outputs,
        )

        paths.update(write_reasoning_outputs(run, args.output))
    audit_path = os.path.join(args.output, "audit_summary.json")
    with open(audit_path, "w") as fh:
        json.dump(audit_database(reader.conn), fh, indent=2)
    paths["audit_summary"] = audit_path
    print(f"run {run.run_id}: {len(run.labeled_episodes)} labeled decisions")
    print(f"  reasoning judgments persisted: {persisted}")
    for model, metrics in run.evaluation.metrics.items():
        print(
            f"  {model}: log_loss={metrics['log_loss']:.4f} "
            f"brier={metrics['brier']:.4f} acc={metrics['accuracy']:.3f}"
        )
    print(
        "  M2->M3 log-loss improvement: "
        f"{run.evaluation.improvements['m2_to_m3_log_loss']:+.4f}"
    )
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")
    return 0


def cmd_underreaction(args) -> int:
    import json as _json
    import os as _os

    from polymarket.analysis.attention import (
        distraction_interaction_regressions,
    )
    from polymarket.analysis.news_returns import (
        DecompositionConfig,
        build_interval_records,
        daily_aggregation,
    )
    from polymarket.analysis.underreaction import (
        event_absorption,
        run_drift_regressions,
    )
    from polymarket.contracts.schema import connect, ensure_paper_schema

    conn = connect(args.db)
    ensure_paper_schema(conn)
    _os.makedirs(args.output, exist_ok=True)
    config = DecompositionConfig(
        bin_seconds=args.bin_seconds,
        mode_run_id=args.mode_run_id,
        screen_basis=args.screen_basis,
    )
    specs = ["all_relevant"]
    if args.mode_run_id:
        specs.append("screened_impactful")
    report: dict = {"specs": {}}
    for spec in specs:
        records = build_interval_records(conn, config, spec)
        news_n = sum(1 for r in records if r.is_news)
        regressions = run_drift_regressions(
            conn, config, spec, records=records
        )
        # placebo distribution over many permutations -> empirical p
        placebo_betas: dict[float, list[float]] = {}
        for seed in range(args.placebo_seeds):
            for result in run_drift_regressions(
                conn, config, spec, records=records,
                placebo_seed=10_000 + seed,
            ):
                placebo_betas.setdefault(
                    result.horizon_seconds, []
                ).append(result.beta_news)
        placebo_summary = {}
        for result in regressions:
            draws = placebo_betas.get(result.horizon_seconds, [])
            if draws:
                exceed = sum(
                    1 for b in draws if abs(b) >= abs(result.beta_news)
                )
                placebo_summary[str(result.horizon_seconds)] = {
                    "n_placebos": len(draws),
                    "mean_beta": sum(draws) / len(draws),
                    "empirical_p_news": (exceed + 1) / (len(draws) + 1),
                }
        # COMPLETE daily table as its own artifact
        daily = daily_aggregation(records)
        with open(
            _os.path.join(args.output, f"daily_{spec}.json"), "w"
        ) as handle:
            _json.dump(daily, handle, indent=2, sort_keys=True)
        report["specs"][spec] = {
            "intervals": len(records),
            "news_intervals": news_n,
            "daily_rows": len(daily),
            "daily_file": f"daily_{spec}.json",
            "drift_regressions": [r.as_dict() for r in regressions],
            "placebo": placebo_summary,
        }
        events = event_absorption(conn, config, spec)
        with open(
            _os.path.join(args.output, f"events_{spec}.jsonl"), "w"
        ) as handle:
            for event in events:
                handle.write(_json.dumps(event, sort_keys=True) + "\n")
        report["specs"][spec]["events"] = len(events)
        report["specs"][spec]["events_with_intervening_news"] = sum(
            1 for e in events if e["intervening_news"]
        )
        interactions = distraction_interaction_regressions(
            conn, config, spec, mode_run_id=args.mode_run_id
        )
        report["specs"][spec]["distraction_interactions"] = interactions
    report["analyst_revision_mechanism"] = (
        "UNTESTED: requires an external expectations series the "
        "pipeline does not have"
    )
    path = _os.path.join(args.output, "underreaction_report.json")
    with open(path, "w") as handle:
        _json.dump(report, handle, indent=2, sort_keys=True)
    for spec, block in report["specs"].items():
        print(f"[{spec}] intervals={block['intervals']} "
              f"news={block['news_intervals']} "
              f"regressions={len(block['drift_regressions'])}")
    print(f"wrote {path}")
    conn.close()
    return 0


def cmd_fit_liquidity_modes(args) -> int:
    import json as _json
    import time as _time

    from polymarket.analysis.liquidity_modes import (
        JumpModelConfig,
        fit_jump_model,
        persist_jump_model,
    )
    from polymarket.contracts.schema import connect, ensure_paper_schema

    conn = connect(args.db)
    ensure_paper_schema(conn)
    fit_cutoff = args.fit_cutoff or _time.time()
    config = JumpModelConfig(
        bin_seconds=args.bin_seconds,
        fixed_lambda=args.fixed_lambda,
    )
    try:
        model = fit_jump_model(conn, fit_cutoff=fit_cutoff, config=config)
    except ValueError as exc:
        print(f"cannot fit: {exc}")
        conn.close()
        return 1
    persist_jump_model(conn, model, fit_cutoff)
    labels = {}
    for (_, _), mode in model.assignments.items():
        label = "calm" if mode == model.calm_mode else "event"
        labels[label] = labels.get(label, 0) + 1
    print(_json.dumps({
        "mode_run_id": model.mode_run_id,
        "lambda": model.lambda_penalty,
        "lambda_selection": model.lambda_selection,
        "train_bars": model.train_bar_count,
        "assigned_bars": len(model.assignments),
        "mode_counts": labels,
        "centroids": model.centroids,
        "calm_mode": model.calm_mode,
    }, indent=2, sort_keys=True))
    conn.close()
    return 0


def cmd_screen_news_impact(args) -> int:
    import json as _json

    from polymarket.analysis.news_impact_screen import screen_news_impact
    from polymarket.contracts.schema import connect, ensure_paper_schema

    conn = connect(args.db)
    ensure_paper_schema(conn)
    mode_run_id = args.mode_run_id
    if mode_run_id is None:
        row = conn.execute(
            "SELECT mode_run_id FROM liquidity_mode_runs "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            print("no fitted mode runs; run fit-liquidity-modes first")
            return 1
        mode_run_id = row[0]
    counters = screen_news_impact(
        conn, mode_run_id, assignment_basis=args.assignment_basis
    )
    print(_json.dumps(counters, indent=2, sort_keys=True))
    conn.close()
    return 0


def cmd_build_liquidity_bars(args) -> int:
    from polymarket.analysis.liquidity_bars import (
        LiquidityBarConfig,
        build_liquidity_bars,
    )
    from polymarket.contracts.schema import connect, ensure_paper_schema

    conn = connect(args.db)
    ensure_paper_schema(conn)
    config = LiquidityBarConfig(bin_seconds=args.bin_seconds)
    condition_ids = args.condition_id or [
        row[0] for row in conn.execute(
            "SELECT DISTINCT condition_id FROM markets"
        )
    ]
    for condition_id in condition_ids:
        start = None
        if not args.rebuild:
            row = conn.execute(
                "SELECT MAX(bin_start) FROM liquidity_bars "
                "WHERE condition_id = ? AND bin_seconds = ?",
                (condition_id, config.bin_seconds),
            ).fetchone()
            # recompute the last bar too: it may have been partial
            start = row[0] if row and row[0] is not None else None
        written = build_liquidity_bars(
            conn, condition_id, start=start, config=config
        )
        stats = conn.execute(
            """
            SELECT COUNT(*),
                   SUM(coverage_complete),
                   AVG(book_observation_count),
                   AVG(book_coverage_fraction)
            FROM liquidity_bars
            WHERE condition_id = ? AND bin_seconds = ?
            """,
            (condition_id, config.bin_seconds),
        ).fetchone()
        total, complete, mean_obs, mean_cov = stats
        complete = complete or 0
        print(
            f"{condition_id[:16]}…: wrote {written} bars | total "
            f"{total} ({complete} complete, {total - complete} "
            f"incomplete) | mean observations "
            f"{(mean_obs or 0):.1f} | mean coverage "
            f"{(mean_cov or 0):.2f}"
        )
    conn.close()
    return 0


def cmd_rescore_news(args) -> int:
    import json as _json

    from polymarket.contracts.schema import connect, ensure_paper_schema
    from polymarket.normalization.rescore import make_scorer, rescore_news

    conn = connect(args.db)
    ensure_paper_schema(conn)
    scorer = make_scorer(args.method, model=args.model)
    print(
        f"rescoring news relevance: method={args.method} "
        f"model_version={getattr(scorer, 'version', '?')}"
    )
    counters = rescore_news(
        conn, scorer, method=args.method, limit=args.limit
    )
    print(_json.dumps(counters, indent=2, sort_keys=True))
    conn.close()
    return 0


def cmd_migrate(args) -> int:
    from polymarket.contracts.schema import connect, ensure_paper_schema

    conn = connect(args.db)
    applied = ensure_paper_schema(conn)
    print("applied:", applied or "nothing (already current)")
    conn.close()
    return 0


def cmd_collect_loop(args) -> int:
    from polymarket.collection.forward import (
        ForwardConfig,
        default_cycle_printer,
        run_loop,
    )
    from polymarket.contracts.schema import connect, init_db

    if os.path.exists(args.db):
        conn = connect(args.db)
    else:
        os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
        conn = init_db(args.db, description="forward collection")
    config = ForwardConfig(
        condition_ids=tuple(args.condition_id),
        book_every=args.book_every,
        trade_every=args.trade_every,
        market_every=args.market_every,
        news_every_seconds=args.news_every,
        activity_every_seconds=args.activity_every,
        activity_wallets=args.activity_wallets,
        news_queries=tuple(args.news_query),
        trade_pages_per_cycle=args.trade_pages,
    )
    duration = args.duration_hours * 3600.0 if args.duration_hours else None
    print(
        f"forward collection: {len(config.condition_ids)} markets | "
        f"books every {config.book_every:g}s, trades every "
        f"{config.trade_every:g}s, news every "
        f"{config.news_every_seconds:g}s, activity every "
        f"{config.activity_every_seconds:g}s"
        + (f" | for {args.duration_hours:g}h" if duration else "")
        + (f" | {args.cycles} cycles" if args.cycles else "")
        + " (Ctrl-C stops after the current cycle)"
    )
    state = run_loop(
        conn, config, max_cycles=args.cycles,
        duration_seconds=duration, on_cycle=default_cycle_printer,
    )
    print(
        f"done: {state.cycle_index} cycles"
        + (f", failures: {state.failures}" if state.failures else "")
    )
    conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket.cli",
        description="Polymarket research pipeline (read-only; no trading)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create schema in a new or existing db")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("collect", help="collect live data (network required)")
    p.add_argument("--db", required=True)
    p.add_argument("--surface", required=True, choices=["trades"])
    p.add_argument("--condition-id")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("normalize", help="normalize all stored raw responses")
    p.add_argument("--db", required=True)
    p.add_argument(
        "--news-llm",
        action="store_true",
        help="use a local Ollama model for news extraction and relevance",
    )
    p.add_argument(
        "--llm-model",
        default="qwen3:8b",
        help="Ollama model used with --news-llm",
    )
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("build-synthetic", help="build the synthetic fixture db")
    p.add_argument("--db", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_build_synthetic)

    p = sub.add_parser("audit", help="audit a database")
    p.add_argument("--db", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser(
        "underreaction-analysis",
        help="news/non-news decomposition and drift regressions "
             "(Pervasive Underreaction)",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bin-seconds", type=float, default=900.0)
    p.add_argument("--mode-run-id", default=None,
                   help="enables the screened_impactful spec and the "
                        "event-mode prevalence proxy")
    p.add_argument("--screen-basis", default="retrospective_smoothed",
                   choices=["retrospective_smoothed", "online_filtered"])
    p.add_argument("--placebo-seeds", type=int, default=20)
    p.set_defaults(func=cmd_underreaction)

    p = sub.add_parser(
        "fit-liquidity-modes",
        help="fit the two-state liquidity jump model (paper section 3)",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--bin-seconds", type=float, default=300.0)
    p.add_argument("--fit-cutoff", type=float, default=None,
                   help="train on bars strictly before this epoch "
                        "(default: now)")
    p.add_argument("--fixed-lambda", type=float, default=None,
                   help="skip persistence-target selection")
    p.set_defaults(func=cmd_fit_liquidity_modes)

    p = sub.add_parser(
        "screen-news-impact",
        help="calm->event news impact screen (paper section 4)",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--mode-run-id", default=None,
                   help="default: the most recent fitted run")
    p.add_argument("--assignment-basis", default="online_filtered",
                   choices=["online_filtered", "retrospective_smoothed"],
                   help="online_filtered makes a TRUE availability "
                        "claim; retrospective is for labelled offline "
                        "paper analyses")
    p.set_defaults(func=cmd_screen_news_impact)

    p = sub.add_parser(
        "build-liquidity-bars",
        help="build five-minute liquidity bars (paper data contract)",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--condition-id", action="append", default=None,
                   help="repeatable; default: every market in the db")
    p.add_argument("--bin-seconds", type=float, default=300.0)
    p.add_argument("--rebuild", action="store_true",
                   help="recompute all bars instead of incrementally")
    p.set_defaults(func=cmd_build_liquidity_bars)

    p = sub.add_parser(
        "rescore-news",
        help="re-score stored news claims with a chosen relevance scorer",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--method", choices=["rule", "ollama"], required=True)
    p.add_argument("--model", default=None,
                   help="ollama model name (default qwen3:8b)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N new judgments (resumable)")
    p.set_defaults(func=cmd_rescore_news)

    p = sub.add_parser(
        "migrate", help="upgrade an existing db to the current schema"
    )
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser(
        "collect-loop",
        help="forward collection: books/trades/status every N minutes",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--condition-id", action="append", required=True)
    p.add_argument("--book-every", type=float, default=60.0,
                   help="seconds between order-book snapshots")
    p.add_argument("--trade-every", type=float, default=300.0,
                   help="seconds between incremental trade pulls")
    p.add_argument("--market-every", type=float, default=300.0,
                   help="seconds between market-status refreshes")
    p.add_argument("--news-every", type=float, default=300.0,
                   help="seconds between news-feed pulls")
    p.add_argument("--activity-every", type=float, default=3600.0,
                   help="seconds between wallet-activity refreshes")
    p.add_argument("--duration-hours", type=float, default=None)
    p.add_argument("--cycles", type=int, default=None,
                   help="stop after N loop ticks (tests/smoke)")
    p.add_argument("--activity-wallets", type=int, default=30)
    p.add_argument("--news-query", action="append", default=[])
    p.add_argument("--trade-pages", type=int, default=5)
    p.set_defaults(func=cmd_collect_loop)

    p = sub.add_parser("run-analysis", help="replay decisions and fit models")
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--end-time", type=float, default=None)
    p.add_argument("--interval", type=float, default=3600.0)
    p.add_argument("--embargo", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--reasoning-model",
        help="path to a reasoning model artifact (reasoning_model.json)",
    )
    p.add_argument(
        "--reasoning-target",
        choices=["direction", "occurrence", "both"],
        default="direction",
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Layer-1 driver attribution only (no template posterior)",
    )
    p.add_argument(
        "--run-id",
        help="stable run identifier; reruns with the same id replace "
             "their own judgments (idempotent)",
    )
    p.set_defaults(func=cmd_run_analysis)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
