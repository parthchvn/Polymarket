"""Write machine-readable run outputs and the database audit summary."""

from __future__ import annotations

import csv
import dataclasses
import json
import sqlite3
from pathlib import Path

from polymarket.analysis.replay import ReplayRun
from polymarket.contracts.schema import REQUIRED_TABLES


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def write_run_outputs(run: ReplayRun, output_dir: str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    predictions_path = out / "predictions.csv"
    rows = run.evaluation.per_decision if run.evaluation else []
    with predictions_path.open("w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("decision_id,time,label\n")
    paths["predictions"] = str(predictions_path)

    metrics = {
        "run_id": run.run_id,
        "n_episodes": len(run.episodes),
        "n_labeled": len(run.labeled_episodes),
        "metrics": run.evaluation.metrics if run.evaluation else {},
        "improvements": run.evaluation.improvements if run.evaluation else {},
        "folds": run.evaluation.folds if run.evaluation else [],
        "placebos": dataclasses.asdict(run.placebos) if run.placebos else None,
        "bootstraps": [dataclasses.asdict(b) for b in run.bootstraps],
        "attributions": run.attributions,
        "notes": run.notes,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default))
    paths["metrics"] = str(metrics_path)

    config_path = out / "config.json"
    config_path.write_text(json.dumps(run.config, indent=2))
    paths["config"] = str(config_path)

    if run.driver_attributions:
        from polymarket.analysis.reasoning import attribution_report

        reasoning_path = out / "reasoning.json"
        reasoning_path.write_text(
            json.dumps(
                {
                    "reasoning_run_id": run.run_id,
                    "method": "predictive driver attribution",
                    "note": (
                        "predictive attribution, not mechanism inference; "
                        "template fields are reserved for a later layer"
                    ),
                    "records": attribution_report(run.driver_attributions),
                },
                indent=2,
            )
        )
        paths["reasoning"] = str(reasoning_path)

    from polymarket.analysis.features import feature_manifest

    manifest_path = out / "feature_manifest.json"
    manifest_path.write_text(json.dumps(feature_manifest(), indent=2))
    paths["feature_manifest"] = str(manifest_path)
    return paths


# ---------------------------------------------------------------------------
def audit_database(conn: sqlite3.Connection) -> dict:
    def one(sql: str, *args) -> sqlite3.Row | None:
        return conn.execute(sql, args).fetchone()

    def value(sql: str, *args, default=0):
        row = one(sql, *args)
        return row[0] if row and row[0] is not None else default

    report: dict = {}
    row = one(
        "SELECT schema_version, parser_version FROM schema_metadata "
        "ORDER BY schema_version DESC LIMIT 1"
    )
    report["schema_version"] = row["schema_version"] if row else None
    report["parser_version"] = row["parser_version"] if row else None

    def safe_count(table: str):
        try:
            return value(f"SELECT COUNT(*) FROM {table}")
        except sqlite3.OperationalError:
            return None  # table absent in a legacy database

    report["table_row_counts"] = {
        table: safe_count(table) for table in REQUIRED_TABLES
    }
    report["raw_responses_by_collector"] = {
        r["collector"]: r["n"]
        for r in conn.execute(
            "SELECT collector, COUNT(*) AS n FROM raw_responses GROUP BY collector"
        )
    }
    report["http_status_distribution"] = {
        str(r["http_status"]): r["n"]
        for r in conn.execute(
            "SELECT http_status, COUNT(*) AS n FROM raw_responses "
            "GROUP BY http_status"
        )
    }
    report["collector_gaps"] = {
        "total": value("SELECT COUNT(*) FROM collector_gaps"),
        "unresolved": value(
            "SELECT COUNT(*) FROM collector_gaps WHERE resolved_at IS NULL"
        ),
    }
    report["backfill"] = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM backfill_windows GROUP BY status"
        )
    }
    report["lineage"] = {
        "actor_legs_missing_lineage": value(
            "SELECT COUNT(*) FROM actor_trade_legs WHERE raw_response_id IS NULL"
        ),
        "executions_missing_lineage": value(
            "SELECT COUNT(*) FROM canonical_executions WHERE raw_response_id IS NULL"
        ),
    }
    report["unknown_role_legs"] = value(
        "SELECT COUNT(*) FROM actor_trade_legs WHERE liquidity_role = 'unknown'"
    )
    report["unresolved_position_events"] = value(
        "SELECT COUNT(*) FROM position_events "
        "WHERE accounting_confidence = 'unresolved'"
    )
    report["markets_missing_outcome_mappings"] = value(
        """
        SELECT COUNT(*) FROM markets m
        WHERE NOT EXISTS (
            SELECT 1 FROM outcome_tokens ot
            WHERE ot.condition_id = m.condition_id
        )
        """
    )
    report["contract_version_counts"] = {
        r["market_id"]: r["n"]
        for r in conn.execute(
            "SELECT market_id, COUNT(*) AS n FROM contract_versions "
            "GROUP BY market_id"
        )
    }
    lag = one(
        """
        SELECT AVG(first_observed_at - source_published_at) AS mean_lag,
               MAX(first_observed_at - source_published_at) AS max_lag
        FROM news_articles WHERE source_published_at IS NOT NULL
        """
    )
    report["news_ingestion_lag"] = {
        "mean": lag["mean_lag"] if lag else None,
        "max": lag["max_lag"] if lag else None,
    }
    report["position_snapshot_wallets"] = value(
        "SELECT COUNT(DISTINCT wallet) FROM position_snapshots"
    )
    report["taker_legs"] = value(
        "SELECT COUNT(*) FROM actor_trade_legs WHERE liquidity_role = 'taker'"
    )
    return report
