"""One-command reasoning pipeline — collection health through the
acceptance-gate report.

Every stage runs in order, tolerates the refusals the pipeline is
built to emit, and records its status; a refusal in one stage never
crashes the stages that don't depend on it.  The output is a single
``acceptance_report.json`` stating, per gate, exactly what is
established, pending, or refused — the report an advisor reads.

Gates (from the research plan):

1. **Predictive value** — the latent reasoning model beats the C-only
   baseline on held-out actors and periods (from
   ``train-latent-reasoning``); refused honestly on thin samples.
2. **Human agreement** — model primary templates vs unanimous human
   gold labels (pending until an annotation batch is imported).
3. **Stability** — latent cluster stability across seeds.
4. **Honest abstention** — the DRC status distribution
   (accepted / ambiguous / insufficient_context / counterfactual
   failures) with the abstention-vs-coverage relationship, so
   abstention is shown to track missing context rather than being a
   constant shrug.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any

from polymarket.analysis.versioning import feature_version_hash


def _stage(report: dict, name: str, status: str, **detail: Any) -> None:
    report["stages"][name] = {"status": status, **detail}


def run_reasoning_pipeline(
    db_path: str,
    output_dir: str,
    *,
    reasoning_model_path: str | None = None,
    end_time: float | None = None,
    placebo_seeds: int = 5,
    latent_dim: int = 8,
    annotation_batch_id: str | None = None,
    skip: set[str] | None = None,
) -> dict:
    from polymarket.contracts.schema import connect, ensure_paper_schema

    skip = skip or set()
    os.makedirs(output_dir, exist_ok=True)
    conn = connect(db_path)
    conn.row_factory = sqlite3.Row
    end = end_time or time.time()
    report: dict = {
        "pipeline_started_at": time.time(),
        "db": db_path,
        "feature_version": feature_version_hash(),
        "stages": {},
        "gates": {},
        "data_readiness": {},
    }

    # ---- 1. schema -------------------------------------------------------
    applied = ensure_paper_schema(conn)
    _stage(report, "migrate", "ok", applied=applied)

    # ---- 2. normalize ----------------------------------------------------
    if "normalize" in skip:
        _stage(report, "normalize", "skipped")
    else:
        from polymarket.normalization.normalizer import Normalizer
        from polymarket.normalization.reconciliation import (
            reconcile_roles,
        )

        results = Normalizer(conn).normalize_all()
        diagnostics = reconcile_roles(conn)
        _stage(
            report, "normalize", "ok",
            raw_responses=len(results),
            unknown_roles=diagnostics.unknown_remaining,
        )

    # ---- 3. liquidity bars (300s + 900s) ---------------------------------
    from polymarket.analysis.liquidity_bars import (
        LiquidityBarConfig,
        build_liquidity_bars,
    )

    conditions = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT condition_id FROM markets"
        )
    ]
    bar_detail = {}
    for bin_seconds in (300.0, 900.0):
        written = 0
        for condition_id in conditions:
            last = conn.execute(
                "SELECT MAX(bin_start) FROM liquidity_bars "
                "WHERE condition_id = ? AND bin_seconds = ?",
                (condition_id, bin_seconds),
            ).fetchone()[0]
            written += build_liquidity_bars(
                conn, condition_id, start=last,
                config=LiquidityBarConfig(bin_seconds=bin_seconds),
            )
        complete = conn.execute(
            "SELECT COUNT(*) FROM liquidity_bars "
            "WHERE bin_seconds = ? AND coverage_complete = 1",
            (bin_seconds,),
        ).fetchone()[0]
        bar_detail[f"{int(bin_seconds)}s"] = {
            "written": written, "complete_total": complete,
        }
    _stage(report, "liquidity_bars", "ok", **bar_detail)
    report["data_readiness"]["complete_bars"] = bar_detail

    # ---- 4. jump model ---------------------------------------------------
    mode_run_id = None
    if "modes" in skip:
        _stage(report, "liquidity_modes", "skipped")
    else:
        from polymarket.analysis.liquidity_modes import (
            fit_jump_model,
            persist_jump_model,
        )

        try:
            model = fit_jump_model(conn, fit_cutoff=end)
            persist_jump_model(conn, model, end)
            mode_run_id = model.mode_run_id
            _stage(
                report, "liquidity_modes", "ok",
                mode_run_id=mode_run_id,
                lambda_penalty=model.lambda_penalty,
                assigned_bars=len(model.assignments),
            )
        except ValueError as exc:
            _stage(report, "liquidity_modes", "refused",
                   reason=str(exc))

    # ---- 5. impact screens -----------------------------------------------
    if mode_run_id is None:
        _stage(report, "impact_screens", "skipped",
               reason="no fitted mode run")
    else:
        from polymarket.analysis.news_impact_screen import (
            screen_news_impact,
        )

        counters = {}
        for basis in ("online_filtered", "retrospective_smoothed"):
            counters[basis] = screen_news_impact(
                conn, mode_run_id, assignment_basis=basis
            )
        _stage(report, "impact_screens", "ok", **{
            basis: {k: v for k, v in c.items()
                    if isinstance(v, int)}
            for basis, c in counters.items()
        })

    # ---- 6. replay + reasoning + DRC + outcomes --------------------------
    if "analysis" in skip:
        _stage(report, "run_analysis", "skipped")
    else:
        from polymarket.analysis.reader import SQLiteNormalizedReader
        from polymarket.analysis.replay import run_replay

        reasoning_model = None
        if reasoning_model_path:
            from polymarket.analysis.reasoning_artifact import (
                ArtifactVersionMismatch,
                load_reasoning_model,
            )

            try:
                reasoning_model, _artifact = load_reasoning_model(
                    reasoning_model_path
                )
            except (FileNotFoundError, ArtifactVersionMismatch) as exc:
                _stage(report, "run_analysis", "refused",
                       reason=f"reasoning artifact: {exc}")
        if reasoning_model is not None or not reasoning_model_path:
            reader = SQLiteNormalizedReader(conn)
            run = run_replay(
                reader, end_time=end, mode_run_id=mode_run_id,
                reasoning_model=reasoning_model,
                reasoning_target="both" if reasoning_model else
                "direction",
            )
            if run.evaluation is None:
                _stage(report, "run_analysis", "refused",
                       reason="; ".join(run.notes)
                       or "insufficient labeled decisions")
            else:
                from polymarket.analysis.reporting import (
                    write_run_outputs,
                )

                paths = write_run_outputs(run, output_dir)
                drc_statuses: dict[str, int] = {}
                if reasoning_model is not None:
                    from polymarket.analysis.reasoning_reconstruction import write_reasoning_outputs

                    paths.update(write_reasoning_outputs(
                        run, output_dir, outcomes_conn=conn,
                    ))
                    for record in run.drc_records:
                        status = record["R"]["status"]
                        drc_statuses[status] = (
                            drc_statuses.get(status, 0) + 1
                        )
                _stage(
                    report, "run_analysis", "ok",
                    labeled_decisions=len(run.labeled_episodes),
                    metrics={
                        name: metrics.get("log_loss")
                        for name, metrics in
                        run.evaluation.metrics.items()
                    },
                    drc_status_counts=drc_statuses,
                )
                report["data_readiness"]["decisions"] = len(
                    run.labeled_episodes
                )
                report["_drc_records"] = run.drc_records

    # ---- 7. underreaction analysis ---------------------------------------
    if "underreaction" in skip:
        _stage(report, "underreaction", "skipped")
    else:
        from polymarket.analysis.news_returns import (
            DecompositionConfig,
            build_interval_records,
        )
        from polymarket.analysis.underreaction import (
            run_drift_regressions,
        )

        config = DecompositionConfig(mode_run_id=mode_run_id)
        records = build_interval_records(conn, config)
        regressions = run_drift_regressions(
            conn, config, records=records
        )
        if not regressions:
            _stage(report, "underreaction", "refused",
                   reason=f"{len(records)} intervals: too few "
                          "censoring-admissible observations")
        else:
            _stage(report, "underreaction", "ok", horizons=[
                {
                    "h": r.horizon_seconds,
                    "beta_news": r.beta_news,
                    "note": r.inference_note,
                }
                for r in regressions
            ])

    # ---- 8. latent reasoning (gates 1 + 3) -------------------------------
    from polymarket.analysis.latent_reasoning import (
        assemble_decision_matrix,
        evaluate_latent_reasoning,
    )

    data = assemble_decision_matrix(
        conn, end_time=end, mode_run_id=mode_run_id
    )
    latent = evaluate_latent_reasoning(
        data, latent_dim=latent_dim
    )
    _stage(report, "latent_reasoning", latent["status"],
           **({"note": latent["note"]}
              if latent["status"] == "refused" else {}))
    report["data_readiness"]["actors"] = len(
        set(data["actors"].tolist())
    )
    if latent["status"] == "evaluated":
        report["gates"]["gate_1_predictive_value"] = {
            "status": "evaluated",
            "passed": latent["gate_1_predictive_value"],
            "splits": latent["splits"],
        }
        stabilities = [
            block["cluster_stability"]
            for block in latent["splits"].values()
            if block.get("status") == "evaluated"
        ]
        report["gates"]["gate_3_stability"] = {
            "status": "evaluated",
            "min_cluster_stability": min(stabilities)
            if stabilities else None,
        }
    else:
        report["gates"]["gate_1_predictive_value"] = {
            "status": "refused", "reason": latent["note"],
        }
        report["gates"]["gate_3_stability"] = {
            "status": "refused", "reason": latent["note"],
        }

    # ---- 9. human agreement (gate 2) -------------------------------------
    from polymarket.analysis.annotation import gold_labels

    batch_id = annotation_batch_id
    if batch_id is None:
        row = conn.execute(
            "SELECT batch_id FROM annotation_batches "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        batch_id = row[0] if row else None
    gold = gold_labels(conn, batch_id) if batch_id else {}
    if not gold:
        report["gates"]["gate_2_human_agreement"] = {
            "status": "pending",
            "reason": "no imported annotation consensus yet "
                      "(export-annotation-batch -> label -> "
                      "import-annotations)",
        }
    else:
        drc_records = report.pop("_drc_records", [])
        by_decision = {
            record["D"].get("decision_id"): record["R"].get(
                "primary_template"
            )
            for record in drc_records
        }
        shared = [
            (gold[d], by_decision[d]) for d in gold
            if by_decision.get(d) is not None
        ]
        if shared:
            agreement = sum(
                1 for human, model in shared if human == model
            ) / len(shared)
            report["gates"]["gate_2_human_agreement"] = {
                "status": "evaluated",
                "n_shared": len(shared),
                "agreement_with_consensus": agreement,
            }
        else:
            report["gates"]["gate_2_human_agreement"] = {
                "status": "pending",
                "reason": f"{len(gold)} gold labels but no "
                          "overlapping accepted model outputs",
            }
    report.pop("_drc_records", None)

    # ---- 10. honest abstention (gate 4) ----------------------------------
    analysis_stage = report["stages"].get("run_analysis", {})
    status_counts = analysis_stage.get("drc_status_counts") or {}
    total = sum(status_counts.values())
    if total:
        abstained = sum(
            n for status, n in status_counts.items()
            if status != "accepted"
        )
        report["gates"]["gate_4_honest_abstention"] = {
            "status": "evaluated",
            "status_counts": status_counts,
            "abstention_rate": abstained / total,
            "note": (
                "abstention should FALL as coverage grows; compare "
                "across pipeline runs over time"
            ),
        }
    else:
        report["gates"]["gate_4_honest_abstention"] = {
            "status": "pending",
            "reason": "no DRC records this run",
        }

    # ---- write -----------------------------------------------------------
    report["pipeline_finished_at"] = time.time()
    path = os.path.join(output_dir, "acceptance_report.json")
    with open(path, "w") as handle:
        json.dump(_clean(report), handle, indent=2, sort_keys=True)
    report["report_path"] = path
    conn.close()
    return report


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
