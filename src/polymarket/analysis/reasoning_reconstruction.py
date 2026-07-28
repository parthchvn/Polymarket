"""Reasoning reconstruction over a real replay run.

Consumes Layer-1 driver attributions from ``run_replay`` and produces
the full (D, C, R) surface: template posteriors, fixed-model context
counterfactuals, structured DRC records with the ACTUAL as-of contract
version and market status, deterministic rationales, and (optionally)
the occurrence target over the at-risk interval grid.  Ambiguous and
unresolved outputs are preserved, never dropped.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import DecisionEpisode
from polymarket.analysis.drc import build_drc_record
from polymarket.analysis.features import compute_features
from polymarket.analysis.models import chronological_folds
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.reasoning import (
    ATTRIBUTION_FEATURES,
    DEFAULT_ATTRIBUTION_CONFIG,
    AttributionConfig,
    _fit_channel_models,
    news_evidence,
    run_driver_attribution,
)
from polymarket.analysis.reasoning_counterfactuals import (
    COUNTERFACTUAL_NAMES,
    run_counterfactuals,
)
from polymarket.analysis.reasoning_posterior import (
    DEFAULT_POSTERIOR_CONFIG,
    PosteriorConfig,
    TemplateModel,
    infer_posterior,
    posterior_features,
)
from polymarket.analysis.reasoning_targets import (
    build_at_risk_opportunities,
    occurrence_labels,
)
from polymarket.analysis.versioning import version_block

if TYPE_CHECKING:  # pragma: no cover
    from polymarket.analysis.replay import ReplayRun

REASONING_TARGETS = ("direction", "occurrence", "both")


def reasoning_versions(
    attribution_config: AttributionConfig = DEFAULT_ATTRIBUTION_CONFIG,
    posterior_config: PosteriorConfig = DEFAULT_POSTERIOR_CONFIG,
) -> dict[str, str]:
    return version_block(
        attribution_config, posterior_config, COUNTERFACTUAL_NAMES,
        {"posterior_l2": posterior_config.l2, "layer1_l2": 1.0},
    )


def _market_status_context(
    reader: SQLiteNormalizedReader, episode: DecisionEpisode
) -> tuple[int | None, dict[str, Any]]:
    """The ACTUAL as-of decision environment: contract version, market
    open state, coverage certification, blocking gaps and the outcome-
    token mapping in force."""
    t = episode.anchor_time
    contract_version: int | None = None
    status: dict[str, Any] = {"as_of": t}
    if episode.market_id:
        contract = reader.contract_asof(episode.market_id, t)
        if contract is not None:
            contract_version = int(contract["version_seq"])
        row = reader.market_status_asof(episode.market_id, t)
        if row is not None:
            status.update(
                trading_enabled=bool(row["trading_enabled"]),
                closed=bool(row["closed"]),
                resolved=bool(row["resolved"]),
                status_effective_from=row["effective_from"],
            )
    gaps = reader.blocking_gaps(
        episode.condition_id, episode.interval_start, episode.interval_end
    )
    status["blocking_gap_count"] = len(gaps)
    status["coverage_certified"] = not gaps
    tokens = reader.outcome_tokens_asof(episode.condition_id, t)
    status["outcome_token_mapping"] = [
        {
            "asset": row["asset"],
            "outcome_sign": row["outcome_sign"],
            "mapping_effective_from": row["mapping_effective_from"],
        }
        for row in tokens
    ]
    return contract_version, status


def _fold_full_models(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    times: list[float],
    *,
    n_folds: int,
    embargo_seconds: float,
    l2: float = 1.0,
) -> dict[int, Any]:
    """Deterministically refit the per-fold FULL models (identical folds
    to Layer 1) so counterfactuals score under a fixed model whose
    training strictly precedes each evaluated decision."""
    X = np.asarray(
        [[row[n] for n in ATTRIBUTION_FEATURES] for row in feature_rows],
        dtype=float,
    )
    y = np.asarray(labels, dtype=float)
    models: dict[int, Any] = {}
    for fold in chronological_folds(
        np.asarray(times), n_folds=n_folds, embargo_seconds=embargo_seconds
    ):
        model, _ = _fit_channel_models(X, y, fold.train_indices, l2)
        models[fold.fold_index] = model if model.fit_ok else None
    return models


def run_reasoning_reconstruction(
    run: "ReplayRun",
    reader: SQLiteNormalizedReader,
    model: TemplateModel,
    *,
    reasoning_target: str = "direction",
    attribution_config: AttributionConfig = DEFAULT_ATTRIBUTION_CONFIG,
    posterior_config: PosteriorConfig = DEFAULT_POSTERIOR_CONFIG,
    occurrence_interval_seconds: float = 6 * 3600.0,
) -> None:
    """Populate run.template_posteriors / counterfactual_results /
    drc_records (direction) and the occurrence surfaces, in place."""
    if reasoning_target not in REASONING_TARGETS:
        raise ValueError(
            f"reasoning_target must be one of {REASONING_TARGETS}"
        )
    versions = reasoning_versions(attribution_config, posterior_config)
    run.config["reasoning"] = {
        "target": reasoning_target,
        "versions": versions,
        "posterior_config": dataclasses.asdict(posterior_config),
        "calibration_version": model.calibration_version,
    }

    if reasoning_target in ("direction", "both"):
        _direction_reconstruction(
            run, reader, model, versions,
            posterior_config=posterior_config,
        )
    if reasoning_target in ("occurrence", "both"):
        _occurrence_reconstruction(
            run, reader, model, versions,
            attribution_config=attribution_config,
            posterior_config=posterior_config,
            interval_seconds=occurrence_interval_seconds,
        )


def _direction_reconstruction(
    run: "ReplayRun",
    reader: SQLiteNormalizedReader,
    model: TemplateModel,
    versions: dict[str, str],
    *,
    posterior_config: PosteriorConfig,
) -> None:
    labels = [
        1.0 if episode.direction == "positive" else -1.0
        for episode in run.labeled_episodes
    ]
    times = [episode.anchor_time for episode in run.labeled_episodes]
    fold_models = _fold_full_models(
        run.feature_rows, labels, times,
        n_folds=run.config.get("n_folds", 3),
        embargo_seconds=run.config.get("embargo_seconds", 0.0),
    )
    by_id = {a.decision_id: a for a in run.driver_attributions}
    for episode, features, label in zip(
        run.labeled_episodes, run.feature_rows, labels
    ):
        layer1 = by_id.get(episode.decision_id)
        if layer1 is None:
            continue
        evidence = run.evidence_by_decision.get(episode.decision_id, [])
        x = posterior_features(features, episode, layer1=layer1,
                               evidence=evidence)
        posterior, posterior_status = infer_posterior(
            model, x,
            layer1_status=layer1.status,
            coverage_complete=layer1.coverage_complete,
            config=posterior_config,
        )
        fold_model = (
            fold_models.get(layer1.fold_index)
            if layer1.fold_index is not None else None
        )
        cf_result = (
            run_counterfactuals(fold_model, features, label)
            if fold_model is not None else None
        )
        contract_version, market_status = _market_status_context(
            reader, episode
        )
        record = build_drc_record(
            episode=episode, features=features, evidence=evidence,
            layer1=layer1, posterior=posterior,
            posterior_status=posterior_status, cf_result=cf_result,
            versions=versions, reasoning_target="direction",
            contract_version=contract_version, market_status=market_status,
        )
        run.template_posteriors.append(posterior)
        run.counterfactual_results.append(cf_result)
        run.drc_records.append(record)


def _occurrence_reconstruction(
    run: "ReplayRun",
    reader: SQLiteNormalizedReader,
    model: TemplateModel,
    versions: dict[str, str],
    *,
    attribution_config: AttributionConfig,
    posterior_config: PosteriorConfig,
    interval_seconds: float,
) -> None:
    end_time = run.config.get("end_time")
    if end_time is None:
        run.notes.append(
            "occurrence reasoning skipped: replay end_time unavailable"
        )
        return
    opportunities = build_at_risk_opportunities(
        reader, end_time=end_time, interval_seconds=interval_seconds
    )
    run.occurrence_opportunities = opportunities
    if len(opportunities) < attribution_config.min_train_rows:
        run.notes.append(
            f"occurrence reasoning: only {len(opportunities)} at-risk "
            "intervals; attributions will be insufficient_context"
        )
    features = []
    evidence_by_id: dict[str, list[dict]] = {}
    for opportunity in opportunities:
        context = build_context(
            reader, opportunity.episode,
            relevance_availability="retrospective_source",
            mode_run_id=run.config.get("mode_run_id"),
        )
        features.append(compute_features(context, opportunity.episode))
        evidence_by_id[opportunity.episode.decision_id] = news_evidence(
            context
        )
    labels = occurrence_labels(opportunities)
    times = [o.interval_start for o in opportunities]
    ids = [o.episode.decision_id for o in opportunities]
    if not opportunities:
        return
    attributions = run_driver_attribution(
        features, labels, times, ids, evidence_by_id,
        reasoning_run_id=run.run_id, config=attribution_config,
        coverage_by_decision={
            o.episode.decision_id: o.episode.coverage for o in opportunities
        },
    )
    run.occurrence_attributions = attributions
    by_id = {a.decision_id: a for a in attributions}
    fold_models = _fold_full_models(features, labels, times, n_folds=3,
                                    embargo_seconds=0.0)
    for opportunity, feature_row, label in zip(
        opportunities, features, labels
    ):
        if not opportunity.traded:
            continue  # reasoning explains observed trades; the no-trade
            # intervals train/evaluate the occurrence head itself
        layer1 = by_id.get(opportunity.episode.decision_id)
        if layer1 is None:
            continue
        evidence = evidence_by_id.get(opportunity.episode.decision_id, [])
        x = posterior_features(
            feature_row, opportunity.episode, layer1=layer1,
            evidence=evidence,
        )
        posterior, posterior_status = infer_posterior(
            model, x,
            layer1_status=layer1.status,
            coverage_complete=layer1.coverage_complete,
            config=posterior_config,
        )
        fold_model = (
            fold_models.get(layer1.fold_index)
            if layer1.fold_index is not None else None
        )
        cf_result = (
            run_counterfactuals(fold_model, feature_row, label)
            if fold_model is not None else None
        )
        contract_version, market_status = _market_status_context(
            reader, opportunity.episode
        )
        record = build_drc_record(
            episode=opportunity.episode, features=feature_row,
            evidence=evidence, layer1=layer1, posterior=posterior,
            posterior_status=posterior_status, cf_result=cf_result,
            versions=versions, reasoning_target="occurrence",
            contract_version=contract_version, market_status=market_status,
        )
        run.occurrence_drc_records.append(record)


# ---------------------------------------------------------------------------
def write_reasoning_outputs(
    run: "ReplayRun", output_dir: str,
    *, outcomes_conn=None, outcome_bin_seconds: float = 900.0,
) -> dict[str, str]:
    """drc_records.jsonl, occurrence_drc_records.jsonl and a summary.

    When ``outcomes_conn`` is supplied, the ex-post outcome layer O is
    attached as a FINAL pass — features, attributions, posteriors and
    counterfactuals are already frozen inside the records, so realised
    post-decision drift can never flow back into P(R|D,C)."""
    if outcomes_conn is not None:
        from polymarket.analysis.outcomes import attach_outcomes

        attach_outcomes(
            run.drc_records, outcomes_conn,
            bin_seconds=outcome_bin_seconds,
        )
        attach_outcomes(
            run.occurrence_drc_records, outcomes_conn,
            bin_seconds=outcome_bin_seconds,
        )
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    def write_jsonl(name: str, records: list[dict]) -> None:
        path = os.path.join(output_dir, name)
        with open(path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        paths[name.removesuffix(".jsonl")] = path

    write_jsonl("drc_records.jsonl", run.drc_records)
    write_jsonl("occurrence_drc_records.jsonl", run.occurrence_drc_records)

    def summarize(records: list[dict]) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        templates: dict[str, int] = {}
        for record in records:
            reasoning = record["R"]
            statuses[reasoning["status"]] = (
                statuses.get(reasoning["status"], 0) + 1
            )
            primary = reasoning["primary_template"]
            if primary:
                templates[primary] = templates.get(primary, 0) + 1
        return {
            "n_records": len(records),
            "status_counts": statuses,
            "primary_template_counts": templates,
        }

    summary = {
        "run_id": run.run_id,
        "reasoning": run.config.get("reasoning", {}),
        "direction": summarize(run.drc_records),
        "occurrence": summarize(run.occurrence_drc_records),
        "at_risk_intervals": len(run.occurrence_opportunities),
        "traded_intervals": sum(
            1 for o in run.occurrence_opportunities if o.traded
        ),
    }
    summary_path = os.path.join(output_dir, "reasoning_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    paths["reasoning_summary"] = summary_path
    return paths
