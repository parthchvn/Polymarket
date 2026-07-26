"""End-to-end replay: decisions -> strict contexts -> features -> nested
models -> placebos -> uncertainty -> report files."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import DecisionEpisode, build_decision_episodes
from polymarket.analysis.features import (
    NEWS_DECAY_MAX_AGE,
    compute_features,
    feature_manifest,
    news_decay_config,
)
from polymarket.analysis.models import EvaluationResult, evaluate_nested_models
from polymarket.analysis.news_attribution import attribute_decision
from polymarket.analysis.placebos import PlaceboSuiteResult, run_placebo_suite
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.reasoning import (
    DriverAttribution,
    news_evidence,
    run_driver_attribution,
)
from polymarket.analysis.uncertainty import (
    BootstrapCI,
    cluster_bootstrap,
    moving_block_bootstrap,
)


@dataclass
class ReplayRun:
    run_id: str
    config: dict
    episodes: list[DecisionEpisode] = field(default_factory=list)
    labeled_episodes: list[DecisionEpisode] = field(default_factory=list)
    feature_rows: list[dict] = field(default_factory=list)
    attributions: list[dict] = field(default_factory=list)
    driver_attributions: list[DriverAttribution] = field(default_factory=list)
    evidence_by_decision: dict[str, list[dict]] = field(default_factory=dict)
    evaluation: EvaluationResult | None = None
    placebos: PlaceboSuiteResult | None = None
    bootstraps: list[BootstrapCI] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_replay(
    reader: SQLiteNormalizedReader,
    *,
    end_time: float | None = None,
    interval_seconds: float = 3600.0,
    mixed_threshold: float = 0.2,
    n_folds: int = 3,
    embargo_seconds: float = 0.0,
    seed: int = 1337,
    run_id: str | None = None,
    news_lookback: float = 86400.0,
    news_decay_half_lives: dict[str, float] | None = None,
    news_decay_max_age: float = NEWS_DECAY_MAX_AGE,
) -> ReplayRun:
    end_time = end_time or time.time()
    run = ReplayRun(
        run_id=run_id or f"run-{int(time.time())}",
        config={
            "end_time": end_time,
            "interval_seconds": interval_seconds,
            "mixed_threshold": mixed_threshold,
            "n_folds": n_folds,
            "embargo_seconds": embargo_seconds,
            "seed": seed,
            **news_decay_config(
                news_lookback=news_lookback,
                news_decay_half_lives=news_decay_half_lives,
                news_decay_max_age=news_decay_max_age,
            ),
        },
    )
    run.episodes = build_decision_episodes(
        reader,
        end_time=end_time,
        interval_seconds=interval_seconds,
        mixed_threshold=mixed_threshold,
    )
    labels: list[float] = []
    times: list[float] = []
    ids: list[str] = []
    markets: list[str] = []
    actors: list[str] = []
    for episode in run.episodes:
        if episode.direction is None:
            # mixed / empty activity has no directional label; excluded
            # from the direction model but retained in the episode list.
            continue
        context = build_context(reader, episode)
        features = compute_features(
            context,
            episode,
            news_lookback=news_lookback,
            news_decay_half_lives=news_decay_half_lives,
            news_decay_max_age=news_decay_max_age,
        )
        run.labeled_episodes.append(episode)
        run.feature_rows.append(features)
        labels.append(1.0 if episode.direction == "positive" else -1.0)
        times.append(episode.anchor_time)
        ids.append(episode.decision_id)
        markets.append(episode.market_id or episode.condition_id)
        actors.append(episode.actor_id)
        run.evidence_by_decision[episode.decision_id] = news_evidence(context)
        attribution = attribute_decision(context, episode)
        run.attributions.append(
            {
                "decision_id": episode.decision_id,
                "label": attribution.label,
                "top_event_family": (
                    attribution.top_candidate.event_family_id
                    if attribution.top_candidate else None
                ),
                "attribution_score": (
                    attribution.top_candidate.attribution_score
                    if attribution.top_candidate else None
                ),
                "confidence": attribution.confidence,
                "alternatives": len(attribution.alternatives),
                "notes": attribution.notes,
            }
        )
    if len(run.feature_rows) < 4:
        run.notes.append(
            "too few labeled decisions for chronological evaluation"
        )
        return run
    run.evaluation = evaluate_nested_models(
        run.feature_rows, labels, times, ids,
        n_folds=n_folds, embargo_seconds=embargo_seconds,
    )
    run.placebos = run_placebo_suite(
        run.feature_rows, labels, times, ids, markets, actors,
        baseline=run.evaluation, seed=seed,
        n_folds=n_folds, embargo_seconds=embargo_seconds,
    )
    run.driver_attributions = run_driver_attribution(
        run.feature_rows, labels, times, ids, run.evidence_by_decision,
        reasoning_run_id=run.run_id,
        n_folds=n_folds, embargo_seconds=embargo_seconds,
    )
    actor_map = {i: a for i, a in zip(ids, actors)}
    market_map = {i: m for i, m in zip(ids, markets)}
    run.bootstraps = [
        cluster_bootstrap(run.evaluation.per_decision, "actor", actor_map, seed=seed),
        cluster_bootstrap(run.evaluation.per_decision, "market", market_map, seed=seed),
        moving_block_bootstrap(run.evaluation.per_decision, seed=seed),
    ]
    return run


def manifest() -> dict:
    return feature_manifest()
