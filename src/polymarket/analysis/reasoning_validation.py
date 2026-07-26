"""Synthetic reasoning validation: train / calibrate / evaluate by world.

Splits are BY WORLD SEED.  The template model is trained on train worlds,
temperature-calibrated on validation worlds, and evaluated on test
worlds; thresholds are never tuned on the held-out test worlds.

Writes: reasoning_metrics.json, reasoning_confusion_matrix.csv,
reasoning_calibration.json, reasoning_failures.json,
reasoning_manifest.json, dr_validation_records.jsonl.

Run: python -m polymarket.analysis.reasoning_validation OUTPUT_DIR
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import tempfile
from typing import Any

import numpy as np

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import build_decision_episodes
from polymarket.analysis.drc import build_drc_record, persist_reasoning_records
from polymarket.analysis.features import compute_features
from polymarket.analysis.models import Fold, chronological_folds
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.reasoning import (
    ATTRIBUTION_FEATURES,
    DEFAULT_ATTRIBUTION_CONFIG,
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
    POSTERIOR_FEATURES,
    infer_posterior,
    posterior_features,
    train_template_model,
)
from polymarket.analysis.reasoning_targets import (
    build_at_risk_opportunities,
    occurrence_labels,
)
from polymarket.analysis.reasoning_templates import TEMPLATE_NAMES
from polymarket.analysis.versioning import version_block
from polymarket.contracts.schema import PARSER_VERSION
from polymarket.synthetic.reasoning_worlds import SETUP_LABEL, build_world

DEFAULT_TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_VAL_SEEDS = (6, 7)
DEFAULT_TEST_SEEDS = (8, 9)


# ---------------------------------------------------------------------------
def _world_dataset(seed: int, workdir: str) -> dict[str, Any]:
    """Build one world and compute strict per-decision rows (episodes,
    features, evidence, posterior base features).  Layer 1 attribution is
    attached separately by ``attach_cross_world_layer1`` so it can be
    trained on POOLED rows from OTHER worlds (worlds are independent
    universes; the split is by world seed)."""
    path = os.path.join(workdir, f"reasoning-world-{seed}.sqlite")
    conn, meta = build_world(seed, path)
    reader = SQLiteNormalizedReader(conn)
    episodes = build_decision_episodes(reader, end_time=meta["end_time"])
    labeled = [e for e in episodes if e.direction is not None]
    rows: list[dict[str, Any]] = []
    for episode in labeled:
        context = build_context(reader, episode)
        features = compute_features(context, episode)
        rows.append({
            "episode": episode,
            "features": features,
            "evidence": news_evidence(context),
            "label": 1.0 if episode.direction == "positive" else -1.0,
            "true_template": meta["labels"].get(
                (episode.actor_id, episode.anchor_time)
            ),
        })
    return {"seed": seed, "conn": conn, "reader": reader, "meta": meta,
            "rows": rows}


def attach_cross_world_layer1(
    train_worlds: list[dict[str, Any]],
    target_world: dict[str, Any],
    *,
    config=DEFAULT_ATTRIBUTION_CONFIG,
) -> None:
    """Attach Layer 1 attributions, counterfactuals and posterior
    features to a target world's rows, with models trained on the pooled
    rows of the OTHER worlds (leave-target-out when the target is a
    training world).  Stability is leave-one-world-out."""
    pool = [w for w in train_worlds if w["seed"] != target_world["seed"]]
    pooled_rows = [r for w in pool for r in w["rows"]]
    pooled_block = [w["seed"] for w in pool for _ in w["rows"]]
    all_rows = pooled_rows + target_world["rows"]
    block_ids = np.asarray(
        pooled_block + [target_world["seed"]] * len(target_world["rows"])
    )
    features = [r["features"] for r in all_rows]
    labels = [r["label"] for r in all_rows]
    times = [r["episode"].anchor_time for r in all_rows]
    ids = [r["episode"].decision_id for r in all_rows]
    n_pool = len(pooled_rows)
    fold = Fold(
        fold_index=0, train_end=float("nan"), eval_start=float("nan"),
        eval_end=float("nan"),
        train_indices=np.arange(n_pool),
        eval_indices=np.arange(n_pool, len(all_rows)),
    )
    attributions = run_driver_attribution(
        features, labels, times, ids,
        {r["episode"].decision_id: r["evidence"] for r in all_rows},
        reasoning_run_id=f"cross-world-{target_world['seed']}",
        config=config,
        coverage_by_decision={
            r["episode"].decision_id: r["episode"].coverage for r in all_rows
        },
        folds=[fold],
        stability_block_ids=block_ids,
    )
    by_id = {a.decision_id: a for a in attributions}
    # fixed pooled model for context counterfactuals (same training set)
    X = np.asarray(
        [[r["features"][n] for n in ATTRIBUTION_FEATURES]
         for r in pooled_rows], dtype=float,
    )
    y = np.asarray([r["label"] for r in pooled_rows])
    cf_model, _ = _fit_channel_models(X, y, np.arange(len(pooled_rows)), 1.0)
    for row in target_world["rows"]:
        decision_id = row["episode"].decision_id
        row["layer1"] = by_id.get(decision_id)
        row["cf"] = (
            run_counterfactuals(cf_model, row["features"], row["label"])
            if cf_model.fit_ok else None
        )
        row["posterior_x"] = posterior_features(
            row["features"], row["episode"],
            layer1=row["layer1"], evidence=row["evidence"],
        )


def _trainable(rows: list[dict]) -> tuple[list[dict], list[str]]:
    xs, ys = [], []
    for row in rows:
        label = row["true_template"]
        if label in (None, SETUP_LABEL):
            continue
        xs.append(row["posterior_x"])
        ys.append(label)
    return xs, ys


# ---------------------------------------------------------------------------
def run_reasoning_validation(
    output_dir: str,
    *,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    val_seeds: tuple[int, ...] = DEFAULT_VAL_SEEDS,
    test_seeds: tuple[int, ...] = DEFAULT_TEST_SEEDS,
    workdir: str | None = None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    workdir = workdir or tempfile.mkdtemp(prefix="reasoning-validation-")
    overlap = (set(train_seeds) | set(val_seeds)) & set(test_seeds)
    if overlap:
        raise ValueError(f"test worlds must be disjoint: {overlap}")

    train = [_world_dataset(seed, workdir) for seed in train_seeds]
    val = [_world_dataset(seed, workdir) for seed in val_seeds]
    test = [_world_dataset(seed, workdir) for seed in test_seeds]
    for world in train:
        attach_cross_world_layer1(train, world)   # leave-one-world-out
    for world in val + test:
        attach_cross_world_layer1(train, world)   # trained on train worlds

    train_x, train_y = [], []
    for world in train:
        xs, ys = _trainable(world["rows"])
        train_x += xs
        train_y += ys
    model = train_template_model(train_x, train_y)
    val_x, val_y = [], []
    for world in val:
        xs, ys = _trainable(world["rows"])
        val_x += xs
        val_y += ys
    model.calibrate(
        np.asarray([[x[n] for n in POSTERIOR_FEATURES] for x in val_x]),
        val_y,
    )

    versions = version_block(
        DEFAULT_ATTRIBUTION_CONFIG, DEFAULT_POSTERIOR_CONFIG,
        COUNTERFACTUAL_NAMES,
        {"posterior_l2": DEFAULT_POSTERIOR_CONFIG.l2,
         "layer1_l2": 1.0, "n_folds": 3},
    )

    # ---- evaluate on held-out test worlds (no tuning here) --------------
    records: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for world in test:
        world_records = []
        for row in world["rows"]:
            true_template = row["true_template"]
            if true_template in (None, SETUP_LABEL):
                continue
            layer1 = row["layer1"]
            posterior, posterior_status = infer_posterior(
                model, row["posterior_x"],
                layer1_status=layer1.status,
                coverage_complete=layer1.coverage_complete,
            )
            record = build_drc_record(
                episode=row["episode"], features=row["features"],
                evidence=row["evidence"], layer1=layer1,
                posterior=posterior, posterior_status=posterior_status,
                cf_result=row["cf"], versions=versions,
                reasoning_target="direction",
            )
            record["world_seed"] = world["seed"]
            record["true_template"] = true_template
            records.append(record)
            world_records.append(record)
            eval_rows.append({
                "record": record, "row": row,
                "true": true_template, "posterior": posterior,
            })
        persist_reasoning_records(
            world["conn"], world_records,
            reasoning_run_id=f"validation-{world['seed']}",
        )

    # ---- occurrence head on the first test world ------------------------
    occurrence_summary = _occurrence_summary(test[0], model, versions, records)

    metrics = _metrics(eval_rows)
    metrics["occurrence"] = occurrence_summary
    calibration = _calibration(eval_rows)
    failures = [
        {
            "decision_id": entry["record"]["decision_id"],
            "world_seed": entry["record"]["world_seed"],
            "true_template": entry["true"],
            "predicted": max(
                entry["posterior"].probabilities,
                key=entry["posterior"].probabilities.get,
            ),
            "status": entry["record"]["R"]["status"],
            "posterior": entry["posterior"].probabilities,
        }
        for entry in eval_rows
        if max(entry["posterior"].probabilities,
               key=entry["posterior"].probabilities.get) != entry["true"]
        or entry["record"]["R"]["status"] not in ("accepted", "ambiguous")
    ]
    manifest = {
        "parser_version_never_reused_as_feature_version": PARSER_VERSION,
        "versions": versions,
        "attribution_config": dataclasses.asdict(DEFAULT_ATTRIBUTION_CONFIG),
        "posterior_config": dataclasses.asdict(DEFAULT_POSTERIOR_CONFIG),
        "counterfactuals": list(COUNTERFACTUAL_NAMES),
        "splits": {
            "train_world_seeds": list(train_seeds),
            "validation_world_seeds": list(val_seeds),
            "test_world_seeds": list(test_seeds),
            "note": "split by world seed; thresholds never tuned on test",
        },
        "posterior_feature_names": POSTERIOR_FEATURES,
        "template_ontology": list(TEMPLATE_NAMES),
        "temperature": model.temperature,
        "calibration_version": model.calibration_version,
        "train_examples": len(train_y),
        "validation_examples": len(val_y),
        "test_examples": len(eval_rows),
    }

    _write_json(os.path.join(output_dir, "reasoning_metrics.json"), metrics)
    _write_json(
        os.path.join(output_dir, "reasoning_calibration.json"), calibration
    )
    _write_json(
        os.path.join(output_dir, "reasoning_failures.json"), failures
    )
    _write_json(
        os.path.join(output_dir, "reasoning_manifest.json"), manifest
    )
    _write_confusion_csv(
        os.path.join(output_dir, "reasoning_confusion_matrix.csv"), eval_rows
    )
    with open(
        os.path.join(output_dir, "dr_validation_records.jsonl"), "w"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"metrics": metrics, "manifest": manifest,
            "records": records, "model": model}


# ---------------------------------------------------------------------------
def _occurrence_summary(world, model, versions, records) -> dict[str, Any]:
    from polymarket.analysis.models import LogisticModel
    from polymarket.analysis.reasoning import ATTRIBUTION_FEATURES

    reader = world["reader"]
    opportunities = build_at_risk_opportunities(
        reader, end_time=world["meta"]["end_time"]
    )
    labels = occurrence_labels(opportunities)
    features = []
    for opportunity in opportunities:
        context = build_context(reader, opportunity.episode)
        features.append(compute_features(context, opportunity.episode))
    X = np.asarray(
        [[f[n] for n in ATTRIBUTION_FEATURES] for f in features], dtype=float
    )
    y = np.asarray(labels)
    times = np.asarray([o.interval_start for o in opportunities])
    correct = total = 0
    for fold in chronological_folds(times, n_folds=3):
        head = LogisticModel(feature_names=ATTRIBUTION_FEATURES, l2=1.0).fit(
            X[fold.train_indices], y[fold.train_indices]
        )
        if not head.fit_ok:
            continue
        probs = head.predict_proba(X[fold.eval_indices])
        predictions = np.where(probs >= 0.5, 1.0, -1.0)
        correct += int(np.sum(predictions == y[fold.eval_indices]))
        total += len(fold.eval_indices)
    # occurrence-target DRC records for traded intervals only
    occurrence_records = 0
    for opportunity, feature_row in zip(opportunities, features):
        if not opportunity.traded:
            continue
        x = posterior_features(feature_row, opportunity.episode)
        posterior, status = infer_posterior(
            model, x, layer1_status="ambiguous", coverage_complete=True
        )
        from polymarket.analysis.reasoning import DriverAttribution

        placeholder = DriverAttribution(
            decision_id=opportunity.episode.decision_id,
            reasoning_run_id="occurrence",
            observed_action_probability=float("nan"),
            logit_contributions={}, group_attributions={},
            top_evidence=[], primary_channel=None,
            status="ambiguous", confidence=0.0,
        )
        record = build_drc_record(
            episode=opportunity.episode, features=feature_row,
            evidence=[], layer1=placeholder, posterior=posterior,
            posterior_status=status, cf_result=None, versions=versions,
            reasoning_target="occurrence",
        )
        record["world_seed"] = world["seed"]
        records.append(record)
        occurrence_records += 1
    return {
        "at_risk_intervals": len(opportunities),
        "traded_intervals": int(sum(1 for o in opportunities if o.traded)),
        "no_trade_intervals": int(
            sum(1 for o in opportunities if not o.traded)
        ),
        "holdout_hit_rate": correct / total if total else None,
        "occurrence_drc_records": occurrence_records,
        "note": "no-trade intervals enter the occurrence head only; "
                "direction modelling requires clean single-direction "
                "taker trades",
    }


def _metrics(eval_rows: list[dict]) -> dict[str, Any]:
    per_template: dict[str, dict[str, int]] = {}
    statuses: dict[str, int] = {}
    cf_pass = cf_total = 0
    agreement_values = []
    false_accepts = mixed_total = 0
    by_seed: dict[int, dict[str, int]] = {}
    for entry in eval_rows:
        true = entry["true"]
        record = entry["record"]
        predicted = max(
            entry["posterior"].probabilities,
            key=entry["posterior"].probabilities.get,
        )
        stats = per_template.setdefault(
            true, {"tp": 0, "fn": 0, "fp": 0}
        )
        if predicted == true:
            stats["tp"] += 1
        else:
            stats["fn"] += 1
            per_template.setdefault(
                predicted, {"tp": 0, "fn": 0, "fp": 0}
            )["fp"] += 1
        status = record["R"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
        seed_stats = by_seed.setdefault(
            record["world_seed"], {"n": 0, "correct": 0}
        )
        seed_stats["n"] += 1
        seed_stats["correct"] += int(predicted == true)
        if record["R"]["primary_template"] is not None:
            cf_total += 1
            if not record["R"]["counterfactual_failures"]:
                cf_pass += 1
            agreement_values.append(record["R"]["agreement_score"])
        if true == "MIXED_OR_UNRESOLVED":
            mixed_total += 1
            if (
                record["R"]["status"] == "accepted"
                and record["R"]["primary_template"] not in (
                    None, "MIXED_OR_UNRESOLVED"
                )
            ):
                false_accepts += 1
    f1_values = []
    recall = {}
    for template, stats in per_template.items():
        tp, fn, fp = stats["tp"], stats["fn"], stats["fp"]
        template_recall = tp / (tp + fn) if (tp + fn) else None
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if template_recall is not None:
            recall[template] = template_recall
            denominator = precision + template_recall
            f1_values.append(
                2 * precision * template_recall / denominator
                if denominator else 0.0
            )
    total = len(eval_rows)
    return {
        "n_test_decisions": total,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "per_template_recall": recall,
        "status_counts": statuses,
        "ambiguous_rate": statuses.get("ambiguous", 0) / total if total else None,
        "false_accept_rate_mixed": (
            false_accepts / mixed_total if mixed_total else None
        ),
        "counterfactual_pass_rate": cf_pass / cf_total if cf_total else None,
        "mean_agreement_score": (
            float(np.mean(agreement_values)) if agreement_values else None
        ),
        "coverage_exclusions": statuses.get("insufficient_context", 0),
        "accuracy_by_world_seed": {
            str(seed): stats["correct"] / stats["n"]
            for seed, stats in by_seed.items()
        },
    }


def _calibration(eval_rows: list[dict]) -> dict[str, Any]:
    bins = [[] for _ in range(5)]
    for entry in eval_rows:
        probabilities = entry["posterior"].probabilities
        predicted = max(probabilities, key=probabilities.get)
        top_p = probabilities[predicted]
        index = min(4, int(top_p * 5))
        bins[index].append((top_p, float(predicted == entry["true"])))
    bin_reports = []
    weighted_error = 0.0
    total = sum(len(b) for b in bins)
    for i, bucket in enumerate(bins):
        if not bucket:
            bin_reports.append({"bin": i, "n": 0})
            continue
        confidence = float(np.mean([c for c, _ in bucket]))
        accuracy = float(np.mean([a for _, a in bucket]))
        weighted_error += abs(confidence - accuracy) * len(bucket) / total
        bin_reports.append({
            "bin": i, "n": len(bucket),
            "mean_top_probability": confidence,
            "top1_accuracy": accuracy,
        })
    return {
        "top1_calibration_error": weighted_error,
        "bins": bin_reports,
        "note": "temperature fitted on validation worlds only",
    }


def _write_confusion_csv(path: str, eval_rows: list[dict]) -> None:
    matrix: dict[tuple[str, str], int] = {}
    for entry in eval_rows:
        predicted = max(
            entry["posterior"].probabilities,
            key=entry["posterior"].probabilities.get,
        )
        key = (entry["true"], predicted)
        matrix[key] = matrix.get(key, 0) + 1
    names = [n for n in TEMPLATE_NAMES]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *names])
        for true in names:
            writer.writerow(
                [true, *[matrix.get((true, predicted), 0)
                         for predicted in names]]
            )


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "reasoning_validation"
    result = run_reasoning_validation(target_dir)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
