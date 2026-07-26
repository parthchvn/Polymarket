"""Template recovery on held-out worlds, counterfactual validity, and
posterior / persistence / rationale invariants."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from polymarket.analysis.drc import build_drc_record, persist_reasoning_records
from polymarket.analysis.rationale import render_rationale
from polymarket.analysis.reasoning import ATTRIBUTION_FEATURES, DEFAULT_ATTRIBUTION_CONFIG
from polymarket.analysis.reasoning_counterfactuals import (
    COUNTERFACTUAL_NAMES,
    run_counterfactuals,
)
from polymarket.analysis.reasoning_posterior import (
    POSTERIOR_FEATURES,
    PosteriorConfig,
    infer_posterior,
    train_template_model,
)
from polymarket.analysis.reasoning_templates import TEMPLATES
from polymarket.analysis.reasoning_validation import (
    _fit_channel_models,
    _trainable,
    run_reasoning_validation,
)
from polymarket.analysis.versioning import (
    reasoning_method_version_hash,
    template_ontology_version_hash,
    version_block,
)
from polymarket.contracts.schema import PARSER_VERSION

PURE_ARCHETYPES = {
    "FRESH_NEWS_RESPONSE",
    "PERSISTENT_NEWS_ADJUSTMENT",
    "MARKET_MOMENTUM",
    "CONTRARIAN_REVERSAL",
    "INVENTORY_REBALANCING",
    "POSITION_BUILDING",
    "LIQUIDITY_TIMING",
    "ACTOR_PRIOR",
}


def _test_predictions(reasoning_worlds, trained_template_model):
    model = trained_template_model
    predictions = []
    for row in reasoning_worlds["test"]["rows"]:
        truth = row["true_template"]
        if truth in (None, "SETUP"):
            continue
        x = np.asarray(
            [[row["posterior_x"][n] for n in POSTERIOR_FEATURES]]
        )
        probs = model.predict_proba(x)[0]
        predictions.append(
            (truth, model.classes[int(np.argmax(probs))], row)
        )
    return predictions


# ---------------------------------------------------------------------------
# template recovery on the held-out world
# ---------------------------------------------------------------------------
def test_each_pure_archetype_recovers_its_template(
    reasoning_worlds, trained_template_model
):
    predictions = _test_predictions(reasoning_worlds, trained_template_model)
    for archetype in PURE_ARCHETYPES:
        relevant = [p for p in predictions if p[0] == archetype]
        assert relevant, f"no test decisions for {archetype}"
        hits = sum(1 for truth, predicted, _ in relevant if predicted == truth)
        assert hits / len(relevant) >= 0.5, (
            f"{archetype}: {hits}/{len(relevant)} recovered"
        )


def test_overall_recovery_beats_chance_by_a_wide_margin(
    reasoning_worlds, trained_template_model
):
    predictions = _test_predictions(reasoning_worlds, trained_template_model)
    hits = sum(1 for truth, predicted, _ in predictions if predicted == truth)
    assert hits / len(predictions) >= 0.6  # chance is ~1/9


def test_mixed_actor_remains_ambiguous(
    reasoning_worlds, trained_template_model
):
    model = trained_template_model
    mixed_rows = [
        row for row in reasoning_worlds["test"]["rows"]
        if row["true_template"] == "MIXED_OR_UNRESOLVED"
    ]
    assert mixed_rows
    confidently_wrong = 0
    for row in mixed_rows:
        posterior, status = infer_posterior(
            model, row["posterior_x"],
            layer1_status=row["layer1"].status,
            coverage_complete=row["layer1"].coverage_complete,
        )
        specific = posterior.primary_template not in (
            None, "MIXED_OR_UNRESOLVED"
        )
        if status == "accepted" and specific:
            confidently_wrong += 1
    # a mixed/random actor must not be confidently assigned a specific
    # mechanism for the majority of its decisions
    assert confidently_wrong / len(mixed_rows) <= 0.25


# ---------------------------------------------------------------------------
# counterfactual validity
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cf_environment(reasoning_worlds):
    pooled = [r for w in reasoning_worlds["train"] for r in w["rows"]]
    X = np.asarray(
        [[r["features"][n] for n in ATTRIBUTION_FEATURES] for r in pooled]
    )
    y = np.asarray([r["label"] for r in pooled])
    model, _ = _fit_channel_models(X, y, np.arange(len(pooled)), 1.0)
    assert model.fit_ok
    return model


def _mean_cf_delta(reasoning_worlds, model, template, intervention):
    deltas = []
    for row in reasoning_worlds["test"]["rows"]:
        if row["true_template"] != template:
            continue
        result = run_counterfactuals(model, row["features"], row["label"])
        deltas.append(result.deltas[intervention])
    assert deltas
    return float(np.mean(deltas))


@pytest.mark.parametrize(
    "template,intervention",
    [
        ("FRESH_NEWS_RESPONSE", "remove_fresh_news"),
        ("PERSISTENT_NEWS_ADJUSTMENT", "remove_persistent_news"),
        ("MARKET_MOMENTUM", "flatten_market_trend"),
        ("INVENTORY_REBALANCING", "neutralise_position"),
    ],
)
def test_required_counterfactual_hurts_its_template(
    reasoning_worlds, cf_environment, template, intervention
):
    assert _mean_cf_delta(
        reasoning_worlds, cf_environment, template, intervention
    ) > 0.0


def test_irrelevant_interventions_do_not_mechanically_dominate(
    reasoning_worlds, cf_environment
):
    relevant = _mean_cf_delta(
        reasoning_worlds, cf_environment,
        "FRESH_NEWS_RESPONSE", "remove_fresh_news",
    )
    irrelevant = _mean_cf_delta(
        reasoning_worlds, cf_environment,
        "FRESH_NEWS_RESPONSE", "neutralise_position",
    )
    assert relevant > irrelevant


def test_failed_required_counterfactual_removes_primary(
    reasoning_worlds, trained_template_model
):
    row = next(
        r for r in reasoning_worlds["test"]["rows"]
        if r["true_template"] == "FRESH_NEWS_RESPONSE"
    )
    posterior, status = infer_posterior(
        trained_template_model, row["posterior_x"],
        layer1_status="accepted", coverage_complete=True,
        config=PosteriorConfig(min_top_probability=0.0, min_top_margin=0.0,
                               max_entropy=99.0),
    )
    assert posterior.primary_template is not None
    failing = copy.deepcopy(row["cf"])
    for name in TEMPLATES[posterior.primary_template].required_counterfactuals:
        failing.deltas[name] = -1.0
    versions = version_block(
        DEFAULT_ATTRIBUTION_CONFIG, PosteriorConfig(), COUNTERFACTUAL_NAMES, {}
    )
    record = build_drc_record(
        episode=row["episode"], features=row["features"],
        evidence=row["evidence"], layer1=row["layer1"],
        posterior=posterior, posterior_status=status,
        cf_result=failing, versions=versions,
    )
    assert record["R"]["status"] == "counterfactual_failure"
    assert record["R"]["primary_template"] is None


# ---------------------------------------------------------------------------
# posterior, persistence, versioning, rationale
# ---------------------------------------------------------------------------
def test_posterior_probabilities_sum_to_one(
    reasoning_worlds, trained_template_model
):
    for row in reasoning_worlds["test"]["rows"][:10]:
        posterior, _ = infer_posterior(
            trained_template_model, row["posterior_x"]
        )
        assert abs(sum(posterior.probabilities.values()) - 1.0) < 1e-9


def test_template_inference_is_deterministic(reasoning_worlds):
    xs, ys = [], []
    for world in reasoning_worlds["train"]:
        world_x, world_y = _trainable(world["rows"])
        xs += world_x
        ys += world_y
    a = train_template_model(xs, ys)
    b = train_template_model(xs, ys)
    assert np.allclose(a.weights, b.weights)
    x = reasoning_worlds["test"]["rows"][0]["posterior_x"]
    pa, _ = infer_posterior(a, x)
    pb, _ = infer_posterior(b, x)
    assert pa.probabilities == pb.probabilities


def test_world_seed_train_test_separation_is_enforced(tmp_path):
    with pytest.raises(ValueError):
        run_reasoning_validation(
            str(tmp_path), train_seeds=(0, 1), val_seeds=(2,),
            test_seeds=(1,),
        )


def test_version_hash_changes_with_configuration():
    base = reasoning_method_version_hash(
        DEFAULT_ATTRIBUTION_CONFIG, PosteriorConfig(), COUNTERFACTUAL_NAMES,
        {"l2": 1.0},
    )
    changed = reasoning_method_version_hash(
        DEFAULT_ATTRIBUTION_CONFIG, PosteriorConfig(min_top_probability=0.5),
        COUNTERFACTUAL_NAMES, {"l2": 1.0},
    )
    assert base != changed
    assert base.startswith("reason-")
    assert template_ontology_version_hash().startswith("ontology-")


def test_feature_version_is_not_parser_version():
    versions = version_block(
        DEFAULT_ATTRIBUTION_CONFIG, PosteriorConfig(), COUNTERFACTUAL_NAMES, {}
    )
    assert versions["feature_version"] != PARSER_VERSION
    assert versions["feature_version"].startswith("feat-")


def _one_record(reasoning_worlds, trained_template_model):
    row = next(
        r for r in reasoning_worlds["test"]["rows"]
        if r["true_template"] in PURE_ARCHETYPES
    )
    posterior, status = infer_posterior(
        trained_template_model, row["posterior_x"],
        layer1_status=row["layer1"].status,
        coverage_complete=row["layer1"].coverage_complete,
    )
    versions = version_block(
        DEFAULT_ATTRIBUTION_CONFIG, PosteriorConfig(), COUNTERFACTUAL_NAMES, {}
    )
    return build_drc_record(
        episode=row["episode"], features=row["features"],
        evidence=row["evidence"], layer1=row["layer1"],
        posterior=posterior, posterior_status=status,
        cf_result=row["cf"], versions=versions,
    )


def test_persistence_is_idempotent(reasoning_worlds, trained_template_model):
    record = _one_record(reasoning_worlds, trained_template_model)
    conn = reasoning_worlds["test"]["conn"]
    persist_reasoning_records(conn, [record], reasoning_run_id="idem")
    persist_reasoning_records(conn, [record], reasoning_run_id="idem")
    count = conn.execute(
        "SELECT COUNT(*) FROM reasoning_judgments WHERE reasoning_run_id = ?",
        ("idem",),
    ).fetchone()[0]
    assert count == 1


def test_json_round_trip_preserves_all_fields(
    reasoning_worlds, trained_template_model
):
    record = _one_record(reasoning_worlds, trained_template_model)
    assert json.loads(json.dumps(record, sort_keys=True)) == record


def test_rationale_contains_no_unsupported_evidence(
    reasoning_worlds, trained_template_model
):
    record = _one_record(reasoning_worlds, trained_template_model)
    stripped = copy.deepcopy(record)
    stripped["C"]["news_evidence"] = []
    text = render_rationale(stripped)
    assert "hours" not in text  # news-age claims need evidence rows
    assert "behavioural inference" in text


def test_no_confident_rationale_for_non_accepted_records(
    reasoning_worlds, trained_template_model
):
    record = _one_record(reasoning_worlds, trained_template_model)
    for status in ("ambiguous", "insufficient_context",
                   "counterfactual_failure",
                   "attribution_template_disagreement"):
        failed = copy.deepcopy(record)
        failed["R"]["status"] = status
        text = render_rationale(failed)
        assert "most consistent with" not in text
        assert "behavioural inference" in text
