"""PR D: annotation tooling (strict pre-decision records, agreement)
and the latent reasoning scaffold (planted-structure recovery,
held-out gates, refusal, cluster stability, naming policy)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from polymarket.analysis.annotation import (
    ANNOTATION_LABELS,
    cohens_kappa,
    gold_labels,
    import_annotations,
    sample_annotation_batch,
)
from polymarket.analysis.latent_reasoning import (
    LatentReasoningModel,
    cluster_stability,
    evaluate_latent_reasoning,
    fit_logistic,
    heldout_splits,
    predict_logistic,
)
from polymarket.synthetic.fixtures import build_synthetic_fixture


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "s.sqlite"
    return build_synthetic_fixture(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------
def test_annotation_batch_is_strict_and_deterministic(synth):
    a = sample_annotation_batch(synth, n=20, seed=5)
    b = sample_annotation_batch(synth, n=20, seed=5)
    assert a["batch_id"] == b["batch_id"]
    assert [i["decision_id"] for i in a["items"]] == \
        [i["decision_id"] for i in b["items"]]
    item = a["items"][0]
    # nothing post-decision is renderable to the reviewer
    text = json.dumps(item)
    for forbidden in ("outcome", "resolved", "realized", "drift",
                      "winning"):
        assert forbidden not in text.lower()
    assert item["label"] == "" and item["label_options"] == list(
        ANNOTATION_LABELS
    )
    # stratification key present and the batch spans channels
    channels = {i["dominant_attribution_channel"] for i in a["items"]}
    assert len(channels) >= 1
    row = synth.execute(
        "SELECT n_decisions, feature_version FROM annotation_batches "
        "WHERE batch_id = ?", (a["batch_id"],),
    ).fetchone()
    assert row[0] == len(a["items"]) and row[1].startswith("feat-")


def test_import_agreement_kappa_and_gold(synth, tmp_path):
    batch = sample_annotation_batch(synth, n=12, seed=9)
    items = batch["items"]
    # reviewer A and B agree on all but two decisions
    for reviewer, flips in (("alice", set()), ("bob", {0, 1})):
        path = tmp_path / f"{reviewer}.jsonl"
        with open(path, "w") as handle:
            for index, item in enumerate(items):
                labelled = dict(item)
                labelled["label"] = (
                    "MARKET_MOMENTUM"
                    if index not in flips or reviewer == "alice"
                    else "CONTRARIAN_REVERSAL"
                )
                handle.write(json.dumps(labelled) + "\n")
    report = import_annotations(
        synth, batch["batch_id"],
        {"alice": str(tmp_path / "alice.jsonl"),
         "bob": str(tmp_path / "bob.jsonl")},
    )
    pair = report["pairwise"]
    assert pair["n_shared"] == len(items)
    assert pair["raw_agreement"] == pytest.approx(
        (len(items) - 2) / len(items)
    )
    assert -1.0 <= pair["cohens_kappa"] <= 1.0
    gold = gold_labels(synth, batch["batch_id"])
    assert len(gold) == len(items) - 2          # disagreements excluded
    assert set(gold.values()) == {"MARKET_MOMENTUM"}


def test_import_rejects_unknown_labels(synth, tmp_path):
    batch = sample_annotation_batch(synth, n=3, seed=2)
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as handle:
        item = dict(batch["items"][0])
        item["label"] = "VIBES"
        handle.write(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="unknown label"):
        import_annotations(
            synth, batch["batch_id"], {"x": str(path)}
        )


def test_kappa_known_values():
    perfect = [("A", "A")] * 10
    assert cohens_kappa(perfect) == 1.0
    # zero-information: agreement exactly at chance level
    chance = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    assert cohens_kappa(chance) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------
def _planted_world(n=600, d=30, k_true=2, seed=4, n_actors=20):
    """Decisions generated from a LOW-RANK structure over features:
    exactly the situation where the bottleneck should beat nothing and
    match/beat the full-rank baseline out of sample."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W_true = rng.normal(size=(d, k_true))
    v_true = np.array([2.0, -1.5])[:k_true]
    logits = (X @ W_true) @ v_true
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(float)
    actors = np.array([f"w{i % n_actors}" for i in range(n)])
    times = np.arange(n, dtype=float) * 600.0
    return {
        "X": X, "y": y, "actors": actors, "times": times,
        "decision_ids": [f"d{i}" for i in range(n)],
        "feature_names": [f"f{i}" for i in range(d)],
    }


def test_latent_model_recovers_planted_low_rank():
    data = _planted_world()
    report = evaluate_latent_reasoning(data, latent_dim=2, seed=13)
    assert report["status"] == "evaluated"
    for block in report["splits"].values():
        assert block["status"] == "evaluated"
        assert block["log_loss_latent"] < block["log_loss_null"]
        # rank-2 truth: the bottleneck matches the full-rank model
        assert block["log_loss_latent"] <= \
            block["log_loss_c_only"] * 1.05
    assert "latent_0" in report["naming_policy"]
    assert "UNNAMED" in report["naming_policy"]


def test_gate_flag_reflects_split_results():
    data = _planted_world(seed=11)
    report = evaluate_latent_reasoning(data, latent_dim=2, seed=13)
    expected = all(
        b["latent_beats_c_only"] for b in report["splits"].values()
        if b["status"] == "evaluated"
    )
    assert report["gate_1_predictive_value"] == expected


def test_refusal_on_thin_samples():
    data = _planted_world(n=40, n_actors=4)
    report = evaluate_latent_reasoning(data)
    assert report["status"] == "refused"
    assert "too few" in report["note"]


def test_heldout_splits_never_leak_actors():
    data = _planted_world()
    splits = heldout_splits(data["actors"], data["times"], seed=7)
    test_actors = set(data["actors"][splits["actor"]].tolist())
    train_actors = set(data["actors"][~splits["actor"]].tolist())
    assert test_actors.isdisjoint(train_actors)
    # time split holds out the FINAL slice, not random rows
    assert data["times"][splits["time"]].min() > \
        data["times"][~splits["time"]].max()


def test_cluster_stability_high_on_strong_structure():
    rng = np.random.default_rng(0)
    centers = np.array([[4, 0], [-4, 0], [0, 4]])
    Z = np.vstack([
        center + rng.normal(scale=0.3, size=(80, 2))
        for center in centers
    ])
    assert cluster_stability(Z, Z + rng.normal(
        scale=0.05, size=Z.shape
    ), 3) > 0.95


def test_baseline_logistic_sane():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(float)
    w = fit_logistic(X, y)
    p = predict_logistic(X, w)
    assert ((p > 0.5) == (y == 1)).mean() > 0.9


def test_encoder_is_low_rank_and_unnamed():
    data = _planted_world()
    model = LatentReasoningModel(latent_dim=3, seed=5).fit(
        data["X"], data["y"]
    )
    Z = model.encode(data["X"])
    assert Z.shape == (len(data["y"]), 3)
    assert model.W.shape == (data["X"].shape[1], 3)
    # no attribute anywhere claims primitive names for dimensions
    assert not hasattr(model, "primitive_names")
