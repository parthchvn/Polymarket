"""PR E: the one-command pipeline — stage isolation, honest gate
states, the acceptance report, and the DRC-backed abstention gate."""

from __future__ import annotations

import json

import pytest

from polymarket.analysis.pipeline import run_reasoning_pipeline
from polymarket.synthetic.fixtures import build_synthetic_fixture


@pytest.fixture()
def synth_db(tmp_path):
    path = str(tmp_path / "pipe.sqlite")
    conn = build_synthetic_fixture(path, overwrite=True)
    conn.close()
    return path


def test_pipeline_end_to_end_honest_statuses(synth_db, tmp_path):
    out = str(tmp_path / "out")
    report = run_reasoning_pipeline(synth_db, out)
    stages = report["stages"]
    # completed stages
    assert stages["migrate"]["status"] == "ok"
    assert stages["normalize"]["status"] == "ok"
    assert stages["liquidity_bars"]["status"] == "ok"
    assert stages["run_analysis"]["status"] == "ok"
    # honest refusals on the tiny fixture, each with a reason
    assert stages["liquidity_modes"]["status"] == "refused"
    assert "complete training bars" in stages["liquidity_modes"]["reason"]
    # a refusal upstream is ISOLATION, not cascade: screens skipped
    # with the reason, analysis still ran
    assert stages["impact_screens"]["status"] == "skipped"
    assert stages["latent_reasoning"]["status"] == "refused"
    # gates present with honest states
    gates = report["gates"]
    assert gates["gate_1_predictive_value"]["status"] == "refused"
    assert "too few" in gates["gate_1_predictive_value"]["reason"]
    assert gates["gate_2_human_agreement"]["status"] == "pending"
    assert gates["gate_4_honest_abstention"]["status"] == "pending"
    # the report is on disk, valid JSON, and carries data readiness
    loaded = json.load(open(report["report_path"]))
    assert loaded["feature_version"].startswith("feat-")
    assert "complete_bars" in loaded["data_readiness"]
    assert loaded["data_readiness"]["actors"] >= 1
    assert "_drc_records" not in loaded          # internal key stripped


def test_pipeline_skip_flags(synth_db, tmp_path):
    report = run_reasoning_pipeline(
        synth_db, str(tmp_path / "out"),
        skip={"normalize", "underreaction", "analysis", "modes"},
    )
    assert report["stages"]["normalize"]["status"] == "skipped"
    assert report["stages"]["underreaction"]["status"] == "skipped"
    assert report["stages"]["run_analysis"]["status"] == "skipped"
    assert report["stages"]["liquidity_modes"]["status"] == "skipped"


def test_pipeline_refuses_bad_artifact_without_crashing(
    synth_db, tmp_path
):
    report = run_reasoning_pipeline(
        synth_db, str(tmp_path / "out"),
        reasoning_model_path=str(tmp_path / "missing.json"),
    )
    stage = report["stages"]["run_analysis"]
    assert stage["status"] == "refused"
    assert "reasoning artifact" in stage["reason"]
    # downstream latent evaluation still ran (refused on size, not
    # crashed)
    assert report["stages"]["latent_reasoning"]["status"] in (
        "refused", "evaluated",
    )


def test_gate_2_uses_gold_labels_when_present(synth_db, tmp_path):
    import sqlite3

    from polymarket.analysis.annotation import sample_annotation_batch

    conn = sqlite3.connect(synth_db)
    conn.row_factory = sqlite3.Row
    batch = sample_annotation_batch(conn, n=5, seed=1)
    now = 1.0
    for item in batch["items"]:
        for reviewer in ("a", "b"):
            conn.execute(
                "INSERT OR REPLACE INTO annotations (batch_id, "
                "decision_id, reviewer, label, imported_at) VALUES "
                "(?, ?, ?, 'MARKET_MOMENTUM', ?)",
                (batch["batch_id"], item["decision_id"], reviewer,
                 now),
            )
    conn.commit()
    conn.close()
    report = run_reasoning_pipeline(
        synth_db, str(tmp_path / "out"),
        annotation_batch_id=batch["batch_id"],
    )
    gate = report["gates"]["gate_2_human_agreement"]
    # gold labels exist; without accepted model DRC records the gate
    # stays PENDING with the honest overlap explanation
    assert gate["status"] == "pending"
    assert "gold labels" in gate["reason"]
