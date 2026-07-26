"""End-to-end CLI proof: a synthetic normalized database + a saved
reasoning model artifact produce full, persisted, deterministic and
idempotent DRC records through ``python -m polymarket.cli run-analysis``."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from polymarket.analysis.reasoning import DEFAULT_ATTRIBUTION_CONFIG
from polymarket.analysis.reasoning_artifact import (
    ArtifactVersionMismatch,
    build_artifact_payload,
    load_reasoning_model,
    save_reasoning_model,
)
from polymarket.analysis.reasoning_posterior import DEFAULT_POSTERIOR_CONFIG
from polymarket.analysis.reasoning_reconstruction import reasoning_versions
from polymarket.contracts.schema import PARSER_VERSION


@pytest.fixture(scope="module")
def cli_environment(reasoning_worlds, trained_template_model,
                    tmp_path_factory):
    base = tmp_path_factory.mktemp("cli-e2e")
    artifact_path = str(base / "reasoning_model.json")
    save_reasoning_model(
        build_artifact_payload(
            trained_template_model,
            attribution_config=DEFAULT_ATTRIBUTION_CONFIG,
            posterior_config=DEFAULT_POSTERIOR_CONFIG,
            versions=reasoning_versions(),
            train_world_seeds=(0, 1, 2),
            validation_world_seeds=(),
        ),
        artifact_path,
    )
    db_path = str(base / "world.sqlite")
    copy = sqlite3.connect(db_path)
    reasoning_worlds["test"]["conn"].backup(copy)
    copy.close()
    return {
        "artifact": artifact_path,
        "db": db_path,
        "base": base,
        "end_time": reasoning_worlds["test"]["meta"]["end_time"],
    }


def _run_cli(env, output_dir, *extra):
    return subprocess.run(
        [sys.executable, "-m", "polymarket.cli", "run-analysis",
         "--db", env["db"], "--output", str(output_dir),
         "--end-time", str(env["end_time"]),
         "--run-id", "cli-e2e",
         "--reasoning-model", env["artifact"],
         "--reasoning-target", "both", *extra],
        capture_output=True, text=True,
    )


def test_cli_produces_and_persists_full_drc_records(cli_environment):
    out = cli_environment["base"] / "out1"
    result = _run_cli(cli_environment, out)
    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in (out / "drc_records.jsonl").read_text().splitlines()
    ]
    assert records
    for record in records:
        assert record["R"]["template_posterior"]
        assert record["C"]["contract_version"] is not None
        assert "coverage_certified" in record["C"]["market_status"]
        assert record["versions"]["feature_version"].startswith("feat-")
        assert record["versions"]["feature_version"] != PARSER_VERSION
    assert (out / "occurrence_drc_records.jsonl").exists()
    summary = json.loads((out / "reasoning_summary.json").read_text())
    assert summary["direction"]["n_records"] == len(records)

    conn = sqlite3.connect(cli_environment["db"])
    rows = conn.execute(
        "SELECT template_posterior_json, counterfactual_json, "
        "feature_version, model_version FROM reasoning_judgments "
        "WHERE reasoning_run_id = 'cli-e2e'"
    ).fetchall()
    assert rows
    for posterior_json, cf_json, feature_version, model_version in rows:
        assert json.loads(posterior_json)["probabilities"]
        assert feature_version.startswith("feat-")
        assert model_version.startswith("reason-")
    assert any(json.loads(r[1]).get("deltas") for r in rows if r[1])


def test_cli_is_deterministic_and_idempotent(cli_environment):
    out_a = cli_environment["base"] / "det-a"
    out_b = cli_environment["base"] / "det-b"
    assert _run_cli(cli_environment, out_a).returncode == 0
    conn = sqlite3.connect(cli_environment["db"])
    count_first = conn.execute(
        "SELECT COUNT(*) FROM reasoning_judgments "
        "WHERE reasoning_run_id = 'cli-e2e'"
    ).fetchone()[0]
    assert _run_cli(cli_environment, out_b).returncode == 0
    count_second = conn.execute(
        "SELECT COUNT(*) FROM reasoning_judgments "
        "WHERE reasoning_run_id = 'cli-e2e'"
    ).fetchone()[0]
    assert count_first == count_second  # idempotent under a stable run id
    assert (
        (out_a / "drc_records.jsonl").read_text()
        == (out_b / "drc_records.jsonl").read_text()
    )  # deterministic


def test_artifact_refuses_feature_hash_mismatch(cli_environment, tmp_path):
    payload = json.loads(open(cli_environment["artifact"]).read())
    payload["versions"]["feature_version"] = "feat-0000000000000000"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ArtifactVersionMismatch):
        load_reasoning_model(str(tampered))
    result = subprocess.run(
        [sys.executable, "-m", "polymarket.cli", "run-analysis",
         "--db", cli_environment["db"], "--output", str(tmp_path / "o"),
         "--reasoning-model", str(tampered)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not match the current feature manifest" in result.stderr


def test_no_reasoning_fallback_never_uses_parser_version(
    cli_environment,
):
    out = cli_environment["base"] / "layer1-only"
    result = subprocess.run(
        [sys.executable, "-m", "polymarket.cli", "run-analysis",
         "--db", cli_environment["db"], "--output", str(out),
         "--end-time", str(cli_environment["end_time"]),
         "--no-reasoning"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    conn = sqlite3.connect(cli_environment["db"])
    versions = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT feature_version FROM reasoning_judgments"
        )
    }
    assert PARSER_VERSION not in versions
    assert all(v.startswith("feat-") for v in versions)


def test_round_trip_artifact_reproduces_inference(
    cli_environment, trained_template_model, reasoning_worlds
):
    import numpy as np

    from polymarket.analysis.reasoning_posterior import POSTERIOR_FEATURES

    loaded, _payload = load_reasoning_model(cli_environment["artifact"])
    row = reasoning_worlds["test"]["rows"][0]
    x = np.asarray([[row["posterior_x"][n] for n in POSTERIOR_FEATURES]])
    original = trained_template_model.predict_proba(x)[0]
    reloaded = loaded.predict_proba(x)[0]
    assert np.allclose(original, reloaded)
