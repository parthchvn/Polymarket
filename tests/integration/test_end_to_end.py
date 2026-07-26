"""29.11 end-to-end workflow test on the synthetic world."""

import json
import os

from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.replay import run_replay
from polymarket.analysis.reporting import audit_database, write_run_outputs
from polymarket.synthetic import scenarios as sc


def test_full_workflow(synthetic_db_path, tmp_path):
    reader = SQLiteNormalizedReader(synthetic_db_path)
    run = run_replay(
        reader,
        end_time=sc.BASE + 80 * sc.HOUR,
        seed=1337,
        run_id="test-run",
    )
    assert len(run.labeled_episodes) >= 6
    assert run.evaluation is not None
    for model in ("M0", "M1", "M2", "M3"):
        assert model in run.evaluation.metrics
        assert 0.0 <= run.evaluation.metrics[model]["brier"] <= 1.0
    assert run.placebos is not None and len(run.placebos.results) == 5
    assert len(run.bootstraps) == 3
    assert run.attributions

    # news-driven decision: w1 at BASE+30h attributes to the debate family
    w1_news = [
        a for a in run.attributions
        if a["top_event_family"] is not None
        and any(e.actor_id == sc.W1 and e.anchor_time == sc.BASE + 30 * sc.HOUR
                and e.decision_id == a["decision_id"]
                for e in run.labeled_episodes)
    ]
    assert w1_news, "expected candidate attribution for the news-driven decision"
    assert all(a["label"] == "candidate attribution" for a in run.attributions)

    # market-state-driven decision: w2 at BASE+10h has no news attribution
    w2_early = [
        a for a in run.attributions
        if any(e.actor_id == sc.W2 and e.anchor_time == sc.BASE + 10 * sc.HOUR
               and e.decision_id == a["decision_id"]
               for e in run.labeled_episodes)
    ]
    assert w2_early and w2_early[0]["top_event_family"] is None

    output_dir = str(tmp_path / "outputs")
    paths = write_run_outputs(run, output_dir)
    for key in ("predictions", "metrics", "config", "feature_manifest"):
        assert os.path.exists(paths[key]), key
    metrics = json.loads(open(paths["metrics"]).read())
    assert metrics["placebos"]["results"]
    assert metrics["run_id"] == "test-run"

    audit = audit_database(reader.conn)
    assert audit["schema_version"] == 1
    assert audit["unknown_role_legs"] == 2


def test_run_replay_deterministic(synthetic_db_path):
    reader = SQLiteNormalizedReader(synthetic_db_path)
    a = run_replay(reader, end_time=sc.BASE + 80 * sc.HOUR, run_id="x")
    b = run_replay(reader, end_time=sc.BASE + 80 * sc.HOUR, run_id="x")
    assert a.evaluation.per_decision == b.evaluation.per_decision
    assert a.evaluation.metrics == b.evaluation.metrics
