"""Layer-1 predictive driver attribution tests.

No Ollama required; everything is deterministic and synthetic.
"""

import json
import math

import numpy as np
import pytest

from polymarket.analysis.features import ALL_FEATURES
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.reasoning import (
    ATTRIBUTION_FEATURES,
    ATTRIBUTION_GROUPS,
    STATUS_VALUES,
    load_reasoning_judgments,
    persist_driver_attributions,
    run_driver_attribution,
)
from polymarket.analysis.replay import run_replay
from polymarket.synthetic import scenarios as sc


def test_attribution_groups_partition_all_features_exactly():
    seen: list[str] = []
    for names in ATTRIBUTION_GROUPS.values():
        seen.extend(names)
    assert len(seen) == len(set(seen)), "feature assigned to two channels"
    assert set(seen) == set(ALL_FEATURES), "partition must cover all features"
    assert set(ATTRIBUTION_FEATURES) == set(ALL_FEATURES)


def test_fresh_and_persistent_news_are_separate_channels():
    fresh = set(ATTRIBUTION_GROUPS["fresh_news"])
    persistent = set(ATTRIBUTION_GROUPS["persistent_news"])
    assert not fresh & persistent
    assert "news_decay_signed_6h" in fresh
    assert "news_decay_signed_72h" in persistent
    assert "news_decay_signed_168h" in persistent


def _synthetic_rows(n=24, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels, times, ids = [], [], [], []
    for i in range(n):
        row = {name: 0.0 for name in ALL_FEATURES}
        signal = rng.normal()
        row["news_decay_signed_72h"] = signal          # persistent channel
        row["mkt_return_short"] = 0.1 * rng.normal()   # weak noise channel
        rows.append(row)
        labels.append(1.0 if signal > 0 else -1.0)
        times.append(1000.0 + i * 60.0)
        ids.append(f"d{i}")
    return rows, labels, times, ids


def test_ablating_the_informative_channel_hurts_most():
    rows, labels, times, ids = _synthetic_rows()
    records = run_driver_attribution(
        rows, labels, times, ids, {}, reasoning_run_id="r1"
    )
    evaluated = [r for r in records if r.status != "insufficient_context"]
    assert evaluated
    # the label is a deterministic function of the persistent-news channel,
    # so its ablation delta should dominate for most evaluated decisions
    wins = sum(
        1 for r in evaluated
        if max(r.group_attributions, key=r.group_attributions.get)
        == "persistent_news"
    )
    assert wins / len(evaluated) > 0.7
    for record in evaluated:
        assert set(record.group_attributions) == set(ATTRIBUTION_GROUPS)
        assert set(record.logit_contributions) == set(ATTRIBUTION_GROUPS)
        assert 0.0 <= record.observed_action_probability <= 1.0
        assert record.status in STATUS_VALUES


def test_burn_in_decisions_marked_insufficient_context():
    rows, labels, times, ids = _synthetic_rows()
    records = run_driver_attribution(
        rows, labels, times, ids, {}, reasoning_run_id="r1"
    )
    assert records[0].status == "insufficient_context"
    assert records[0].primary_channel is None
    assert math.isnan(records[0].observed_action_probability)


def test_attribution_is_deterministic():
    rows, labels, times, ids = _synthetic_rows()
    a = run_driver_attribution(rows, labels, times, ids, {},
                               reasoning_run_id="r1")
    b = run_driver_attribution(rows, labels, times, ids, {},
                               reasoning_run_id="r1")
    for ra, rb in zip(a, b):
        assert ra.group_attributions == rb.group_attributions
        assert ra.logit_contributions == rb.logit_contributions
        assert ra.status == rb.status


# ---------------------------------------------------------------------------
def test_replay_produces_attributions_and_strict_evidence(synthetic_db_path):
    reader = SQLiteNormalizedReader(synthetic_db_path)
    run = run_replay(reader, end_time=sc.BASE + 80 * sc.HOUR, run_id="reason")
    assert run.driver_attributions
    assert len(run.driver_attributions) == len(run.labeled_episodes)

    by_decision = {r.decision_id: r for r in run.driver_attributions}
    episodes = {e.decision_id: e for e in run.labeled_episodes}

    # w1's news-driven decision 30 minutes after the debate article: its
    # evidence must contain the debate/polls family with the right age
    # and direction, computed strictly pre-decision.
    w1_id = next(
        d for d, e in episodes.items()
        if e.actor_id == sc.W1 and e.anchor_time == sc.BASE + 30 * sc.HOUR
    )
    w1 = by_decision[w1_id]
    assert w1.top_evidence, "expected news evidence for the news-driven case"
    top = w1.top_evidence[0]
    assert top["direction"] == 1.0
    assert top["age_hours"] == pytest.approx(0.5, abs=0.01)
    assert top["semantic_relevance"] == pytest.approx(0.4)
    # aligned move endpoints both precede the decision (no leakage): the
    # value exists only because pre-news prices exist in the strict context
    assert top["aligned_move_since_news"] is not None

    # w2's early market-driven decision has no news evidence at all
    w2_id = next(
        d for d, e in episodes.items()
        if e.actor_id == sc.W2 and e.anchor_time == sc.BASE + 10 * sc.HOUR
    )
    assert by_decision[w2_id].top_evidence == []

    for record in run.driver_attributions:
        assert record.status in STATUS_VALUES


def test_persistence_round_trip(synthetic_db_path, tmp_path):
    import shutil

    db = str(tmp_path / "reason.sqlite")
    shutil.copy(synthetic_db_path, db)
    reader = SQLiteNormalizedReader(db)
    run = run_replay(reader, end_time=sc.BASE + 80 * sc.HOUR, run_id="persist")
    inserted = persist_driver_attributions(
        reader.conn, run.driver_attributions, feature_version="1.0.0"
    )
    assert inserted == len(run.driver_attributions)
    # idempotent
    assert persist_driver_attributions(
        reader.conn, run.driver_attributions, feature_version="1.0.0"
    ) == 0
    loaded = load_reasoning_judgments(reader.conn, "persist")
    assert len(loaded) == inserted
    for record in loaded:
        assert record["status"] in STATUS_VALUES
        assert record["primary_template"] is None  # reserved for Layer 2
        assert isinstance(record["driver_attribution_json"], dict)
        assert isinstance(record["evidence_json"], list)
        assert record["model_version"].startswith("driver-attribution")


def test_reasoning_output_file_written(synthetic_db_path, tmp_path):
    from polymarket.analysis.reporting import write_run_outputs

    reader = SQLiteNormalizedReader(synthetic_db_path)
    run = run_replay(reader, end_time=sc.BASE + 80 * sc.HOUR, run_id="files")
    paths = write_run_outputs(run, str(tmp_path / "out"))
    assert "reasoning" in paths
    payload = json.loads(open(paths["reasoning"]).read())
    assert payload["method"] == "predictive driver attribution"
    assert payload["records"]
    assert "not mechanism inference" in payload["note"]
