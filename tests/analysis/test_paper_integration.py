"""PR C: paper-derived state in strict decision context, the paper
feature group, gate tightening, and the ex-post outcome layer O."""

from __future__ import annotations

import sys
import time

import pytest

sys.path.insert(0, "tests/analysis")

import test_liquidity_modes as modes_t
from test_liquidity_modes import BIN, COND, T0, _fitted, _news_family

from polymarket.analysis.features import ALL_FEATURES, FEATURE_GROUPS
from polymarket.analysis.outcomes import (
    OUTCOME_CONTRACT,
    attach_outcomes,
    compute_outcomes,
)
from polymarket.analysis.paper_features import build_paper_state
from polymarket.contracts.schema import init_db


@pytest.fixture()
def conn(tmp_path):
    return init_db(str(tmp_path / "integ.sqlite"), description="integ")


def _world_with_screens(conn):
    modes_t._regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)                    # cutoff = T0 + 60*BIN
    _news_family(conn, "fam-impact", T0 + 100 * BIN + 42)
    from polymarket.analysis.news_impact_screen import screen_news_impact

    screen_news_impact(conn, model.mode_run_id)
    return model


def test_paper_state_strict_asof(conn):
    model = _world_with_screens(conn)
    # decision long after the impactful screen became available
    t_late = T0 + 150 * BIN
    state = build_paper_state(conn, COND, t_late, model.mode_run_id)
    assert state["liquidity_mode"] is not None
    assert state["liquidity_mode"]["closed_at"] <= t_late
    assert state["screens_evaluated"] >= 1
    assert len(state["impact_screens"]) == 1
    assert state["initial_response_so_far"] is not None
    # decision BEFORE the screen was available: the screen must vanish
    screen_available = T0 + 102 * BIN        # end of bin t+1
    state_early = build_paper_state(
        conn, COND, screen_available, model.mode_run_id
    )
    assert state_early["impact_screens"] == []
    # no mode run: explicit absence, attention still computed
    state_none = build_paper_state(conn, COND, t_late, None)
    assert state_none["liquidity_mode"] is None
    assert "claim_count_24h" in state_none["attention"]


def test_paper_features_and_missingness(conn):
    from polymarket.analysis.context import DecisionContext
    from polymarket.analysis.decisions import DecisionEpisode
    from polymarket.analysis.features import compute_features

    model = _world_with_screens(conn)
    t_late = T0 + 150 * BIN

    def episode_at(t):
        return DecisionEpisode(
            decision_id="d1", actor_id="w1", market_id="m-jump",
            condition_id=COND, anchor_time=t, interval_start=t - 60,
            interval_end=t, positive_quantity_change=1.0,
            gross_quantity=1.0, direction="positive",
            mixed_activity_ratio=0.0,
        )

    def context_with(paper_state, t):
        return DecisionContext(
            decision_id="d1", actor_id="w1", market_id="m-jump",
            condition_id=COND, decision_time=t, contract=None,
            market_status=None, market_state=[], execution_activity=[],
            order_books=[], actor_history=[], position={},
            articles=[], event_families=[], relevance=[],
            paper_state=paper_state,
        )

    state = build_paper_state(conn, COND, t_late, model.mode_run_id)
    features = compute_features(
        context_with(state, t_late), episode_at(t_late)
    )
    assert set(features) == set(ALL_FEATURES)
    assert features["impact_screen_available"] == 1.0
    assert features["impactful_news_count"] == 1.0
    assert features["impactful_news_probability"] == 1.0
    assert features["impact_screen_contradiction"] == 0.0
    assert features["liq_mode_missing"] == 0.0
    assert features["initial_response_missing"] == 0.0
    # without a mode run everything degrades to explicit missingness
    empty = compute_features(
        context_with(build_paper_state(conn, COND, t_late, None),
                     t_late),
        episode_at(t_late),
    )
    assert empty["liq_mode_missing"] == 1.0
    assert empty["impact_screen_available"] == 0.0
    assert empty["impact_screen_contradiction"] == 0.0
    assert empty["initial_response_missing"] == 1.0


def test_contradiction_flag_when_screens_find_nothing(conn):
    modes_t._regime_world(conn, event_windows=((30, 40),))
    model = _fitted(conn)
    _news_family(conn, "fam-quiet", T0 + 70 * BIN + 42)  # calm news
    from polymarket.analysis.news_impact_screen import screen_news_impact

    screen_news_impact(conn, model.mode_run_id)
    state = build_paper_state(
        conn, COND, T0 + 90 * BIN, model.mode_run_id
    )
    assert state["screens_evaluated"] == 1
    assert state["impact_screens"] == []      # screened, NOT impactful
    from polymarket.analysis.reasoning_templates import TEMPLATES

    persistent = TEMPLATES["PERSISTENT_NEWS_ADJUSTMENT"]
    assert "impact_screen_contradiction" in \
        persistent.incompatible_conditions


def test_outcomes_layer_and_no_leakage(conn):
    modes_t._regime_world(conn, event_windows=((30, 40),))
    # outcome bars at 900s granularity for the O layer
    for i in range(200):
        conn.execute(
            "INSERT OR REPLACE INTO liquidity_bars (condition_id, "
            "bin_start, bin_end, bin_seconds, logit_open, logit_high, "
            "logit_low, logit_close, realized_variance, "
            "turnover_notional, spread_mean, spread_ticks_mean, "
            "best_book_size_mean, total_depth_mean, imbalance_mean, "
            "book_observation_count, expected_book_observation_count, "
            "book_coverage_fraction, blocking_gap, execution_count, "
            "coverage_complete, feature_version, computed_at) VALUES "
            "(?, ?, ?, 900, 0, 0, 0, ?, 0.0001, 10, 0.02, 2, 200, 800, "
            "0, 15, 15, 1.0, 0, 1, 1, 'fv', ?)",
            (COND, T0 + i * 900.0, T0 + i * 900.0 + 900.0,
             0.001 * i, time.time()),
        )
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "('m-jump', ?, 'Q?', 1, 0, 'h', 'p', 2, ?)", (COND, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO market_status_versions (market_id, "
        "effective_from, first_observed_at, trading_enabled, closed, "
        "resolved, raw_response_id, parser_version, schema_version, "
        "normalized_at) VALUES ('m-jump', 0, 0, 1, 0, 0, 1, 'p', 2, ?)",
        (now,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO contract_versions (market_id, "
        "version_seq, effective_from, first_observed_at, question, "
        "rules_text, content_hash, raw_response_id, parser_version, "
        "schema_version, normalized_at) VALUES ('m-jump', 1, 0, 0, "
        "'Q?', 'r', 'ch-m', 1, 'p', 2, ?)", (now,),
    )
    conn.commit()
    decision_time = T0 + 50 * 900.0 + 10
    outcome = compute_outcomes(
        conn, condition_id=COND, decision_time=decision_time,
        direction="positive",
    )
    one_hour = outcome["horizons"]["3600s"]
    assert one_hour["status"] == "observed"
    assert one_hour["realized_drift"] > 0
    assert one_hour["same_direction_continuation"] is True
    assert outcome["contract"] == OUTCOME_CONTRACT
    # attach pass: O rides on records, features untouched
    records = [{
        "D": {"condition_id": COND, "decision_time": decision_time,
              "direction": "positive"},
        "C": {"features": {"mkt_last_price": 0.5}},
    }]
    attached = attach_outcomes(records, conn)
    assert attached == 1
    assert "O" in records[0]
    assert records[0]["C"] == {"features": {"mkt_last_price": 0.5}}
    # STRUCTURAL no-leakage: no outcome key is a feature, and the
    # feature/reasoning modules never import the outcome layer
    outcome_keys = {"realized_drift", "same_direction_continuation",
                    "ex_post_continuation_rate"}
    assert outcome_keys.isdisjoint(set(ALL_FEATURES))
    for module in ("features", "reasoning", "reasoning_templates",
                   "context", "paper_features"):
        source = open(
            f"src/polymarket/analysis/{module}.py"
        ).read()
        assert "outcomes" not in source.replace(
            "paper_outcomes_never_in_C", ""
        ), module


def test_feature_manifest_includes_paper_group():
    assert "paper" in FEATURE_GROUPS
    from polymarket.analysis.versioning import feature_version_hash

    assert feature_version_hash()  # manifest change -> new hash exists
