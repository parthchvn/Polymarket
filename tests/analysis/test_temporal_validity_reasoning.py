"""Temporal validity of R: nothing that happens AFTER the decision can
alter the reasoning inputs — not later news, later market state, nor
future wallet trades."""

from __future__ import annotations

import sqlite3

import pytest

from polymarket.analysis.context import build_context
from polymarket.analysis.features import compute_features
from polymarket.analysis.reasoning import news_evidence
from polymarket.analysis.reasoning_posterior import posterior_features


@pytest.fixture()
def world_copy(reasoning_worlds, tmp_path):
    """A disposable copy of the held-out world: mutations never leak
    into the shared session fixture."""
    target = str(tmp_path / "world-copy.sqlite")
    destination = sqlite3.connect(target)
    reasoning_worlds["test"]["conn"].backup(destination)
    destination.row_factory = sqlite3.Row
    from polymarket.analysis.reader import SQLiteNormalizedReader

    return {
        "conn": destination,
        "reader": SQLiteNormalizedReader(destination),
        "rows": reasoning_worlds["test"]["rows"],
    }


def _reasoning_inputs(world, episode):
    context = build_context(world["reader"], episode)
    features = compute_features(context, episode)
    evidence = news_evidence(context)
    return features, evidence, posterior_features(
        features, episode, evidence=evidence
    )


def _pick_episode(world):
    rows = [
        row for row in world["rows"]
        if row["true_template"] == "PERSISTENT_NEWS_ADJUSTMENT"
    ]
    assert rows
    return rows[len(rows) // 2]["episode"]


def test_post_decision_news_cannot_alter_reasoning(world_copy):
    world = world_copy
    episode = _pick_episode(world)
    before = _reasoning_inputs(world, episode)
    t = episode.anchor_time
    conn = world["conn"]
    conn.execute(
        """
        INSERT INTO relevance_judgments
            (event_family_id, market_id, contract_version_seq, computed_at,
             rel_class, rel_score, direction, novelty, surprise, method,
             model_version, evidence_json)
        VALUES ('fam-future', ?, 1, ?, 'supports_negative', 0.9, -1.0,
                NULL, NULL, 'rule', 'rules-1', '{}')
        """,
        (episode.market_id, t + 60.0),
    )
    conn.commit()
    after = _reasoning_inputs(world, episode)
    assert after == before


def test_post_decision_market_state_cannot_alter_reasoning(world_copy):
    world = world_copy
    episode = _pick_episode(world)
    before = _reasoning_inputs(world, episode)
    conn = world["conn"]
    conn.execute(
        """
        INSERT INTO market_state
            (condition_id, ts, positive_price, volume, spread, depth,
             imbalance, state_source, coverage_complete, raw_response_id,
             parser_version, schema_version, normalized_at)
        VALUES (?, ?, 0.99, 1000.0, 0.001, 9999.0, 0.0, 'derived', 1,
                NULL, '1.0.0', 1, ?)
        """,
        (episode.condition_id, episode.anchor_time + 60.0,
         episode.anchor_time + 60.0),
    )
    conn.commit()
    after = _reasoning_inputs(world, episode)
    assert after == before


def test_future_wallet_trades_cannot_alter_reasoning(world_copy):
    world = world_copy
    episode = _pick_episode(world)
    before = _reasoning_inputs(world, episode)
    conn = world["conn"]
    conn.execute(
        """
        INSERT INTO actor_trade_legs
            (actor_leg_id, source_record_id, candidate_fingerprint,
             transaction_hash, transaction_log_index,
             transaction_occurrence, proxy_wallet, condition_id, asset,
             outcome_label, outcome_sign, side, size, price, ts,
             liquidity_role, role_confidence, raw_response_id,
             raw_record_index, raw_record_hash, parser_version,
             schema_version, normalized_at)
        VALUES ('future-leg', NULL, 'fp-future', '0xfuture', 1, 1, ?, ?,
                'w8-yes', 'Yes', 1, 'BUY', 50.0, 0.5, ?, 'taker',
                'exact', 1, 0, 'h', '1.0.0', 1, ?)
        """,
        (episode.actor_id, episode.condition_id,
         episode.anchor_time + 120.0, episode.anchor_time + 120.0),
    )
    conn.commit()
    after = _reasoning_inputs(world, episode)
    assert after == before
