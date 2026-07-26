"""Occurrence versus direction separation: at-risk intervals feed only
the occurrence head; maker/unknown legs and mixed-direction episodes
never become direction decisions."""

from __future__ import annotations

from polymarket.analysis.decisions import build_decision_episodes
from polymarket.analysis.reasoning_targets import (
    build_at_risk_opportunities,
    occurrence_labels,
)
from polymarket.synthetic.reasoning_worlds import MAKER


def test_maker_and_unknown_legs_never_create_primary_decisions(
    reasoning_worlds,
):
    world = reasoning_worlds["test"]
    episodes = build_decision_episodes(
        world["reader"], end_time=world["meta"]["end_time"]
    )
    actors = {episode.actor_id for episode in episodes}
    assert MAKER not in actors  # the maker counterparty never decides
    roles = world["conn"].execute(
        "SELECT DISTINCT liquidity_role FROM actor_trade_legs "
        "WHERE proxy_wallet = ?",
        (MAKER,),
    ).fetchall()
    assert roles and all(role[0] != "taker" for role in roles)


def test_no_trade_intervals_enter_occurrence_model_only(reasoning_worlds):
    world = reasoning_worlds["test"]
    opportunities = build_at_risk_opportunities(
        world["reader"], end_time=world["meta"]["end_time"]
    )
    labels = occurrence_labels(opportunities)
    no_trade = [
        opportunity for opportunity, label in zip(opportunities, labels)
        if label < 0
    ]
    assert no_trade  # the grid contains genuine no-trade intervals
    # every pseudo-episode is direction=None: structurally ineligible for
    # the direction model, which requires an observed clean taker trade
    assert all(o.episode.direction is None for o in opportunities)
    direction_ids = {
        episode.decision_id
        for episode in build_decision_episodes(
            world["reader"], end_time=world["meta"]["end_time"]
        )
        if episode.direction is not None
    }
    assert direction_ids.isdisjoint(
        {o.episode.decision_id for o in opportunities}
    )


def test_at_risk_intervals_respect_market_status_and_coverage(
    reasoning_worlds,
):
    world = reasoning_worlds["test"]
    opportunities = build_at_risk_opportunities(
        world["reader"], end_time=world["meta"]["end_time"]
    )
    end = world["meta"]["end_time"]
    for opportunity in opportunities[:50]:
        assert opportunity.interval_start < opportunity.interval_end <= end
        state = world["reader"].market_state_before(
            opportunity.condition_id, opportunity.interval_start, 24 * 3600.0
        )
        assert state  # coverage was certified at the interval start


def test_mixed_direction_trades_stay_out_of_direction_model(
    reasoning_worlds, tmp_path
):
    import sqlite3

    from polymarket.analysis.reader import SQLiteNormalizedReader

    target = str(tmp_path / "mixed.sqlite")
    conn = sqlite3.connect(target)
    reasoning_worlds["test"]["conn"].backup(conn)
    conn.row_factory = sqlite3.Row
    t = reasoning_worlds["test"]["meta"]["end_time"] - 5 * 86400.0
    for leg_index, (side, size) in enumerate(
        (("BUY", 5.0), ("SELL", 5.0))
    ):
        conn.execute(
            """
            INSERT INTO actor_trade_legs
                (actor_leg_id, source_record_id, candidate_fingerprint,
                 transaction_hash, transaction_log_index,
                 transaction_occurrence, proxy_wallet, condition_id,
                 asset, outcome_label, outcome_sign, side, size, price,
                 ts, liquidity_role, role_confidence, raw_response_id,
                 raw_record_index, raw_record_hash, parser_version,
                 schema_version, normalized_at)
            VALUES (?, NULL, ?, ?, 1, 1, '0x-mixer', ?, 'w8-yes', 'Yes',
                    1, ?, ?, 0.5, ?, 'taker', 'exact', 1, 0, 'h',
                    '1.0.0', 1, ?)
            """,
            (f"mix-{leg_index}", f"fp-mix-{leg_index}",
             f"0xmix{leg_index}",
             reasoning_worlds["test"]["meta"]["condition_id"], side, size,
             t + leg_index * 60.0, t),
        )
    conn.commit()
    episodes = build_decision_episodes(
        SQLiteNormalizedReader(conn),
        end_time=reasoning_worlds["test"]["meta"]["end_time"],
    )
    mixer = [e for e in episodes if e.actor_id == "0x-mixer"]
    assert len(mixer) == 1
    # perfectly offsetting activity: |net| / gross = 0 < threshold
    assert mixer[0].direction is None
    assert mixer[0].mixed_activity_ratio == 0.0
