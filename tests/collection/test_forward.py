"""Forward-collection loop: scheduling, failure isolation, bounded gap
recording, restart cursor derivation, and book-mid state derivation.
All offline — surfaces are stubbed; no network."""

from __future__ import annotations

import json
import time

import pytest

from polymarket.collection.forward import (
    DOWNTIME_FACTOR,
    ForwardConfig,
    LoopState,
    derive_trade_cursor,
    discover_assets,
    record_downtime_gaps,
    run_cycle,
    run_loop,
)
from polymarket.collection.raw_store import (
    finish_collector_run,
    insert_raw_response,
    start_collector_run,
)
from polymarket.contracts.schema import init_db

CONDITIONS = ("0xc-a", "0xc-b")


@pytest.fixture()
def conn(tmp_path):
    return init_db(str(tmp_path / "loop.sqlite"), description="loop test")


def _config(**overrides):
    defaults = dict(
        condition_ids=CONDITIONS, interval_seconds=300.0,
        activity_every=3, news_every=2, news_queries=("q",),
    )
    defaults.update(overrides)
    return ForwardConfig(**defaults)


def _stub(log, name, fail_on: set[int] | None = None):
    def surface(conn, config, state, window):
        if fail_on and state.cycle_index in fail_on:
            raise RuntimeError(f"{name} boom")
        log.append((state.cycle_index, name, window))
        return f"{name} ok"
    return surface


def test_every_cycle_vs_every_k_scheduling(conn):
    log: list = []
    config = _config()
    surfaces = {
        "books": (_stub(log, "books"), "books"),
        "activity": (
            lambda c, cfg, st, w: (
                "skip" if st.cycle_index % cfg.activity_every else
                (log.append((st.cycle_index, "activity", w)) or "ok")
            ),
            None,
        ),
    }
    state = LoopState()
    for _ in range(6):
        run_cycle(conn, config, state, surfaces=surfaces, now=time.time())
    books_cycles = [c for c, n, _ in log if n == "books"]
    activity_cycles = [c for c, n, _ in log if n == "activity"]
    assert books_cycles == [0, 1, 2, 3, 4, 5]       # every cycle
    assert activity_cycles == [0, 3]                 # every K cycles


def test_failure_isolation_and_bounded_gap(conn):
    log: list = []
    config = _config()
    surfaces = {
        "trades": (_stub(log, "trades", fail_on={1}), "trades"),
        "books": (_stub(log, "books"), "books"),
    }
    state = LoopState()
    t0 = 1_000_000.0
    run_cycle(conn, config, state, surfaces=surfaces, now=t0)
    report = run_cycle(conn, config, state, surfaces=surfaces, now=t0 + 300)
    assert report["trades"].startswith("FAILED")
    assert report["books"] == "books ok"             # isolation held
    assert state.failures == {"trades": 1}
    gaps = conn.execute(
        "SELECT surface, object_id, gap_start, gap_end, resolved_at "
        "FROM collector_gaps ORDER BY object_id"
    ).fetchall()
    assert len(gaps) == len(CONDITIONS)              # one per market
    for surface, object_id, gap_start, gap_end, resolved_at in gaps:
        assert surface == "trades"
        assert object_id in CONDITIONS
        assert (gap_start, gap_end) == (t0, t0 + 300)  # BOUNDED window
        assert resolved_at is None                   # blocks that window
    # a bounded gap blocks exactly the affected span, nothing else
    from polymarket.analysis.reader import SQLiteNormalizedReader

    reader = SQLiteNormalizedReader(conn)
    assert reader.blocking_gaps("0xc-a", t0 + 100, t0 + 200)
    assert not reader.blocking_gaps("0xc-a", t0 + 400, t0 + 500)


def test_third_cycle_after_failure_continues(conn):
    log: list = []
    config = _config()
    surfaces = {"trades": (_stub(log, "trades", fail_on={1}), "trades")}
    state = LoopState()
    t0 = 1_000_000.0
    for i in range(3):
        run_cycle(conn, config, state, surfaces=surfaces, now=t0 + 300 * i)
    assert [c for c, _, _ in log] == [0, 2]
    assert conn.execute(
        "SELECT COUNT(*) FROM collector_gaps"
    ).fetchone()[0] == len(CONDITIONS)               # only the failed cycle


def _store_taker_response(conn, received_at):
    run_id = start_collector_run(conn, "trades_taker", {"t": received_at})
    insert_raw_response(
        conn, collector_run_id=run_id, collector="trades_taker",
        base_url="stub://", endpoint="trades", params={"t": received_at},
        requested_at=received_at - 1, received_at=received_at,
        http_status=200, headers={}, payload=b"[]",
    )
    finish_collector_run(conn, run_id, "succeeded")


def test_trade_cursor_derived_from_database(conn):
    assert derive_trade_cursor(conn) is None
    _store_taker_response(conn, 1_000_000.0)
    _store_taker_response(conn, 1_000_600.0)
    assert derive_trade_cursor(conn) == 1_000_600.0


def test_downtime_gap_recorded_on_restart(conn):
    config = _config(interval_seconds=300.0)
    now = 2_000_000.0
    assert record_downtime_gaps(conn, config, now) == 0  # fresh db: none
    _store_taker_response(conn, now - 10_000.0)          # long downtime
    assert record_downtime_gaps(conn, config, now) == len(CONDITIONS)
    gaps = conn.execute(
        "SELECT surface, gap_start, gap_end FROM collector_gaps"
    ).fetchall()
    assert {g[0] for g in gaps} == {"trades", "books"}
    assert all(g[1] == now - 10_000.0 and g[2] == now for g in gaps)
    # recent activity within the downtime factor records nothing
    _store_taker_response(conn, now - config.interval_seconds)
    before = conn.execute("SELECT COUNT(*) FROM collector_gaps").fetchone()[0]
    assert record_downtime_gaps(
        conn, config, now + DOWNTIME_FACTOR * config.interval_seconds * 0.4
    ) == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM collector_gaps"
    ).fetchone()[0] == before


def test_run_loop_respects_max_cycles_without_sleeping(conn):
    config = _config()
    calls: list = []
    surfaces = {"books": (_stub(calls, "books"), "books")}
    import polymarket.collection.forward as forward

    original = forward.DEFAULT_SURFACES
    forward.DEFAULT_SURFACES = surfaces
    slept: list = []
    try:
        state = run_loop(
            conn, config, max_cycles=3,
            sleep_fn=slept.append, now_fn=time.time,
        )
    finally:
        forward.DEFAULT_SURFACES = original
    assert state.cycle_index == 3
    assert len(slept) == 2                            # no sleep after last


def test_discover_assets_from_raw_markets_payload(conn):
    payload = [{
        "conditionId": "0xc-a",
        "clobTokenIds": json.dumps(["tok-yes", "tok-no"]),
        "outcomes": json.dumps(["Yes", "No"]),
    }]
    run_id = start_collector_run(conn, "markets", {})
    insert_raw_response(
        conn, collector_run_id=run_id, collector="markets",
        base_url="stub://", endpoint="markets", params={},
        requested_at=1.0, received_at=2.0, http_status=200, headers={},
        payload=json.dumps(payload).encode(),
    )
    finish_collector_run(conn, run_id, "succeeded")
    assets = discover_assets(conn, ("0xc-a", "0xc-b"))
    assert assets["0xc-a"] == ["tok-yes", "tok-no"]
    assert assets["0xc-b"] == []


def test_book_mid_state_derivation(conn):
    from polymarket.contracts.schema import PARSER_VERSION, SCHEMA_VERSION
    from polymarket.normalization.markets import (
        derive_market_state_from_books,
    )

    now = time.time()
    conn.execute(
        "INSERT INTO outcome_tokens (condition_id, asset, outcome_label, "
        "outcome_sign, mapping_effective_from, mapping_confidence, "
        "raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) "
        "VALUES ('0xc-a', 'tok-yes', 'Yes', 1, 0, 'exact', 1, 0, 'h', ?, ?, ?)",
        (PARSER_VERSION, SCHEMA_VERSION, now),
    )
    for i, (bid, ask) in enumerate([(0.40, 0.44), (0.42, 0.46)]):
        conn.execute(
            "INSERT INTO order_book_snapshots (asset, observed_at, "
            "best_bid, best_ask, spread, bid_depth, ask_depth, imbalance, "
            "raw_response_id, parser_version, schema_version, "
            "normalized_at) VALUES ('tok-yes', ?, ?, ?, ?, 100, 120, 0.1, "
            "1, ?, ?, ?)",
            (1000.0 + i * 60, bid, ask, ask - bid,
             PARSER_VERSION, SCHEMA_VERSION, now),
        )
    conn.commit()
    written = derive_market_state_from_books(conn, "0xc-a")
    assert written == 2
    rows = conn.execute(
        "SELECT ts, positive_price, spread, depth, state_source "
        "FROM market_state WHERE state_source='book_mid' ORDER BY ts"
    ).fetchall()
    assert [round(r[1], 3) for r in rows] == [0.42, 0.44]  # mids
    assert rows[0][3] == 220.0                              # summed depth
    # idempotent re-derivation
    assert derive_market_state_from_books(conn, "0xc-a") == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM market_state WHERE state_source='book_mid'"
    ).fetchone()[0] == 2
