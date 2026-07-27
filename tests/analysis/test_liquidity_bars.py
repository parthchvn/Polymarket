"""Paper data contract: five-minute liquidity bars and the canonical
market-series source policy."""

from __future__ import annotations

import math
import time

import pytest

from polymarket.analysis.liquidity_bars import (
    LiquidityBarConfig,
    build_liquidity_bars,
    logit,
)
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.contracts.schema import (
    PARSER_VERSION,
    SCHEMA_VERSION,
    init_db,
)

COND = "0xbar-cond"
T0 = 1_000_000.0 - (1_000_000.0 % 300)  # aligned bin start


@pytest.fixture()
def conn(tmp_path):
    conn = init_db(str(tmp_path / "bars.sqlite"), description="bars")
    now = time.time()
    conn.execute(
        "INSERT INTO outcome_tokens (condition_id, asset, outcome_label, "
        "outcome_sign, mapping_effective_from, mapping_confidence, "
        "raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) "
        "VALUES (?, 'tok-yes', 'Yes', 1, 0, 'exact', 1, 0, 'h', ?, ?, ?)",
        (COND, PARSER_VERSION, SCHEMA_VERSION, now),
    )
    return conn


def _book(conn, observed_at, bid, ask, *, bid_size=50.0, ask_size=70.0,
          bid_depth=500.0, ask_depth=400.0, tick=0.01):
    conn.execute(
        "INSERT INTO order_book_snapshots (asset, observed_at, best_bid, "
        "best_ask, spread, bid_depth, ask_depth, imbalance, best_bid_size, "
        "best_ask_size, tick_size, raw_response_id, parser_version, "
        "schema_version, normalized_at) VALUES ('tok-yes', ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, 1, ?, ?, ?)",
        (observed_at, bid, ask, ask - bid, bid_depth, ask_depth,
         (bid_depth - ask_depth) / (bid_depth + ask_depth),
         bid_size, ask_size, tick, PARSER_VERSION, SCHEMA_VERSION,
         time.time()),
    )


def _execution(conn, ts, notional):
    columns = [r[1] for r in conn.execute(
        "PRAGMA table_info(canonical_executions)"
    )]
    values = {
        "execution_id": f"e-{ts}", "transaction_hash": f"0x{ts}",
        "condition_id": COND, "notional": notional, "ts": ts,
        "positive_price": 0.5, "positive_side": "BUY", "size": 1.0,
        "raw_response_id": 1, "raw_record_index": 0,
        "raw_record_hash": "h", "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION, "normalized_at": time.time(),
        "transaction_log_index": 1, "transaction_occurrence": 1,
        "reconciliation_status": "direct", "taker_wallet": "w",
    }
    present = [c for c in columns if c in values]
    conn.execute(
        f"INSERT INTO canonical_executions ({','.join(present)}) "
        f"VALUES ({','.join('?' for _ in present)})",
        [values[c] for c in present],
    )


def test_bar_logit_ohlc_realized_variance_and_turnover(conn):
    mids = [(0.40, 0.44), (0.42, 0.46), (0.38, 0.42)]  # mid .42 .44 .40
    for i, (bid, ask) in enumerate(mids):
        _book(conn, T0 + 60 * i, bid, ask)
    _execution(conn, T0 + 30, 12.5)
    _execution(conn, T0 + 200, 7.5)
    conn.commit()
    assert build_liquidity_bars(
        conn, COND,
        config=LiquidityBarConfig(min_book_observations=3),
    ) >= 1
    row = conn.execute(
        "SELECT logit_open, logit_high, logit_low, logit_close, "
        "realized_variance, turnover_notional, spread_mean, "
        "spread_ticks_mean, best_book_size_mean, total_depth_mean, "
        "book_observation_count, execution_count, coverage_complete "
        "FROM liquidity_bars WHERE condition_id = ? AND bin_start = ?",
        (COND, T0),
    ).fetchone()
    logits = [logit(0.42), logit(0.44), logit(0.40)]
    assert row[0] == pytest.approx(logits[0])
    assert row[1] == pytest.approx(max(logits))
    assert row[2] == pytest.approx(min(logits))
    assert row[3] == pytest.approx(logits[-1])
    expected_rv = (logits[1] - logits[0]) ** 2 + (logits[2] - logits[1]) ** 2
    assert row[4] == pytest.approx(expected_rv)
    assert row[5] == pytest.approx(20.0)             # turnover notional
    assert row[6] == pytest.approx(0.04)             # spread mean
    assert row[7] == pytest.approx(4.0)              # spread in ticks
    assert row[8] == pytest.approx(60.0)             # mean best-level size
    assert row[9] == pytest.approx(900.0)            # mean TOTAL depth
    assert (row[10], row[11]) == (3, 2)
    assert row[12] == 1


def test_logit_clipping_keeps_extreme_prices_finite():
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))
    assert logit(0.0) == pytest.approx(-logit(1.0))


def test_bar_without_books_records_turnover_but_incomplete(conn):
    _execution(conn, T0 + 30, 5.0)
    conn.commit()
    build_liquidity_bars(conn, COND)
    # empty-bin grid regularity is covered separately below
    row = conn.execute(
        "SELECT logit_close, spread_mean, turnover_notional, "
        "coverage_complete FROM liquidity_bars WHERE bin_start = ?",
        (T0,),
    ).fetchone()
    assert row[0] is None and row[1] is None      # missing is not zero
    assert row[2] == pytest.approx(5.0)
    assert row[3] == 0                            # not coverage-complete


def test_blocking_gap_makes_bar_incomplete(conn):
    from polymarket.collection.raw_store import record_gap

    _book(conn, T0 + 10, 0.40, 0.44)
    record_gap(conn, collector="forward-loop", surface="books",
               object_id=COND, gap_start=T0 + 100, gap_end=T0 + 200,
               reason="test")
    conn.commit()
    build_liquidity_bars(conn, COND)
    complete = conn.execute(
        "SELECT coverage_complete FROM liquidity_bars WHERE bin_start = ?",
        (T0,),
    ).fetchone()[0]
    assert complete == 0


def test_bars_are_idempotent(conn):
    _book(conn, T0 + 10, 0.40, 0.44)
    conn.commit()
    build_liquidity_bars(conn, COND)
    build_liquidity_bars(conn, COND)
    assert conn.execute(
        "SELECT COUNT(*) FROM liquidity_bars WHERE condition_id = ?",
        (COND,),
    ).fetchone()[0] == conn.execute(
        "SELECT COUNT(DISTINCT bin_start) FROM liquidity_bars "
        "WHERE condition_id = ?", (COND,),
    ).fetchone()[0]


def test_custom_bin_seconds_key(conn):
    _book(conn, T0 + 10, 0.40, 0.44)
    conn.commit()
    build_liquidity_bars(conn, COND)
    build_liquidity_bars(
        conn, COND, config=LiquidityBarConfig(bin_seconds=900.0)
    )
    sizes = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT bin_seconds FROM liquidity_bars"
        )
    }
    assert sizes == {300.0, 900.0}                # coexist under the key


# ---------------------------------------------------------------------------
def _state(conn, ts, price, source):
    conn.execute(
        "INSERT INTO market_state (condition_id, ts, positive_price, "
        "volume, spread, depth, imbalance, state_source, "
        "coverage_complete, raw_response_id, parser_version, "
        "schema_version, normalized_at) VALUES (?, ?, ?, NULL, NULL, "
        "NULL, NULL, ?, 1, NULL, ?, ?, ?)",
        (COND, ts, price, source, PARSER_VERSION, SCHEMA_VERSION,
         time.time()),
    )


def test_market_series_source_policy(conn):
    _state(conn, T0 + 10, 0.50, "executions")
    _state(conn, T0 + 20, 0.52, "executions")
    _state(conn, T0 + 15, 0.51, "book_mid")
    conn.commit()
    reader = SQLiteNormalizedReader(conn)
    cutoff, lookback = T0 + 100, 3600.0

    rows, source = reader.market_series_before(
        COND, cutoff, lookback, policy="book_only"
    )
    assert source == "book_mid" and len(rows) == 1   # never mixed

    rows, source = reader.market_series_before(
        COND, cutoff, lookback, policy="book_preferred"
    )
    assert source == "book_mid" and len(rows) == 1

    rows, source = reader.market_series_before(
        COND, cutoff, lookback, policy="execution_only"
    )
    assert source == "executions" and len(rows) == 2

    with pytest.raises(ValueError):
        reader.market_series_before(COND, cutoff, lookback, policy="mixed")


def test_book_preferred_falls_back_with_flag(conn):
    _state(conn, T0 + 10, 0.50, "executions")
    conn.commit()
    reader = SQLiteNormalizedReader(conn)
    rows, source = reader.market_series_before(
        COND, T0 + 100, 3600.0, policy="book_preferred"
    )
    assert source == "executions" and rows           # flagged fallback
    rows, source = reader.market_series_before(
        "0x-none", T0 + 100, 3600.0, policy="book_preferred"
    )
    assert source == "none" and rows == []



def test_coverage_needs_min_observations_and_variance(conn):
    # 3 observations with default min=4: values exist, coverage does not
    for i, (bid, ask) in enumerate([(0.40, 0.44), (0.42, 0.46),
                                    (0.38, 0.42)]):
        _book(conn, T0 + 60 * i, bid, ask)
    conn.commit()
    build_liquidity_bars(conn, COND)
    row = conn.execute(
        "SELECT book_observation_count, expected_book_observation_count, "
        "book_coverage_fraction, coverage_complete, blocking_gap "
        "FROM liquidity_bars WHERE bin_start = ?", (T0,),
    ).fetchone()
    assert row[0] == 3 and row[1] == 5
    assert row[2] == pytest.approx(3 / 5)
    assert row[3] == 0 and row[4] == 0
    # a single observation can never be complete: no within-bin return
    conn.execute("DELETE FROM liquidity_bars")
    conn.execute(
        "DELETE FROM order_book_snapshots WHERE observed_at > ?", (T0,),
    )
    conn.commit()
    build_liquidity_bars(
        conn, COND, config=LiquidityBarConfig(min_book_observations=1)
    )
    row = conn.execute(
        "SELECT realized_variance, coverage_complete FROM liquidity_bars "
        "WHERE bin_start = ?", (T0,),
    ).fetchone()
    assert tuple(row) == (None, 0)


def test_regular_grid_includes_empty_bins(conn):
    _book(conn, T0 + 10, 0.40, 0.44)
    _book(conn, T0 + 3 * 300 + 10, 0.42, 0.46)   # two empty bins between
    conn.commit()
    build_liquidity_bars(conn, COND)
    rows = conn.execute(
        "SELECT bin_start, book_observation_count, coverage_complete "
        "FROM liquidity_bars ORDER BY bin_start"
    ).fetchall()
    assert [r[0] for r in rows] == [T0, T0 + 300, T0 + 600, T0 + 900]
    assert [r[1] for r in rows] == [1, 0, 0, 1]   # empties are EXPLICIT
    assert all(r[2] == 0 for r in rows)


def test_realized_variance_seeded_with_previous_close(conn):
    for i, (bid, ask) in enumerate(
        [(0.40, 0.44), (0.40, 0.44), (0.40, 0.44), (0.40, 0.44)]
    ):
        _book(conn, T0 + 60 * i, bid, ask)        # bin 1: flat, close .42
    for i, (bid, ask) in enumerate(
        [(0.48, 0.52), (0.48, 0.52), (0.48, 0.52), (0.48, 0.52)]
    ):
        _book(conn, T0 + 300 + 60 * i, bid, ask)  # bin 2: jump to .50
    conn.commit()
    build_liquidity_bars(conn, COND)
    rv2 = conn.execute(
        "SELECT realized_variance FROM liquidity_bars WHERE bin_start = ?",
        (T0 + 300,),
    ).fetchone()[0]
    jump = (logit(0.50) - logit(0.42)) ** 2
    assert rv2 == pytest.approx(jump)             # first return retained
