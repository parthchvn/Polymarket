"""Underreaction decomposition: interval classification, drift
regressions recover planted continuation, placebo destroys it,
absorption events, distraction interaction, and clustered-SE sanity."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

from polymarket.analysis.attention import (
    compute_distraction,
    distraction_interaction_regression,
)
from polymarket.analysis.news_returns import (
    DecompositionConfig,
    build_interval_records,
    daily_aggregation,
)
from polymarket.analysis.underreaction import (
    event_absorption,
    ols_clustered,
    run_drift_regressions,
)
from polymarket.contracts.schema import init_db

COND = "0xur"
BIN = 900.0
T0 = 1_700_000_000.0 - (1_700_000_000.0 % 900)
CONFIG = DecompositionConfig(bin_seconds=BIN)


@pytest.fixture()
def conn(tmp_path):
    return init_db(str(tmp_path / "ur.sqlite"), description="ur")


def _bar(conn, condition_id, bin_start, logit_close, turnover=50.0):
    conn.execute(
        "INSERT OR REPLACE INTO liquidity_bars (condition_id, "
        "bin_start, bin_end, bin_seconds, logit_open, logit_high, "
        "logit_low, logit_close, realized_variance, turnover_notional, "
        "spread_mean, spread_ticks_mean, best_book_size_mean, "
        "total_depth_mean, imbalance_mean, book_observation_count, "
        "expected_book_observation_count, book_coverage_fraction, "
        "blocking_gap, execution_count, coverage_complete, "
        "feature_version, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
        "0.0001, ?, 0.02, 2.0, 200, 800, 0, 15, 15, 1.0, 0, 3, 1, "
        "'fv', ?)",
        (condition_id, bin_start, bin_start + BIN, BIN, logit_close,
         logit_close, logit_close, logit_close, turnover, time.time()),
    )


def _claim(conn, claim_id, ts, condition_id=COND, market_id=None,
           rel_class="supports_positive"):
    market_id = market_id or f"m-{condition_id}"
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "(?, ?, 'Q?', 1, 0, 'h', 'p', 2, ?)",
        (market_id, condition_id, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO news_articles (article_id, source_id, "
        "source_url, source_published_at, first_observed_at, "
        "download_completed_at, timestamp_source, timestamp_confidence, "
        "headline, body, content_hash, raw_response_id, "
        "raw_record_index, raw_record_hash, parser_version, "
        "schema_version, normalized_at) VALUES (?, 's', 'u', ?, ?, ?, "
        "'feed', 0.8, ?, ?, ?, 1, 0, 'h', 'p', 2, ?)",
        (f"art-{claim_id}", ts, ts, ts, claim_id, claim_id,
         f"ch-{claim_id}", now),
    )
    conn.execute(
        "INSERT INTO news_claims (claim_id, article_id, claim_text, "
        "entities_json, quantities_json, first_available_at, "
        "extractor_version, confidence) VALUES (?, ?, ?, '[]', '[]', "
        "?, 'x', 0.9)",
        (claim_id, f"art-{claim_id}", claim_id, ts),
    )
    conn.execute(
        "INSERT OR IGNORE INTO event_families (event_family_id, label, "
        "earliest_available_at, created_by, created_at) VALUES "
        "(?, ?, ?, 't', ?)", (f"fam-{claim_id}", claim_id, ts, now),
    )
    conn.execute(
        "INSERT INTO claim_edges (edge_id, claim_id, event_family_id, "
        "edge_type, effective_from, evidence, confidence) VALUES "
        "(?, ?, ?, 'new', ?, 'k', 0.5)",
        (f"edge-{claim_id}", claim_id, f"fam-{claim_id}", ts),
    )
    conn.execute(
        "INSERT INTO relevance_judgments (relevance_judgment_id, "
        "claim_id, event_family_id, market_id, contract_version_seq, "
        "source_effective_at, scored_at, computed_at, rel_class, "
        "rel_score, direction, method, model_version) VALUES "
        "(?, ?, ?, ?, 1, ?, ?, ?, ?, 0.9, 1.0, 'rule', 'v')",
        (f"rj-{claim_id}", claim_id, f"fam-{claim_id}", market_id, ts,
         now, ts, rel_class),
    )


def _continuation_world(conn, n=400, news_every=10, seed=7,
                        condition_id=COND):
    """News intervals: initial move of VARYING size m (identification
    comes from cross-event size variation), then a concentrated
    CONTINUATION of 0.5*m in the next interval.  Non-news shocks of
    varying size REVERSE fully next interval."""
    rng = random.Random(seed)
    level, drift_per_bin, drift_left = 0.0, 0.0, 0
    for i in range(n):
        shock = 0.0
        if i > 0 and i % news_every == 0:
            _claim(conn, f"{condition_id}-news-{i}",
                   T0 + i * BIN + 30.0, condition_id)
            m = rng.uniform(0.02, 0.10) * rng.choice([1, 1, 1, -1])
            shock, drift_per_bin, drift_left = m, 0.5 * m, 1
        elif i % news_every == 5:
            reversal = rng.uniform(0.02, 0.05) * rng.choice([-1, 1])
            level += reversal          # reverses next interval
            _bar(conn, condition_id, T0 + i * BIN, level)
            level -= reversal
            continue
        if drift_left > 0 and shock == 0.0:
            shock = drift_per_bin
            drift_left -= 1
        level += shock + rng.gauss(0, 0.001)
        _bar(conn, condition_id, T0 + i * BIN, level)
    conn.commit()


def test_interval_classification_and_decomposition(conn):
    _continuation_world(conn)
    records = build_interval_records(conn, CONFIG)
    news = [r for r in records if r.is_news]
    assert 30 <= len(news) <= 45              # ~ every 10th interval
    sample = news[0]
    assert sample.news_claims and sample.r_news == sample.ret
    assert sample.r_nonnews == 0.0
    quiet = next(r for r in records if not r.is_news)
    assert quiet.r_news == 0.0 and quiet.r_nonnews == quiet.ret
    days = daily_aggregation(records)
    assert sum(d["news_intervals"] for d in days) == len(news)


def test_gap_breaks_return_pair(conn):
    _bar(conn, COND, T0, 0.0)
    _bar(conn, COND, T0 + BIN, 0.1)
    _bar(conn, COND, T0 + 3 * BIN, 0.2)       # hole at bin 2
    conn.commit()
    records = build_interval_records(conn, CONFIG)
    assert [r.bin_start for r in records] == [T0 + BIN]  # no bridging


def test_drift_regression_recovers_continuation(conn):
    _continuation_world(conn)
    _continuation_world(conn, seed=17, condition_id="0xur2")
    results = run_drift_regressions(
        conn, CONFIG, horizons=(BIN, 2 * BIN, 3 * BIN)
    )
    assert len(results) == 3
    for result in results:
        # planted structure: the news response CONTINUES strongly; the
        # non-news loading is far smaller (reversals dominate it at
        # short horizons; at longer horizons it mixes with drift bins,
        # matching the paper's "behaves differently" phrasing)
        assert result.beta_news > 0.2, result.as_dict()
        assert result.beta_nonnews < result.beta_news * 0.3, \
            result.as_dict()
        t_day = result.beta_news / result.se_news_by_cluster["utc_day"]
        assert t_day > 2
    # the full continuation is present from the first horizon on
    assert results[1].beta_news >= results[0].beta_news * 0.8


def test_placebo_destroys_the_news_loading(conn):
    """A single circular shift is a high-variance draw, so the placebo
    is averaged over seeds — its MEAN loading collapses toward zero
    while the real alignment stays put."""
    _continuation_world(conn)
    _continuation_world(conn, seed=17, condition_id="0xur2")
    real = run_drift_regressions(conn, CONFIG, horizons=(4 * BIN,))[0]
    placebos = [
        run_drift_regressions(
            conn, CONFIG, horizons=(4 * BIN,), placebo_seed=seed
        )[0].beta_news
        for seed in (11, 23, 47, 91, 137, 251, 999)
    ]
    mean_placebo = sum(placebos) / len(placebos)
    assert abs(mean_placebo) < abs(real.beta_news) * 0.4
    assert real.beta_news > max(
        0.2, sorted(abs(b) for b in placebos)[len(placebos) // 2] * 0.9
    )


def test_event_absorption_records(conn):
    _continuation_world(conn)
    events = event_absorption(
        conn, CONFIG, initial_horizon=BIN, drift_horizon=4 * BIN
    )
    covered = [e for e in events if e["coverage_complete"]]
    assert len(covered) >= 25
    continuation_rate = sum(
        1 for e in covered if e["same_direction_continuation"]
    ) / len(covered)
    assert continuation_rate > 0.7            # planted drift
    fractions = [e["absorption_fraction"] for e in covered]
    assert all(0 < f < 1 for f in fractions)
    # initial m vs concentrated drift 0.5m: absorption near 2/3
    assert 0.5 < sum(fractions) / len(fractions) < 0.85
    assert "NOT a statistic from the paper" in covered[0][
        "absorption_note"
    ]


def test_absorption_missing_coverage_is_missing(conn):
    _bar(conn, COND, T0, 0.0)
    _bar(conn, COND, T0 + BIN, 0.05)
    _claim(conn, "lonely", T0 + BIN + 10)
    conn.commit()
    # the drift endpoint (24h out) has no fresh close: missing, not 0
    events = event_absorption(conn, CONFIG, drift_horizon=24 * 3600.0)
    assert len(events) == 1
    assert events[0]["coverage_complete"] is False
    assert events[0]["absorption_fraction"] is None   # never zero
    # a claim before any coverage is missing on the PRE side too
    _claim(conn, "too-early", T0 - 5000.0)
    conn.commit()
    events = event_absorption(conn, CONFIG, drift_horizon=BIN)
    early = next(e for e in events if e["claim_id"] == "too-early")
    assert early["coverage_complete"] is False


def test_distraction_interaction_positive_when_planted(conn):
    """Two half-worlds: in the HIGH-distraction half (many unrelated
    claims) the drift after news is doubled.  The interaction picks up
    the planted mechanism."""
    rng = random.Random(3)
    level = 0.0
    for i in range(400):
        high = i >= 200
        if high and i % 2 == 0:  # unrelated claims for another market
            _claim(conn, f"noise-{i}", T0 + i * BIN + 5.0,
                   condition_id="0xother")
        if i % 10 == 0 and i > 0:
            _claim(conn, f"n-{i}", T0 + i * BIN + 30.0)
            level += 0.03
        elif (i % 10) in (1, 2, 3, 4):
            level += (0.02 if high else 0.01)  # stronger drift when high
        level += rng.gauss(0, 0.001)
        _bar(conn, COND, T0 + i * BIN, level)
    conn.commit()
    out = distraction_interaction_regression(
        conn, CONFIG, horizon=4 * BIN
    )
    assert out is not None
    assert out["beta_news_x_distraction"] > 0
    assert out["n"] > 300


def test_compute_distraction_proxies(conn):
    _claim(conn, "own-1", T0 + 100.0)                 # own family
    _claim(conn, "other-1", T0 + 200.0, condition_id="0xother")
    _bar(conn, COND, T0 + BIN, 0.0)
    _bar(conn, COND, T0 + 2 * BIN, 0.01)
    conn.commit()
    records = build_interval_records(conn, CONFIG)
    proxies = compute_distraction(conn, records)
    assert proxies[0]["cross_market_claim_count"] == 2
    assert proxies[0]["unrelated_family_count"] == 1  # own excluded
    assert isinstance(proxies[0]["weekend"], bool)


def test_screened_spec_uses_only_impactful_claims(conn):
    _bar(conn, COND, T0, 0.0)
    for i in range(1, 6):
        _bar(conn, COND, T0 + i * BIN, 0.01 * i)
    _claim(conn, "hit", T0 + 2 * BIN + 10)
    _claim(conn, "dud", T0 + 3 * BIN + 10)
    now = time.time()
    conn.execute(
        "INSERT INTO liquidity_mode_runs (mode_run_id, fit_cutoff, "
        "bin_seconds, lambda_penalty, lambda_selection, centroids_json, "
        "reference_stats_json, calm_mode, train_bar_count, config_json, "
        "model_version, created_at) VALUES ('run-x', 0, 300, 1, "
        "'fixed', '[]', '{}', 0, 10, '{}', 'v', ?)", (now,),
    )
    for claim_id, transition in (("hit", 1), ("dud", 0)):
        conn.execute(
            "INSERT INTO news_impact_screens (mode_run_id, claim_id, "
            "event_family_id, condition_id, assignment_basis, "
            "news_time, arrival_bin_start, transition_detected, "
            "impact_score, screen_status, screen_available_at, "
            "model_effective_from, screen_model_version, created_at) "
            "VALUES ('run-x', ?, ?, ?, 'retrospective_smoothed', ?, 0, "
            "?, ?, 'screened', 0, 0, 'v', ?)",
            (claim_id, f"fam-{claim_id}", COND,
             T0 + (2 if claim_id == 'hit' else 3) * BIN + 10,
             transition, float(transition), now),
        )
    conn.commit()
    config = DecompositionConfig(
        bin_seconds=BIN, mode_run_id="run-x",
    )
    records = build_interval_records(conn, config, "screened_impactful")
    news = [r for r in records if r.is_news]
    assert len(news) == 1
    assert news[0].news_claims == ["hit"]             # dud excluded


def test_clustered_se_sanity():
    """With independent clusters and homoskedastic noise, clustered SEs
    land near the classical ones; with perfectly correlated duplicated
    clusters, the clustered SE is materially larger."""
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    fit = ols_clustered(
        y, X, clusters={"iid": np.arange(n)}
    )
    classical = math.sqrt(
        float(np.sum((y - X @ fit["beta"]) ** 2) / (n - 2)
              * np.linalg.pinv(X.T @ X)[1, 1])
    )
    assert fit["se"]["iid"][1] == pytest.approx(classical, rel=0.15)
    # duplicate every observation: same info, doubled n; clustering by
    # duplicate-pair must keep the SE near the original, while iid
    # treatment would understate it
    X2 = np.vstack([X, X])
    y2 = np.concatenate([y, y])
    pair = np.concatenate([np.arange(n), np.arange(n)])
    dup = ols_clustered(y2, X2, clusters={
        "pair": pair, "iid": np.arange(2 * n),
    })
    assert dup["se"]["pair"][1] > dup["se"]["iid"][1] * 1.2
