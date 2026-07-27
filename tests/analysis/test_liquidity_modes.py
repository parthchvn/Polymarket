"""Liquidity jump model + news impact screen: DP optimality, synthetic
regime recovery, deterministic refits, training-only stationarization,
lambda selection, eq-4.1 screening, and strict screen availability."""

from __future__ import annotations

import itertools
import math
import random
import time

import pytest

from polymarket.analysis.liquidity_modes import (
    VARIABLES,
    FittedJumpModel,
    JumpModelConfig,
    dp_assign,
    fit_jump_model,
    fit_reference_stats,
    persist_jump_model,
    stationarize,
)
from polymarket.analysis.news_impact_screen import (
    impactful_news_asof,
    screen_news_impact,
)
from polymarket.contracts.schema import init_db

COND = "0xjump"
BIN = 300.0
T0 = 1_700_000_000.0 - (1_700_000_000.0 % 300)


@pytest.fixture()
def conn(tmp_path):
    return init_db(str(tmp_path / "modes.sqlite"), description="modes")


def _insert_bar(conn, condition_id, bin_start, *, spread_ticks, turnover,
                volatility, best_size, complete=1):
    conn.execute(
        "INSERT OR REPLACE INTO liquidity_bars (condition_id, bin_start, "
        "bin_end, bin_seconds, logit_open, logit_high, logit_low, "
        "logit_close, realized_variance, turnover_notional, spread_mean, "
        "spread_ticks_mean, best_book_size_mean, total_depth_mean, "
        "imbalance_mean, book_observation_count, "
        "expected_book_observation_count, book_coverage_fraction, "
        "blocking_gap, execution_count, coverage_complete, "
        "feature_version, computed_at) VALUES (?, ?, ?, ?, 0, 0, 0, 0, "
        "?, ?, ?, ?, ?, ?, 0, 5, 5, 1.0, 0, 3, ?, 'fv', ?)",
        (condition_id, bin_start, bin_start + BIN, BIN,
         volatility ** 2, turnover, spread_ticks * 0.01, spread_ticks,
         best_size, best_size * 4, complete, time.time()),
    )


def _regime_world(conn, *, n_bins=240, event_windows=((100, 112), (180, 190)),
                  condition_id=COND, seed=5):
    """Calm baseline with planted high-volatility/turnover/spread event
    windows.  Returns the true mode labels."""
    rng = random.Random(seed)
    truth = []
    for i in range(n_bins):
        in_event = any(a <= i < b for a, b in event_windows)
        truth.append("event" if in_event else "calm")
        scale = 6.0 if in_event else 1.0
        _insert_bar(
            conn, condition_id, T0 + i * BIN,
            spread_ticks=rng.uniform(1.0, 1.4) * (2.0 if in_event else 1.0),
            turnover=rng.uniform(80, 120) * scale,
            volatility=rng.uniform(0.008, 0.012) * scale,
            best_size=rng.uniform(180, 220) / (2.0 if in_event else 1.0),
        )
    conn.commit()
    return truth


# ---------------------------------------------------------------------------
def test_dp_assignment_matches_brute_force():
    rng = random.Random(1)
    X = [tuple(rng.uniform(-2, 2) for _ in range(4)) for _ in range(9)]
    centroids = [[-1.0] * 4, [1.0] * 4]
    for lam in (0.0, 0.7, 3.0):
        modes = dp_assign(X, centroids, lam)

        def objective(assignment):
            fit = sum(
                sum((a - b) ** 2 for a, b in zip(x, centroids[m]))
                for x, m in zip(X, assignment)
            )
            switches = sum(
                1 for a, b in zip(assignment, assignment[1:]) if a != b
            )
            return fit + lam * switches

        best = min(
            itertools.product((0, 1), repeat=len(X)), key=objective
        )
        assert objective(modes) == pytest.approx(objective(list(best)))


def test_lambda_zero_reduces_to_kmeans_assignment():
    X = [(-1.0, 0, 0, 0), (1.0, 0, 0, 0), (-1.1, 0, 0, 0)]
    centroids = [[-1.0, 0, 0, 0], [1.0, 0, 0, 0]]
    assert dp_assign(X, centroids, 0.0) == [0, 1, 0]  # nearest centroid


def test_recovers_planted_regimes_and_labels_calm(conn):
    truth = _regime_world(conn)
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    predicted = {
        bin_start: ("calm" if mode == model.calm_mode else "event")
        for (condition, bin_start), mode in model.assignments.items()
    }
    agreement = sum(
        1 for i, label in enumerate(truth)
        if predicted.get(T0 + i * BIN) == label
    ) / len(truth)
    assert agreement > 0.9
    sigma_index = VARIABLES.index("volatility")
    assert (model.centroids[model.calm_mode][sigma_index]
            < model.centroids[1 - model.calm_mode][sigma_index])


def test_fit_is_deterministic(conn):
    _regime_world(conn)
    kwargs = dict(fit_cutoff=T0 + 500 * BIN,
                  config=JumpModelConfig(fixed_lambda=1.0))
    a = fit_jump_model(conn, **kwargs)
    b = fit_jump_model(conn, **kwargs)
    assert a.mode_run_id == b.mode_run_id
    assert a.centroids == b.centroids
    assert a.assignments == b.assignments


def test_higher_lambda_reduces_switches(conn):
    _regime_world(conn, seed=9)
    switches = {}
    for lam in (0.0, 8.0):
        model = fit_jump_model(
            conn, fit_cutoff=T0 + 500 * BIN,
            config=JumpModelConfig(fixed_lambda=lam),
        )
        ordered = [
            mode for (_, bin_start), mode in sorted(
                model.assignments.items(), key=lambda kv: kv[0][1]
            )
        ]
        switches[lam] = sum(
            1 for a, b in zip(ordered, ordered[1:]) if a != b
        )
    assert switches[8.0] <= switches[0.0]


def test_persistence_target_lambda_selection(conn):
    _regime_world(conn, seed=3)
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(lambda_candidates=(0.0, 1.0, 4.0)),
    )
    assert model.lambda_selection == "persistence_target"
    assert model.lambda_penalty in (0.0, 1.0, 4.0)


def test_stationarization_fitted_on_training_only():
    train = [
        {"condition_id": COND, "bin_start": T0 + i * BIN,
         "spread_ticks": 1.0, "turnover": 100.0, "volatility": 0.01,
         "best_book_size": 200.0}
        for i in range(30)
    ]
    stats = fit_reference_stats(train, JumpModelConfig())
    # a wildly different LATER bar must not shift training statistics
    later = {"condition_id": COND, "bin_start": T0 + 999 * BIN,
             "spread_ticks": 50.0, "turnover": 9999.0,
             "volatility": 0.5, "best_book_size": 1.0}
    x = stationarize(later, stats, JumpModelConfig())
    assert x is not None
    assert x[VARIABLES.index("turnover")] > 0     # z-scored vs TRAIN medians
    # a condition absent from training standardizes via POOLED stats
    unknown = dict(later, condition_id="0xother")
    transferred = stationarize(unknown, stats, JumpModelConfig())
    assert transferred is not None
    assert all(abs(z) <= 10.0 for z in transferred[:-1])  # winsorized


def test_incomplete_bars_break_dp_chains(conn):
    _regime_world(conn, n_bins=60, event_windows=())
    # a hole in the middle: bar removed entirely
    conn.execute(
        "DELETE FROM liquidity_bars WHERE bin_start = ?",
        (T0 + 30 * BIN,),
    )
    conn.commit()
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    assert (COND, T0 + 30 * BIN) not in model.assignments
    assert (COND, T0 + 29 * BIN) in model.assignments


def test_fit_refuses_thin_coverage(conn):
    _regime_world(conn, n_bins=5, event_windows=())
    with pytest.raises(ValueError, match="training bars"):
        fit_jump_model(conn, fit_cutoff=T0 + 500 * BIN)


def test_persist_and_reload_round_trip(conn):
    _regime_world(conn)
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    persist_jump_model(conn, model, T0 + 500 * BIN)
    from polymarket.analysis.liquidity_modes import load_jump_model_run

    run = load_jump_model_run(conn, model.mode_run_id)
    assert run["centroids"] == model.centroids
    assert run["calm_mode"] == model.calm_mode
    n = conn.execute(
        "SELECT COUNT(*) FROM liquidity_mode_assignments "
        "WHERE mode_run_id = ?", (model.mode_run_id,),
    ).fetchone()[0]
    assert n == len(model.assignments)
    # idempotent
    persist_jump_model(conn, model, T0 + 500 * BIN)
    assert conn.execute(
        "SELECT COUNT(*) FROM liquidity_mode_assignments "
        "WHERE mode_run_id = ?", (model.mode_run_id,),
    ).fetchone()[0] == n


# ---------------------------------------------------------------------------
def _news_family(conn, family_id, news_time, market_id="m-jump"):
    """A family with ONE claim arriving at news_time (the screening
    unit is the claim x market)."""
    now = time.time()
    claim_id = f"claim-{family_id}"
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "(?, ?, 'Q?', 1, 0, 'h', 'p', 2, ?)", (market_id, COND, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO news_articles (article_id, source_id, "
        "source_url, source_published_at, first_observed_at, "
        "download_completed_at, timestamp_source, timestamp_confidence, "
        "headline, body, content_hash, raw_response_id, "
        "raw_record_index, raw_record_hash, parser_version, "
        "schema_version, normalized_at) VALUES (?, 's', 'u', ?, ?, ?, "
        "'feed', 0.8, ?, ?, ?, 1, 0, 'h', 'p', 2, ?)",
        (f"art-{family_id}", news_time, news_time, news_time,
         family_id, family_id, f"ch-{family_id}", now),
    )
    conn.execute(
        "INSERT INTO news_claims (claim_id, article_id, claim_text, "
        "entities_json, quantities_json, first_available_at, "
        "extractor_version, confidence) VALUES (?, ?, ?, '[]', '[]', "
        "?, 'x', 0.9)",
        (claim_id, f"art-{family_id}", family_id, news_time),
    )
    conn.execute(
        "INSERT INTO event_families (event_family_id, label, "
        "earliest_available_at, created_by, created_at) VALUES "
        "(?, ?, ?, 't', ?)", (family_id, family_id, news_time, now),
    )
    conn.execute(
        "INSERT INTO claim_edges (edge_id, claim_id, event_family_id, "
        "edge_type, effective_from, evidence, confidence) VALUES "
        "(?, ?, ?, 'new', ?, 'k', 0.5)",
        (f"edge-{family_id}", claim_id, family_id, news_time),
    )
    conn.execute(
        "INSERT INTO relevance_judgments (relevance_judgment_id, "
        "claim_id, event_family_id, market_id, contract_version_seq, "
        "source_effective_at, scored_at, computed_at, rel_class, "
        "rel_score, direction, method, model_version) VALUES "
        "(?, ?, ?, ?, 1, ?, ?, ?, 'background', 0.3, 0.0, 'rule', 'v')",
        (f"rj-{family_id}", claim_id, family_id, market_id, news_time,
         now, news_time),
    )
    conn.commit()


def _fitted(conn, fit_cutoff=None) -> FittedJumpModel:
    """Fit on the EARLY part of the world (through the first training
    event window) so that screened news, which arrives later, is
    online-valid under the model-availability gate."""
    cutoff = fit_cutoff if fit_cutoff is not None else T0 + 60 * BIN
    model = fit_jump_model(
        conn, fit_cutoff=cutoff,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    persist_jump_model(conn, model, cutoff)
    return model


def test_screen_detects_calm_to_event_at_boundary(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    # news inside bin 100 (first event bin): calm(99)->event(100) is the
    # boundary immediately before the arrival bin -> impactful
    _news_family(conn, "fam-hit", T0 + 100 * BIN + 42)
    # news deep inside calm: no adjacent transition
    _news_family(conn, "fam-miss", T0 + 70 * BIN + 42)
    counters = screen_news_impact(conn, model.mode_run_id)
    assert counters["screened"] == 2 and counters["impactful"] == 1
    rows = {
        row["event_family_id"]: row for row in conn.execute(
            "SELECT * FROM news_impact_screens WHERE mode_run_id = ?",
            (model.mode_run_id,),
        )
    }
    hit = rows["fam-hit"]
    assert hit["transition_detected"] == 1
    assert (hit["pre_mode_label"], hit["arrival_mode_label"]) \
        == ("calm", "event")
    assert rows["fam-miss"]["transition_detected"] == 0


def test_screen_detects_jump_after_arrival_bin(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    # news in bin 99 (still calm), event starts at bin 100:
    # calm(99)->event(100) is the boundary immediately AFTER arrival
    _news_family(conn, "fam-after", T0 + 99 * BIN + 10)
    screen_news_impact(conn, model.mode_run_id)
    row = conn.execute(
        "SELECT * FROM news_impact_screens WHERE event_family_id = ?",
        ("fam-after",),
    ).fetchone()
    assert row["transition_detected"] == 1
    assert (row["arrival_mode_label"], row["post_mode_label"]) \
        == ("calm", "event")


def test_event_to_calm_is_not_impactful(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    _news_family(conn, "fam-cooldown", T0 + 111 * BIN + 10)
    screen_news_impact(conn, model.mode_run_id)
    row = conn.execute(
        "SELECT transition_detected FROM news_impact_screens "
        "WHERE event_family_id = 'fam-cooldown'"
    ).fetchone()
    assert row[0] == 0                       # event->calm never triggers


def test_screen_availability_is_strictly_after_next_bin(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    news_time = T0 + 100 * BIN + 42
    _news_family(conn, "fam-online", news_time)
    screen_news_impact(conn, model.mode_run_id)
    row = conn.execute(
        "SELECT screen_available_at, arrival_bin_start FROM "
        "news_impact_screens WHERE event_family_id = 'fam-online'"
    ).fetchone()
    assert row[0] == row[1] + 2 * BIN        # end of bin t+1
    # as-of accessor honors strict availability
    early = impactful_news_asof(conn, COND, row[0], model.mode_run_id)
    later = impactful_news_asof(
        conn, COND, row[0] + 1.0, model.mode_run_id
    )
    assert [r["event_family_id"] for r in early] == []
    assert "fam-online" in {r["event_family_id"] for r in later}


def test_missing_neighbor_bins_mark_insufficient_coverage(conn):
    _regime_world(conn, n_bins=50, event_windows=())
    model = _fitted(conn)
    _news_family(conn, "fam-outside", T0 + 400 * BIN)  # far past coverage
    counters = screen_news_impact(conn, model.mode_run_id)
    assert counters["insufficient_coverage"] == 1
    row = conn.execute(
        "SELECT screen_status, transition_detected FROM "
        "news_impact_screens WHERE event_family_id = 'fam-outside'"
    ).fetchone()
    assert tuple(row) == ("insufficient_coverage", 0)


def test_math_sanity_volatility_from_variance():
    assert math.isclose(math.sqrt(0.0004), 0.02)


def test_online_decoder_never_uses_future_bars():
    """The filtered mode at t must be identical whatever happens after
    t — the property the availability claim rests on."""
    from polymarket.analysis.liquidity_modes import online_assign

    rng = random.Random(11)
    X = [tuple(rng.uniform(-2, 2) for _ in range(5)) for _ in range(30)]
    centroids = [[-1.0] * 5, [1.0] * 5]
    full = online_assign(X, centroids, 2.0)
    for t in (1, 7, 15, 29):
        prefix = online_assign(X[:t], centroids, 2.0)
        assert prefix == full[:t]           # future never leaks backward
    # while the SMOOTHED decoder is allowed to differ on prefixes
    smoothed_full = dp_assign(X, centroids, 2.0)
    assert len(smoothed_full) == len(X)


def test_online_and_smoothed_assignments_both_persisted(conn):
    _regime_world(conn)
    model = _fitted(conn)
    row = conn.execute(
        "SELECT mode, mode_online, mode_label, mode_label_online "
        "FROM liquidity_mode_assignments WHERE mode_run_id = ? LIMIT 1",
        (model.mode_run_id,),
    ).fetchone()
    assert row["mode_label"] in ("calm", "event")
    assert row["mode_label_online"] in ("calm", "event")


def test_sparse_turnover_does_not_explode(conn):
    """90% zero-turnover bars with a genuine volatility event window:
    the zero-aware transform + presence indicator must keep turnover
    z-scores bounded, and turnover sparsity must not override the
    volatility signal — the reviewer's dominant-distance failure mode."""
    rng = random.Random(21)
    for i in range(200):
        in_event = 150 <= i < 160
        traded = in_event or rng.random() < 0.1
        _insert_bar(
            conn, COND, T0 + i * BIN,
            spread_ticks=rng.uniform(1.0, 1.4) * (2.0 if in_event else 1),
            turnover=rng.uniform(50, 150) if traded else 0.0,
            volatility=rng.uniform(0.008, 0.012) * (6 if in_event else 1),
            best_size=rng.uniform(180, 220),
        )
    conn.commit()
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    from polymarket.analysis.liquidity_modes import (
        _load_raw_bars,
        stationarize,
    )

    bars = _load_raw_bars(conn, JumpModelConfig())
    zs = [
        stationarize(bar, model.reference_stats, model.config)
        for bar in bars
    ]
    turnover_index = VARIABLES.index("turnover")
    assert all(abs(z[turnover_index]) <= 10.0 for z in zs if z)
    labels = {
        bin_start: ("calm" if m == model.calm_mode else "event")
        for (_, bin_start), m in model.assignments.items()
    }
    # the volatility event window is recovered ...
    event_hits = sum(
        1 for i in range(150, 160)
        if labels.get(T0 + i * BIN) == "event"
    )
    assert event_hits >= 8
    # ... and QUIET zero-turnover bars are NOT dragged into event mode
    # by turnover presence/absence alone
    calm_zone = [labels[T0 + i * BIN] for i in range(0, 140)
                 if T0 + i * BIN in labels]
    assert calm_zone.count("calm") / len(calm_zone) > 0.9


def test_transfer_to_market_absent_from_training(conn):
    _regime_world(conn, condition_id=COND)
    # a NEW market appears only after the fit cutoff
    for i in range(20):
        _insert_bar(
            conn, "0xnew", T0 + 600 * BIN + i * BIN,
            spread_ticks=1.2, turnover=100.0, volatility=0.01,
            best_size=200.0,
        )
    conn.commit()
    model = fit_jump_model(
        conn, fit_cutoff=T0 + 500 * BIN,
        config=JumpModelConfig(fixed_lambda=1.0),
    )
    new_market = [
        key for key in model.assignments if key[0] == "0xnew"
    ]
    assert len(new_market) == 20            # pooled fallback transfers


def test_screen_basis_online_vs_retrospective_rows(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    _news_family(conn, "fam-basis", T0 + 100 * BIN + 42)
    a = screen_news_impact(
        conn, model.mode_run_id, assignment_basis="online_filtered"
    )
    b = screen_news_impact(
        conn, model.mode_run_id,
        assignment_basis="retrospective_smoothed",
    )
    assert a["assignment_basis"] == "online_filtered"
    assert b["assignment_basis"] == "retrospective_smoothed"
    bases = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT assignment_basis FROM news_impact_screens"
        )
    }
    assert bases == {"online_filtered", "retrospective_smoothed"}
    with pytest.raises(ValueError):
        screen_news_impact(conn, model.mode_run_id, "whenever")


def test_asof_accessor_filters_by_basis(conn):
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn)
    _news_family(conn, "fam-asof", T0 + 100 * BIN + 42)
    screen_news_impact(
        conn, model.mode_run_id,
        assignment_basis="retrospective_smoothed",
    )
    cutoff = T0 + 200 * BIN
    online_rows = impactful_news_asof(
        conn, COND, cutoff, model.mode_run_id
    )
    assert online_rows == []                # no online screens exist yet
    screen_news_impact(conn, model.mode_run_id)
    assert impactful_news_asof(conn, COND, cutoff, model.mode_run_id)


def test_online_screen_requires_model_to_predate_news(conn):
    """A model trained after an event must not claim to have screened
    it online; retrospective screening is exempt by definition."""
    _regime_world(conn, event_windows=((30, 40), (100, 112)))
    model = _fitted(conn, fit_cutoff=T0 + 150 * BIN)  # AFTER the news
    _news_family(conn, "fam-past", T0 + 100 * BIN + 42)
    online = screen_news_impact(conn, model.mode_run_id)
    assert online["model_unavailable"] == 1
    assert online["screened"] == 0
    row = conn.execute(
        "SELECT screen_status, transition_detected, model_effective_from "
        "FROM news_impact_screens WHERE assignment_basis = "
        "'online_filtered'"
    ).fetchone()
    assert row["screen_status"] == "model_unavailable"
    assert row["transition_detected"] == 0
    assert row["model_effective_from"] == T0 + 150 * BIN
    # never surfaced by the online as-of accessor
    assert impactful_news_asof(
        conn, COND, T0 + 500 * BIN, model.mode_run_id
    ) == []
    # retrospective basis screens it, labelled as such
    retro = screen_news_impact(
        conn, model.mode_run_id,
        assignment_basis="retrospective_smoothed",
    )
    assert retro["screened"] == 1 and retro["impactful"] == 1


def test_single_clear_boundary_is_partial_not_negative(conn):
    """calm->calm on the only observable boundary must NOT count as a
    reliable non-impact: the missing boundary could have jumped."""
    _regime_world(conn, n_bins=90, event_windows=((30, 40),))
    model = _fitted(conn)
    # news in the LAST assigned bin: post bin missing, pre->arrival
    # observable and calm->calm
    _news_family(conn, "fam-edge", T0 + 89 * BIN + 10)
    counters = screen_news_impact(conn, model.mode_run_id)
    assert counters["partial_coverage"] == 1
    row = conn.execute(
        "SELECT screen_status, transition_detected FROM "
        "news_impact_screens WHERE event_family_id = 'fam-edge' "
        "AND assignment_basis = 'online_filtered'"
    ).fetchone()
    assert tuple(row) == ("partial_coverage", 0)
    # but a single observable boundary that DOES transition is still
    # impactful: coverage starts at the arrival bin (pre missing), and
    # the after-boundary jumps calm->event
    conn.execute("DELETE FROM news_impact_screens")
    _insert_bar(                       # arrival bin: calm profile
        conn, "0xedge", T0 + 200 * BIN, spread_ticks=1.2,
        turnover=100.0, volatility=0.01, best_size=200.0,
    )
    for i in (201, 202):               # then a genuine event
        _insert_bar(
            conn, "0xedge", T0 + i * BIN, spread_ticks=2.4,
            turnover=600.0, volatility=0.06, best_size=100.0,
        )
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "('m-edge', '0xedge', 'Q?', 1, 0, 'h', 'p', 2, 1)",
    )
    conn.commit()
    model2 = _fitted(conn)
    _news_family(conn, "fam-edge-hit", T0 + 200 * BIN + 10,
                 market_id="m-edge")
    screen_news_impact(conn, model2.mode_run_id)
    row = conn.execute(
        "SELECT transition_detected, screen_status, pre_mode_label "
        "FROM news_impact_screens WHERE event_family_id = "
        "'fam-edge-hit' AND assignment_basis = 'online_filtered'"
    ).fetchone()
    assert row["pre_mode_label"] is None            # truly one boundary
    assert (row["transition_detected"], row["screen_status"]) \
        == (1, "screened")
