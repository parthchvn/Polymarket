"""29.10 model tests."""

import numpy as np
import pytest

from polymarket.analysis.features import ALL_FEATURES
from polymarket.analysis.models import (
    accuracy,
    brier_score,
    calibration_bins,
    chronological_folds,
    evaluate_nested_models,
    log_loss,
)


def _rows(n, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels, times, ids = [], [], [], []
    for i in range(n):
        row = {name: 0.0 for name in ALL_FEATURES}
        signal = rng.normal()
        row["mkt_last_price"] = 0.5 + 0.1 * signal
        row["news_rel_max"] = max(0.0, signal)
        rows.append(row)
        labels.append(1.0 if signal > 0 else -1.0)
        times.append(1000.0 + i * 60.0)
        ids.append(f"d{i}")
    return rows, labels, times, ids


def test_chronological_folds_are_ordered_with_embargo():
    times = np.array([100.0 + i * 10 for i in range(20)])
    folds = chronological_folds(times, n_folds=3, embargo_seconds=15.0)
    assert folds
    for fold in folds:
        train_times = times[fold.train_indices]
        eval_times = times[fold.eval_indices]
        assert train_times.max() < eval_times.min()
        # embargo: no training point within 15s before evaluation start
        assert eval_times.min() - train_times.max() >= 15.0


def test_no_random_split_deterministic_predictions():
    rows, labels, times, ids = _rows(24)
    a = evaluate_nested_models(rows, labels, times, ids)
    b = evaluate_nested_models(rows, labels, times, ids)
    assert a.per_decision == b.per_decision
    assert a.metrics == b.metrics


def test_metric_calculations():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    assert accuracy(y, p) == 1.0
    assert brier_score(y, p) == pytest.approx(np.mean([0.01, 0.01, 0.04, 0.04]))
    assert log_loss(y, p) == pytest.approx(
        -np.mean(np.log([0.9, 0.9, 0.8, 0.8]))
    )
    bins = calibration_bins(y, p, n_bins=10)
    assert sum(b["count"] for b in bins) == 4


def test_improvement_reported_and_folds_saved():
    rows, labels, times, ids = _rows(30)
    result = evaluate_nested_models(rows, labels, times, ids, n_folds=3)
    assert "m2_to_m3_log_loss" in result.improvements
    assert result.folds and all("train_end" in f for f in result.folds)


def test_placebo_seeds_recorded():
    from polymarket.analysis.placebos import run_placebo_suite

    rows, labels, times, ids = _rows(30)
    baseline = evaluate_nested_models(rows, labels, times, ids)
    suite = run_placebo_suite(
        rows, labels, times, ids,
        ["m"] * len(rows), ["a"] * len(rows),
        baseline=baseline, seed=99,
    )
    names = {r.name for r in suite.results}
    assert {"shuffled_event_market_links", "pseudo_event_times",
            "irrelevant_market_news", "actor_permutation",
            "future_lead_diagnostic"} <= names
    assert all(isinstance(r.seed, int) for r in suite.results)
