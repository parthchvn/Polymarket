"""Layer 1 validity: optimiser checks, acceptance gates, stability,
exact logit reconstruction."""

from __future__ import annotations

import numpy as np

from polymarket.analysis.models import LogisticModel
from polymarket.analysis.reasoning import (
    ATTRIBUTION_FEATURES,
    AttributionConfig,
    _decomposition,
    run_driver_attribution,
)

RNG = np.random.default_rng(7)


def _rows(X: np.ndarray) -> list[dict[str, float]]:
    return [
        {name: float(x[i]) for i, name in enumerate(ATTRIBUTION_FEATURES)}
        for x in X
    ]


def _base_dataset(n: int = 40, informative: bool = True):
    X = RNG.normal(size=(n, len(ATTRIBUTION_FEATURES)))
    index = ATTRIBUTION_FEATURES.index("news_rel_max")
    if informative:
        y = np.where(X[:, index] > 0, 1.0, -1.0)
    else:
        y = RNG.choice([-1.0, 1.0], size=n)  # pure noise: null signal
    times = np.arange(n, dtype=float) * 3600.0
    ids = [f"d{i}" for i in range(n)]
    return X, y, times, ids


def test_fit_diagnostics_reject_failed_optimiser():
    X = RNG.normal(size=(30, 3))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    model = LogisticModel(feature_names=["a", "b", "c"]).fit(
        X, y, max_iter=0
    )
    assert model.diagnostics is not None
    assert not model.fit_ok  # zero iterations => optimiser did not converge


def test_fit_diagnostics_reject_non_finite_inputs():
    X = RNG.normal(size=(30, 3))
    X[5, 1] = np.nan
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    model = LogisticModel(feature_names=["a", "b", "c"]).fit(X, y)
    assert not model.fit_ok


def test_failed_optimiser_cannot_produce_accepted_attribution():
    X, y, times, ids = _base_dataset()
    X[3, 0] = np.nan  # poisons every fit that includes this row
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, {}, reasoning_run_id="t"
    )
    assert records
    for record in records:
        assert record.status != "accepted"
        assert record.status in ("insufficient_context", "counterfactual_failure")
        assert record.primary_channel is None


def test_null_signal_produces_no_accepted_attributions():
    # even with plenty of rows, pure-noise labels never yield accepts
    X, y, times, ids = _base_dataset(n=180, informative=False)
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, {}, reasoning_run_id="t"
    )
    assert all(record.status != "accepted" for record in records)


def test_underdetermined_fit_is_never_trusted():
    # fewer training rows than min_train_rows => insufficient_context,
    # regardless of how large the apparent ablation delta is
    X, y, times, ids = _base_dataset(n=40, informative=True)
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, {}, reasoning_run_id="t"
    )
    assert records
    assert all(record.status != "accepted" for record in records)


def test_informative_signal_with_enough_rows_is_accepted():
    # positive control: the gates must not be vacuous
    X, y, times, ids = _base_dataset(n=200, informative=True)
    evidence = {
        decision_id: [{"event_family_id": "f", "direction": 1,
                       "age_hours": 2.0}]
        for decision_id in ids
    }
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, evidence,
        reasoning_run_id="t",
    )
    accepted = [r for r in records if r.status == "accepted"]
    assert accepted
    assert all(
        record.primary_channel == "fresh_news" for record in accepted
    )


def test_tiny_ablation_delta_fails_threshold():
    X, y, times, ids = _base_dataset(n=200, informative=True)
    strict = AttributionConfig(
        min_ablation_delta=1e9, min_margin_ratio=0.0,
        min_stability=0.0, n_stability_refits=2,
        ambiguity_entropy_threshold=1.8, min_train_rows=1,
    )
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, {},
        reasoning_run_id="t", config=strict,
    )
    evaluated = [r for r in records if r.status != "insufficient_context"]
    assert evaluated
    assert all(record.status != "accepted" for record in evaluated)


def test_equal_correlated_channels_become_ambiguous():
    n = 200
    X = RNG.normal(size=(n, len(ATTRIBUTION_FEATURES))) * 0.01
    signal = RNG.normal(size=n)
    # identical signal placed in two different channels
    X[:, ATTRIBUTION_FEATURES.index("news_rel_max")] = signal
    X[:, ATTRIBUTION_FEATURES.index("mkt_return_short")] = signal
    y = np.where(signal > 0, 1.0, -1.0)
    times = np.arange(n, dtype=float) * 3600.0
    records = run_driver_attribution(
        _rows(X), list(y), list(times), [f"d{i}" for i in range(n)], {},
        reasoning_run_id="t",
    )
    evaluated = [r for r in records if r.group_attributions]
    assert evaluated
    # ablating either channel alone barely matters: never accepted
    assert all(record.status != "accepted" for record in evaluated)
    assert any(record.status == "ambiguous" for record in evaluated)


def test_stability_falls_under_unstable_resamples():
    n = 200
    X = RNG.normal(size=(n, len(ATTRIBUTION_FEATURES))) * 0.01
    news = ATTRIBUTION_FEATURES.index("news_rel_max")
    trend = ATTRIBUTION_FEATURES.index("mkt_return_short")
    y = np.empty(n)
    # the informative channel ALTERNATES between contiguous time blocks,
    # so leave-one-block-out refits disagree about the top channel
    for block in range(4):
        rows = slice(block * (n // 4), (block + 1) * (n // 4))
        signal = RNG.normal(size=n // 4)
        column = news if block % 2 == 0 else trend
        X[rows, column] = signal * 3
        y[rows] = np.where(signal > 0, 1.0, -1.0)
    times = np.arange(n, dtype=float) * 3600.0
    records = run_driver_attribution(
        _rows(X), list(y), list(times), [f"d{i}" for i in range(n)], {},
        reasoning_run_id="t",
    )
    stabilities = [
        record.attribution_stability for record in records
        if np.isfinite(record.attribution_stability)
    ]
    assert stabilities
    assert min(stabilities) < 1.0  # instability is detected, not hidden


def test_logit_decomposition_reconstructs_probability():
    X, y, _times, _ids = _base_dataset()
    model = LogisticModel(feature_names=ATTRIBUTION_FEATURES).fit(X, y)
    for i in range(0, len(X), 7):
        decomposition = _decomposition(model, X[i])
        direct = float(model.predict_proba(X[i].reshape(1, -1))[0])
        reconstructed = 1.0 / (1.0 + np.exp(
            -(decomposition["intercept"]
              + sum(decomposition["channel_logit_contributions"].values()))
        ))
        assert abs(reconstructed - direct) < 1e-9
        assert abs(decomposition["reconstructed_probability"] - direct) < 1e-9


def test_prediction_and_attribution_confidence_are_separate():
    X, y, times, ids = _base_dataset(n=200)
    records = run_driver_attribution(
        _rows(X), list(y), list(times), ids, {}, reasoning_run_id="t"
    )
    evaluated = [r for r in records if r.group_attributions]
    assert any(
        abs(r.prediction_confidence - r.attribution_confidence) > 1e-6
        for r in evaluated
    )
    for record in evaluated:
        assert 0.0 <= record.attribution_confidence <= 1.0
