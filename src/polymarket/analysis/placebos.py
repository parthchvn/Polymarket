"""Placebo tests and negative controls.

Each placebo perturbs the news channel (or actor assignment) with a
recorded seed and re-evaluates the nested suite; a real news effect
should shrink toward zero under the placebos while the future-lead
diagnostic deliberately uses shifted future information as a leakage
detector (labelled diagnostic only, never a headline result).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from polymarket.analysis.features import FEATURE_GROUPS
from polymarket.analysis.models import EvaluationResult, evaluate_nested_models

NEWS_FEATURES = FEATURE_GROUPS["news"]

EvalFn = Callable[[list[dict]], EvaluationResult]


@dataclass
class PlaceboResult:
    name: str
    seed: int
    m2_to_m3_log_loss: float
    m2_to_m3_brier: float
    note: str = ""


@dataclass
class PlaceboSuiteResult:
    baseline_m2_to_m3_log_loss: float
    results: list[PlaceboResult] = field(default_factory=list)


def _evaluate(
    rows: list[dict], labels, times, ids, *, n_folds: int, embargo: float
) -> EvaluationResult:
    return evaluate_nested_models(
        rows, labels, times, ids, n_folds=n_folds, embargo_seconds=embargo
    )


def _swap_news(rows: list[dict], permutation: list[int]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        source = rows[permutation[i]]
        for name in NEWS_FEATURES:
            new_row[name] = source[name]
        out.append(new_row)
    return out


def run_placebo_suite(
    feature_rows: list[dict],
    labels: list[float],
    times: list[float],
    decision_ids: list[str],
    market_ids: list[str],
    actor_ids: list[str],
    *,
    baseline: EvaluationResult,
    seed: int = 1337,
    n_folds: int = 3,
    embargo_seconds: float = 0.0,
) -> PlaceboSuiteResult:
    suite = PlaceboSuiteResult(
        baseline_m2_to_m3_log_loss=baseline.improvements["m2_to_m3_log_loss"]
    )
    n = len(feature_rows)

    def record(name: str, rows: list[dict], note: str, sub_seed: int) -> None:
        try:
            res = _evaluate(
                rows, labels, times, decision_ids,
                n_folds=n_folds, embargo=embargo_seconds,
            )
            suite.results.append(
                PlaceboResult(
                    name=name, seed=sub_seed,
                    m2_to_m3_log_loss=res.improvements["m2_to_m3_log_loss"],
                    m2_to_m3_brier=res.improvements["m2_to_m3_brier"],
                    note=note,
                )
            )
        except ValueError as exc:
            suite.results.append(
                PlaceboResult(name=name, seed=sub_seed,
                              m2_to_m3_log_loss=float("nan"),
                              m2_to_m3_brier=float("nan"),
                              note=f"skipped: {exc}")
            )

    # 26.1 shuffled event-market links: permute news features across
    # decisions in different markets (preserves marginal timing structure)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    record(
        "shuffled_event_market_links", _swap_news(feature_rows, perm),
        "news features reassigned across markets", seed,
    )

    # 26.2 pseudo-event times: shift news-age features within valid windows
    rng2 = random.Random(seed + 1)
    shifted = []
    for row in feature_rows:
        new_row = dict(row)
        if new_row.get("news_missing", 1.0) == 0.0:
            new_row["news_age_hours"] = new_row["news_age_hours"] + rng2.uniform(1.0, 12.0)
        shifted.append(new_row)
    record("pseudo_event_times", shifted, "news ages shifted earlier in time", seed + 1)

    # 26.3 irrelevant-market news: blank every relevance-derived channel
    # (raw and decayed) while keeping article counts, as if all observed
    # news had been judged irrelevant to the market.
    relevance_channels = tuple(
        name for name in NEWS_FEATURES
        if not name.endswith("_missing")
        and name not in ("news_article_count", "news_source_diversity",
                         "news_ingestion_lag")
    )
    irrelevant = []
    for row in feature_rows:
        new_row = dict(row)
        for name in relevance_channels:
            new_row[name] = 0.0
        for name in ("news_missing", "news_recent_missing",
                     "news_decay_missing"):
            new_row[name] = 1.0
        irrelevant.append(new_row)
    record("irrelevant_market_news", irrelevant,
           "relevance channel replaced by irrelevant-news coding", seed)

    # 26.4 actor permutation within time strata: permute actor features
    rng3 = random.Random(seed + 2)
    order = sorted(range(n), key=lambda i: times[i])
    actor_features = FEATURE_GROUPS["actor"] + ["base_actor_positive_rate",
                                                "base_actor_positive_rate_missing"]
    permuted_rows = [dict(r) for r in feature_rows]
    stratum = 4
    for start in range(0, n, stratum):
        block = order[start:start + stratum]
        shuffled_block = block[:]
        rng3.shuffle(shuffled_block)
        for dst, src in zip(block, shuffled_block):
            for name in actor_features:
                permuted_rows[dst][name] = feature_rows[src][name]
    record("actor_permutation", permuted_rows,
           "actor features permuted within time strata", seed + 2)

    # 26.5 future-lead negative control (diagnostic only): give each
    # decision the news features of the NEXT decision in time.  A large
    # apparent gain here indicates leakage sensitivity, not a result.
    future_perm = list(range(n))
    for pos in range(n - 1):
        future_perm[order[pos]] = order[pos + 1]
    future_perm[order[-1]] = order[-1]
    record("future_lead_diagnostic", _swap_news(feature_rows, future_perm),
           "DIAGNOSTIC ONLY: deliberately future-shifted news features", seed)

    return suite
