"""Layer 1: per-decision PREDICTIVE DRIVER ATTRIBUTION.

This is deliberately named predictive driver attribution, not
"reasoning": it answers "which feature channels moved the model's
probability of the observed action?", which does not by itself identify
the mechanism.  A large news contribution could reflect immediate
reaction, delayed underreaction, continuation of a news-caused price
move, or correlation with an unobserved signal.  Mechanism inference
(template posteriors) is a separate, later layer; the schema and record
shape here are structured to support it (template fields exist but stay
NULL in Layer 1).

Channels
--------
The news channel is split so a fresh reaction and delayed underreaction
remain distinguishable:

* ``fresh_news``       — raw 24h-window components + the 6h-half-life
  decayed signals (short ``g_fresh`` kernel);
* ``persistent_news``  — the 24h/72h/168h decayed signals (slow drift
  kernel) and the decay missingness flags;
* ``liquidity``        — spread/depth/imbalance/execution-rate, split out
  of general market trend;
* ``market_trend``, ``position``, ``actor``, ``base`` — as before.

Method
------
For each chronological fold (same expanding-window + embargo discipline
as the nested suite), a full L2 logistic model is fitted on all
features, and one ablated model per channel is fitted with that
channel's features removed.  For every evaluation-block decision we
report:

* exact standardized logit contributions per channel from the full
  model:  c_ig = sum_{k in g} beta_k * x_std_ik;
* refit group-ablation deltas:
  Delta_ig = log P(D_i | C_i) - log P(D_i | C_i minus C_g),
  which double as counterfactual results ("what if this channel were
  uninformative?").

Temporal validity: every model that scores a decision is trained only on
strictly earlier decisions (embargoed), and all evidence fields are
computed from the strict pre-decision context — post-trade price or
liquidity responses never appear in a decision's record.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polymarket.analysis.context import DecisionContext
from polymarket.analysis.features import (
    FEATURE_GROUPS,
    NEWS_DECAY_MAX_AGE,
    decayed_news_signals,
    relevance_confidence,
)
from polymarket.analysis.models import Fold, LogisticModel, chronological_folds
from polymarket.collection.canonical import canonical_json, namespace_id

REASONING_METHOD_VERSION = "driver-attribution-1.0.0"

# Partition of ALL_FEATURES into attribution channels.  fresh vs
# persistent news are separated on purpose (see module docstring).
ATTRIBUTION_GROUPS: dict[str, list[str]] = {
    "base": list(FEATURE_GROUPS["base"]),
    "actor": list(FEATURE_GROUPS["actor"]),
    "market_trend": [
        "mkt_last_price", "mkt_last_price_missing", "mkt_return_short",
        "mkt_return_long", "mkt_volatility",
        "mkt_state_from_executions", "mkt_volume",
    ],
    "liquidity": [
        "mkt_spread", "mkt_spread_missing", "mkt_depth", "mkt_imbalance",
        "mkt_execution_rate",
    ],
    "position": list(FEATURE_GROUPS["position"]),
    "fresh_news": [
        "news_rel_max", "news_rel_sum", "news_direction",
        "news_novelty_max", "news_surprise_max", "news_age_hours",
        "news_source_diversity", "news_confirmation_count",
        "news_contradiction_count", "news_article_count",
        "news_ingestion_lag", "news_recent_missing",
        "news_decay_signed_6h", "news_decay_positive_6h",
        "news_decay_negative_6h",
    ],
    "persistent_news": [
        "news_missing", "news_decay_missing",
        "news_decay_signed_24h", "news_decay_positive_24h",
        "news_decay_negative_24h",
        "news_decay_signed_72h", "news_decay_positive_72h",
        "news_decay_negative_72h",
        "news_decay_signed_168h", "news_decay_positive_168h",
        "news_decay_negative_168h",
    ],
}

ATTRIBUTION_FEATURES = [n for g in ATTRIBUTION_GROUPS.values() for n in g]

STATUS_VALUES = (
    "accepted", "ambiguous", "insufficient_context",
    "attribution_template_disagreement", "counterfactual_failure",
)

_EPS = 1e-12
_NULL_QUANTILE = 0.95  # family-wise permutation-null quantile


@dataclass(frozen=True)
class AttributionConfig:
    """Acceptance rules for driver attribution.  All values are recorded
    in config.json and the reasoning manifest — no hidden thresholds."""

    min_ablation_delta: float = 0.05
    min_margin_ratio: float = 0.15
    min_stability: float = 0.6
    n_stability_refits: int = 4
    ambiguity_entropy_threshold: float = 1.8
    n_null_permutations: int = 3
    # an attribution claim from an underdetermined fit (fewer training
    # rows than min_train_rows, default > n_features) is never trusted
    min_train_rows: int = 55
    # the claimed channel must be informative FOLD-WIDE: its mean
    # ablation delta across the fold's evaluation decisions must be
    # material (on noise, ablation helps held-out points on average)
    min_fold_mean_delta: float = 0.02


DEFAULT_ATTRIBUTION_CONFIG = AttributionConfig()


@dataclass
class DriverAttribution:
    decision_id: str
    reasoning_run_id: str
    observed_action_probability: float
    logit_contributions: dict[str, float]
    group_attributions: dict[str, float]  # refit ablation log-prob deltas
    top_evidence: list[dict[str, Any]]
    primary_channel: str | None
    status: str
    confidence: float                    # = attribution_confidence
    fold_index: int | None = None
    notes: list[str] = field(default_factory=list)
    # separated confidences and diagnostics (never conflated):
    prediction_confidence: float = float("nan")
    attribution_confidence: float = 0.0
    top_ablation_delta: float = float("nan")
    top_vs_second_margin: float = float("nan")
    attribution_stability: float = float("nan")
    intercept: float = float("nan")
    total_logit: float = float("nan")
    reconstructed_probability: float = float("nan")
    fit_ok: bool = False
    coverage_complete: bool = True


# ---------------------------------------------------------------------------
# Strict pre-decision news evidence (no post-trade responses).
def news_evidence(
    context: DecisionContext,
    *,
    max_families: int = 3,
    max_age_seconds: float = NEWS_DECAY_MAX_AGE,
) -> list[dict[str, Any]]:
    """Per-event-family evidence computed only from the strict context.

    ``aligned_move_since_news`` is the positive-price movement between the
    last market state observed BEFORE the family became available and the
    last market state observed BEFORE the decision, signed by the news
    direction.  Both endpoints precede the decision time, so no post-trade
    absorption leaks in.  Components are reported separately; no combined
    "absorbed fraction" is invented.
    """
    t = context.decision_time
    family_available: dict[str, float] = {
        fam["event_family_id"]: fam["earliest_available_at"]
        for fam in context.event_families
    }
    prices = [
        (s["ts"], s["positive_price"])
        for s in context.market_state
        if s["positive_price"] is not None
    ]

    by_family: dict[str, list[Any]] = {}
    for row in context.relevance:
        if row["rel_class"] == "irrelevant":
            continue
        age = t - row["computed_at"]
        if age <= 0 or age > max_age_seconds:
            continue
        by_family.setdefault(row["event_family_id"], []).append(row)

    evidence: list[dict[str, Any]] = []
    for family_id, rows in by_family.items():
        top = max(rows, key=lambda r: r["rel_score"])
        available_at = family_available.get(family_id, top["computed_at"])
        age_seconds = t - available_at
        fresh = decayed_news_signals(
            rows, decision_time=t, half_life_seconds=6 * 3600.0,
            max_age_seconds=max_age_seconds,
        )
        drift = decayed_news_signals(
            rows, decision_time=t, half_life_seconds=72 * 3600.0,
            max_age_seconds=max_age_seconds,
        )
        pre_news = [p for ts, p in prices if ts < available_at]
        pre_decision = [p for ts, p in prices]
        aligned_move = None
        if pre_news and pre_decision:
            aligned_move = (pre_decision[-1] - pre_news[-1]) * top["direction"]
        evidence.append(
            {
                "event_family_id": family_id,
                "age_hours": age_seconds / 3600.0,
                "semantic_relevance": float(top["rel_score"]),
                "direction": float(top["direction"]),
                "judgment_confidence": relevance_confidence(top),
                "fresh_signal_signed": fresh["signed"],
                "persistent_signal_signed": drift["signed"],
                "aligned_move_since_news": aligned_move,
                "supporting_judgments": len(rows),
            }
        )
    evidence.sort(
        key=lambda e: -(abs(e["persistent_signal_signed"])
                        + abs(e["fresh_signal_signed"]))
    )
    return evidence[:max_families]


# ---------------------------------------------------------------------------
def _log_prob_observed(p_positive: float, label: float) -> float:
    p = p_positive if label > 0 else 1.0 - p_positive
    return float(np.log(max(p, _EPS)))


def _decomposition(model: LogisticModel, x: np.ndarray) -> dict[str, Any]:
    """Exact standardized logit decomposition: sigmoid(intercept + sum of
    channel contributions) reconstructs predict_proba to numerical
    tolerance (tested)."""
    x_std = model.standardizer.transform(x.reshape(1, -1))[0]
    weights = model.weights[1:]
    index = {name: i for i, name in enumerate(model.feature_names)}
    contributions: dict[str, float] = {}
    for channel, names in ATTRIBUTION_GROUPS.items():
        contributions[channel] = float(
            sum(weights[index[n]] * x_std[index[n]] for n in names)
        )
    intercept = float(model.weights[0])
    total_logit = intercept + sum(contributions.values())
    return {
        "intercept": intercept,
        "channel_logit_contributions": contributions,
        "total_logit": total_logit,
        "reconstructed_probability": float(
            1.0 / (1.0 + np.exp(-total_logit))
        ),
    }


def _fit_channel_models(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, l2: float
) -> tuple[LogisticModel, dict[str, tuple[LogisticModel, list[int]]]]:
    full = LogisticModel(feature_names=ATTRIBUTION_FEATURES, l2=l2).fit(
        X[train_idx], y[train_idx]
    )
    ablated: dict[str, tuple[LogisticModel, list[int]]] = {}
    for channel, names in ATTRIBUTION_GROUPS.items():
        keep = [i for i, n in enumerate(ATTRIBUTION_FEATURES) if n not in names]
        kept_names = [ATTRIBUTION_FEATURES[i] for i in keep]
        model = LogisticModel(feature_names=kept_names, l2=l2).fit(
            X[train_idx][:, keep], y[train_idx]
        )
        ablated[channel] = (model, keep)
    return full, ablated


def _deltas_for(
    full: LogisticModel,
    ablated: dict[str, tuple[LogisticModel, list[int]]],
    X: np.ndarray,
    y: np.ndarray,
    i: int,
) -> tuple[float, dict[str, float]]:
    p_full = float(full.predict_proba(X[i].reshape(1, -1))[0])
    logp_full = _log_prob_observed(p_full, y[i])
    deltas: dict[str, float] = {}
    for channel, (model, keep) in ablated.items():
        p_ab = float(model.predict_proba(X[i, keep].reshape(1, -1))[0])
        deltas[channel] = logp_full - _log_prob_observed(p_ab, y[i])
    return p_full, deltas


def _stability_resamples(
    train_idx: np.ndarray,
    n_resamples: int,
    block_ids: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Deterministic leave-one-block-out resamples of the training
    indices.  Blocks are contiguous time blocks by default, or explicit
    block ids (e.g. leave-one-world-out) when provided.  Individual rows
    are never randomly split."""
    if len(train_idx) < 4 or n_resamples <= 0:
        return []
    if block_ids is not None:
        unique = sorted(set(block_ids[train_idx].tolist()))
        resamples = []
        for block in unique[: max(n_resamples, len(unique))]:
            mask = block_ids[train_idx] != block
            if mask.sum() >= 2:
                resamples.append(train_idx[mask])
        return resamples
    blocks = np.array_split(np.arange(len(train_idx)), n_resamples)
    resamples = []
    for block in blocks:
        mask = np.ones(len(train_idx), dtype=bool)
        mask[block] = False
        if mask.sum() >= 2:
            resamples.append(train_idx[mask])
    return resamples


def _coverage_complete(coverage: dict[str, Any] | None) -> bool:
    if not coverage:
        return True
    if not coverage.get("position_history_complete", True):
        return False
    if coverage.get("unmapped_outcome_legs", 0):
        return False
    if coverage.get("blocking_gap_count", 0):
        return False
    return True


def _attribution_confidence(
    *,
    top_delta: float,
    margin_ratio: float,
    stability: float,
    fits_ok: bool,
    evidence_ok: bool,
    config: AttributionConfig,
) -> float:
    """Confidence in the CLAIMED DRIVER — distinct from prediction
    confidence.  Deterministic heuristic combining ablation magnitude,
    separation, refit stability, fit success and evidence availability;
    each factor saturates at twice its acceptance threshold."""
    if not fits_ok or not np.isfinite(top_delta) or top_delta <= 0:
        return 0.0
    magnitude = min(1.0, top_delta / max(2 * config.min_ablation_delta, _EPS))
    separation = min(1.0, margin_ratio / max(2 * config.min_margin_ratio, _EPS))
    stability_factor = stability if np.isfinite(stability) else 0.0
    evidence_factor = 1.0 if evidence_ok else 0.5
    return float(
        max(0.0, min(1.0, magnitude * separation * stability_factor
                     * evidence_factor))
    )


def run_driver_attribution(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    times: list[float],
    decision_ids: list[str],
    evidence_by_decision: dict[str, list[dict[str, Any]]],
    *,
    reasoning_run_id: str,
    n_folds: int = 3,
    embargo_seconds: float = 0.0,
    l2: float = 1.0,
    config: AttributionConfig = DEFAULT_ATTRIBUTION_CONFIG,
    coverage_by_decision: dict[str, dict[str, Any]] | None = None,
    folds: list[Fold] | None = None,
    stability_block_ids: np.ndarray | None = None,
) -> list[DriverAttribution]:
    y = np.asarray(labels, dtype=float)
    t = np.asarray(times, dtype=float)
    X_all = np.asarray(
        [[row[n] for n in ATTRIBUTION_FEATURES] for row in feature_rows],
        dtype=float,
    )
    coverage_by_decision = coverage_by_decision or {}
    if folds is None:
        folds = chronological_folds(
            t, n_folds=n_folds, embargo_seconds=embargo_seconds
        )

    records: dict[int, DriverAttribution] = {}
    for fold in folds:
        fold_channel_sums: dict[str, float] = {}
        fold_deltas_cache: dict[int, tuple[float, dict[str, float]]] = {}
        full, ablated = _fit_channel_models(X_all, y, fold.train_indices, l2)
        fits_ok = full.fit_ok and all(m.fit_ok for m, _ in ablated.values())
        resamples = _stability_resamples(
            fold.train_indices, config.n_stability_refits,
            block_ids=stability_block_ids,
        )
        resample_models = [
            _fit_channel_models(X_all, y, idx, l2) for idx in resamples
        ]
        # permutation-null reference: refit under deterministically
        # permuted training labels; an attribution is only trusted when
        # its observed top ablation delta exceeds every null top delta
        # for the same decision (guards against overfit noise channels)
        null_rng = np.random.default_rng(1000 + fold.fold_index)
        null_models = []
        for _ in range(max(0, config.n_null_permutations)):
            y_null = y.copy()
            y_null[fold.train_indices] = null_rng.permutation(
                y[fold.train_indices]
            )
            null_models.append(
                (_fit_channel_models(X_all, y_null, fold.train_indices, l2),
                 y_null)
            )
        # family-wise permutation-null threshold for this fold: pool the
        # top-channel deltas of EVERY eval decision under every permuted
        # refit; an observed attribution is only trusted when its delta
        # clears the whole null pool with a safety factor
        null_pool: list[float] = []
        for (n_full, n_ablated), y_null in null_models:
            if not (n_full.fit_ok
                    and all(m.fit_ok for m, _ in n_ablated.values())):
                continue
            for j in fold.eval_indices:
                _, n_deltas = _deltas_for(
                    n_full, n_ablated, X_all, y_null, int(j)
                )
                null_pool.append(max(n_deltas.values()))
        null_threshold = (
            float(np.quantile(np.asarray(null_pool), _NULL_QUANTILE))
            if null_pool else 0.0
        )

        for i in fold.eval_indices:
            i = int(i)
            fold_deltas_cache[i] = _deltas_for(full, ablated, X_all, y, i)
            for channel, delta in fold_deltas_cache[i][1].items():
                fold_channel_sums[channel] = (
                    fold_channel_sums.get(channel, 0.0) + delta
                )
        n_eval = max(1, len(fold.eval_indices))
        fold_channel_means = {
            channel: total / n_eval
            for channel, total in fold_channel_sums.items()
        }

        for i in fold.eval_indices:
            i = int(i)
            evidence = evidence_by_decision.get(decision_ids[i], [])
            coverage = coverage_by_decision.get(decision_ids[i])
            coverage_ok = _coverage_complete(coverage)
            p_full, deltas = fold_deltas_cache[i]
            decomposition = _decomposition(full, X_all[i])
            observed_p = p_full if y[i] > 0 else 1.0 - p_full

            ranked = sorted(deltas.items(), key=lambda kv: -kv[1])
            primary_channel, top_delta = ranked[0]
            second_delta = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = top_delta - second_delta
            margin_ratio = margin / top_delta if top_delta > 0 else 0.0

            decision_null_top = 0.0
            for (n_full, n_ablated), y_null in null_models:
                if not (n_full.fit_ok
                        and all(m.fit_ok for m, _ in n_ablated.values())):
                    continue
                _, n_deltas = _deltas_for(n_full, n_ablated, X_all, y_null, i)
                decision_null_top = max(
                    decision_null_top, max(n_deltas.values())
                )
            beats_null = top_delta > max(null_threshold, decision_null_top)
            fold_mean_ok = (
                fold_channel_means.get(primary_channel, 0.0)
                >= config.min_fold_mean_delta
            )

            # deterministic block-resample stability of the top channel
            agreements = 0
            valid = 0
            for r_full, r_ablated in resample_models:
                if not (r_full.fit_ok
                        and all(m.fit_ok for m, _ in r_ablated.values())):
                    continue
                _, r_deltas = _deltas_for(r_full, r_ablated, X_all, y, i)
                valid += 1
                if max(r_deltas, key=r_deltas.get) == primary_channel:
                    agreements += 1
            stability = agreements / valid if valid else 0.0

            evidence_ok = (
                bool(evidence)
                if primary_channel in ("fresh_news", "persistent_news")
                else True
            )
            notes: list[str] = []
            if not fits_ok:
                failed_full = not full.fit_ok
                status = (
                    "insufficient_context" if failed_full
                    else "counterfactual_failure"
                )
                primary_channel = None
                notes.append(
                    "optimiser failure in "
                    + ("full model" if failed_full else "an ablated model")
                    + "; attribution rejected"
                )
            elif top_delta <= 0:
                status = "counterfactual_failure"
                primary_channel = None
                notes.append(
                    "no channel ablation reduces the observed-action "
                    "probability; attribution not trusted"
                )
            elif not coverage_ok:
                status = "insufficient_context"
                primary_channel = None
                notes.append("context coverage incomplete")
            elif len(fold.train_indices) < config.min_train_rows:
                status = "insufficient_context"
                notes.append(
                    f"training rows {len(fold.train_indices)} below "
                    f"min_train_rows {config.min_train_rows}: "
                    "underdetermined fit is never trusted"
                )
            elif (
                top_delta < config.min_ablation_delta
                or margin_ratio < config.min_margin_ratio
                or stability < config.min_stability
                or not beats_null
                or not fold_mean_ok
                or not evidence_ok
            ):
                status = "ambiguous"
                notes.append(
                    "acceptance thresholds not met: "
                    f"delta={top_delta:.4f} margin_ratio={margin_ratio:.3f} "
                    f"stability={stability:.2f} beats_null={beats_null} "
                    f"fold_mean_ok={fold_mean_ok} evidence={evidence_ok}"
                )
            else:
                status = "accepted"

            attribution_conf = _attribution_confidence(
                top_delta=top_delta, margin_ratio=margin_ratio,
                stability=stability, fits_ok=fits_ok,
                evidence_ok=evidence_ok, config=config,
            )
            records[i] = DriverAttribution(
                decision_id=decision_ids[i],
                reasoning_run_id=reasoning_run_id,
                observed_action_probability=observed_p,
                logit_contributions=decomposition[
                    "channel_logit_contributions"
                ],
                group_attributions=deltas,
                top_evidence=evidence,
                primary_channel=primary_channel if status == "accepted"
                else (primary_channel if status == "ambiguous" else None),
                status=status,
                confidence=attribution_conf,
                fold_index=fold.fold_index,
                notes=notes,
                prediction_confidence=observed_p,
                attribution_confidence=attribution_conf,
                top_ablation_delta=top_delta,
                top_vs_second_margin=margin,
                attribution_stability=stability,
                intercept=decomposition["intercept"],
                total_logit=decomposition["total_logit"],
                reconstructed_probability=decomposition[
                    "reconstructed_probability"
                ],
                fit_ok=fits_ok,
                coverage_complete=coverage_ok,
            )

    covered = set(records)
    for i in range(len(decision_ids)):
        if i not in covered:
            records[i] = DriverAttribution(
                decision_id=decision_ids[i],
                reasoning_run_id=reasoning_run_id,
                observed_action_probability=float("nan"),
                logit_contributions={},
                group_attributions={},
                top_evidence=evidence_by_decision.get(decision_ids[i], []),
                primary_channel=None,
                status="insufficient_context",
                confidence=0.0,
                notes=["decision precedes the first evaluation block"],
            )
    return [records[i] for i in sorted(records)]


# ---------------------------------------------------------------------------
def persist_driver_attributions(
    conn: sqlite3.Connection,
    attributions: list[DriverAttribution],
    *,
    feature_version: str,
) -> int:
    now = time.time()
    inserted = 0
    for record in attributions:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO reasoning_judgments
                (reasoning_judgment_id, decision_id, reasoning_run_id,
                 primary_template, template_posterior_json,
                 driver_attribution_json, evidence_json, counterfactual_json,
                 rationale_text, agreement_score, confidence, status,
                 model_version, feature_version, computed_at)
            VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
            """,
            (
                namespace_id(
                    "reasoning", record.reasoning_run_id, record.decision_id
                ),
                record.decision_id,
                record.reasoning_run_id,
                canonical_json(
                    {
                        "observed_action_probability": (
                            None
                            if np.isnan(record.observed_action_probability)
                            else record.observed_action_probability
                        ),
                        "primary_channel": record.primary_channel,
                        "logit_contributions": record.logit_contributions,
                        "group_attributions": record.group_attributions,
                        "notes": record.notes,
                    }
                ),
                canonical_json(record.top_evidence),
                canonical_json(record.group_attributions),
                record.confidence,
                record.status,
                REASONING_METHOD_VERSION,
                feature_version,
                now,
            ),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def attribution_report(attributions: list[DriverAttribution]) -> list[dict]:
    out = []
    for record in attributions:
        out.append(
            {
                "decision_id": record.decision_id,
                "status": record.status,
                "primary_channel": record.primary_channel,
                "observed_action_probability": (
                    None
                    if np.isnan(record.observed_action_probability)
                    else record.observed_action_probability
                ),
                "group_attributions": record.group_attributions,
                "logit_contributions": record.logit_contributions,
                "top_evidence": record.top_evidence,
                "confidence": record.confidence,
                "prediction_confidence": (
                    None if np.isnan(record.prediction_confidence)
                    else record.prediction_confidence
                ),
                "attribution_confidence": record.attribution_confidence,
                "top_ablation_delta": (
                    None if np.isnan(record.top_ablation_delta)
                    else record.top_ablation_delta
                ),
                "top_vs_second_margin": (
                    None if np.isnan(record.top_vs_second_margin)
                    else record.top_vs_second_margin
                ),
                "attribution_stability": (
                    None if np.isnan(record.attribution_stability)
                    else record.attribution_stability
                ),
                "intercept": (
                    None if np.isnan(record.intercept) else record.intercept
                ),
                "total_logit": (
                    None if np.isnan(record.total_logit)
                    else record.total_logit
                ),
                "reconstructed_probability": (
                    None if np.isnan(record.reconstructed_probability)
                    else record.reconstructed_probability
                ),
                "fit_ok": record.fit_ok,
                "coverage_complete": record.coverage_complete,
                "fold_index": record.fold_index,
                "notes": record.notes,
                "label": "predictive driver attribution (not mechanism "
                         "inference)",
            }
        )
    return out


def load_reasoning_judgments(
    conn: sqlite3.Connection, reasoning_run_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM reasoning_judgments WHERE reasoning_run_id = ? "
        "ORDER BY decision_id",
        (reasoning_run_id,),
    ).fetchall()
    out = []
    for row in rows:
        record = dict(row)
        for key in ("driver_attribution_json", "evidence_json",
                    "counterfactual_json"):
            if record.get(key):
                record[key] = json.loads(record[key])
        out.append(record)
    return out
