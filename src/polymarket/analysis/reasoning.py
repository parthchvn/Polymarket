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
from polymarket.analysis.models import LogisticModel, chronological_folds
from polymarket.collection.canonical import canonical_json, namespace_id

REASONING_METHOD_VERSION = "driver-attribution-1.0.0"

# Partition of ALL_FEATURES into attribution channels.  fresh vs
# persistent news are separated on purpose (see module docstring).
ATTRIBUTION_GROUPS: dict[str, list[str]] = {
    "base": list(FEATURE_GROUPS["base"]),
    "actor": list(FEATURE_GROUPS["actor"]),
    "market_trend": [
        "mkt_last_price", "mkt_last_price_missing", "mkt_return_short",
        "mkt_return_long", "mkt_volatility", "mkt_volume",
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

_AMBIGUITY_MARGIN = 0.2  # top-two ablation deltas within 20% => ambiguous
_EPS = 1e-12


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
    confidence: float
    fold_index: int | None = None
    notes: list[str] = field(default_factory=list)


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


def _channel_logit_contributions(
    model: LogisticModel, x: np.ndarray
) -> dict[str, float]:
    x_std = model.standardizer.transform(x.reshape(1, -1))[0]
    weights = model.weights[1:]  # skip intercept
    index = {name: i for i, name in enumerate(model.feature_names)}
    out: dict[str, float] = {}
    for channel, names in ATTRIBUTION_GROUPS.items():
        out[channel] = float(
            sum(weights[index[n]] * x_std[index[n]] for n in names)
        )
    return out


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
) -> list[DriverAttribution]:
    y = np.asarray(labels, dtype=float)
    t = np.asarray(times, dtype=float)
    X_all = np.asarray(
        [[row[n] for n in ATTRIBUTION_FEATURES] for row in feature_rows],
        dtype=float,
    )
    folds = chronological_folds(t, n_folds=n_folds, embargo_seconds=embargo_seconds)

    records: dict[int, DriverAttribution] = {}
    for fold in folds:
        full = LogisticModel(feature_names=ATTRIBUTION_FEATURES, l2=l2).fit(
            X_all[fold.train_indices], y[fold.train_indices]
        )
        ablated: dict[str, tuple[LogisticModel, list[int]]] = {}
        for channel, names in ATTRIBUTION_GROUPS.items():
            keep = [
                i for i, n in enumerate(ATTRIBUTION_FEATURES) if n not in names
            ]
            kept_names = [ATTRIBUTION_FEATURES[i] for i in keep]
            model = LogisticModel(feature_names=kept_names, l2=l2).fit(
                X_all[fold.train_indices][:, keep], y[fold.train_indices]
            )
            ablated[channel] = (model, keep)

        for i in fold.eval_indices:
            i = int(i)
            p_full = float(full.predict_proba(X_all[i].reshape(1, -1))[0])
            logp_full = _log_prob_observed(p_full, y[i])
            deltas: dict[str, float] = {}
            for channel, (model, keep) in ablated.items():
                p_ab = float(
                    model.predict_proba(X_all[i, keep].reshape(1, -1))[0]
                )
                deltas[channel] = logp_full - _log_prob_observed(p_ab, y[i])
            observed_p = p_full if y[i] > 0 else 1.0 - p_full

            ranked = sorted(deltas.items(), key=lambda kv: -kv[1])
            primary_channel, top_delta = ranked[0]
            status = "accepted"
            notes: list[str] = []
            if top_delta <= 0:
                status = "counterfactual_failure"
                primary_channel = None
                notes.append(
                    "no channel ablation reduces the observed-action "
                    "probability; attribution not trusted"
                )
            elif (
                len(ranked) > 1
                and ranked[1][1] > 0
                and (top_delta - ranked[1][1]) <= _AMBIGUITY_MARGIN * top_delta
            ):
                status = "ambiguous"
                notes.append(
                    f"top channels {ranked[0][0]!r} and {ranked[1][0]!r} "
                    "are within the ambiguity margin"
                )
            records[i] = DriverAttribution(
                decision_id=decision_ids[i],
                reasoning_run_id=reasoning_run_id,
                observed_action_probability=observed_p,
                logit_contributions=_channel_logit_contributions(
                    full, X_all[i]
                ),
                group_attributions=deltas,
                top_evidence=evidence_by_decision.get(decision_ids[i], []),
                primary_channel=primary_channel,
                status=status,
                confidence=observed_p,
                fold_index=fold.fold_index,
                notes=notes,
            )

    # decisions never inside an evaluation block (initial training burn-in)
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
