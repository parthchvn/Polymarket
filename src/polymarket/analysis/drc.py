"""Structured (D, C, R) record assembly, agreement scoring, persistence.

R is a calibrated posterior over structured reasoning hypotheses most
consistent with the observed decision D and the strict pre-decision
context C under the fitted model.  It is a BEHAVIOURAL INFERENCE and is
never a claim about the trader's private thoughts.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import numpy as np

from polymarket.analysis.decisions import DecisionEpisode
from polymarket.analysis.reasoning import DriverAttribution
from polymarket.analysis.reasoning_counterfactuals import (
    DEFAULT_MATERIALITY_DELTA,
    CounterfactualResult,
    counterfactual_report,
    required_counterfactuals_pass,
)
from polymarket.analysis.reasoning_posterior import TemplatePosterior
from polymarket.analysis.reasoning_templates import TEMPLATES
from polymarket.collection.canonical import canonical_json, namespace_id

STATUS_ORDER = (
    "insufficient_context",
    "counterfactual_failure",
    "attribution_template_disagreement",
    "ambiguous",
    "accepted",
)


# ---------------------------------------------------------------------------
def evidence_flags(
    features: dict[str, float],
    episode: DecisionEpisode,
    evidence: list[dict] | None,
) -> set[str]:
    """Which required-evidence tokens are satisfied by the strict context."""
    flags: set[str] = set()
    evidence = evidence or []
    ages = [e["age_hours"] for e in evidence]
    if any(a < 24.0 for a in ages):
        flags.add("recent_relevant_news")
    if any(24.0 <= a < 28 * 24.0 for a in ages):
        flags.add("aged_relevant_news")
    if abs(features.get("mkt_return_long", 0.0)) >= 0.01:
        flags.add("recent_price_trend")
    net = features.get("pos_net_proposition", 0.0)
    direction = (
        1.0 if episode.direction == "positive"
        else -1.0 if episode.direction == "negative" else 0.0
    )
    if abs(net) > 1.0:
        flags.add("existing_exposure")
        if direction and np.sign(net) == -direction:
            flags.add("exposure_reduction")
        if direction and np.sign(net) == direction:
            flags.add("exposure_increase")
    if features.get("mkt_spread_missing", 1.0) == 0.0:
        flags.add("order_book_state")
    if features.get("act_category_trade_count", 0.0) > 0:
        flags.add("actor_history")
    return flags


def compute_agreement(
    layer1: DriverAttribution,
    template_name: str,
    posterior: TemplatePosterior,
    cf_result: CounterfactualResult | None,
    satisfied_evidence: set[str],
    *,
    materiality_delta: float = DEFAULT_MATERIALITY_DELTA,
) -> tuple[float, bool]:
    """Agreement between Layer 1 dominant channels, template-expected
    channels, template counterfactual results and available evidence.

    Returns (agreement_score in [0, 1], disagreement flag).  Disagreement
    is flagged when an ACCEPTED Layer 1 attribution names a channel the
    template does not expect, the template's expected channels carry no
    material ablation delta, and the posterior is not explicitly mixed —
    disagreement is surfaced, never smoothed into a narrative.
    """
    template = TEMPLATES[template_name]

    if not template.expected_channels:  # MIXED_OR_UNRESOLVED
        return 0.0, False

    deltas = layer1.group_attributions or {}
    expected_material = any(
        deltas.get(channel, 0.0) >= materiality_delta
        for channel in template.expected_channels
    )
    if layer1.primary_channel in template.expected_channels:
        channel_component = 1.0
    elif expected_material:
        channel_component = 0.6
    else:
        channel_component = 0.0

    if cf_result is not None and template.required_counterfactuals:
        passed = sum(
            1 for name in template.required_counterfactuals
            if cf_result.deltas.get(name, float("-inf")) >= materiality_delta
        )
        cf_component = passed / len(template.required_counterfactuals)
    else:
        cf_component = 0.0

    if template.required_evidence:
        evidence_component = sum(
            1 for token in template.required_evidence
            if token in satisfied_evidence
        ) / len(template.required_evidence)
    else:
        evidence_component = 1.0

    score = float((channel_component + cf_component + evidence_component) / 3)

    mixed = posterior.probabilities.get("MIXED_OR_UNRESOLVED", 0.0) >= 0.25
    disagreement = (
        layer1.status == "accepted"
        and layer1.primary_channel is not None
        and layer1.primary_channel not in template.expected_channels
        and not expected_material
        and not mixed
    )
    return score, disagreement


def reasoning_confidence(
    status: str,
    layer1: DriverAttribution,
    posterior: TemplatePosterior | None,
    agreement_score: float,
) -> float:
    """Final reasoning confidence for the top-level ``confidence`` column
    — never merely the observed-action probability."""
    if status != "accepted" or posterior is None:
        return 0.0
    top_p = max(posterior.probabilities.values(), default=0.0)
    return float(
        max(0.0, min(1.0, layer1.attribution_confidence * top_p
                     * (0.5 + 0.5 * agreement_score)))
    )


# ---------------------------------------------------------------------------
def build_drc_record(
    *,
    episode: DecisionEpisode,
    features: dict[str, float],
    evidence: list[dict] | None,
    layer1: DriverAttribution,
    posterior: TemplatePosterior | None,
    posterior_status: str,
    cf_result: CounterfactualResult | None,
    versions: dict[str, str],
    reasoning_target: str = "direction",
) -> dict[str, Any]:
    """Assemble the full structured (D, C, R) record with final status."""
    evidence = evidence or []
    satisfied = evidence_flags(features, episode, evidence)

    # --- resolve final status, primary template, agreement ---------------
    primary = posterior.primary_template if posterior else None
    status = posterior_status
    agreement_score = 0.0
    cf_failures: list[str] = []
    if primary is not None:
        ok, cf_failures = (
            required_counterfactuals_pass(primary, cf_result)
            if cf_result is not None else (False, ["counterfactuals_missing"])
        )
        if not ok:
            status, primary = "counterfactual_failure", None
        else:
            agreement_score, disagreement = compute_agreement(
                layer1, primary, posterior, cf_result, satisfied
            )
            if disagreement:
                status, primary = "attribution_template_disagreement", None
    if layer1.status == "insufficient_context":
        status = "insufficient_context"
        primary = None

    confidence = reasoning_confidence(status, layer1, posterior, agreement_score)

    assumptions = [
        "R is inferred behaviourally under the fitted model; it is not "
        "the actor's private mental state.",
    ]
    if layer1.coverage_complete is False:
        assumptions.append("context coverage incomplete")

    def clean(value: float) -> float | None:
        return None if not np.isfinite(value) else float(value)

    record = {
        "decision_id": episode.decision_id,
        "reasoning_target": reasoning_target,
        "D": {
            "actor_id": episode.actor_id,
            "condition_id": episode.condition_id,
            "decision_time": episode.anchor_time,
            "liquidity_role": "taker",
            "direction": episode.direction,
            "gross_quantity": episode.gross_quantity,
        },
        "C": {
            "contract_version": None,
            "market_summary": {
                "last_price": clean(features.get("mkt_last_price", float("nan"))),
                "return_short": clean(features.get("mkt_return_short", float("nan"))),
                "return_long": clean(features.get("mkt_return_long", float("nan"))),
                "spread": clean(features.get("mkt_spread", float("nan"))),
                "depth": clean(features.get("mkt_depth", float("nan"))),
            },
            "position_summary": {
                "net_proposition": features.get("pos_net_proposition"),
                "gross_exposure": features.get("pos_gross_exposure"),
                "history_incomplete": features.get("pos_history_incomplete"),
            },
            "news_evidence": evidence,
            "actor_history_summary": {
                "recent_trade_count": features.get("act_recent_trade_count"),
                "category_trade_count": features.get("act_category_trade_count"),
                "positive_rate": features.get("base_actor_positive_rate"),
            },
            "coverage": dict(episode.coverage),
            "satisfied_evidence": sorted(satisfied),
        },
        "R": {
            "primary_template": primary,
            "template_posterior": (
                posterior.probabilities if posterior else {}
            ),
            "posterior_entropy": posterior.entropy if posterior else None,
            "posterior_top_margin": posterior.top_margin if posterior else None,
            "calibration_version": (
                posterior.calibration_version if posterior else None
            ),
            "driver_attribution": {
                "primary_channel": layer1.primary_channel,
                "status": layer1.status,
                "group_attributions": layer1.group_attributions,
                "logit_contributions": layer1.logit_contributions,
                "intercept": clean(layer1.intercept),
                "total_logit": clean(layer1.total_logit),
                "reconstructed_probability": clean(
                    layer1.reconstructed_probability
                ),
                "top_ablation_delta": clean(layer1.top_ablation_delta),
                "top_vs_second_margin": clean(layer1.top_vs_second_margin),
                "attribution_stability": clean(layer1.attribution_stability),
                "fit_ok": layer1.fit_ok,
            },
            "counterfactuals": (
                counterfactual_report(cf_result) if cf_result else {}
            ),
            "counterfactual_failures": cf_failures,
            "agreement_score": agreement_score,
            "prediction_confidence": clean(layer1.prediction_confidence),
            "attribution_confidence": layer1.attribution_confidence,
            "reasoning_confidence": confidence,
            "status": status,
            "assumptions": assumptions,
        },
        "versions": dict(versions),
    }
    return record


# ---------------------------------------------------------------------------
def persist_reasoning_records(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    *,
    reasoning_run_id: str,
) -> int:
    """Idempotently persist full DRC records into reasoning_judgments.

    The top-level ``confidence`` column stores the final REASONING
    confidence.  Re-running with the same reasoning_run_id replaces the
    prior rows for those decisions (idempotent)."""
    from polymarket.analysis.rationale import render_rationale

    now = time.time()
    written = 0
    for record in records:
        rationale = render_rationale(record)
        judgment_id = namespace_id(
            "reasoning", reasoning_run_id, record["decision_id"],
            record["reasoning_target"],
        )
        conn.execute(
            "DELETE FROM reasoning_judgments WHERE reasoning_judgment_id = ?",
            (judgment_id,),
        )
        conn.execute(
            """
            INSERT INTO reasoning_judgments
                (reasoning_judgment_id, decision_id, reasoning_run_id,
                 primary_template, template_posterior_json,
                 driver_attribution_json, evidence_json, counterfactual_json,
                 rationale_text, agreement_score, confidence, status,
                 model_version, feature_version, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                judgment_id,
                record["decision_id"],
                reasoning_run_id,
                record["R"]["primary_template"],
                canonical_json({
                    "reasoning_target": record["reasoning_target"],
                    "probabilities": record["R"]["template_posterior"],
                    "entropy": record["R"]["posterior_entropy"],
                    "top_margin": record["R"]["posterior_top_margin"],
                    "calibration_version": record["R"]["calibration_version"],
                }),
                canonical_json(record["R"]["driver_attribution"]),
                canonical_json(record["C"]["news_evidence"]),
                canonical_json(record["R"]["counterfactuals"]),
                rationale,
                record["R"]["agreement_score"],
                record["R"]["reasoning_confidence"],
                record["R"]["status"],
                record["versions"]["reasoning_method_version"],
                record["versions"]["feature_version"],
                now,
            ),
        )
        written += 1
    conn.commit()
    return written
