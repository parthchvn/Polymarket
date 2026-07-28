"""Feature construction from strict decision contexts.

Feature groups are kept separate (base rates, market state, actor
history, position, news) and news components are NOT multiplied into one
opaque score.  Every group carries missingness indicators; missing data
is encoded as explicit indicator + neutral fill, never as a silent
substantive zero.
"""

from __future__ import annotations

import json
import math
from typing import Any

from polymarket.analysis.context import DecisionContext
from polymarket.analysis.decisions import DecisionEpisode, proposition_change

FEATURE_GROUPS = {
    "base": [
        "base_market_age", "base_time_to_resolution",
        "base_actor_positive_rate", "base_actor_positive_rate_missing",
    ],
    "market": [
        "mkt_last_price", "mkt_last_price_missing",
        "mkt_return_short", "mkt_return_long", "mkt_volatility",
        "mkt_state_from_executions",
        "mkt_volume", "mkt_spread", "mkt_spread_missing",
        "mkt_depth", "mkt_imbalance", "mkt_execution_rate",
    ],
    "actor": [
        "act_recent_trade_count", "act_recent_gross_volume",
        "act_recent_net_prop_change", "act_time_since_last_trade",
        "act_time_since_last_trade_missing", "act_category_trade_count",
    ],
    "position": [
        "pos_positive_tokens", "pos_negative_tokens", "pos_net_proposition",
        "pos_gross_exposure", "pos_distance_from_neutral",
        "pos_history_incomplete", "pos_unresolved_event_count",
    ],
    "news": [
        "news_rel_max", "news_rel_sum", "news_direction",
        "news_novelty_max", "news_surprise_max", "news_age_hours",
        "news_source_diversity", "news_confirmation_count",
        "news_contradiction_count", "news_article_count",
        "news_ingestion_lag", "news_missing",
        "news_recent_missing", "news_decay_missing",
        "news_decay_signed_6h", "news_decay_positive_6h",
        "news_decay_negative_6h",
        "news_decay_signed_24h", "news_decay_positive_24h",
        "news_decay_negative_24h",
        "news_decay_signed_72h", "news_decay_positive_72h",
        "news_decay_negative_72h",
        "news_decay_signed_168h", "news_decay_positive_168h",
        "news_decay_negative_168h",
    ],
    "paper": [
        "liq_mode_event", "liq_mode_missing", "liq_mode_age_hours",
        "event_mode_prevalence", "event_mode_prevalence_missing",
        "impact_screen_available", "impactful_news_count",
        "impactful_news_probability", "impact_screen_contradiction",
        "initial_response_so_far", "initial_response_missing",
        "attention_claim_load", "attention_unrelated_load",
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# ---------------------------------------------------------------------------
# News time decay.
#
# The permanent semantic relevance judgment (relevance_judgments.rel_score)
# is NEVER modified.  A decision-specific dynamic weight is recomputed for
# every decision from the age of the news at that decision, using half-life
# decay over multiple horizons so the model can learn whether short-lived
# or persistent news matters.  These half-lives are modelling choices, not
# established causal parameters (see docs/RESEARCH_ASSUMPTIONS.md).

NEWS_DECAY_HALF_LIVES = {
    "6h": 6 * 3600.0,
    "24h": 24 * 3600.0,
    "72h": 72 * 3600.0,
    "168h": 168 * 3600.0,
}

NEWS_DECAY_MAX_AGE = 28 * 86400.0

NEWS_DECAY_AGGREGATION = "event_family_max_positive_negative"


def half_life_decay(age_seconds: float, half_life_seconds: float) -> float:
    """Half-life decay weight: 0.5 ** (age / half_life)."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    return 2.0 ** (-float(age_seconds) / float(half_life_seconds))


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read a field from a sqlite3.Row or a plain dict."""
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


def relevance_confidence(row: Any) -> float:
    """Confidence stored in evidence_json, defaulting safely to 1.0.

    Malformed or missing JSON must never crash feature construction.
    """
    raw = _row_get(row, "evidence_json")
    if not raw:
        return 1.0
    try:
        evidence = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return 1.0
    if not isinstance(evidence, dict):
        return 1.0
    confidence = evidence.get("confidence", 1.0)
    try:
        return min(max(float(confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def decayed_news_signals(
    relevance_rows: Any,
    *,
    decision_time: float,
    half_life_seconds: float,
    max_age_seconds: float = NEWS_DECAY_MAX_AGE,
) -> dict[str, float]:
    """Event-family-deduplicated, time-decayed news signals.

    Row eligibility (defensive even though the strict reader already
    enforces the temporal rule): age = decision_time - computed_at must be
    strictly positive — rows exactly at or after the decision time
    contribute nothing — and at most ``max_age_seconds``; irrelevant rows
    are excluded.

    Per-row signed evidence is
    ``rel_score * direction * novelty * confidence * decay`` with all
    inputs clamped.  Rows are grouped by event_family_id; within a family
    the positive and negative components each take the MAX row
    contribution (duplicate articles must not multiply the signal), and
    distinct families sum (independent events may add).  Positive and
    negative evidence are kept separate so contradictions stay visible.
    """
    family_positive: dict[str, float] = {}
    family_negative: dict[str, float] = {}
    for index, row in enumerate(relevance_rows):
        if _row_get(row, "rel_class") == "irrelevant":
            continue
        computed_at = _row_get(row, "computed_at")
        if computed_at is None:
            continue
        age_seconds = decision_time - float(computed_at)
        if age_seconds <= 0 or age_seconds > max_age_seconds:
            continue
        relevance = min(max(float(_row_get(row, "rel_score", 0.0)), 0.0), 1.0)
        direction = min(max(float(_row_get(row, "direction", 0.0)), -1.0), 1.0)
        novelty_raw = _row_get(row, "novelty")
        novelty = (
            1.0 if novelty_raw is None
            else min(max(float(novelty_raw), 0.0), 1.0)
        )
        confidence = relevance_confidence(row)
        decay = half_life_decay(age_seconds, half_life_seconds)
        contribution = relevance * direction * novelty * confidence * decay
        family_key = _row_get(row, "event_family_id") or f"unfamilied:{index}"
        if contribution > 0:
            family_positive[family_key] = max(
                family_positive.get(family_key, 0.0), contribution
            )
            family_negative.setdefault(family_key, 0.0)
        else:
            family_negative[family_key] = max(
                family_negative.get(family_key, 0.0), -contribution
            )
            family_positive.setdefault(family_key, 0.0)
    total_positive = sum(family_positive.values())
    total_negative = sum(family_negative.values())
    return {
        "signed": total_positive - total_negative,
        "positive": total_positive,
        "negative": total_negative,
        "family_count": float(len(family_positive)),
    }


def _sign_by_asset(context: DecisionContext) -> dict[str, int]:
    signs: dict[str, int] = {}
    for row in context.actor_history:
        if row["condition_id"] == context.condition_id and row["outcome_sign"]:
            signs[row["asset"]] = row["outcome_sign"]
    return signs


def compute_features(
    context: DecisionContext,
    episode: DecisionEpisode,
    *,
    recent_window: float = 86400.0,
    short_horizon: float = 3600.0,
    news_lookback: float = 86400.0,
    news_decay_half_lives: dict[str, float] | None = None,
    news_decay_max_age: float = NEWS_DECAY_MAX_AGE,
) -> dict[str, float]:
    t = context.decision_time
    f: dict[str, float] = {name: 0.0 for name in ALL_FEATURES}

    # ---- base rates ------------------------------------------------------
    market = context.contract or {}
    created = market.get("effective_from")
    f["base_market_age"] = max(t - created, 0.0) if created else 0.0
    resolution_time = market.get("resolution_time")
    f["base_time_to_resolution"] = (
        max(resolution_time - t, 0.0) if resolution_time else 0.0
    )
    # actor historical positive-direction rate (temporally available only:
    # computed from history strictly before t)
    directed = [
        proposition_change(r["side"], r["outcome_sign"], r["size"])
        for r in context.actor_history
        if r["outcome_sign"] is not None
    ]
    if directed:
        f["base_actor_positive_rate"] = sum(
            1.0 for c in directed if c > 0
        ) / len(directed)
        f["base_actor_positive_rate_missing"] = 0.0
    else:
        f["base_actor_positive_rate"] = 0.5
        f["base_actor_positive_rate_missing"] = 1.0

    # ---- market state ----------------------------------------------------
    states = context.market_state
    f["mkt_state_from_executions"] = (
        1.0 if context.coverage.get("market_series_source") == "executions"
        else 0.0
    )
    prices = [
        (s["ts"], s["positive_price"])
        for s in states
        if s["positive_price"] is not None
    ]
    if prices:
        last_ts, last_price = prices[-1]
        f["mkt_last_price"] = last_price
        f["mkt_last_price_missing"] = 0.0
        short = [p for ts, p in prices if ts >= t - short_horizon]
        if len(short) >= 2:
            f["mkt_return_short"] = short[-1] - short[0]
        if len(prices) >= 2:
            f["mkt_return_long"] = prices[-1][1] - prices[0][1]
            mean = sum(p for _, p in prices) / len(prices)
            f["mkt_volatility"] = (
                sum((p - mean) ** 2 for _, p in prices) / len(prices)
            ) ** 0.5
    else:
        f["mkt_last_price"] = 0.5
        f["mkt_last_price_missing"] = 1.0
    executions = context.execution_activity
    f["mkt_volume"] = sum(e["size"] or 0.0 for e in executions)
    f["mkt_execution_rate"] = float(len(executions))
    books = context.order_books
    spreads = [b["spread"] for b in books if b["spread"] is not None]
    if spreads:
        f["mkt_spread"] = sum(spreads) / len(spreads)
        f["mkt_spread_missing"] = 0.0
    else:
        f["mkt_spread_missing"] = 1.0
    depths = [
        (b["bid_depth"] or 0.0) + (b["ask_depth"] or 0.0)
        for b in books
        if b["bid_depth"] is not None or b["ask_depth"] is not None
    ]
    f["mkt_depth"] = sum(depths) / len(depths) if depths else 0.0
    imbalances = [b["imbalance"] for b in books if b["imbalance"] is not None]
    f["mkt_imbalance"] = (
        sum(imbalances) / len(imbalances) if imbalances else 0.0
    )

    # ---- actor history ---------------------------------------------------
    recent = [r for r in context.actor_history if r["ts"] >= t - recent_window]
    f["act_recent_trade_count"] = float(len(recent))
    f["act_recent_gross_volume"] = sum(r["size"] for r in recent)
    f["act_recent_net_prop_change"] = sum(
        proposition_change(r["side"], r["outcome_sign"], r["size"])
        for r in recent
        if r["outcome_sign"] is not None
    )
    if context.actor_history:
        f["act_time_since_last_trade"] = t - context.actor_history[-1]["ts"]
        f["act_time_since_last_trade_missing"] = 0.0
    else:
        f["act_time_since_last_trade"] = recent_window
        f["act_time_since_last_trade_missing"] = 1.0
    f["act_category_trade_count"] = float(
        sum(1 for r in context.actor_history
            if r["condition_id"] == context.condition_id)
    )

    # ---- position --------------------------------------------------------
    signs = _sign_by_asset(context)
    balances: dict[str, float] = context.position.get("balances", {})
    positive = sum(v for a, v in balances.items() if signs.get(a) == 1)
    negative = sum(v for a, v in balances.items() if signs.get(a) == -1)
    f["pos_positive_tokens"] = positive
    f["pos_negative_tokens"] = negative
    f["pos_net_proposition"] = positive - negative
    f["pos_gross_exposure"] = abs(positive) + abs(negative)
    f["pos_distance_from_neutral"] = abs(positive - negative)
    f["pos_history_incomplete"] = 0.0 if context.position.get("complete") else 1.0
    f["pos_unresolved_event_count"] = float(
        context.position.get("unresolved_event_count", 0)
    )

    # ---- news (components kept separate; zero-preserving sparse coding) --
    # Raw recent-window (24h) features are retained for backwards
    # compatibility and interpretability; decayed features below extend
    # the horizon to news_decay_max_age with half-life weighting.
    relevant = [
        r for r in context.relevance
        if r["computed_at"] < t  # defensive: never at/after decision time
        and r["computed_at"] >= t - news_lookback
        and r["rel_class"] not in ("irrelevant",)
    ]
    f["news_recent_missing"] = 0.0 if relevant else 1.0
    if relevant:
        f["news_rel_max"] = max(r["rel_score"] for r in relevant)
        f["news_rel_sum"] = sum(r["rel_score"] for r in relevant)
        top = max(relevant, key=lambda r: r["rel_score"])
        f["news_direction"] = top["direction"]
        f["news_novelty_max"] = max(r["novelty"] or 0.0 for r in relevant)
        f["news_surprise_max"] = max(r["surprise"] or 0.0 for r in relevant)
        f["news_age_hours"] = (t - top["computed_at"]) / 3600.0
        by_family: dict[str, list[float]] = {}
        for r in relevant:
            by_family.setdefault(r["event_family_id"], []).append(r["direction"])
        f["news_confirmation_count"] = float(
            sum(len(ds) - 1 for ds in by_family.values() if len(ds) > 1)
        )
        f["news_contradiction_count"] = float(
            sum(
                1 for ds in by_family.values()
                if any(d > 0 for d in ds) and any(d < 0 for d in ds)
            )
        )

    # ---- decayed news signals (event-family deduplicated) ---------------
    half_lives = (
        dict(NEWS_DECAY_HALF_LIVES)
        if news_decay_half_lives is None
        else dict(news_decay_half_lives)
    )
    decay_family_count = 0.0
    for label, half_life_seconds in half_lives.items():
        signals = decayed_news_signals(
            context.relevance,
            decision_time=t,
            half_life_seconds=half_life_seconds,
            max_age_seconds=news_decay_max_age,
        )
        f[f"news_decay_signed_{label}"] = signals["signed"]
        f[f"news_decay_positive_{label}"] = signals["positive"]
        f[f"news_decay_negative_{label}"] = signals["negative"]
        decay_family_count = max(decay_family_count, signals["family_count"])
    f["news_decay_missing"] = 0.0 if decay_family_count > 0 else 1.0
    # news_missing now means: no news information is available to ANY news
    # feature used by the model (i.e. it equals news_decay_missing).  The
    # old 24-hour-only semantics live in news_recent_missing.
    f["news_missing"] = f["news_decay_missing"]

    articles = [
        a for a in context.articles
        if a["first_observed_at"] >= t - news_lookback
    ]
    f["news_article_count"] = float(len(articles))
    f["news_source_diversity"] = float(len({a["source_id"] for a in articles}))
    lags = [
        a["first_observed_at"] - a["source_published_at"]
        for a in articles
        if a["source_published_at"] is not None
    ]
    f["news_ingestion_lag"] = sum(lags) / len(lags) if lags else 0.0
    # ---- paper-derived state (strict as-of; PR C integration) --------
    paper = getattr(context, "paper_state", {}) or {}
    mode = paper.get("liquidity_mode")
    f["liq_mode_event"] = (
        1.0 if mode and mode["mode_label_online"] == "event" else 0.0
    )
    f["liq_mode_missing"] = 0.0 if mode else 1.0
    f["liq_mode_age_hours"] = (
        mode["age_seconds"] / 3600.0 if mode else 0.0
    )
    prevalence = paper.get("event_mode_prevalence")
    f["event_mode_prevalence"] = (
        prevalence if prevalence is not None else 0.0
    )
    f["event_mode_prevalence_missing"] = (
        0.0 if prevalence is not None else 1.0
    )
    screens = paper.get("impact_screens", [])
    evaluated = paper.get("screens_evaluated", 0)
    f["impact_screen_available"] = 1.0 if evaluated > 0 else 0.0
    f["impactful_news_count"] = float(len(screens))
    f["impactful_news_probability"] = (
        max((s_["impact_score"] for s_ in screens), default=0.0)
    )
    # screens ran for this market and found NO impactful news: the
    # persistent-adjustment story is contradicted by the market's own
    # liquidity reaction
    f["impact_screen_contradiction"] = (
        1.0 if evaluated > 0 and not screens else 0.0
    )
    initial = paper.get("initial_response_so_far")
    f["initial_response_so_far"] = initial if initial is not None else 0.0
    f["initial_response_missing"] = 0.0 if initial is not None else 1.0
    attention = paper.get("attention", {})
    f["attention_claim_load"] = math.log1p(
        attention.get("claim_count_24h", 0)
    )
    f["attention_unrelated_load"] = math.log1p(
        attention.get("unrelated_family_count_24h", 0)
    )

    return {name: float(value) for name, value in f.items()}


def feature_subset(features: dict[str, float], groups: list[str]) -> dict[str, float]:
    names = [n for g in groups for n in FEATURE_GROUPS[g]]
    return {n: features[n] for n in names}


def news_decay_config(
    *,
    news_lookback: float = 86400.0,
    news_decay_half_lives: dict[str, float] | None = None,
    news_decay_max_age: float = NEWS_DECAY_MAX_AGE,
) -> dict[str, Any]:
    half_lives = (
        dict(NEWS_DECAY_HALF_LIVES)
        if news_decay_half_lives is None
        else dict(news_decay_half_lives)
    )
    return {
        "news_recent_lookback_seconds": news_lookback,
        "news_decay_max_age_seconds": news_decay_max_age,
        "news_decay_half_lives_seconds": half_lives,
        "news_decay_aggregation": NEWS_DECAY_AGGREGATION,
    }


def feature_manifest() -> dict[str, Any]:
    return {
        "groups": FEATURE_GROUPS,
        "all_features": ALL_FEATURES,
        "news_decay": news_decay_config(),
    }
