"""Feature construction from strict decision contexts.

Feature groups are kept separate (base rates, market state, actor
history, position, news) and news components are NOT multiplied into one
opaque score.  Every group carries missingness indicators; missing data
is encoded as explicit indicator + neutral fill, never as a silent
substantive zero.
"""

from __future__ import annotations

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
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


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
    f["mkt_volume"] = sum(s["volume"] or 0.0 for s in states)
    f["mkt_execution_rate"] = len(states)
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
    relevant = [
        r for r in context.relevance
        if r["computed_at"] >= t - news_lookback
        and r["rel_class"] not in ("irrelevant",)
    ]
    if relevant:
        f["news_missing"] = 0.0
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
    else:
        f["news_missing"] = 1.0

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
    return {name: float(value) for name, value in f.items()}


def feature_subset(features: dict[str, float], groups: list[str]) -> dict[str, float]:
    names = [n for g in groups for n in FEATURE_GROUPS[g]]
    return {n: features[n] for n in names}


def feature_manifest() -> dict[str, Any]:
    return {"groups": FEATURE_GROUPS, "all_features": ALL_FEATURES}
