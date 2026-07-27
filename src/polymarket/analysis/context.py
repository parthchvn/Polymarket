"""Replay context for one decision.

Everything in the context is strictly before the decision time; the
constructor runs assert_no_future_information as a runtime guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polymarket.analysis.decisions import DecisionEpisode
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.temporal import assert_no_future_information


@dataclass
class DecisionContext:
    decision_id: str
    actor_id: str
    market_id: str | None
    condition_id: str
    decision_time: float
    contract: dict | None
    market_status: dict | None
    market_state: list[dict]
    order_books: list[dict]
    actor_history: list[dict]
    position: dict
    articles: list[dict]
    event_families: list[dict]
    relevance: list[dict]
    coverage: dict = field(default_factory=dict)


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


def build_context(
    reader: SQLiteNormalizedReader,
    episode: DecisionEpisode,
    *,
    market_state_lookback: float = 86400.0,
) -> DecisionContext:
    t = episode.anchor_time
    contract = None
    status = None
    if episode.market_id:
        c = reader.contract_asof(episode.market_id, t)
        contract = dict(c) if c else None
        s = reader.market_status_asof(episode.market_id, t)
        status = dict(s) if s else None

    books: list[dict] = []
    for token in reader.outcome_tokens_asof(episode.condition_id, t):
        book = reader.order_book_before(token["asset"], t)
        if book is not None:
            books.append(dict(book))

    context = DecisionContext(
        decision_id=episode.decision_id,
        actor_id=episode.actor_id,
        market_id=episode.market_id,
        condition_id=episode.condition_id,
        decision_time=t,
        contract=contract,
        market_status=status,
        market_state=[],
        order_books=books,
        actor_history=_rows(reader.actor_trade_legs_before(t, actor=episode.actor_id)),
        position=reader.position_asof(episode.actor_id, episode.condition_id, t),
        articles=_rows(reader.articles_asof(t)),
        event_families=_rows(reader.event_families_asof(t)),
        relevance=[],
        coverage=dict(episode.coverage),
    )
    series_rows, series_source = reader.market_series_before(
        episode.condition_id, t, market_state_lookback,
        policy="book_preferred",
    )
    context.market_state = _rows(series_rows)
    context.coverage["market_series_source"] = series_source
    if episode.market_id and contract is not None:
        snapshot_rows, version_fallback = reader.relevance_snapshot_asof(
            episode.market_id, contract["version_seq"], t
        )
        context.relevance = _rows(snapshot_rows)
        context.coverage["relevance_version_fallback"] = version_fallback
    assert_no_future_information(
        {
            "contract": context.contract,
            "market_status": context.market_status,
            "market_state": context.market_state,
            "order_books": context.order_books,
            "actor_history": context.actor_history,
            "articles": context.articles,
            "event_families": context.event_families,
            "relevance": context.relevance,
        },
        t,
    )
    return context
