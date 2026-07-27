"""Forward collection loop: continuous, incremental, gap-aware.

Runs repeated collection cycles against live Polymarket surfaces for a
configured set of markets:

* every cycle: market metadata/status, incremental taker + expanded
  trades (``after`` cursor with overlap), and an order-book snapshot for
  every outcome token — the book history that gives real mid-price
  state;
* every ``activity_every`` cycles: activity for the most active taker
  wallets;
* every ``news_every`` cycles: the configured news feeds.

Robustness rules:

* each surface is isolated — one failure never kills the cycle; a
  failed surface records a BOUNDED collector gap covering that cycle's
  window, so coverage certification excludes exactly the affected span;
* on startup, downtime since the previous run (detected from the newest
  stored response) is recorded as a gap per market;
* the trade cursor is derived from the database (newest trades_taker
  response time), so restarts resume incrementally with an overlap and
  never silently skip a window.

Read-only throughout: no keys, no orders, no trading.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Callable

from polymarket.collection.activity import collect_activity
from polymarket.collection.books import collect_book
from polymarket.collection.client import ObservingClient
from polymarket.collection.markets import collect_markets
from polymarket.collection.news import collect_google_news_rss
from polymarket.collection.raw_store import (
    finish_collector_run,
    record_gap,
    start_collector_run,
)
from polymarket.collection.trades import collect_trades

TRADE_OVERLAP_SECONDS = 120.0
DOWNTIME_FACTOR = 2.0  # gap recorded when idle > factor * interval


@dataclass(frozen=True)
class ForwardConfig:
    """Per-surface cadences in SECONDS.

    The screening paper needs five-minute liquidity bars built from
    many within-bin book observations (average spread, average
    best-level book size, within-bin volatility), so books sample much
    faster than everything else.  The loop ticks at the fastest cadence
    and each surface runs when due — trade calls are never repeated at
    book frequency."""

    condition_ids: tuple[str, ...]
    book_every: float = 60.0
    trade_every: float = 300.0
    market_every: float = 300.0
    news_every_seconds: float = 300.0
    activity_every_seconds: float = 3600.0
    activity_wallets: int = 30
    news_queries: tuple[str, ...] = ()
    trade_pages_per_cycle: int = 5
    activity_pages: int = 3

    @property
    def interval_seconds(self) -> float:
        """The loop tick = the fastest configured cadence."""
        cadences = [self.book_every, self.trade_every, self.market_every]
        if self.news_queries:
            cadences.append(self.news_every_seconds)
        cadences.append(self.activity_every_seconds)
        return max(1.0, min(cadences))


@dataclass
class LoopState:
    cycle_index: int = 0
    last_cycle_start: float | None = None
    assets_by_condition: dict[str, list[str]] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    last_run: dict[str, float] = field(default_factory=dict)

    def due(self, surface: str, cadence: float, now: float) -> bool:
        last = self.last_run.get(surface)
        return last is None or now - last >= cadence - 1e-6

    def mark(self, surface: str, now: float) -> None:
        self.last_run[surface] = now


# ---------------------------------------------------------------------------
# database-derived state (restart safety)
# ---------------------------------------------------------------------------
def derive_trade_cursor(conn) -> float | None:
    """Newest trades_taker response time — the incremental cursor."""
    return derive_surface_cursor(conn, ("trades_taker",))


def derive_surface_cursor(
    conn, collectors: tuple[str, ...]
) -> float | None:
    """Newest stored response time across the given collectors."""
    placeholders = ",".join("?" for _ in collectors)
    row = conn.execute(
        f"SELECT MAX(received_at) FROM raw_responses "
        f"WHERE collector IN ({placeholders})",
        collectors,
    ).fetchone()
    return row[0]


# gap surface -> (collectors that feed it, cadence attribute)
_SURFACE_CURSORS: dict[str, tuple[tuple[str, ...], str]] = {
    "trades": (("trades_taker", "trades_expanded"), "trade_every"),
    "books": (("books",), "book_every"),
    "markets": (("markets",), "market_every"),
    "news": (("news:google-rss",), "news_every_seconds"),
    "activity": (("activity",), "activity_every_seconds"),
}


def record_downtime_gaps(conn, config: ForwardConfig, now: float) -> int:
    """Record bounded downtime gaps PER SURFACE, each measured against
    its own last-success cursor and its own cadence.  Books collected
    60 seconds ago must not inherit a gap because trades last ran five
    minutes ago — and vice versa."""
    recorded = 0
    for surface, (collectors, cadence_attr) in _SURFACE_CURSORS.items():
        last = derive_surface_cursor(conn, collectors)
        if last is None:
            continue
        cadence = float(getattr(config, cadence_attr))
        if now - last <= DOWNTIME_FACTOR * cadence:
            continue
        for condition_id in config.condition_ids:
            record_gap(
                conn, collector="forward-loop", surface=surface,
                object_id=condition_id, gap_start=last, gap_end=now,
                reason=f"collection loop downtime ({surface})",
            )
            recorded += 1
    return recorded


def discover_assets(conn, condition_ids: tuple[str, ...]) -> dict[str, list[str]]:
    """Outcome tokens per condition from normalized mappings, falling
    back to the newest raw markets payload before first normalization."""
    import json

    assets: dict[str, list[str]] = {c: [] for c in condition_ids}
    rows = conn.execute(
        "SELECT condition_id, asset FROM outcome_tokens WHERE condition_id "
        f"IN ({','.join('?' for _ in condition_ids)})",
        condition_ids,
    ).fetchall()
    for condition_id, asset in rows:
        assets[condition_id].append(asset)
    if all(assets.values()):
        return assets
    raw = conn.execute(
        "SELECT payload FROM raw_responses WHERE collector = 'markets' "
        "ORDER BY raw_response_id DESC LIMIT 1"
    ).fetchone()
    if raw:
        try:
            for market in json.loads(bytes(raw[0])):
                condition_id = market.get("conditionId")
                if condition_id in assets and not assets[condition_id]:
                    token_ids = market.get("clobTokenIds")
                    if isinstance(token_ids, str):
                        token_ids = json.loads(token_ids)
                    tokens = market.get("tokens") or [
                        {"token_id": t} for t in (token_ids or [])
                    ]
                    assets[condition_id] = [
                        t.get("token_id") for t in tokens if t.get("token_id")
                    ]
        except (ValueError, TypeError):
            pass
    return assets


def _wallets_from_raw_trades(conn, condition_ids, limit: int) -> list[str]:
    import json
    from collections import Counter

    counts: Counter = Counter()
    rows = conn.execute(
        "SELECT payload FROM raw_responses WHERE collector = 'trades_taker' "
        "ORDER BY raw_response_id DESC LIMIT 10"
    ).fetchall()
    wanted = set(condition_ids)
    for (payload,) in rows:
        try:
            for record in json.loads(bytes(payload)):
                if record.get("conditionId") in wanted:
                    counts[record.get("proxyWallet")] += 1
        except (ValueError, TypeError):
            continue
    counts.pop(None, None)
    return [wallet for wallet, _ in counts.most_common(limit)]


def top_taker_wallets(conn, condition_ids, limit: int) -> list[str]:
    placeholders = ",".join("?" for _ in condition_ids)
    rows = conn.execute(
        f"""
        SELECT proxy_wallet, COUNT(*) AS n FROM actor_trade_legs
        WHERE condition_id IN ({placeholders})
        GROUP BY proxy_wallet ORDER BY n DESC, proxy_wallet LIMIT ?
        """,
        (*condition_ids, limit),
    ).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# surfaces (each isolated; injectable for tests)
# ---------------------------------------------------------------------------
def _observed(conn, collector: str, params: dict, fn: Callable, **kwargs):
    run_id = start_collector_run(conn, collector, params)
    try:
        with ObservingClient(conn, collector, run_id) as client:
            outcome = fn(client, **kwargs)
    except Exception:
        finish_collector_run(conn, run_id, "failed")
        raise
    finish_collector_run(conn, run_id, "succeeded",
                         note=getattr(outcome, "note", None))
    return outcome


def surface_markets(conn, config: ForwardConfig, state: LoopState,
                    window: tuple[float, float]) -> str:
    if not state.due("markets", config.market_every, window[1]):
        return "(not due)"
    _observed(conn, "markets", {"condition_ids": list(config.condition_ids)},
              collect_markets, condition_ids=list(config.condition_ids))
    state.mark("markets", window[1])
    return "markets ok"


def surface_trades(conn, config: ForwardConfig, state: LoopState,
                   window: tuple[float, float]) -> str:
    if not state.due("trades", config.trade_every, window[1]):
        return "(not due)"
    cursor = derive_trade_cursor(conn)
    start_ts = (
        max(0.0, cursor - TRADE_OVERLAP_SECONDS)
        if cursor is not None else None
    )
    total = 0
    for condition_id in config.condition_ids:
        for taker_only in (True, False):
            collector = "trades_taker" if taker_only else "trades_expanded"
            outcome = _observed(
                conn, collector,
                {"condition_id": condition_id, "takerOnly": taker_only,
                 "after": start_ts},
                collect_trades, condition_id=condition_id,
                taker_only=taker_only, start_ts=start_ts,
                max_pages=config.trade_pages_per_cycle,
            )
            total += getattr(outcome, "record_count", 0) or 0
    state.mark("trades", window[1])
    return f"trades +{total}"


def surface_books(conn, config: ForwardConfig, state: LoopState,
                  window: tuple[float, float]) -> str:
    if not state.due("books", config.book_every, window[1]):
        return "(not due)"
    if not state.assets_by_condition:
        state.assets_by_condition = discover_assets(
            conn, config.condition_ids
        )
    count = 0
    for condition_id, assets in state.assets_by_condition.items():
        for asset in assets:
            _observed(conn, "books", {"asset": asset},
                      collect_book, asset=asset)
            count += 1
    if count == 0:
        raise RuntimeError("no outcome tokens known yet for any market")
    state.mark("books", window[1])
    return f"books x{count}"


def surface_activity(conn, config: ForwardConfig, state: LoopState,
                     window: tuple[float, float]) -> str:
    if not state.due("activity", config.activity_every_seconds, window[1]):
        return "(not due)"
    wallets = top_taker_wallets(
        conn, config.condition_ids, config.activity_wallets
    )
    if not wallets:
        # fresh database: normalized legs do not exist yet, so derive
        # wallets from the newest raw taker payload instead of skipping
        wallets = _wallets_from_raw_trades(
            conn, config.condition_ids, config.activity_wallets
        )
    for wallet in wallets:
        _observed(conn, "activity", {"wallet": wallet},
                  collect_activity, wallet=wallet,
                  max_pages=config.activity_pages)
    state.mark("activity", window[1])
    return f"activity x{len(wallets)}"


def surface_news(conn, config: ForwardConfig, state: LoopState,
                 window: tuple[float, float]) -> str:
    if not config.news_queries:
        return "news (none configured)"
    if not state.due("news", config.news_every_seconds, window[1]):
        return "(not due)"
    total = 0
    for query in config.news_queries:
        total += collect_google_news_rss(conn, query)
    state.mark("news", window[1])
    return f"news +{total}"


# surface name -> (fn, gap surface label or None when a failure is
# harmless for coverage certification)
DEFAULT_SURFACES: dict[str, tuple[Callable, str | None]] = {
    "markets": (surface_markets, "markets"),
    "trades": (surface_trades, "trades"),
    "books": (surface_books, "books"),
    "activity": (surface_activity, None),
    "news": (surface_news, None),
}


# ---------------------------------------------------------------------------
def run_cycle(
    conn,
    config: ForwardConfig,
    state: LoopState,
    surfaces: dict[str, tuple[Callable, str | None]] | None = None,
    now: float | None = None,
) -> dict[str, str]:
    """One collection cycle.  Every surface is isolated: a failure logs
    a bounded gap for its window and the cycle continues."""
    surfaces = surfaces if surfaces is not None else DEFAULT_SURFACES
    cycle_start = now if now is not None else time.time()
    window = (
        state.last_cycle_start
        if state.last_cycle_start is not None
        else cycle_start - config.interval_seconds,
        cycle_start,
    )
    report: dict[str, str] = {}
    for name, (fn, gap_surface) in surfaces.items():
        try:
            report[name] = fn(conn, config, state, window)
        except Exception as exc:  # noqa: BLE001 - isolation is the contract
            report[name] = f"FAILED: {exc}"
            state.failures[name] = state.failures.get(name, 0) + 1
            if gap_surface is not None:
                for condition_id in config.condition_ids:
                    record_gap(
                        conn, collector="forward-loop",
                        surface=gap_surface, object_id=condition_id,
                        gap_start=window[0], gap_end=window[1],
                        reason=f"cycle surface failure: {exc}"[:200],
                    )
    state.last_cycle_start = cycle_start
    state.cycle_index += 1
    return report


def run_loop(
    conn,
    config: ForwardConfig,
    *,
    max_cycles: int | None = None,
    duration_seconds: float | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.time,
    on_cycle: Callable[[int, dict[str, str]], None] | None = None,
) -> LoopState:
    """Run cycles until max_cycles / duration / SIGINT."""
    state = LoopState()
    started = now_fn()
    gaps = record_downtime_gaps(conn, config, started)
    if gaps:
        print(f"recorded downtime gaps for {gaps} markets")
    stop = {"flag": False}

    def _sigint(_signum, _frame):
        stop["flag"] = True
        print("\nstopping after the current cycle ...")

    previous = signal.signal(signal.SIGINT, _sigint)
    try:
        while not stop["flag"]:
            cycle_start = now_fn()
            report = run_cycle(conn, config, state, now=cycle_start)
            if on_cycle is not None:
                on_cycle(state.cycle_index, report)
            if max_cycles is not None and state.cycle_index >= max_cycles:
                break
            if (duration_seconds is not None
                    and now_fn() - started >= duration_seconds):
                break
            elapsed = now_fn() - cycle_start
            sleep_fn(max(0.0, config.interval_seconds - elapsed))
    finally:
        signal.signal(signal.SIGINT, previous)
    return state


def default_cycle_printer(cycle_index: int, report: dict[str, str]) -> None:
    stamp = time.strftime("%H:%M:%S")
    summary = " | ".join(f"{k}: {v}" for k, v in report.items())
    print(f"[{stamp}] cycle {cycle_index}: {summary}", flush=True)
