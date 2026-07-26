"""Synthetic worlds with KNOWN reasoning mechanisms.

Each world (one per seed) writes raw-like responses through the SAME raw
store and the SAME production normalizer into the SAME schema, then
labels every scripted actor decision with the mechanism that is
responsible BY CONSTRUCTION.  Worlds are fully deterministic in their
seed.  Train/validation/test splits are BY WORLD SEED — no episode from
one world may ever appear on both sides.

Archetypes (one actor each): fresh-news, persistent-news (delayed
adjustment), momentum, contrarian, inventory-rebalancing,
position-building, liquidity-timing, actor-prior, and a mixed/random
actor whose decisions must remain ambiguous.  Weak-signal and
correlated-signal cases are included (late fresh reactions, very old
persistent reactions, momentum trades that occasionally coincide with
news).

The existing deterministic fixture (scenarios.py) is untouched.
"""

from __future__ import annotations

import json
import random
import sqlite3

from polymarket.analysis.versioning import SYNTHETIC_GENERATOR_VERSION
from polymarket.collection.raw_store import (
    finish_collector_run,
    insert_raw_response,
    start_collector_run,
)
from polymarket.contracts.schema import init_db
from polymarket.normalization.markets import derive_market_state_from_executions
from polymarket.normalization.normalizer import Normalizer
from polymarket.normalization.reconciliation import reconcile_roles

GENERATOR_VERSION = SYNTHETIC_GENERATOR_VERSION

W_BASE = 1_720_000_000.0
HOUR = 3600.0
DAY = 86400.0
HORIZON_DAYS = 40

SETUP_LABEL = "SETUP"  # excluded from training and evaluation

ACTORS = {
    "fresh": "0xa-fresh",
    "persistent": "0xa-persist",
    "momentum": "0xa-mom",
    "contrarian": "0xa-contra",
    "inventory": "0xa-inv",
    "building": "0xa-build",
    "liquidity": "0xa-liq",
    "prior": "0xa-prior",
    "mixed": "0xa-mix",
}
MAKER = "0xa-mm"
CROWD = "0xa-crowd"


def _condition(seed: int) -> str:
    return f"0xw{seed}-cond"


def _trade(tx, cond, asset, wallet, side, size, price, ts, log_index=1):
    return {
        "transactionHash": tx, "conditionId": cond, "asset": asset,
        "proxyWallet": wallet, "side": side, "size": float(size),
        "price": round(float(price), 4), "timestamp": float(ts),
        "logIndex": log_index,
    }


class WorldScript:
    """Deterministic script for one world seed."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(10_000 + seed)
        self.cond = _condition(seed)
        self.yes = f"w{seed}-yes"
        self.no = f"w{seed}-no"
        self._build_price_path()
        self._build_news()
        self._build_liquidity_windows()
        self.actor_trades: list[tuple] = []  # (wallet, side, asset, size, ts, label)
        self._script_actors()

    # -- price path --------------------------------------------------------
    def _build_price_path(self) -> None:
        n_segments = 10
        segment = HORIZON_DAYS * DAY / n_segments
        self.segments: list[tuple[float, float, float]] = []  # start, end, slope/day
        price = 0.5
        self.anchors: list[tuple[float, float]] = [(0.0, price)]
        for i in range(n_segments):
            slope = self.rng.choice([-1, 1]) * self.rng.uniform(0.02, 0.06)
            start, end = i * segment, (i + 1) * segment
            self.segments.append((start, end, slope))
            price = min(0.9, max(0.1, price + slope * segment / DAY))
            self.anchors.append((end, price))

    def price(self, offset: float) -> float:
        price = 0.5
        for start, end, slope in self.segments:
            span = min(offset, end) - start
            if span <= 0:
                break
            price += slope * span / DAY
        return min(0.9, max(0.1, price))

    def trend(self, offset: float, horizon: float = 12 * HOUR) -> float:
        return self.price(offset) - self.price(max(0.0, offset - horizon))

    # -- news --------------------------------------------------------------
    def _build_news(self) -> None:
        self.articles: list[dict] = []
        for k in range(6):
            offset = (2 + k * 5.5 + self.rng.uniform(0, 1.5)) * DAY
            direction = 1 if self.rng.random() < 0.5 else -1
            marker = f"Zq{self.seed}x{k}"
            if direction > 0:
                headline = f"Alice Carter {marker} wins election debate"
                body = f"Alice Carter {marker} wins the latest election debate."
            else:
                headline = f"Alice Carter {marker} trails election polls"
                body = f"Alice Carter {marker} trails in the latest election polls."
            self.articles.append(
                {"offset": offset, "direction": direction,
                 "record": {
                     "id": f"n-{self.seed}-{k}",
                     "url": f"https://wire.example/{self.seed}/{k}",
                     "publishedAt": W_BASE + offset - 0.2 * HOUR,
                     "timestampSource": "feed", "timestampConfidence": 0.9,
                     "headline": headline, "body": body,
                 }}
            )

    def _flat(self, offset: float) -> bool:
        return abs(self.trend(offset, horizon=24 * HOUR)) < 0.015

    def _quiet(self, offset: float) -> float:
        """Nearest offset >= the given one that is flat and news-free, so
        the labelled mechanism is unconfounded BY CONSTRUCTION."""
        for k in range(0, 30):
            candidate = offset + k * 6 * HOUR
            if candidate >= (HORIZON_DAYS - 1) * DAY:
                break
            if self._flat(candidate) and self._news_free(candidate):
                return candidate
        return offset

    def _news_free(self, offset: float, window: float = 30 * HOUR) -> bool:
        return all(
            not (0 < offset - a["offset"] < window) for a in self.articles
        )

    # -- liquidity ---------------------------------------------------------
    def _build_liquidity_windows(self) -> None:
        self.liquidity_windows = sorted(
            self.rng.uniform(3, HORIZON_DAYS - 3) * DAY for _ in range(5)
        )

    def _tight(self, offset: float) -> bool:
        return any(0 <= offset - w < 6 * HOUR for w in self.liquidity_windows)

    def books_payloads(self) -> list[tuple[float, list[dict]]]:
        payloads = []
        offset = 1 * HOUR
        while offset < HORIZON_DAYS * DAY:
            mid = self.price(offset)
            if self._tight(offset):
                half_spread, depth = 0.005, 500.0
            else:
                half_spread, depth = 0.04, 60.0
            payloads.append(
                (W_BASE + offset, [{
                    "asset": self.yes,
                    "bids": [{"price": round(mid - half_spread, 4), "size": depth}],
                    "asks": [{"price": round(mid + half_spread, 4), "size": depth}],
                }])
            )
            offset += 6 * HOUR
        return payloads

    # -- actor scripting ---------------------------------------------------
    def _add(self, wallet, side, size, offset, label, asset=None) -> None:
        asset = asset or self.yes
        self.actor_trades.append(
            (wallet, side, asset, size, W_BASE + offset, label)
        )

    def _dir_trade(self, wallet, direction, size, offset, label,
                   unwind: bool = True) -> None:
        # positive-proposition direction expressed on the YES token
        self._add(wallet, "BUY" if direction > 0 else "SELL",
                  size, offset, label)
        if unwind:
            # scripted unwind (SETUP, excluded from evaluation) keeps the
            # signal-driven archetypes flat so their NEXT labelled trade
            # is not confounded by stale exposure
            self._add(wallet, "SELL" if direction > 0 else "BUY",
                      size, offset + 8 * HOUR, SETUP_LABEL)

    def _script_actors(self) -> None:
        rng = self.rng
        # FRESH_NEWS_RESPONSE: quick reaction; one deliberately weak (late)
        for k, article in enumerate(self.articles[::2]):
            lag = (0.3 if k < 2 else 4.0 + rng.uniform(0, 2)) * HOUR
            self._dir_trade(ACTORS["fresh"], article["direction"],
                            5 + rng.uniform(0, 2), article["offset"] + lag,
                            "FRESH_NEWS_RESPONSE")
        # PERSISTENT_NEWS_ADJUSTMENT: 30-60h later, no fresher news around
        for k, article in enumerate(self.articles[1::2]):
            lag = (30 + rng.uniform(0, 30)) * HOUR
            offset = article["offset"] + lag
            if self._news_free(offset, window=24 * HOUR):
                self._dir_trade(ACTORS["persistent"], article["direction"],
                                4 + rng.uniform(0, 2), offset,
                                "PERSISTENT_NEWS_ADJUSTMENT")
        # MARKET_MOMENTUM / CONTRARIAN_REVERSAL at strong-trend, news-free times
        trend_offsets = []
        for start, end, _slope in self.segments:
            candidate = start + 0.7 * (end - start)
            realized = self.trend(candidate, horizon=24 * HOUR)
            # select by REALIZED trailing trend (price clipping can flatten
            # or even reverse the nominal segment slope)
            if abs(realized) >= 0.02 and self._news_free(candidate):
                trend_offsets.append((candidate, realized))
        for candidate, realized in trend_offsets[:4]:
            self._dir_trade(ACTORS["momentum"], 1 if realized > 0 else -1,
                            3 + rng.uniform(0, 2), candidate,
                            "MARKET_MOMENTUM")
            self._dir_trade(ACTORS["contrarian"], -1 if realized > 0 else 1,
                            3 + rng.uniform(0, 2), candidate + 1.5 * HOUR,
                            "CONTRARIAN_REVERSAL")
        # INVENTORY_REBALANCING: build then reduce
        t0 = 4 * DAY + rng.uniform(0, DAY)
        self._add(ACTORS["inventory"], "BUY", 8.0, t0, SETUP_LABEL)
        self._add(ACTORS["inventory"], "BUY", 6.0, t0 + 5 * HOUR, SETUP_LABEL)
        for j, lag_days in enumerate((6.0, 12.0, 20.0)):
            offset = self._quiet(
                t0 + lag_days * DAY + rng.uniform(0, 4) * HOUR
            )
            self._add(ACTORS["inventory"], "SELL", 4.0 - j,
                      offset + j * HOUR, "INVENTORY_REBALANCING")
        # POSITION_BUILDING: keep adding in the same direction
        t1 = 3 * DAY + rng.uniform(0, DAY)
        self._add(ACTORS["building"], "BUY", 5.0, t1, SETUP_LABEL)
        for lag_days in (0.5, 7.0, 15.0, 25.0):
            offset = self._quiet(t1 + lag_days * DAY + rng.uniform(0, 3) * HOUR)
            self._add(ACTORS["building"], "BUY", 4.0, offset,
                      "POSITION_BUILDING")
        # LIQUIDITY_TIMING: trade just after tight-book snapshots, direction
        # alternating (execution timing, not directional belief)
        book_times = []
        t = 1 * HOUR
        while t < HORIZON_DAYS * DAY:
            book_times.append(t)
            t += 6 * HOUR
        for j, window in enumerate(self.liquidity_windows[:4]):
            tight_books = [
                b for b in book_times if window <= b < window + 6 * HOUR
            ]
            if not tight_books:
                continue
            offset = tight_books[0] + 0.3 * HOUR  # just after a TIGHT book
            if self._news_free(offset):
                self._add(ACTORS["liquidity"], "BUY" if j % 2 == 0 else "SELL",
                          2.0, offset, "LIQUIDITY_TIMING")
        # ACTOR_PRIOR: habitual positive buys at flat, news-free times
        prior_offsets = [d * DAY for d in (6.3, 13.7, 21.4, 29.9, 35.2)]
        for j, raw_offset in enumerate(prior_offsets):
            offset = self._quiet(raw_offset)
            # tiny habitual size: net exposure stays below the dust
            # threshold, so ACTOR_PRIOR is not confounded with
            # POSITION_BUILDING (which is a deliberate correlated pair)
            self._add(ACTORS["prior"], "BUY", 0.15,
                      offset + rng.uniform(0, 2) * HOUR,
                      SETUP_LABEL if j == 0 else "ACTOR_PRIOR")
        # MIXED: random times, random directions — must stay ambiguous
        for _ in range(4):
            self._add(ACTORS["mixed"], rng.choice(["BUY", "SELL"]),
                      2 + rng.uniform(0, 3),
                      rng.uniform(2, HORIZON_DAYS - 1) * DAY,
                      "MIXED_OR_UNRESOLVED")
        self.actor_trades.sort(key=lambda t: t[4])

    # -- payload assembly --------------------------------------------------
    def market_payload(self) -> list[dict]:
        return [{
            "id": f"mkt-w{self.seed}", "conditionId": self.cond,
            "question": "Will Alice Carter win the election?",
            "category": "politics",
            "rules": "Resolves positive if Alice Carter wins the election.",
            "resolutionSource": "official-results",
            "resolutionTime": W_BASE + (HORIZON_DAYS + 30) * DAY,
            "createdAt": W_BASE - DAY,
            "tradingEnabled": True, "closed": False, "resolved": False,
            "winningAsset": None, "isCombo": False,
            "tokens": [
                {"token_id": self.yes, "outcome": "Yes", "sign": 1},
                {"token_id": self.no, "outcome": "No", "sign": -1},
            ],
        }]

    def crowd_trades(self) -> list[dict]:
        records = []
        offset = 0.5 * HOUR
        k = 0
        while offset < HORIZON_DAYS * DAY:
            side = "BUY" if self.rng.random() < 0.5 else "SELL"
            records.append(
                _trade(f"0xc{self.seed}-{k}", self.cond, self.yes, CROWD,
                       side, 1.0, self.price(offset), W_BASE + offset)
            )
            offset += 2 * HOUR
            k += 1
        return records

    def taker_and_expanded(self) -> tuple[list[dict], list[dict]]:
        taker = self.crowd_trades()
        expanded = []
        for j, (wallet, side, asset, size, ts, _label) in enumerate(
            self.actor_trades
        ):
            tx = f"0xa{self.seed}-{j}"
            price = self.price(ts - W_BASE)
            record = _trade(tx, self.cond, asset, wallet, side, size, price, ts)
            taker.append(record)
            expanded.append(record)
            maker_side = "SELL" if side == "BUY" else "BUY"
            expanded.append(
                _trade(tx, self.cond, asset, MAKER, maker_side, size, price, ts)
            )
        return taker, expanded

    def activity_payload(self) -> list[dict]:
        events = []
        for j, (wallet, side, asset, size, ts, _label) in enumerate(
            self.actor_trades
        ):
            events.append(
                {"type": "TRADE", "proxyWallet": wallet,
                 "conditionId": self.cond, "asset": asset, "side": side,
                 "size": size, "price": self.price(ts - W_BASE),
                 "timestamp": ts, "transactionHash": f"0xa{self.seed}-{j}"}
            )
        return events

    def true_labels(self) -> dict[tuple[str, float], str]:
        return {
            (wallet, ts): label
            for wallet, _side, _asset, _size, ts, label in self.actor_trades
        }


def build_world(seed: int, path: str) -> tuple[sqlite3.Connection, dict]:
    """Build one normalized world database plus its ground-truth labels."""
    script = WorldScript(seed)
    conn = init_db(path, description=f"reasoning world seed={seed}")
    run_id = start_collector_run(
        conn, "synthetic",
        {"generator": GENERATOR_VERSION, "seed": seed},
    )
    end = W_BASE + (HORIZON_DAYS + 1) * DAY

    def insert(collector, endpoint, params, records, observed_at):
        return insert_raw_response(
            conn, collector_run_id=run_id, collector=collector,
            base_url="synthetic://reasoning-world", endpoint=endpoint,
            params=params, requested_at=observed_at - 0.5,
            received_at=observed_at, http_status=200,
            headers={"x-synthetic-world": str(seed)},
            payload=json.dumps(records, sort_keys=True).encode(),
        )

    insert("markets", "markets", {}, script.market_payload(), W_BASE)
    taker, expanded = script.taker_and_expanded()
    insert("trades_taker", "trades", {"takerOnly": "true"}, taker, end)
    insert("trades_expanded", "trades", {"takerOnly": "false"}, expanded, end)
    insert("activity", "activity", {}, script.activity_payload(), end)
    for observed_at, records in script.books_payloads():
        insert("books", "book", {"asset": script.yes}, records, observed_at)
    for article in script.articles:
        insert("news:wire", "news_feed", {}, [article["record"]],
               W_BASE + article["offset"])
    finish_collector_run(conn, run_id, "succeeded")

    normalizer = Normalizer(conn)
    results = normalizer.normalize_all()
    errors = [e for r in results for e in r.errors]
    if errors:
        raise RuntimeError(f"world {seed} normalization errors: {errors}")
    reconcile_roles(conn)
    derive_market_state_from_executions(conn, script.cond, bucket_seconds=3600.0)
    return conn, {
        "seed": seed,
        "condition_id": script.cond,
        "end_time": end,
        "labels": script.true_labels(),
    }
