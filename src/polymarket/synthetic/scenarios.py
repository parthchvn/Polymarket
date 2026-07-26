"""Deterministic synthetic scenario.

All timestamps derive from BASE; no randomness anywhere.  The scenario
includes every case required by the brief: two markets, signed outcome
tokens, three-plus actors, taker and maker legs, canonical executions,
split/merge/redeem, snapshots, status changes (including a resolution and
a contract revision), order books, relevant and irrelevant news, one
deliberately ambiguous maker/taker case, one collector gap, one
incomplete position history, one news-responsive decision and one
market-state-driven decision.
"""

from __future__ import annotations

BASE = 1_700_000_000.0
HOUR = 3600.0

C1 = "0xc1-election"
C2 = "0xc2-marathon"
T1Y, T1N = "t1-yes", "t1-no"
T2Y, T2N = "t2-yes", "t2-no"

W1, W2, W3 = "0xw1", "0xw2", "0xw3"          # primary actors
WA, WB = "0xwa", "0xwb"                        # ambiguous-case wallets
MM = "0xmm"                                    # maker counterparty
CROWD = "0xcrowd"                              # background taker flow

NEWS_SOURCE = "wire"


def markets_payload_v1() -> list[dict]:
    return [
        {
            "id": "mkt-election",
            "conditionId": C1,
            "question": "Will Alice Carter win the election?",
            "category": "politics",
            "rules": "Resolves positive if Alice Carter wins.",
            "resolutionSource": "official-results",
            "resolutionTime": BASE + 90 * HOUR,
            "createdAt": BASE - 24 * HOUR,
            "tradingEnabled": True,
            "closed": False,
            "resolved": False,
            "winningAsset": None,
            "isCombo": False,
            "tokens": [
                {"token_id": T1Y, "outcome": "Yes", "sign": 1},
                {"token_id": T1N, "outcome": "No", "sign": -1},
            ],
        },
        {
            "id": "mkt-marathon",
            "conditionId": C2,
            "question": "Will it rain during the city marathon on Sunday?",
            "category": "weather",
            "rules": "Resolves positive if measurable rain falls during the marathon.",
            "resolutionSource": "weather-service",
            "resolutionTime": BASE + 50 * HOUR,
            "createdAt": BASE - 24 * HOUR,
            "tradingEnabled": True,
            "closed": False,
            "resolved": False,
            "winningAsset": None,
            "isCombo": False,
            "tokens": [
                {"token_id": T2Y, "outcome": "Yes", "sign": 1},
                {"token_id": T2N, "outcome": "No", "sign": -1},
            ],
        },
    ]


def markets_payload_v2() -> list[dict]:
    """Second observation: revised rules for the election market (new
    contract version) and a resolved marathon market (positive, rain)."""
    records = markets_payload_v1()
    records[0]["rules"] = (
        "Resolves positive if Alice Carter wins. Clarified: runoff counts."
    )
    records[1]["closed"] = True
    records[1]["resolved"] = True
    records[1]["winningAsset"] = T2Y
    return records


def _trade(
    tx: str, condition: str, asset: str, wallet: str, side: str,
    size: float, price: float, ts: float, log_index: int | None,
    record_id: str | None = None,
) -> dict:
    record = {
        "transactionHash": tx,
        "conditionId": condition,
        "asset": asset,
        "proxyWallet": wallet,
        "side": side,
        "size": size,
        "price": price,
        "timestamp": ts,
    }
    if log_index is not None:
        record["logIndex"] = log_index
    if record_id is not None:
        record["id"] = record_id
    return record


# Actor trades: (tx, condition, asset, wallet, side, size, price, hours, log_index)
ACTOR_TRADES = [
    # w2 market-state-driven decision: sells YES after the price slide,
    # with no relevant news in its lookback window before BASE+10h.
    ("0xt-w2a", C1, T1Y, W2, "SELL", 8.0, 0.36, 10.0, 1),
    # w1 buys marathon YES
    ("0xt-w1b", C2, T2Y, W1, "BUY", 5.0, 0.40, 20.0, 1),
    # w2 buys marathon NO (negative-token trade: price complement applies)
    ("0xt-w2b", C2, T2N, W2, "BUY", 6.0, 0.55, 22.0, 1),
    # w1 news-driven decision: buys election YES shortly after the
    # debate-win article first observed at BASE+29.5h.
    ("0xt-w1a", C1, T1Y, W1, "BUY", 12.0, 0.55, 30.0, 1),
    ("0xt-w1a2", C1, T1Y, W1, "BUY", 4.0, 0.57, 30.2, 1),
    # w3 buys election NO after the negative-poll article at BASE+39h.
    ("0xt-w3a", C1, T1N, W3, "BUY", 7.0, 0.45, 40.0, 1),
    # later single-leg episodes
    ("0xt-w1c", C1, T1Y, W1, "BUY", 3.0, 0.62, 50.0, 1),
    ("0xt-w2c", C1, T1Y, W2, "BUY", 5.0, 0.63, 55.0, 1),
    ("0xt-w3b", C2, T2Y, W3, "SELL", 2.0, 0.70, 35.0, 1),
    ("0xt-w2d", C1, T1Y, W2, "SELL", 4.0, 0.61, 60.0, 1),
]

# Background crowd flow shaping the c1 price path (taker view only).
CROWD_TRADES = [
    ("0xt-cr1", C1, T1Y, CROWD, "BUY", 2.0, 0.50, 1.0, 1),
    ("0xt-cr2", C1, T1Y, CROWD, "SELL", 3.0, 0.44, 4.0, 1),
    ("0xt-cr3", C1, T1Y, CROWD, "SELL", 2.5, 0.38, 7.0, 1),
    ("0xt-cr4", C1, T1Y, CROWD, "SELL", 2.0, 0.35, 9.0, 1),
    ("0xt-cr5", C1, T1Y, CROWD, "BUY", 2.0, 0.52, 29.8, 1),
    ("0xt-cr6", C1, T1Y, CROWD, "BUY", 2.0, 0.60, 33.0, 1),
    ("0xt-cr7", C2, T2Y, CROWD, "BUY", 1.5, 0.42, 2.0, 1),
    ("0xt-cr8", C2, T2Y, CROWD, "BUY", 1.0, 0.55, 21.0, 1),
    ("0xt-cr9", C2, T2Y, CROWD, "BUY", 1.0, 0.68, 34.0, 1),
    ("0xt-cr10", C1, T1Y, CROWD, "BUY", 1.0, 0.62, 48.0, 1),
    ("0xt-cr11", C1, T1Y, CROWD, "BUY", 1.0, 0.63, 54.0, 1),
    ("0xt-cr12", C1, T1Y, CROWD, "SELL", 1.0, 0.60, 59.0, 1),
]

# Deliberately ambiguous maker/taker case: two executions in ONE
# transaction with identical size/price/timestamp and no log index —
# expanded legs cannot be uniquely matched and must stay 'unknown'.
AMBIGUOUS_TX = "0xt-ambig"
AMBIGUOUS_TAKER_RECORDS = [
    _trade(AMBIGUOUS_TX, C1, T1Y, WA, "BUY", 2.0, 0.50, BASE + 12 * HOUR, None),
    _trade(AMBIGUOUS_TX, C1, T1Y, WB, "BUY", 2.0, 0.50, BASE + 12 * HOUR, None),
]
AMBIGUOUS_EXPANDED_RECORDS = [
    _trade(AMBIGUOUS_TX, C1, T1Y, WA, "BUY", 2.0, 0.50, BASE + 12 * HOUR, None),
    _trade(AMBIGUOUS_TX, C1, T1Y, WB, "BUY", 2.0, 0.50, BASE + 12 * HOUR, None),
]


def taker_trades_payload() -> list[dict]:
    records = []
    for tx, cond, asset, wallet, side, size, price, hours, li in (
        ACTOR_TRADES + CROWD_TRADES
    ):
        records.append(
            _trade(tx, cond, asset, wallet, side, size, price,
                   BASE + hours * HOUR, li, record_id=f"src-{tx}")
        )
    records.extend(AMBIGUOUS_TAKER_RECORDS)
    return records


def expanded_trades_payload() -> list[dict]:
    """Expanded view: actor taker legs plus maker counterparty legs."""
    records = []
    for tx, cond, asset, wallet, side, size, price, hours, li in ACTOR_TRADES:
        ts = BASE + hours * HOUR
        records.append(_trade(tx, cond, asset, wallet, side, size, price, ts, li,
                              record_id=f"src-{tx}"))
        maker_side = "SELL" if side == "BUY" else "BUY"
        records.append(_trade(tx, cond, asset, MM, maker_side, size, price, ts, li))
    records.extend(AMBIGUOUS_EXPANDED_RECORDS)
    return records


def activity_payload() -> list[dict]:
    events: list[dict] = []
    for tx, cond, asset, wallet, side, size, price, hours, _li in ACTOR_TRADES:
        events.append(
            {"type": "TRADE", "proxyWallet": wallet, "conditionId": cond,
             "asset": asset, "side": side, "size": size, "price": price,
             "timestamp": BASE + hours * HOUR, "transactionHash": tx}
        )
    events.append(
        {"type": "SPLIT", "proxyWallet": W1, "conditionId": C1, "size": 10.0,
         "timestamp": BASE + 5 * HOUR, "transactionHash": "0xa-split"}
    )
    events.append(
        {"type": "MERGE", "proxyWallet": W1, "conditionId": C1, "size": 4.0,
         "timestamp": BASE + 8 * HOUR, "transactionHash": "0xa-merge"}
    )
    # REDEEM after the marathon market resolved (status observed BASE+50h).
    events.append(
        {"type": "REDEEM", "proxyWallet": W1, "conditionId": C2,
         "asset": T2Y, "size": 5.0, "timestamp": BASE + 52 * HOUR,
         "transactionHash": "0xa-redeem"}
    )
    # Incomplete position history: unknown semantics stay unresolved.
    events.append(
        {"type": "CONVERT", "proxyWallet": W3, "conditionId": C1,
         "size": 1.0, "timestamp": BASE + 15 * HOUR,
         "transactionHash": "0xa-convert"}
    )
    return events


def positions_payload() -> list[dict]:
    """Platform snapshots observed at BASE+70h (see fixtures)."""
    return [
        # w1 c1: split +10 both, merge -4 both, buys 12+4+3 YES = +25 yes / +6 no
        {"proxyWallet": W1, "asset": T1Y, "size": 25.0},
        {"proxyWallet": W1, "asset": T1N, "size": 6.0},
        # w1 c2: buy 5 yes, redeem -5 => 0 (omitted: platform reports nothing)
        # w2 c1: -8 +5 -4 = -7 yes
        {"proxyWallet": W2, "asset": T1Y, "size": -7.0},
        {"proxyWallet": W2, "asset": T2N, "size": 6.0},
    ]


def books_payloads() -> list[tuple[float, list[dict]]]:
    return [
        (BASE + 1 * HOUR, [
            {"asset": T1Y,
             "bids": [{"price": 0.49, "size": 100.0}, {"price": 0.48, "size": 50.0}],
             "asks": [{"price": 0.51, "size": 90.0}, {"price": 0.52, "size": 60.0}]},
        ]),
        (BASE + 25 * HOUR, [
            {"asset": T1Y,
             "bids": [{"price": 0.44, "size": 80.0}],
             "asks": [{"price": 0.47, "size": 70.0}]},
            {"asset": T2Y,
             "bids": [{"price": 0.52, "size": 40.0}],
             "asks": [{"price": 0.56, "size": 45.0}]},
        ]),
        (BASE + 45 * HOUR, [
            {"asset": T1Y,
             "bids": [{"price": 0.60, "size": 120.0}],
             "asks": [{"price": 0.63, "size": 110.0}]},
        ]),
    ]


def news_payloads() -> list[tuple[float, list[dict]]]:
    """(first_observed_at, records) pairs — availability is collector time."""
    return [
        (BASE + 9.5 * HOUR, [
            {"id": "n-storm", "url": "https://wire.example/storm",
             "publishedAt": BASE + 9.2 * HOUR, "timestampSource": "feed",
             "timestampConfidence": 0.9,
             "headline": "Storm forecast worsens for city marathon",
             "body": "Forecasters expect heavy rain during the marathon."},
        ]),
        (BASE + 12 * HOUR, [
            {"id": "n-bakery", "url": "https://wire.example/bakery",
             "publishedAt": BASE + 11.8 * HOUR, "timestampSource": "feed",
             "timestampConfidence": 0.9,
             "headline": "Local bakery celebrates anniversary",
             "body": "A beloved bakery marks fifty years of pastries."},
        ]),
        (BASE + 29.5 * HOUR, [
            {"id": "n-debate", "url": "https://wire.example/debate",
             "publishedAt": BASE + 29.0 * HOUR, "timestampSource": "feed",
             "timestampConfidence": 0.9,
             "headline": "Alice Carter wins key debate before election",
             "body": "Alice Carter wins the final debate, boosting her election bid."},
        ]),
        (BASE + 39 * HOUR, [
            {"id": "n-polls", "url": "https://wire.example/polls",
             "publishedAt": BASE + 38.5 * HOUR, "timestampSource": "feed",
             "timestampConfidence": 0.9,
             "headline": "Alice Carter trails in new election polls",
             "body": "Alice Carter trails her rival in the latest election polls."},
        ]),
    ]


# Collector gap: order-book outage for the marathon market.
GAP = {
    "collector": "books",
    "surface": "order_books",
    "object_id": C2,
    "gap_start": BASE + 30 * HOUR,
    "gap_end": BASE + 33 * HOUR,
    "reason": "synthetic collector outage",
}
