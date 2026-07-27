"""Five-minute liquidity bars — the screening paper's data contract.

For each (condition, 5-minute bin) the bar carries the four
liquidity-driven variables the jump model clusters on, adapted from
equities to bounded prediction-market prices:

* **spread** (phi): mean of the last-observed bid-ask spread per book
  observation inside the bin, both raw and in ticks when the tick size
  is known;
* **turnover** (V): total executed notional inside the bin, from
  canonical executions only (actor legs are never mixed in — expanded
  counterparty legs double-count);
* **volatility** (sigma): realized variance of LOG-ODDS mid-price
  moves within the bin.  Prices live on (0, 1), so returns are computed
  on the logit ell = log(p / (1 - p)) with p clipped to
  [CLIP, 1 - CLIP]; the bar also stores the logit OHCL;
* **book size** (B): mean volume at the best bid and ask (the paper's
  definition), stored separately from mean TOTAL depth.

Coverage honesty: ``coverage_complete`` requires at least one book
observation inside the bin and no blocking collector gap overlapping
it.  Bars with zero book observations still record execution turnover
but carry NULL book statistics — missing data is not zero.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

from polymarket.analysis.versioning import feature_version_hash

BIN_SECONDS = 300.0
LOGIT_CLIP = 1e-3


def logit(price: float, clip: float = LOGIT_CLIP) -> float:
    p = min(max(price, clip), 1.0 - clip)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class LiquidityBarConfig:
    bin_seconds: float = BIN_SECONDS
    logit_clip: float = LOGIT_CLIP


def build_liquidity_bars(
    conn: sqlite3.Connection,
    condition_id: str,
    *,
    start: float | None = None,
    end: float | None = None,
    config: LiquidityBarConfig = LiquidityBarConfig(),
) -> int:
    """Build (idempotently upsert) liquidity bars for a condition.

    Book statistics come from the positive outcome token's snapshots;
    turnover comes from canonical executions.  Bin boundaries are
    aligned to epoch multiples of ``bin_seconds``.
    """
    positive = conn.execute(
        "SELECT asset FROM outcome_tokens WHERE condition_id = ? "
        "AND outcome_sign = 1 ORDER BY mapping_effective_from LIMIT 1",
        (condition_id,),
    ).fetchone()
    books = conn.execute(
        """
        SELECT observed_at, best_bid, best_ask, spread, bid_depth,
               ask_depth, imbalance, best_bid_size, best_ask_size,
               tick_size
        FROM order_book_snapshots WHERE asset = ?
        ORDER BY observed_at
        """,
        (positive[0],),
    ).fetchall() if positive else []
    executions = conn.execute(
        "SELECT ts, notional FROM canonical_executions "
        "WHERE condition_id = ? ORDER BY ts",
        (condition_id,),
    ).fetchall()

    if start is None:
        candidates = [r[0] for r in books] + [r[0] for r in executions]
        if not candidates:
            return 0
        start = min(candidates)
    if end is None:
        candidates = [r[0] for r in books] + [r[0] for r in executions]
        end = max(candidates) + config.bin_seconds
    first_bin = math.floor(start / config.bin_seconds) * config.bin_seconds

    bins: dict[float, dict] = {}

    def bin_of(ts: float) -> dict | None:
        bin_start = math.floor(ts / config.bin_seconds) * config.bin_seconds
        if bin_start < first_bin or bin_start >= end:
            return None
        return bins.setdefault(bin_start, {
            "spreads": [], "spread_ticks": [], "best_sizes": [],
            "total_depths": [], "imbalances": [], "logits": [],
            "turnover": 0.0, "executions": 0,
        })

    for (observed_at, bid, ask, spread, bid_depth, ask_depth, imbalance,
         best_bid_size, best_ask_size, tick_size) in books:
        bucket = bin_of(observed_at)
        if bucket is None:
            continue
        if spread is not None:
            bucket["spreads"].append(spread)
            if tick_size:
                bucket["spread_ticks"].append(spread / tick_size)
        if best_bid_size is not None and best_ask_size is not None:
            bucket["best_sizes"].append(
                (best_bid_size + best_ask_size) / 2.0
            )
        if bid_depth is not None and ask_depth is not None:
            bucket["total_depths"].append(bid_depth + ask_depth)
        if imbalance is not None:
            bucket["imbalances"].append(imbalance)
        if bid is not None and ask is not None:
            bucket["logits"].append(
                logit((bid + ask) / 2.0, config.logit_clip)
            )

    for ts, notional in executions:
        bucket = bin_of(ts)
        if bucket is None:
            continue
        bucket["turnover"] += float(notional or 0.0)
        bucket["executions"] += 1

    from polymarket.analysis.reader import SQLiteNormalizedReader

    reader = SQLiteNormalizedReader(conn)
    feature_version = feature_version_hash(
        {"liquidity_bars": {"bin_seconds": config.bin_seconds,
                            "logit_clip": config.logit_clip}}
    )
    now = time.time()
    written = 0

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    for bin_start in sorted(bins):
        bucket = bins[bin_start]
        bin_end = bin_start + config.bin_seconds
        logits = bucket["logits"]
        realized_variance = (
            sum(
                (b - a) ** 2 for a, b in zip(logits, logits[1:])
            ) if len(logits) >= 2 else None
        )
        blocked = bool(
            reader.blocking_gaps(condition_id, bin_start, bin_end)
        )
        coverage_complete = bool(logits) and not blocked
        conn.execute(
            """
            INSERT OR REPLACE INTO liquidity_bars
                (condition_id, bin_start, bin_end, bin_seconds,
                 logit_open, logit_high, logit_low, logit_close,
                 realized_variance, turnover_notional, spread_mean,
                 spread_ticks_mean, best_book_size_mean,
                 total_depth_mean, imbalance_mean,
                 book_observation_count, execution_count,
                 coverage_complete, feature_version, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?)
            """,
            (
                condition_id, bin_start, bin_end, config.bin_seconds,
                logits[0] if logits else None,
                max(logits) if logits else None,
                min(logits) if logits else None,
                logits[-1] if logits else None,
                realized_variance,
                bucket["turnover"],
                mean(bucket["spreads"]),
                mean(bucket["spread_ticks"]),
                mean(bucket["best_sizes"]),
                mean(bucket["total_depths"]),
                mean(bucket["imbalances"]),
                len(logits),
                bucket["executions"],
                int(coverage_complete),
                feature_version,
                now,
            ),
        )
        written += 1
    conn.commit()
    return written
