"""Real-decision annotation tooling — Track A of the real reasoning
dataset.

The synthetic-trained template classifier needs a REAL evaluation set
before any real-data reasoning claim.  This module:

* samples real decision episodes, stratified over (market, time
  bucket, dominant Layer-1 attribution channel) so the batch is not
  one market's one week;
* renders each as a STRICT pre-decision record — everything a human
  reviewer may see is exactly what the model may see (contexts built
  through the same ``build_context`` path, so the temporal assertion
  runs); no outcome, no resolution, no post-decision prices;
* exports reviewer files (JSONL, one blank ``label`` per decision)
  and imports two or more reviewers' files back, computing per-label
  agreement and Cohen's kappa, persisting every judgment and a
  consensus view.

Label set (the ontology under test — 'insufficient_evidence' is a
first-class answer, not a failure):

    FRESH_NEWS_RESPONSE, PERSISTENT_NEWS_ADJUSTMENT, MARKET_MOMENTUM,
    CONTRARIAN_REVERSAL, INVENTORY_REBALANCING, POSITION_BUILDING,
    LIQUIDITY_TIMING, ACTOR_PRIOR, MIXED, INSUFFICIENT_EVIDENCE
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import build_decision_episodes
from polymarket.analysis.features import compute_features
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.analysis.reasoning import (
    ATTRIBUTION_GROUPS,
)
from polymarket.analysis.versioning import feature_version_hash
from polymarket.collection.canonical import canonical_json, namespace_id

ANNOTATION_LABELS = (
    "FRESH_NEWS_RESPONSE", "PERSISTENT_NEWS_ADJUSTMENT",
    "MARKET_MOMENTUM", "CONTRARIAN_REVERSAL", "INVENTORY_REBALANCING",
    "POSITION_BUILDING", "LIQUIDITY_TIMING", "ACTOR_PRIOR",
    "MIXED", "INSUFFICIENT_EVIDENCE",
)


def _dominant_channel(features: dict[str, float]) -> str:
    """Cheap stratification key: the attribution channel with the
    largest mean |z|-ish magnitude of its populated features."""
    best, best_score = "base", -1.0
    for channel, names in ATTRIBUTION_GROUPS.items():
        values = [abs(features.get(n, 0.0)) for n in names
                  if not n.endswith("_missing")]
        score = sum(values) / len(values) if values else 0.0
        if score > best_score:
            best, best_score = channel, score
    return best


def sample_annotation_batch(
    conn: sqlite3.Connection,
    *,
    n: int = 300,
    end_time: float | None = None,
    seed: int = 1337,
    mode_run_id: str | None = None,
) -> dict:
    """Deterministic stratified sample of real decisions with strict
    pre-decision records rendered for human review."""
    import random

    reader = SQLiteNormalizedReader(conn)
    end = end_time or time.time()
    episodes = [
        e for e in build_decision_episodes(reader, end_time=end)
        if e.direction in ("positive", "negative")
    ]
    if not episodes:
        raise ValueError("no labeled decision episodes to sample")
    records = []
    for episode in episodes:
        context = build_context(
            reader, episode, mode_run_id=mode_run_id,
            relevance_availability="online_scored",
        )
        features = compute_features(context, episode)
        records.append((episode, context, features))
    # stratify: (condition, utc-day, dominant channel)
    strata: dict[tuple, list] = {}
    for item in records:
        episode, _context, features = item
        key = (
            episode.condition_id,
            int(episode.anchor_time // 86400),
            _dominant_channel(features),
        )
        strata.setdefault(key, []).append(item)
    rng = random.Random(seed)
    ordered_keys = sorted(strata)
    sampled = []
    while len(sampled) < min(n, len(records)):
        for key in ordered_keys:
            bucket = strata[key]
            if bucket and len(sampled) < min(n, len(records)):
                index = rng.randrange(len(bucket))
                sampled.append(bucket.pop(index))
    batch_id = namespace_id(
        "annotation-batch", seed, n, feature_version_hash(),
        *sorted(episode.decision_id for episode, _c, _f in sampled),
    )
    items = []
    for episode, context, features in sampled:
        items.append({
            "batch_id": batch_id,
            "decision_id": episode.decision_id,
            # ---- what the reviewer sees: strict pre-decision only ----
            "decision": {
                "actor_id": episode.actor_id,
                "condition_id": episode.condition_id,
                "decision_time_utc": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.gmtime(episode.anchor_time),
                ),
                "direction": episode.direction,
                "gross_quantity": episode.gross_quantity,
            },
            "market": {
                "question": (context.contract or {}).get("question"),
                "last_price": features.get("mkt_last_price"),
                "return_short": features.get("mkt_return_short"),
                "return_long": features.get("mkt_return_long"),
                "spread": features.get("mkt_spread"),
            },
            "position": {
                "net_proposition": features.get("pos_net_proposition"),
                "gross_exposure": features.get("pos_gross_exposure"),
            },
            "actor_history": {
                "recent_trade_count": features.get(
                    "act_recent_trade_count"
                ),
                "positive_rate": features.get(
                    "base_actor_positive_rate"
                ),
            },
            "news_available_before_decision": [
                {
                    "family": row.get("event_family_id"),
                    "rel_class": row.get("rel_class"),
                    "rel_score": row.get("rel_score"),
                    "age_hours": None,
                }
                for row in context.relevance
            ],
            "paper_state": {
                "liquidity_mode": (
                    context.paper_state.get("liquidity_mode") or {}
                ).get("mode_label_online"),
                "impactful_screens": len(
                    context.paper_state.get("impact_screens", [])
                ),
            },
            "dominant_attribution_channel": _dominant_channel(features),
            # ---- reviewer fills these ----
            "label": "",
            "confidence": None,
            "notes": "",
            "label_options": list(ANNOTATION_LABELS),
        })
    conn.execute(
        "INSERT OR REPLACE INTO annotation_batches (batch_id, "
        "created_at, sampler_config_json, n_decisions, feature_version)"
        " VALUES (?, ?, ?, ?, ?)",
        (batch_id, time.time(),
         canonical_json({"n": n, "seed": seed, "end_time": end,
                         "mode_run_id": mode_run_id,
                         "stratification":
                             "condition x utc_day x channel"}),
         len(items), feature_version_hash()),
    )
    conn.commit()
    return {"batch_id": batch_id, "items": items}


def import_annotations(
    conn: sqlite3.Connection,
    batch_id: str,
    reviewer_files: dict[str, str],
) -> dict:
    """Import >=1 reviewers' completed JSONL files; persist every
    judgment; compute agreement when >=2 reviewers overlap."""
    now = time.time()
    by_reviewer: dict[str, dict[str, str]] = {}
    for reviewer, path in reviewer_files.items():
        labels: dict[str, str] = {}
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                label = (item.get("label") or "").strip().upper()
                if not label:
                    continue
                if label not in ANNOTATION_LABELS:
                    raise ValueError(
                        f"reviewer {reviewer}: unknown label "
                        f"{label!r} for {item['decision_id']}"
                    )
                labels[item["decision_id"]] = label
                conn.execute(
                    "INSERT OR REPLACE INTO annotations (batch_id, "
                    "decision_id, reviewer, label, confidence, notes, "
                    "imported_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (batch_id, item["decision_id"], reviewer, label,
                     item.get("confidence"), item.get("notes"), now),
                )
        by_reviewer[reviewer] = labels
    conn.commit()
    report: dict = {
        "batch_id": batch_id,
        "reviewers": {
            reviewer: len(labels)
            for reviewer, labels in by_reviewer.items()
        },
    }
    reviewers = sorted(by_reviewer)
    if len(reviewers) >= 2:
        a, b = reviewers[0], reviewers[1]
        shared = sorted(
            set(by_reviewer[a]) & set(by_reviewer[b])
        )
        if shared:
            pairs = [(by_reviewer[a][d], by_reviewer[b][d])
                     for d in shared]
            report["pairwise"] = {
                "reviewers": [a, b],
                "n_shared": len(shared),
                "raw_agreement": sum(
                    1 for x, y in pairs if x == y
                ) / len(pairs),
                "cohens_kappa": cohens_kappa(pairs),
                "per_label_agreement": _per_label(pairs),
            }
            consensus = {
                d: by_reviewer[a][d] for d in shared
                if by_reviewer[a][d] == by_reviewer[b][d]
            }
            report["consensus_labels"] = len(consensus)
    return report


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n)
        for label in set(counts_a) | set(counts_b)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _per_label(pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for label in sorted({a for a, _ in pairs} | {b for _, b in pairs}):
        relevant = [
            (a, b) for a, b in pairs if a == label or b == label
        ]
        agree = sum(1 for a, b in relevant if a == b)
        out[label] = {
            "n": len(relevant),
            "agreement": agree / len(relevant) if relevant else None,
        }
    return out


def gold_labels(
    conn: sqlite3.Connection, batch_id: str
) -> dict[str, str]:
    """Consensus labels: decisions where every reviewer agrees."""
    rows = conn.execute(
        "SELECT decision_id, reviewer, label FROM annotations "
        "WHERE batch_id = ?",
        (batch_id,),
    ).fetchall()
    by_decision: dict[str, set[str]] = {}
    for row in rows:
        by_decision.setdefault(row["decision_id"], set()).add(
            row["label"]
        )
    return {
        decision: labels.pop()
        for decision, labels in by_decision.items()
        if len(labels) == 1
    }
