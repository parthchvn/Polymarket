"""Time-decayed news feature tests.

No Ollama server is required: rows are deterministic fakes and the
synthetic fixture is fully rule-based.
"""

import pytest

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import build_decision_episodes
from polymarket.analysis.features import (
    NEWS_DECAY_HALF_LIVES,
    NEWS_DECAY_MAX_AGE,
    compute_features,
    decayed_news_signals,
    feature_manifest,
    half_life_decay,
    relevance_confidence,
)
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.synthetic import scenarios as sc

HOUR = 3600.0
DAY = 86400.0
T = 1_000_000.0


def _row(
    age_seconds: float,
    *,
    rel_score: float = 0.8,
    direction: float = 1.0,
    novelty: float | None = 1.0,
    family: str | None = "fam-1",
    rel_class: str = "supports_positive",
    evidence_json: str | None = None,
) -> dict:
    return {
        "rel_score": rel_score,
        "direction": direction,
        "novelty": novelty,
        "computed_at": T - age_seconds,
        "event_family_id": family,
        "rel_class": rel_class,
        "evidence_json": evidence_json,
    }


def _signals(rows, half_life=DAY, max_age=NEWS_DECAY_MAX_AGE):
    return decayed_news_signals(
        rows, decision_time=T, half_life_seconds=half_life,
        max_age_seconds=max_age,
    )


# ---- tests 1-3: the decay curve itself ------------------------------------
def test_immediate_news_decays_to_nearly_one():
    assert half_life_decay(1.0, DAY) == pytest.approx(1.0, abs=1e-4)


def test_one_half_life_is_half():
    assert half_life_decay(24 * HOUR, DAY) == pytest.approx(0.5)


def test_two_half_lives_is_quarter():
    assert half_life_decay(48 * HOUR, DAY) == pytest.approx(0.25)


# ---- test 4: the motivating case ------------------------------------------
def test_tomorrows_trade_includes_todays_news_at_half_weight():
    signals = _signals([_row(24 * HOUR, rel_score=0.8)])
    assert signals["positive"] == pytest.approx(0.4)
    assert signals["signed"] == pytest.approx(0.4)
    assert signals["negative"] == 0.0


# ---- tests 5-7: temporal eligibility --------------------------------------
def test_exact_cutoff_row_contributes_zero():
    signals = _signals([_row(0.0)])  # computed_at == decision_time
    assert signals == {"signed": 0.0, "positive": 0.0, "negative": 0.0,
                       "family_count": 0.0}


def test_future_row_contributes_zero():
    signals = _signals([_row(-1.0)])  # computed_at > decision_time
    assert signals["positive"] == 0.0
    assert signals["family_count"] == 0.0


def test_rows_older_than_max_age_excluded():
    signals = _signals([_row(29 * DAY)])
    assert signals["positive"] == 0.0
    just_inside = _signals([_row(27 * DAY)])
    assert just_inside["positive"] > 0.0


def test_irrelevant_rows_excluded():
    signals = _signals([_row(1 * HOUR, rel_class="irrelevant")])
    assert signals["family_count"] == 0.0


# ---- tests 8-10: event-family aggregation ---------------------------------
def test_duplicate_family_takes_max_not_sum():
    one = _signals([_row(2 * HOUR)])
    two = _signals([_row(2 * HOUR), _row(2 * HOUR)])
    assert two["positive"] == pytest.approx(one["positive"])


def test_independent_families_add():
    one = _signals([_row(2 * HOUR, family="fam-1")])
    two = _signals([
        _row(2 * HOUR, family="fam-1"),
        _row(2 * HOUR, family="fam-2"),
    ])
    assert two["positive"] == pytest.approx(2 * one["positive"])
    assert two["family_count"] == 2.0


def test_contradictory_evidence_stays_visible():
    signals = _signals([
        _row(2 * HOUR, direction=1.0),
        _row(2 * HOUR, direction=-1.0, rel_class="supports_negative"),
    ])
    assert signals["positive"] > 0.0
    assert signals["negative"] > 0.0
    assert signals["signed"] == pytest.approx(
        signals["positive"] - signals["negative"]
    )


def test_missing_family_rows_do_not_collapse_together():
    signals = _signals([
        _row(2 * HOUR, family=None),
        _row(2 * HOUR, family=None),
    ])
    # stable per-row fallback keys: two families, additive
    assert signals["family_count"] == 2.0


# ---- tests 11-13: robustness of row fields --------------------------------
def test_missing_novelty_defaults_to_one():
    with_novelty = _signals([_row(2 * HOUR, novelty=1.0)])
    without = _signals([_row(2 * HOUR, novelty=None)])
    assert without["positive"] == pytest.approx(with_novelty["positive"])
    assert without["positive"] > 0.0


def test_malformed_evidence_json_defaults_confidence():
    assert relevance_confidence(_row(1, evidence_json="{not json")) == 1.0
    assert relevance_confidence(_row(1, evidence_json=None)) == 1.0
    assert relevance_confidence(_row(1, evidence_json='"a string"')) == 1.0
    signals = _signals([_row(2 * HOUR, evidence_json="{broken")])
    assert signals["positive"] > 0.0


def test_confidence_scaling_halves_contribution():
    full = _signals([_row(2 * HOUR, evidence_json='{"confidence": 1.0}')])
    half = _signals([_row(2 * HOUR, evidence_json='{"confidence": 0.5}')])
    assert half["positive"] == pytest.approx(full["positive"] / 2)


# ---- test 14: manifest ----------------------------------------------------
def test_feature_manifest_exposes_decay_config():
    manifest = feature_manifest()
    news = set(manifest["groups"]["news"])
    for label in NEWS_DECAY_HALF_LIVES:
        for kind in ("signed", "positive", "negative"):
            assert f"news_decay_{kind}_{label}" in news
    assert {"news_recent_missing", "news_decay_missing",
            "news_missing"} <= news
    decay = manifest["news_decay"]
    assert decay["news_decay_max_age_seconds"] == NEWS_DECAY_MAX_AGE
    assert decay["news_decay_half_lives_seconds"] == NEWS_DECAY_HALF_LIVES
    assert decay["news_decay_aggregation"] == (
        "event_family_max_positive_negative"
    )


# ---- test 15: synthetic end-to-end behaviour ------------------------------
def test_decay_features_on_synthetic_world(synthetic_db_path):
    reader = SQLiteNormalizedReader(synthetic_db_path)
    episodes = build_decision_episodes(reader, end_time=sc.BASE + 80 * sc.HOUR)
    by_key = {}
    for episode in episodes:
        if episode.direction is None:
            continue
        context = build_context(reader, episode)
        by_key[(episode.actor_id, episode.anchor_time)] = compute_features(
            context, episode
        )

    # w1's news-driven decision 30 minutes after the debate article:
    # decay weight is near 1, so the decayed positive signal is close to
    # the raw relevance channel and clearly nonzero.
    w1 = by_key[(sc.W1, sc.BASE + 30 * sc.HOUR)]
    assert w1["news_decay_missing"] == 0.0
    assert w1["news_decay_positive_24h"] > 0.0
    assert w1["news_decay_signed_24h"] > 0.0
    # shorter half-life decays at least as fast as longer ones
    assert w1["news_decay_positive_6h"] <= w1["news_decay_positive_168h"]

    # w2's election decision at BASE+60h is the motivating case: the
    # POSITIVE debate article is ~30.5h old — outside the raw 24h window
    # (only the negative polls article remains there) — yet it still
    # contributes a decayed positive signal, so contradictory evidence
    # from "yesterday" stays visible instead of vanishing at 24h.
    w2 = by_key[(sc.W2, sc.BASE + 60 * sc.HOUR)]
    assert w2["news_recent_missing"] == 0.0
    assert w2["news_direction"] == -1.0        # raw window sees polls only
    assert w2["news_decay_positive_24h"] > 0.0  # decayed debate article
    assert w2["news_decay_negative_24h"] > 0.0  # decayed polls article
    assert w2["news_missing"] == w2["news_decay_missing"] == 0.0

    # persistent semantic score untouched in the database
    stored = reader.conn.execute(
        "SELECT rel_score FROM relevance_judgments "
        "WHERE rel_class = 'supports_positive'"
    ).fetchone()[0]
    assert stored == pytest.approx(0.4)  # unchanged rule-based value
