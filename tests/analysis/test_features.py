"""29.9 feature tests."""

import numpy as np

from polymarket.analysis.context import build_context
from polymarket.analysis.decisions import build_decision_episodes
from polymarket.analysis.features import (
    ALL_FEATURES,
    FEATURE_GROUPS,
    compute_features,
    feature_subset,
)
from polymarket.analysis.models import Standardizer
from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.synthetic import scenarios as sc


def _episode_features(synthetic_db_path):
    reader = SQLiteNormalizedReader(synthetic_db_path)
    episodes = build_decision_episodes(
        reader, end_time=sc.BASE + 80 * sc.HOUR
    )
    out = []
    for episode in episodes:
        if episode.direction is None:
            continue
        context = build_context(reader, episode, relevance_availability="retrospective_source")
        out.append((episode, context, compute_features(context, episode)))
    return out


def test_no_future_data_and_complete_feature_set(synthetic_db_path):
    rows = _episode_features(synthetic_db_path)
    assert rows, "expected labeled episodes"
    for _episode, _context, features in rows:
        assert set(features) == set(ALL_FEATURES)
        assert all(isinstance(v, float) for v in features.values())


def test_lookback_windows_respected(synthetic_db_path):
    rows = _episode_features(synthetic_db_path)
    # w2's decision at BASE+10h: the debate article (BASE+29.5h) must NOT
    # be visible; news for that episode reflects only earlier items.
    early = [f for e, _c, f in rows
             if e.actor_id == sc.W2 and e.anchor_time == sc.BASE + 10 * sc.HOUR]
    assert early, "missing w2 early episode"
    features = early[0]
    # only irrelevant/ambiguous marathon news exists before BASE+10h for
    # the ELECTION market; relevance channel must be empty
    assert features["news_missing"] == 1.0
    # w1's decision at BASE+30h sees the debate article
    late = [f for e, _c, f in rows
            if e.actor_id == sc.W1 and e.anchor_time == sc.BASE + 30 * sc.HOUR]
    assert late and late[0]["news_missing"] == 0.0
    assert late[0]["news_direction"] == 1.0


def test_missingness_indicators(synthetic_db_path):
    rows = _episode_features(synthetic_db_path)
    for _episode, context, features in rows:
        if not context.relevance:
            assert features["news_missing"] == 1.0
        if not context.order_books:
            assert features["mkt_spread_missing"] == 1.0


def test_news_and_market_features_remain_separate():
    overlap = set(FEATURE_GROUPS["news"]) & set(FEATURE_GROUPS["market"])
    assert not overlap
    features = {name: 1.0 for name in ALL_FEATURES}
    news_only = feature_subset(features, ["news"])
    assert set(news_only) == set(FEATURE_GROUPS["news"])


def test_training_only_standardization():
    names = ["x", "x_missing"]
    train = np.array([[1.0, 0.0], [3.0, 1.0]])
    test = np.array([[100.0, 1.0]])
    standardizer = Standardizer(names).fit(train)
    transformed = standardizer.transform(test)
    # mean/std from TRAIN only: (100-2)/1 = 98
    assert abs(transformed[0, 0] - 98.0) < 1e-9
    # binary indicator keeps zero-preserving coding
    assert transformed[0, 1] == 1.0
