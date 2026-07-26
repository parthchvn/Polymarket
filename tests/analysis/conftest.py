"""Shared, session-cached synthetic reasoning worlds for analysis tests.

Building worlds and running the full strict pipeline is the expensive
part, so it happens once per session.  Train and test sets are split by
WORLD SEED (0-2 train, 8 test) — no episode crosses the split.
"""

from __future__ import annotations

import numpy as np
import pytest

from polymarket.analysis.reasoning_posterior import (
    POSTERIOR_FEATURES,
    train_template_model,
)
from polymarket.analysis.reasoning_validation import (
    _trainable,
    _world_dataset,
    attach_cross_world_layer1,
)
from polymarket.synthetic.reasoning_worlds import build_world  # noqa: F401

TRAIN_SEEDS = (0, 1, 2)
TEST_SEED = 8


@pytest.fixture(scope="session")
def reasoning_worlds(tmp_path_factory):
    workdir = str(tmp_path_factory.mktemp("reasoning-worlds"))
    train = [_world_dataset(seed, workdir) for seed in TRAIN_SEEDS]
    test = _world_dataset(TEST_SEED, workdir)
    for world in train:
        attach_cross_world_layer1(train, world)
    attach_cross_world_layer1(train, test)
    return {"train": train, "test": test, "workdir": workdir}


@pytest.fixture(scope="session")
def trained_template_model(reasoning_worlds):
    xs, ys = [], []
    for world in reasoning_worlds["train"]:
        world_x, world_y = _trainable(world["rows"])
        xs += world_x
        ys += world_y
    model = train_template_model(xs, ys)
    # temperature calibrated on a slice of TRAIN data (never on the test
    # world) purely so posterior scales are sensible in tests
    X = np.asarray([[x[n] for n in POSTERIOR_FEATURES] for x in xs])
    model.calibrate(X, ys)
    return model
