from __future__ import annotations

import pytest

from polymarket.contracts.schema import init_db
from polymarket.synthetic.fixtures import build_synthetic_fixture


@pytest.fixture
def db(tmp_path):
    conn = init_db(str(tmp_path / "test.sqlite"))
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def synthetic_db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("fixture") / "synthetic.sqlite")
    conn = build_synthetic_fixture(path, overwrite=True)
    conn.close()
    return path
