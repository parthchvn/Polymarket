"""Build the deterministic synthetic fixture database.

The fixture passes synthetic raw responses through the SAME Normalizer
used for real data, then derives execution-based market state and runs
maker/taker reconciliation.
"""

from __future__ import annotations

import os
import sqlite3

from polymarket.contracts.schema import init_db
from polymarket.normalization.markets import derive_market_state_from_executions
from polymarket.normalization.normalizer import Normalizer
from polymarket.normalization.reconciliation import reconcile_roles
from polymarket.synthetic import scenarios as sc
from polymarket.synthetic.generator import generate_raw_world

DEFAULT_FIXTURE_PATH = "fixtures/synthetic_normalized.sqlite"


def build_synthetic_fixture(
    path: str = DEFAULT_FIXTURE_PATH, *, overwrite: bool = False
) -> sqlite3.Connection:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"{path} exists; pass overwrite=True to rebuild"
            )
        os.remove(path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = init_db(path, description="synthetic fixture")
    generate_raw_world(conn)
    normalizer = Normalizer(conn)
    results = normalizer.normalize_all()
    errors = [e for r in results for e in r.errors]
    if errors:
        raise RuntimeError(f"synthetic normalization errors: {errors}")
    reconcile_roles(conn)
    for condition_id in (sc.C1, sc.C2):
        derive_market_state_from_executions(conn, condition_id)
    return conn
