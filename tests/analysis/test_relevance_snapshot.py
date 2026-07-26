"""As-of relevance snapshot: contract-version matching, strict cutoff,
latest-per-family selection, no evidence multiplication."""

from __future__ import annotations

import pytest

from polymarket.analysis.reader import SQLiteNormalizedReader
from polymarket.contracts.schema import init_db

MARKET = "mkt-x"
T0 = 1_000_000.0


def _insert_judgment(conn, *, family, version_seq, computed_at,
                     rel_class="supports_positive", method="rule",
                     model_version="rules-1"):
    conn.execute(
        """
        INSERT INTO relevance_judgments
            (event_family_id, market_id, contract_version_seq, computed_at,
             rel_class, rel_score, direction, novelty, surprise, method,
             model_version, evidence_json)
        VALUES (?, ?, ?, ?, ?, 0.8, 1.0, NULL, NULL, ?, ?, '{}')
        """,
        (family, MARKET, version_seq, computed_at, rel_class, method,
         model_version),
    )


@pytest.fixture()
def snapshot_db(tmp_path):
    conn = init_db(str(tmp_path / "snap.sqlite"), description="snapshot test")
    # family A: obsolete v1 judgment plus two v2 recomputations
    _insert_judgment(conn, family="fam-A", version_seq=1, computed_at=T0 + 10)
    _insert_judgment(conn, family="fam-A", version_seq=2, computed_at=T0 + 20,
                     rel_class="supports_negative")
    _insert_judgment(conn, family="fam-A", version_seq=2, computed_at=T0 + 30)
    # family B: single v2 judgment exactly AT the decision time
    _insert_judgment(conn, family="fam-B", version_seq=2, computed_at=T0 + 100)
    conn.commit()
    return SQLiteNormalizedReader(conn)


def test_obsolete_contract_version_judgment_is_excluded(snapshot_db):
    rows, fallback = snapshot_db.relevance_snapshot_asof(
        MARKET, 2, T0 + 100
    )
    assert not fallback
    assert all(row["contract_version_seq"] == 2 for row in rows)


def test_judgment_at_exact_decision_time_is_excluded(snapshot_db):
    rows, _ = snapshot_db.relevance_snapshot_asof(MARKET, 2, T0 + 100)
    families = {row["event_family_id"] for row in rows}
    assert "fam-B" not in families  # computed_at == cutoff: excluded
    later, _ = snapshot_db.relevance_snapshot_asof(MARKET, 2, T0 + 101)
    assert "fam-B" in {row["event_family_id"] for row in later}


def test_latest_eligible_judgment_per_family_is_selected(snapshot_db):
    rows, _ = snapshot_db.relevance_snapshot_asof(MARKET, 2, T0 + 100)
    fam_a = [r for r in rows if r["event_family_id"] == "fam-A"]
    assert len(fam_a) == 1
    assert fam_a[0]["computed_at"] == T0 + 30  # newest recomputation wins


def test_repeated_recomputations_do_not_multiply_evidence(snapshot_db):
    rows, _ = snapshot_db.relevance_snapshot_asof(MARKET, 2, T0 + 100)
    families = [row["event_family_id"] for row in rows]
    assert len(families) == len(set(families))  # exactly one per family


def test_intermediate_cutoff_selects_that_eras_judgment(snapshot_db):
    rows, _ = snapshot_db.relevance_snapshot_asof(MARKET, 2, T0 + 25)
    fam_a = [r for r in rows if r["event_family_id"] == "fam-A"]
    assert len(fam_a) == 1
    assert fam_a[0]["computed_at"] == T0 + 20
    assert fam_a[0]["rel_class"] == "supports_negative"


def test_version_fallback_is_flagged_and_optional(snapshot_db):
    # version 3 has no judgments at all
    rows, fallback = snapshot_db.relevance_snapshot_asof(MARKET, 3, T0 + 100)
    assert fallback and rows  # documented fallback path, flagged
    strict, fallback_strict = snapshot_db.relevance_snapshot_asof(
        MARKET, 3, T0 + 100, allow_version_fallback=False
    )
    assert strict == [] and not fallback_strict


def test_method_and_model_version_filters(snapshot_db):
    rows, _ = snapshot_db.relevance_snapshot_asof(
        MARKET, 2, T0 + 100, method="rule", model_version="rules-1"
    )
    assert rows
    none_rows, fallback = snapshot_db.relevance_snapshot_asof(
        MARKET, 2, T0 + 100, method="llm", allow_version_fallback=False
    )
    assert none_rows == []
