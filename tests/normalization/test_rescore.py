"""Versioned news relevance rescoring: never rewrites, resumes,
supersedes in as-of snapshots, and the Ollama scorer parses structured
responses (mocked — no live model)."""

from __future__ import annotations

import json
import time

import pytest

from polymarket.contracts.schema import init_db
from polymarket.normalization.news import RELEVANCE_MODEL_VERSION  # noqa: F401
from polymarket.normalization.rescore import make_scorer, rescore_news

T0 = 1_700_000_000.0


@pytest.fixture()
def conn(tmp_path):
    conn = init_db(str(tmp_path / "rescore.sqlite"), description="rescore")
    now = time.time()
    conn.execute(
        "INSERT INTO markets (market_id, condition_id, question, "
        "raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "('m1', '0xc', 'Will X happen?', 1, 0, 'h', 'p', 1, ?)", (now,),
    )
    conn.execute(
        "INSERT INTO contract_versions (market_id, version_seq, "
        "effective_from, first_observed_at, question, rules_text, "
        "content_hash, raw_response_id, parser_version, schema_version, "
        "normalized_at) VALUES ('m1', 1, ?, ?, 'Will X happen?', "
        "'Resolves YES if X.', 'ch', 1, 'p', 1, ?)",
        (T0 - 1000, T0 - 1000, now),
    )
    for i, headline in enumerate(["X moves closer", "Unrelated topic"]):
        article_id = f"a{i}"
        conn.execute(
            "INSERT INTO news_articles (article_id, source_id, "
            "source_url, source_published_at, first_observed_at, "
            "download_completed_at, timestamp_source, timestamp_confidence, "
            "headline, body, "
            "content_hash, raw_response_id, raw_record_index, "
            "raw_record_hash, parser_version, schema_version, "
            "normalized_at) VALUES (?, 's', 'u', ?, ?, ?, 'feed', 0.8, ?, "
            "?, 'ch', 1, 0, 'h', 'p', 1, ?)",
            (article_id, T0 + i, T0 + i + 5, T0 + i + 5, headline, headline, now),
        )
        conn.execute(
            "INSERT INTO news_claims (claim_id, article_id, claim_text, "
            "entities_json, quantities_json, first_available_at, "
            "extractor_version, confidence) VALUES (?, ?, ?, '[]', "
            "'[]', ?, 'x', 0.9)",
            (f"c{i}", article_id, headline, T0 + i + 5),
        )
        conn.execute(
            "INSERT INTO event_families (event_family_id, label, "
            "earliest_available_at, created_by, created_at) VALUES "
            "(?, ?, ?, 'test', ?)",
            (f"f{i}", f"key{i}", T0 + i + 5, now),
        )
        conn.execute(
            "INSERT INTO claim_edges (edge_id, claim_id, event_family_id, "
            "edge_type, effective_from, evidence, confidence) VALUES "
            "(?, ?, ?, 'new', ?, 'k', 0.5)",
            (f"e{i}", f"c{i}", f"f{i}", T0 + i + 5),
        )
    conn.commit()
    return conn


class FakeScorer:
    version = "fake-scorer-1"

    def __init__(self):
        self.calls = 0

    def score(self, claim_text, question, rules_text):
        self.calls += 1
        relevant = "X" in claim_text
        return {
            "rel_class": "supports_positive" if relevant else "irrelevant",
            "rel_score": 0.9 if relevant else 0.05,
            "direction": 1.0 if relevant else 0.0,
            "evidence": {"note": "fake"},
        }


def test_rescore_writes_versioned_judgments_and_counts(conn):
    counters = rescore_news(conn, FakeScorer(), method="fake")
    assert counters["scored"] == 2
    assert counters["by_class"] == {
        "supports_positive": 1, "irrelevant": 1,
    }
    rows = conn.execute(
        "SELECT method, model_version, computed_at, evidence_json "
        "FROM relevance_judgments ORDER BY event_family_id"
    ).fetchall()
    assert all(r[0] == "fake" and r[1] == "fake-scorer-1" for r in rows)
    # computed_at anchored to article first_observed + method offset
    assert rows[0][2] == pytest.approx(T0 + 5 + 1.0)
    times = conn.execute(
        "SELECT source_effective_at, scored_at FROM relevance_judgments "
        "ORDER BY event_family_id LIMIT 1"
    ).fetchone()
    assert times[0] == pytest.approx(T0 + 5)      # text availability
    assert times[1] > times[0] + 1000             # honest scorer clock


def test_rescore_never_rewrites_and_resumes(conn):
    # pre-existing batch judgment must survive untouched
    conn.execute(
        "INSERT INTO relevance_judgments (event_family_id, market_id, "
        "contract_version_seq, computed_at, rel_class, rel_score, "
        "direction, method, model_version) VALUES ('f0', 'm1', 1, ?, "
        "'background', 0.2, 0.0, 'rule_keyword_overlap', 'rule-1.0.0')",
        (T0 + 5,),
    )
    conn.commit()
    first = rescore_news(conn, FakeScorer(), method="fake")
    assert first["scored"] == 2
    batch = conn.execute(
        "SELECT rel_class FROM relevance_judgments WHERE "
        "method = 'rule_keyword_overlap'"
    ).fetchall()
    assert [tuple(r) for r in batch] == [("background",)]  # untouched
    # resume: everything already scored under this method+version
    scorer = FakeScorer()
    second = rescore_news(conn, scorer, method="fake")
    assert second["scored"] == 0
    assert second["skipped_existing"] == 2
    assert scorer.calls == 0                        # no wasted LLM calls


def test_rescored_judgment_supersedes_in_latest_snapshot(conn):
    conn.execute(
        "INSERT INTO relevance_judgments (event_family_id, market_id, "
        "contract_version_seq, computed_at, rel_class, rel_score, "
        "direction, method, model_version) VALUES ('f0', 'm1', 1, ?, "
        "'background', 0.2, 0.0, 'rule_keyword_overlap', 'rule-1.0.0')",
        (T0 + 5,),
    )
    conn.commit()
    rescore_news(conn, FakeScorer(), method="fake")
    latest = conn.execute(
        "SELECT rel_class FROM relevance_judgments WHERE "
        "event_family_id = 'f0' AND market_id = 'm1' "
        "ORDER BY computed_at DESC LIMIT 1"
    ).fetchone()[0]
    assert latest == "supports_positive"            # rescore wins as-of


def test_scored_judgment_error_isolation(conn):
    class Flaky(FakeScorer):
        def score(self, claim_text, question, rules_text):
            if "Unrelated" in claim_text:
                raise RuntimeError("model timeout")
            return super().score(claim_text, question, rules_text)

    counters = rescore_news(conn, Flaky(), method="flaky")
    assert counters["scored"] == 1
    assert counters["errors"] == 1
    assert counters["error_samples"]


def test_make_scorer_rule_and_unknown():
    assert make_scorer("rule").score("x", "Will X?", None)
    with pytest.raises(ValueError):
        make_scorer("nope")


def test_ollama_scorer_parses_structured_response(monkeypatch):
    """The Ollama relevance scorer with a mocked chat() — deterministic,
    offline, verifies the structured-output contract end to end."""
    import polymarket.normalization.llm_news as llm

    captured = {}

    class FakeMessage:
        content = json.dumps({
            "rel_class": "supports_negative",
            "rel_score": 0.8,
            "direction": -0.7,
            "directness": "direct",
            "confidence": 0.85,
            "supporting_rule_span": "Resolves YES if X.",
            "reasoning_summary": "claim contradicts X",
        })

    class FakeResponse:
        message = FakeMessage()

    def fake_chat(model, messages, format=None, options=None, **kw):
        captured["model"] = model
        captured["format"] = format
        captured["prompt"] = messages[0]["content"]
        return FakeResponse()

    monkeypatch.setattr(llm, "chat", fake_chat)
    scorer = llm.OllamaRelevanceScorer(model="test-model")
    scored = scorer.score("Claim text", "Will X happen?", "Resolves YES if X.")
    assert captured["model"] == "test-model"
    assert "Will X happen?" in captured["prompt"]
    assert scored["rel_class"] == "supports_negative"
    assert scored["direction"] == pytest.approx(-0.7)
    assert scorer.version.startswith("ollama-test-model")



def test_later_claim_in_scored_family_gets_its_own_judgment(conn):
    rescore_news(conn, FakeScorer(), method="fake")
    now = time.time()
    # a follow-up claim arrives in the ALREADY-SCORED family f0
    conn.execute(
        "INSERT INTO news_articles (article_id, source_id, source_url, "
        "source_published_at, first_observed_at, download_completed_at, "
        "timestamp_source, timestamp_confidence, headline, body, "
        "content_hash, raw_response_id, raw_record_index, "
        "raw_record_hash, parser_version, schema_version, normalized_at) "
        "VALUES ('a9', 's', 'u', ?, ?, ?, 'feed', 0.8, 'X confirmed', "
        "'X confirmed', 'ch9', 1, 0, 'h', 'p', 1, ?)",
        (T0 + 500, T0 + 505, T0 + 505, now),
    )
    conn.execute(
        "INSERT INTO news_claims (claim_id, article_id, claim_text, "
        "entities_json, quantities_json, first_available_at, "
        "extractor_version, confidence) VALUES ('c9', 'a9', "
        "'X confirmed', '[]', '[]', ?, 'x', 0.9)", (T0 + 505,),
    )
    conn.execute(
        "INSERT INTO claim_edges (edge_id, claim_id, event_family_id, "
        "edge_type, effective_from, evidence, confidence) VALUES "
        "('e9', 'c9', 'f0', 'confirmation', ?, 'k', 0.5)", (T0 + 505,),
    )
    conn.commit()
    second = rescore_news(conn, FakeScorer(), method="fake")
    assert second["scored"] == 1                  # NOT skipped forever
    assert second["skipped_existing"] == 2
    claim_ids = {
        r[0] for r in conn.execute(
            "SELECT claim_id FROM relevance_judgments "
            "WHERE event_family_id = 'f0'"
        )
    }
    assert claim_ids == {"c0", "c9"}


def test_two_scorers_never_collide(conn):
    class OtherScorer(FakeScorer):
        version = "other-scorer-9"

    rescore_news(conn, FakeScorer(), method="fake")
    counters = rescore_news(conn, OtherScorer(), method="fake")
    assert counters["scored"] == 2                # no silent PK loss
    n = conn.execute(
        "SELECT COUNT(*) FROM relevance_judgments"
    ).fetchone()[0]
    assert n == 4                                  # both runs fully stored
