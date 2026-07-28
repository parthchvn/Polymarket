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


def test_relevance_prompt_version_bumped_with_guidance():
    """The v2 prompt (indirect/background/irrelevant guidance) must be
    reflected in the model version so v1 and v2 judgments never mix
    silently."""
    llm_news = pytest.importorskip("polymarket.normalization.llm_news")
    source = open(llm_news.__file__).read()
    assert "Use indirect when a claim meaningfully informs" in source
    assert "merely because it does not directly" in source
    assert 'relevance-v2"' in source
    assert 'relevance-v1"' not in source


def test_limited_claim_extractor_budget_and_resume(conn):
    from polymarket.normalization.news import LimitedClaimExtractor

    class Fake:
        version = "fake-x-1"

        def __init__(self):
            self.calls = 0

        def extract(self, headline, body):
            self.calls += 1
            return [{"claim_text": f"claim about {headline}",
                     "entities": [], "quantities": []}]

    inner = Fake()
    limited = LimitedClaimExtractor(inner, conn, limit=2)
    articles = [(f"headline {i}", f"body {i}") for i in range(5)]
    outputs = [limited.extract(h, b) for h, b in articles]
    assert inner.calls == 2                       # budget respected
    assert limited.extracted == 2 and limited.deferred == 3
    assert outputs[0] and not outputs[2]
    # simulate persistence of the two extracted articles, then resume:
    # they are skipped without token spend and the budget covers new ones
    now = time.time()
    for i in range(2):
        conn.execute(
            "INSERT INTO news_articles (article_id, source_id, "
            "source_url, source_published_at, first_observed_at, "
            "download_completed_at, timestamp_source, "
            "timestamp_confidence, headline, body, content_hash, "
            "raw_response_id, raw_record_index, raw_record_hash, "
            "parser_version, schema_version, normalized_at) VALUES "
            "(?, 's', 'u', 1, 1, 1, 'feed', 0.8, ?, ?, ?, 1, 0, 'h', "
            "'p', 2, ?)",
            (f"art-lim-{i}", f"headline {i}", f"body {i}",
             f"ch-lim-{i}", now),
        )
        conn.execute(
            "INSERT INTO news_claims (claim_id, article_id, claim_text, "
            "entities_json, quantities_json, first_available_at, "
            "extractor_version, confidence) VALUES (?, ?, 'c', '[]', "
            "'[]', 1, 'fake-x-1', 0.9)",
            (f"cl-lim-{i}", f"art-lim-{i}"),
        )
    conn.commit()
    resumed = LimitedClaimExtractor(Fake(), conn, limit=2)
    for headline, body in articles:
        resumed.extract(headline, body)
    assert resumed.skipped_existing == 2          # no re-spend
    assert resumed.extracted == 2                 # budget on NEW articles
    assert resumed.deferred == 1


def _insert_article_claim(conn, key, text, ts):
    now = time.time()
    conn.execute(
        "INSERT INTO news_articles (article_id, source_id, source_url, "
        "source_published_at, first_observed_at, download_completed_at, "
        "timestamp_source, timestamp_confidence, headline, body, "
        "content_hash, raw_response_id, raw_record_index, "
        "raw_record_hash, parser_version, schema_version, normalized_at)"
        " VALUES (?, 's', 'u', ?, ?, ?, 'feed', 0.8, ?, ?, ?, 1, 0, "
        "'h', 'p', 1, ?)",
        (f"art-{key}", ts, ts, ts, text, text, f"ch-{key}", now),
    )
    conn.execute(
        "INSERT INTO news_claims (claim_id, article_id, claim_text, "
        "entities_json, quantities_json, first_available_at, "
        "extractor_version, confidence) VALUES (?, ?, ?, '[]', '[]', "
        "?, 'x', 0.9)",
        (f"claim-{key}", f"art-{key}", text, ts),
    )
    conn.execute(
        "INSERT INTO event_families (event_family_id, label, "
        "earliest_available_at, created_by, created_at) VALUES "
        "(?, ?, ?, 'test', ?)", (f"fam-{key}", key, ts, now),
    )
    conn.execute(
        "INSERT INTO claim_edges (edge_id, claim_id, event_family_id, "
        "edge_type, effective_from, evidence, confidence) VALUES "
        "(?, ?, ?, 'new', ?, 'k', 0.5)",
        (f"edge-{key}", f"claim-{key}", f"fam-{key}", ts),
    )


class _CannedOllama:
    """Deterministic stand-in for the Ollama transport: returns the
    class the v2 guidance requires for each labelled example."""

    CASES = {
        "Netanyahu and Trump meet to discuss Iran strategy":
            {"rel_class": "indirect", "rel_score": 0.7,
             "direction": 0.2,
             "reasoning_summary": "diplomacy informs likelihood"},
        "US munitions stockpile shortfall may constrain operations":
            {"rel_class": "indirect", "rel_score": 0.6,
             "direction": -0.2,
             "reasoning_summary": "capability constraint"},
        "State funeral held for former minister":
            {"rel_class": "irrelevant", "rel_score": 0.9,
             "direction": 0.0,
             "reasoning_summary": "no causal connection"},
    }

    def score(self, claim_text, question, rules_text):
        payload = self.CASES[claim_text]
        return {
            "rel_class": payload["rel_class"],
            "rel_score": payload["rel_score"],
            "direction": payload["direction"],
            "evidence": {
                "reasoning_summary": payload["reasoning_summary"],
            },
        }

    method = "ollama_llm"
    version = "canned-relevance-v2"


def test_labelled_examples_not_marked_irrelevant(conn):
    """The advisor's regression cases: indirect evidence (diplomacy,
    capability constraints) must never end up 'irrelevant'; a truly
    unconnected story stays irrelevant.  Runs through the full rescore
    persistence path with a canned scorer honouring the v2 contract."""
    for i, text in enumerate(_CannedOllama.CASES):
        _insert_article_claim(conn, f"reg-{i}", text, T0 + 100 + i)
    conn.commit()
    counters = rescore_news(conn, _CannedOllama(), method="ollama")
    assert counters["scored"] == 3
    classes = {
        row[0]: row[1] for row in conn.execute(
            "SELECT c.claim_text, r.rel_class FROM relevance_judgments r "
            "JOIN news_claims c ON c.claim_id = r.claim_id "
            "WHERE r.model_version = 'canned-relevance-v2'"
        )
    }
    assert classes[
        "Netanyahu and Trump meet to discuss Iran strategy"
    ] == "indirect"
    assert classes[
        "US munitions stockpile shortfall may constrain operations"
    ] == "indirect"
    assert classes[
        "State funeral held for former minister"
    ] == "irrelevant"


def test_limited_relevance_scorer_budget(conn):
    from polymarket.normalization.news import LimitedRelevanceScorer

    class Counting:
        method, version = "fake", "f1"

        def __init__(self):
            self.calls = 0

        def score(self, *a):
            self.calls += 1
            return {"rel_class": "background", "rel_score": 0.5,
                    "direction": 0.0, "evidence": {}}

    inner = Counting()
    limited = LimitedRelevanceScorer(inner, limit=2)
    out = [limited.score("c", "q", "r") for _ in range(5)]
    assert inner.calls == 2
    assert out[2] is None and limited.deferred == 3


def test_backfill_llm_claims_resumable(tmp_path):
    """Articles first normalized by the rule extractor get body-level
    claims from the backfill — versioned, budgeted, resumable, and
    with honest availability (claim availability = article
    observation time, judgment scored_at = now)."""
    from polymarket.contracts.schema import init_db
    from polymarket.normalization.news import backfill_llm_claims

    conn = init_db(str(tmp_path / "backfill.sqlite"),
                   description="backfill test")
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "('m-b', '0xb', 'Will X invade?', 1, 0, 'h', 'p', 2, ?)",
        (now,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO contract_versions (market_id, "
        "version_seq, effective_from, first_observed_at, question, "
        "rules_text, content_hash, raw_response_id, parser_version, "
        "schema_version, normalized_at) VALUES ('m-b', 1, 0, 0, "
        "'Will X invade?', 'rules about invasion', 'chb', 1, 'p', 2, ?)",
        (now,),
    )
    for i in range(3):
        _insert_article_claim(conn, f"bf-{i}", f"headline body {i}",
                              1000.0 + i)
    conn.commit()

    class BodyExtractor:
        version = "ollama-test-claims-v9"

        def __init__(self):
            self.calls = 0

        def extract(self, headline, body):
            self.calls += 1
            return [{"claim_text": f"llm claim from {headline}",
                     "entities": ["X"], "quantities": [],
                     "confidence": 0.9}]

    class Scorer:
        method, version = "ollama_llm", "canned-v2"

        def score(self, claim_text, question, rules):
            return {"rel_class": "indirect", "rel_score": 0.6,
                    "direction": 0.1, "evidence": {}}

    extractor = BodyExtractor()
    first = backfill_llm_claims(conn, extractor, Scorer(), limit=2)
    assert first["articles_processed"] == 2
    assert extractor.calls == 2
    # resume: the two covered articles are skipped, third processed
    extractor2 = BodyExtractor()
    second = backfill_llm_claims(conn, extractor2, Scorer(), limit=5)
    assert second["articles_processed"] == 1
    assert extractor2.calls == 1
    rows = conn.execute(
        "SELECT c.first_available_at, r.scored_at FROM news_claims c "
        "JOIN relevance_judgments r ON r.claim_id = c.claim_id "
        "WHERE c.extractor_version = 'ollama-test-claims-v9'"
    ).fetchall()
    assert len(rows) == 3
    for row in rows:
        # availability honesty: text availability vs actual scoring time
        assert row[0] < 2000.0 and row[1] > 2000.0


def test_backfill_commits_per_article(tmp_path):
    """LLM extraction takes minutes per article: progress must be
    durable article-by-article, so interruption loses at most the
    article in flight."""
    import sqlite3 as _sq

    from polymarket.contracts.schema import init_db
    from polymarket.normalization.news import backfill_llm_claims

    db = str(tmp_path / "commit.sqlite")
    conn = init_db(db, description="commit test")
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO markets (market_id, condition_id, "
        "question, raw_response_id, raw_record_index, raw_record_hash, "
        "parser_version, schema_version, normalized_at) VALUES "
        "('m-c', '0xc', 'Q?', 1, 0, 'h', 'p', 2, ?)", (now,),
    )
    for i in range(3):
        _insert_article_claim(conn, f"cm-{i}", f"text {i}", 500.0 + i)
    conn.commit()

    class Crashing:
        version = "crash-v1"

        def __init__(self):
            self.calls = 0

        def extract(self, headline, body):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("ollama died mid-batch")
            return [{"claim_text": f"c from {headline}",
                     "entities": [], "quantities": [],
                     "confidence": 0.9}]

    class Scorer:
        method, version = "ollama_llm", "s1"

        def score(self, *a):
            return {"rel_class": "background", "rel_score": 0.3,
                    "direction": 0.0, "evidence": {}}

    report = backfill_llm_claims(conn, Crashing(), Scorer())
    # the poison article is recorded and SKIPPED, the queue advances
    assert report["articles_failed"] == 1
    assert report["articles_processed"] == 2
    assert "ollama died" in report["failed_examples"][0]
    # completed articles are durable via a SEPARATE connection
    other = _sq.connect(db)
    count = other.execute(
        "SELECT COUNT(*) FROM news_claims "
        "WHERE extractor_version = 'crash-v1'"
    ).fetchone()[0]
    assert count == 2


def test_cap_extracted_claims_bounds_degenerate_output():
    """Constrained decoding can degenerate into huge repetitive claim
    arrays; the cap keeps the best dozen unique claims so one article
    cannot burn minutes of CPU and hundreds of relevance calls."""
    llm_news = pytest.importorskip("polymarket.normalization.llm_news")

    class C:
        def __init__(self, text, conf):
            self.claim_text = text
            self.confidence = conf

    degenerate = [C("same claim", 0.5) for _ in range(5000)]
    degenerate += [C(f"claim {i}", 0.5 + i / 100) for i in range(30)]
    capped = llm_news.cap_extracted_claims(degenerate)
    assert len(capped) == llm_news.MAX_CLAIMS_PER_ARTICLE
    # highest-confidence unique claims kept
    assert capped[0].claim_text == "claim 29"
    assert len({c.claim_text for c in capped}) == len(capped)
