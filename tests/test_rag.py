import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_demo import api, config, providers
from rag_demo.evaluate import evaluate
from rag_demo.ingest import build_index
from rag_demo.rag import NO_ANSWER, RAG
from rag_demo.store import Index, chunk_text

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def settings():
    # min_score is model-specific; 0.1 is the tuned value for the fake embedder.
    return config.Settings(**{**config.load().__dict__, "provider": "fake", "min_score": 0.1})


@pytest.fixture(scope="module")
def rag(settings):
    embedder, chat = providers.build(settings)
    index = build_index(ROOT / "data/docs", settings, embedder)
    return RAG(index, embedder, chat, settings.top_k, settings.min_score)


def test_chunks_respect_size_and_keep_overlap():
    text = "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text("doc.md", text, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + 50 + 2 for c in chunks)
    assert chunks[1].text.startswith(chunks[0].text[-50:])


def test_oversized_paragraph_is_split():
    chunks = chunk_text("doc.md", "x" * 2000, size=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert len(chunks) >= 4


def test_index_roundtrip(tmp_path, rag):
    rag.index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.id for c in loaded.chunks] == [c.id for c in rag.index.chunks]


def test_retrieval_finds_expected_source(rag):
    hits = rag.retrieve("How long is the password reset link valid?")
    assert hits[0].chunk.source == "account-security.md"


def test_refuses_when_nothing_relevant(rag):
    answer = rag.ask("zzzz qqqq")
    assert answer.text == NO_ANSWER
    assert answer.sources == []


def test_eval_harness_baseline(rag):
    cases = [json.loads(l) for l in (ROOT / "evals/questions.jsonl").read_text().splitlines() if l.strip()]
    report = evaluate(rag, cases)
    # Baseline for the keyword-only fake embedder; real embeddings should beat it.
    assert report[f"hit_rate@{rag.top_k}"] >= 0.8


def test_api_ask(rag, monkeypatch):
    monkeypatch.setattr(api, "get_rag", lambda: rag)
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "Do you ship to Canada?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["source"] == "shipping.md"
    assert body["cost_usd"] == 0.0  # no price configured, so no invented cost
    assert client.post("/ask", json={"question": ""}).status_code == 422
