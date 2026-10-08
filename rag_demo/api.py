"""HTTP API. Runs under uvicorn locally and under Lambda through Mangum.

    uvicorn rag_demo.api:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel, Field

from . import config, providers
from .rag import RAG
from .store import Index

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")

# USD per 1K tokens, set per deployment to match the chosen model's price.
# Left at 0 when unset so the API never reports a made-up cost.
PRICE_IN = float(os.environ.get("RAG_PRICE_IN_PER_1K", "0"))
PRICE_OUT = float(os.environ.get("RAG_PRICE_OUT_PER_1K", "0"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    id: str
    source: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retrieval_ms: float
    generation_ms: float


def _fetch_index_from_s3(uri: str, dest: str) -> None:
    """Download s3://bucket/prefix/{chunks.json,vectors.npy} into dest (Lambda: under /tmp)."""
    import boto3

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    os.makedirs(dest, exist_ok=True)
    s3 = boto3.client("s3")
    for name in ("chunks.json", "vectors.npy"):
        s3.download_file(bucket, f"{prefix.rstrip('/')}/{name}", os.path.join(dest, name))


@lru_cache
def get_rag() -> RAG:
    settings = config.load()
    embedder, chat = providers.build(settings)
    if s3_uri := os.environ.get("RAG_INDEX_S3_URI"):
        _fetch_index_from_s3(s3_uri, settings.index_path)
    index = Index.load(settings.index_path)
    return RAG(index, embedder, chat, settings.top_k, settings.min_score)


app = FastAPI(title="RAG demo")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(get_rag().index.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = get_rag().ask(req.question)
    except Exception:
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="model provider error")

    cost = result.input_tokens / 1000 * PRICE_IN + result.output_tokens / 1000 * PRICE_OUT
    # Structured log line for CloudWatch Logs Insights. Metrics only, never the question text.
    log.info(json.dumps({
        "event": "ask",
        "sources": len(result.sources),
        "top_score": round(result.sources[0].score, 4) if result.sources else None,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(cost, 6),
        "retrieval_ms": round(result.retrieval_ms, 1),
        "generation_ms": round(result.generation_ms, 1),
    }))
    return AskResponse(
        answer=result.text,
        sources=[
            Source(id=h.chunk.id, source=h.chunk.source, score=round(h.score, 4), excerpt=h.chunk.text[:300])
            for h in result.sources
        ],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=round(cost, 6),
        retrieval_ms=round(result.retrieval_ms, 1),
        generation_ms=round(result.generation_ms, 1),
    )


handler = Mangum(app)
