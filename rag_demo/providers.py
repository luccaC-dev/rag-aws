"""Embedding and chat providers behind two small protocols.

Bedrock is the production target; Ollama keeps local development free; the fake
provider is deterministic so tests and the eval harness run offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from .config import Settings


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResult:
    text: str
    usage: Usage


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class ChatModel(Protocol):
    def chat(self, system: str, user: str) -> ChatResult: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# --- fake -------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or our "
    "so that the their there this to was we what when where which who will with you your".split()
)


class FakeEmbedder:
    """Hashed bag-of-words. Crude, but deterministic and good enough for keyword-heavy evals."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                if word in _STOPWORDS:
                    continue
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class FakeChat:
    """Echoes the best-matching context so the API is testable without a model."""

    def chat(self, system: str, user: str) -> ChatResult:
        first_source = re.search(r"\[(\d+)\][^\n]*\n(.+)", user)
        text = (
            f"(fake model) Based on [{first_source.group(1)}]: {first_source.group(2)[:200]}"
            if first_source
            else "I don't know based on the provided documents."
        )
        return ChatResult(text=text, usage=Usage(len(user) // 4, len(text) // 4))


# --- ollama -----------------------------------------------------------------


class OllamaEmbedder:
    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model

    def embed(self, texts: list[str]) -> np.ndarray:
        r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
        r.raise_for_status()
        return _normalize(np.asarray(r.json()["embeddings"], dtype=np.float32))


class OllamaChat:
    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model

    def chat(self, system: str, user: str) -> ChatResult:
        r = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,
        )
        r.raise_for_status()
        body = r.json()
        return ChatResult(
            text=body["message"]["content"],
            usage=Usage(body.get("prompt_eval_count", 0), body.get("eval_count", 0)),
        )


# --- bedrock ----------------------------------------------------------------


class BedrockEmbedder:
    def __init__(self, region: str, model: str):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        # Titan v2 embeds one input per request.
        vectors = []
        for text in texts:
            r = self.client.invoke_model(modelId=self.model, body=json.dumps({"inputText": text}))
            vectors.append(json.loads(r["body"].read())["embedding"])
        return _normalize(np.asarray(vectors, dtype=np.float32))


class BedrockChat:
    def __init__(self, region: str, model: str):
        if not model:
            raise ValueError("BEDROCK_CHAT_MODEL is not set; use a model id enabled in your Bedrock console")
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model

    def chat(self, system: str, user: str) -> ChatResult:
        r = self.client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.1},
        )
        text = "".join(part.get("text", "") for part in r["output"]["message"]["content"])
        usage = r.get("usage", {})
        return ChatResult(text=text, usage=Usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0)))


def build(settings: Settings) -> tuple[Embedder, ChatModel]:
    if settings.provider == "bedrock":
        return (
            BedrockEmbedder(settings.aws_region, settings.bedrock_embed_model),
            BedrockChat(settings.aws_region, settings.bedrock_chat_model),
        )
    if settings.provider == "ollama":
        return (
            OllamaEmbedder(settings.ollama_url, settings.ollama_embed_model),
            OllamaChat(settings.ollama_url, settings.ollama_chat_model),
        )
    if settings.provider == "fake":
        return FakeEmbedder(), FakeChat()
    raise ValueError(f"unknown RAG_PROVIDER: {settings.provider}")
