"""Runtime configuration, read once from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # "bedrock" in AWS, "ollama" for local development, "fake" for tests and offline runs.
    provider: str
    index_path: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    min_score: float

    # Bedrock
    aws_region: str
    bedrock_embed_model: str
    bedrock_chat_model: str

    # Ollama
    ollama_url: str
    ollama_embed_model: str
    ollama_chat_model: str


def load() -> Settings:
    env = os.environ.get
    return Settings(
        provider=env("RAG_PROVIDER", "fake"),
        index_path=env("RAG_INDEX_PATH", "data/index"),
        top_k=int(env("RAG_TOP_K", "4")),
        chunk_size=int(env("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(env("RAG_CHUNK_OVERLAP", "120")),
        min_score=float(env("RAG_MIN_SCORE", "0.2")),
        aws_region=env("AWS_REGION", "us-east-1"),
        bedrock_embed_model=env("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
        # No default on purpose: pick a chat model enabled in your Bedrock console.
        bedrock_chat_model=env("BEDROCK_CHAT_MODEL", ""),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
        ollama_embed_model=env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_chat_model=env("OLLAMA_CHAT_MODEL", "llama3.1"),
    )
