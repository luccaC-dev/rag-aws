"""Chunking and a small cosine-similarity index persisted as .npy + .json.

A flat numpy index is plenty for a few thousand chunks and keeps the Lambda
package small. Past that, swap this module for pgvector or OpenSearch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    source: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def chunk_text(source: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split on paragraph boundaries, packing paragraphs up to `size` chars with `overlap` carried over.

    Markdown headings start a new chunk, so a chunk never mixes two sections.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if p.startswith("#") and current:
            chunks.append(current)
            current = ""
        # A single oversized paragraph is hard-split.
        while len(p) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(p[:size])
            p = p[size - overlap :]
        if current and len(current) + len(p) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p if overlap else p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return [Chunk(id=f"{source}#{i}", source=source, text=c) for i, c in enumerate(chunks)]


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int, min_score: float = 0.0) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        path = Path(path)
        chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
        return cls(chunks, np.load(path / "vectors.npy"))
