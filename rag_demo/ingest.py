"""Build the index from a folder of .md / .txt files.

    rag-ingest data/docs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config, providers
from .store import Chunk, Index, chunk_text

BATCH = 32


def build_index(docs_dir: Path, settings: config.Settings, embedder: providers.Embedder) -> Index:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            source = str(path.relative_to(docs_dir))
            chunks += chunk_text(source, path.read_text(encoding="utf-8"), settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise SystemExit(f"no .md or .txt files under {docs_dir}")
    vectors = np.vstack([embedder.embed([c.text for c in chunks[i : i + BATCH]]) for i in range(0, len(chunks), BATCH)])
    return Index(chunks, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()

    settings = config.load()
    embedder, _ = providers.build(settings)
    started = time.perf_counter()
    index = build_index(args.docs_dir, settings, embedder)
    index.save(settings.index_path)
    sources = len({c.source for c in index.chunks})
    print(
        f"indexed {len(index.chunks)} chunks from {sources} files "
        f"into {settings.index_path} in {time.perf_counter() - started:.1f}s (provider={settings.provider})"
    )


if __name__ == "__main__":
    main()
