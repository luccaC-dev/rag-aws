"""Retrieval evaluation: hit rate@k and MRR against a labeled question set.

    rag-eval evals/questions.jsonl

Each line: {"question": "...", "expected_source": "file.md"} or
{"question": "...", "expected_source": null} for questions that should find nothing.
Run it after every change to chunking, embeddings or top_k and compare numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, providers
from .rag import RAG
from .store import Index


def evaluate(rag: RAG, cases: list[dict]) -> dict:
    hits = reciprocal = answerable = 0
    refused_ok = unanswerable = 0
    failures = []
    for case in cases:
        results = rag.retrieve(case["question"])
        sources = [h.chunk.source for h in results]
        expected = case["expected_source"]
        if expected is None:
            unanswerable += 1
            refused_ok += not results
            if results:
                failures.append({"question": case["question"], "expected": None, "got": sources})
            continue
        answerable += 1
        if expected in sources:
            hits += 1
            reciprocal += 1 / (sources.index(expected) + 1)
        else:
            failures.append({"question": case["question"], "expected": expected, "got": sources})
    return {
        "cases": len(cases),
        f"hit_rate@{rag.top_k}": round(hits / answerable, 3) if answerable else None,
        "mrr": round(reciprocal / answerable, 3) if answerable else None,
        "refusal_rate_on_unanswerable": round(refused_ok / unanswerable, 3) if unanswerable else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()

    settings = config.load()
    embedder, chat = providers.build(settings)
    rag = RAG(Index.load(settings.index_path), embedder, chat, settings.top_k, settings.min_score)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(rag, cases), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
