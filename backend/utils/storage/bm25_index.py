"""
Builds the SPARSE (keyword-based) index from data/processed/chunks.jsonl
(produced by chunker.py) using BM25 — captures exact terms, acronyms, and
model/dataset names that embeddings sometimes blur together.

Stays on local disk (a small pickle file) — no cloud storage needed, and it
ships inside the deploy image itself (see Dockerfile), so the API container
can load it directly without a network round trip.

Expected input:  data/processed/chunks.jsonl        (from chunker.py)
Output:          data/processed/bm25_index.pkl
"""
import pickle
import time
from dataclasses import asdict
from typing import List

from rank_bm25 import BM25Okapi

from .config import BM25_INDEX_PATH


def build_sparse_index(chunks: List) -> None:
    """Builds a BM25 keyword index over the given chunks and saves it to disk."""
    print("\n" + "=" * 70)
    print("BUILDING SPARSE (BM25) INDEX")
    print("=" * 70)

    start = time.perf_counter()

    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunk_records = [
        {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "section": chunk.section,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_records}, f)

    elapsed = time.perf_counter() - start
    print(f"Sparse index built: {len(chunks)} chunks indexed -> {BM25_INDEX_PATH}")
    print(f"Time taken: {elapsed:.1f} seconds")


def run_bm25_index(chunks: List) -> None:
    """Entry point used by pipeline.py."""
    build_sparse_index(chunks)


if __name__ == "__main__":
    import json

    from .chunker import Chunk
    from .config import CHUNKS_PATH

    loaded = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded.append(Chunk(**json.loads(line)))
    run_bm25_index(loaded)
