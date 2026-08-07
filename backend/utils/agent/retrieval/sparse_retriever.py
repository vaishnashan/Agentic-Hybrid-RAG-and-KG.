# src/retrieval/sparse_retriever.py
"""
SPARSE (keyword-based) retrieval using the BM25 index built by embed_and_index.py.

BM25 is a classic keyword-ranking algorithm — no model, no embeddings, just word
statistics (how often a word appears in a chunk vs. how rare it is across the whole
corpus). It's fast and it's good at catching exact terms, acronyms, and model/dataset
names that a meaning-based embedding search can sometimes blur together.

This file only SEARCHES the already-built BM25 index (data/processed/bm25_index.pkl)
— it doesn't build anything. That happens in embed_and_index.py.
"""
import pickle
from dataclasses import dataclass
from typing import List

from ..ingestion1.embed_and_index import BM25_INDEX_PATH


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    source: str  # "dense" | "sparse" | "hybrid" | "reranked"


_bm25 = None
_chunks = None


def _load_index():
    """Loads the BM25 index into memory once, reused across calls."""
    global _bm25, _chunks
    if _bm25 is None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No BM25 index found at {BM25_INDEX_PATH}. Run embed_and_index.py first "
                f"(python -m src.ingestion.embed_and_index)."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _chunks = data["chunks"]
    return _bm25, _chunks


def sparse_search(query: str, top_k: int = 10) -> List[RetrievedChunk]:
    """Returns the top_k chunks ranked by BM25 keyword relevance to the query."""
    bm25, chunks = _load_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for chunk, score in ranked:
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=float(score),
                metadata=chunk["metadata"],
                source="sparse",
            )
        )
    return results


if __name__ == "__main__":
    query = "reinforcement learning for tool use"
    print(f"Query: {query}\n")
    for r in sparse_search(query, top_k=5):
        print(f"  score={r.score:.2f} | {r.chunk_id} | {r.metadata.get('title', '')[:50]}")
        print(f"    {r.text[:100]}...")