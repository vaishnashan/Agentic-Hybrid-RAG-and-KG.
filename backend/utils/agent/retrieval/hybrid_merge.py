# src/retrieval/hybrid_merge.py
"""
Combines the dense (Chroma) and sparse (BM25) result lists into one ranked list, using
Reciprocal Rank Fusion (RRF).

Why RRF instead of just averaging the two scores: dense similarity scores (roughly 0-1)
and BM25 scores (unbounded, can be 0-20+) are on completely different scales, so directly
comparing or averaging them is meaningless. RRF sidesteps this by only looking at each
chunk's RANK (position) in each list, not its raw score — a chunk ranked #1 in both lists
beats a chunk ranked #1 in only one list, regardless of the underlying score scales.

Formula: RRF score for a chunk = sum, over every ranked list it appears in, of
  1 / (k + rank)
where k=60 is a standard smoothing constant from the original RRF paper.
"""
from collections import defaultdict
from typing import List

from .sparse_retriever import RetrievedChunk, sparse_search
from .dense_retriever import dense_search


RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: List[List[RetrievedChunk]], k: int = RRF_K
) -> List[RetrievedChunk]:
    """Merges multiple ranked result lists into one, using each item's rank position."""
    scores = defaultdict(float)
    chunk_lookup = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
            chunk_lookup[chunk.chunk_id] = chunk

    fused_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged: List[RetrievedChunk] = []
    for chunk_id, fused_score in fused_ranking:
        base = chunk_lookup[chunk_id]
        merged.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                text=base.text,
                score=fused_score,
                metadata=base.metadata,
                source="hybrid",
            )
        )
    return merged


def hybrid_search(query: str, top_k: int = 10, candidate_k: int = 25) -> List[RetrievedChunk]:
    """Runs both searches and merges them. This is what a caller should normally use."""
    dense_results = dense_search(query, top_k=candidate_k)
    sparse_results = sparse_search(query, top_k=candidate_k)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused[:top_k]


if __name__ == "__main__":
    def fake_chunk(chunk_id, score, source):
        return RetrievedChunk(chunk_id=chunk_id, text=f"text-{chunk_id}", score=score, metadata={}, source=source)

    dense_results = [fake_chunk("A", 0.9, "dense"), fake_chunk("B", 0.8, "dense"), fake_chunk("C", 0.7, "dense")]
    sparse_results = [fake_chunk("B", 5.0, "sparse"), fake_chunk("A", 3.0, "sparse"), fake_chunk("D", 2.0, "sparse")]

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    print("Fusion test — A and B appear in both lists near the top, so should rank above C/D:")
    for r in fused:
        print(f"  {r.chunk_id}: fused_score={r.score:.4f}")