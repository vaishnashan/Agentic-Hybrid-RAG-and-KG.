"""
Cross-encoder reranking of the hybrid-merged candidates — an off-the-shelf
model, no training required.

Why this is a separate step from dense/sparse search: a cross-encoder looks
at the (question, chunk) pair TOGETHER and scores how relevant they are to
each other, which is more accurate than comparing separately-computed
vectors (what dense_retriever.py does). It's too slow to run over the whole
corpus for every query, which is why the pipeline is: cheap wide search
first (hybrid_merge.py), then this expensive but accurate reranking only on
the small merged candidate set.

Note: unlike dense_retriever.py's query embedding, this still loads a small
model in-process (~90MB) — there isn't a cloud reranking API as universal as
HF's embedding endpoint. If your deploy target is extremely constrained,
set RERANKER_ENABLED=false in .env to skip this step entirely (hybrid_merge
results are used as-is); quality drops slightly but startup/memory cost
drops a lot.
"""
import os
from typing import List

from sentence_transformers import CrossEncoder

from .sparse_retriever import RetrievedChunk

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().lower() in ("1", "true", "yes")

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
    """Re-scores and re-orders candidates using the cross-encoder; returns the top_k."""
    if not candidates:
        return []

    if not RERANKER_ENABLED:
        return candidates[:top_k]

    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id, text=chunk.text, score=float(score),
                metadata=chunk.metadata, source="reranked",
            )
        )
    return results


if __name__ == "__main__":
    from .hybrid_merge import hybrid_search

    query = "Which papers propose mechanisms for agent skills to self-evolve or improve over time, and what approach does each take?"
    candidates = hybrid_search(query, top_k=20)
    for r in rerank(query, candidates, top_k=5):
        print(f"  score={r.score:.3f} | {r.chunk_id} | {r.metadata.get('title', '')[:100]}")
