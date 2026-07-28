"""
Unit tests for the retrieval layer. These assume a small pre-built test index —
TODO: add a pytest fixture that builds a tiny in-memory Chroma collection + BM25 index
from a handful of fixture abstracts (data/processed/test_fixtures/) so these tests don't
depend on your full corpus being built.
"""
import pytest

from utils.retrieval.hybrid_merge import reciprocal_rank_fusion
from utils.retrieval.dense_retriever import RetrievedChunk


def _chunk(chunk_id, score):
    return RetrievedChunk(chunk_id=chunk_id, text=f"text-{chunk_id}", score=score, metadata={}, source="dense")


def test_reciprocal_rank_fusion_merges_and_ranks():
    dense_results = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
    sparse_results = [_chunk("b", 5.0), _chunk("a", 3.0), _chunk("d", 2.0)]

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    fused_ids = [c.chunk_id for c in fused]

    # "a" and "b" appear in both lists near the top -> should outrank "c"/"d" (appear once)
    assert fused_ids.index("a") < fused_ids.index("c")
    assert fused_ids.index("b") < fused_ids.index("d")


def test_reciprocal_rank_fusion_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
