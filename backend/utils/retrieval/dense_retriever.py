# src/retrieval/dense_retriever.py
"""
DENSE (embedding/meaning-based) retrieval using the local, persisted Chroma index
built by embed_and_index.py.

The query is embedded with the SAME model used to embed the chunks (bge-small-en-v1.5)
— this is important: if the query and the chunks aren't embedded with the same model,
their vectors aren't comparable and search quality collapses.

This file only SEARCHES the already-built Chroma index (data/processed/chroma_db/)
— it doesn't build anything. That happens in embed_and_index.py.
"""
from typing import List

from sentence_transformers import SentenceTransformer

from ..storage.ingestion1.embed_and_index import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_chroma_client,
)

from .sparse_retriever import RetrievedChunk

_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def dense_search(query: str, top_k: int = 10) -> List[RetrievedChunk]:
    """Returns the top_k chunks ranked by embedding similarity to the query."""
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1 - dist  # Chroma returns cosine distance; convert to similarity
        out.append(
            RetrievedChunk(chunk_id=cid, text=doc, score=similarity, metadata=meta, source="dense")
        )
    return out


if __name__ == "__main__":
    query = "how do agents learn to use tools"
    print(f"Query: {query}\n")
    for r in dense_search(query, top_k=5):
        print(f"  score={r.score:.3f} | {r.chunk_id} | {r.metadata.get('title', '')[:50]}")
        print(f"    {r.text[:100]}...")