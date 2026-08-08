"""
DENSE (embedding/meaning-based) retrieval using the Chroma Cloud index built
by utils/storage/dense_index.py.

The query is embedded via query_embedder.py — a CLOUD inference API call for
the same model used to embed the chunks, NOT a local model load — so the API
process serving /ask stays light enough to deploy without a VM. See
query_embedder.py's docstring for why this is split from index-time embedding.

This file only SEARCHES the already-built Chroma Cloud index — it doesn't
build anything. That happens in utils/storage/dense_index.py.
"""
from typing import List

from ..storage.config import CHROMA_COLLECTION_NAME
from ..storage.chroma_client import get_chroma_client
from .query_embedder import embed_query
from .sparse_retriever import RetrievedChunk

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def dense_search(query: str, top_k: int = 10) -> List[RetrievedChunk]:
    """Returns the top_k chunks ranked by embedding similarity to the query."""
    query_embedding = embed_query(query)
    collection = _get_collection()

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
