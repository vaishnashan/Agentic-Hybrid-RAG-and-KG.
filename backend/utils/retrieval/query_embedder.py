"""
Embeds a QUESTION at query time — deliberately separate from how chunks get
embedded at index time (utils/storage/dense_index.py, which runs
sentence-transformers locally during the pipeline build).

Why two different code paths for what's conceptually "the same embedding
step": index-time embedding runs once, offline, as part of the pipeline —
loading a ~130MB local model there is fine. Query-time embedding runs on
EVERY /ask request in the deployed API — loading the same model in-process
there means shipping torch + model weights into the API's runtime image,
which is exactly the VM-sized footprint this project is trying to avoid on
a lightweight host (Render/Railway/Fly free tiers, serverless, etc.).

So at query time we call a CLOUD inference API for the SAME model
(BAAI/bge-small-en-v1.5) instead of loading it locally. Using the same model
name is not optional — the index vectors and the query vector must live in
the same embedding space or search quality collapses, no matter how good
either individual embedding is.

Provider is swappable via EMBEDDING_PROVIDER in .env:
  "huggingface" (default) — HF Inference API, running the exact same model
                             used at index time. Requires HF_API_TOKEN.
  "local"                  — falls back to loading sentence-transformers
                             in-process. Only really useful for local dev/
                             testing; defeats the "no VM needed" goal if
                             used in a real deployment.

If you'd rather standardize on a different cloud embedding provider (Cohere,
Jina, OpenAI, etc.) instead of the HF Inference API, this is the ONLY file
that needs to change — everything downstream (dense_retriever.py, and
therefore hybrid_merge.py/reranker.py/the agent) just calls embed_query()
and doesn't care how the vector was produced. Note that switching providers
means the query embedding would come from a DIFFERENT model than the one
used to build the Chroma index — you'd need to re-embed the index with that
provider too (rebuild via utils/storage/dense_index.py) to keep the vector
spaces consistent.
"""
import logging
from typing import List

import requests

from ..storage.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
    HF_API_TOKEN,
    HF_FEATURE_EXTRACTION_URL,
)

logger = logging.getLogger("query_embedder")

_local_model = None  # lazy-loaded only if EMBEDDING_PROVIDER == "local"


def _embed_with_huggingface(query: str) -> List[float]:
    if not HF_API_TOKEN:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=huggingface but HF_API_TOKEN is not set. Add an "
            "HF access token to your .env (https://huggingface.co/settings/tokens)."
        )

    response = requests.post(
        HF_FEATURE_EXTRACTION_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": query, "options": {"wait_for_model": True}},
        timeout=20,
    )
    response.raise_for_status()
    embedding = response.json()

    # sentence-transformers-compatible models on the HF Inference API return
    # the already-pooled sentence vector directly: List[float]. Some model/
    # runtime combinations instead return per-token vectors (List[List[float]])
    # — mean-pool those ourselves so the caller always gets one flat vector.
    if embedding and isinstance(embedding[0], list):
        num_tokens = len(embedding)
        embedding = [sum(dim) / num_tokens for dim in zip(*embedding)]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        logger.warning(
            f"HF embedding returned {len(embedding)} dimensions, expected "
            f"{EMBEDDING_DIMENSIONS} for {EMBEDDING_MODEL_NAME} — check "
            f"EMBEDDING_MODEL_NAME matches what dense_index.py was built with."
        )

    return embedding


def _embed_with_local_model(query: str) -> List[float]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model {EMBEDDING_MODEL_NAME} (EMBEDDING_PROVIDER=local)...")
        _local_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _local_model.encode([query], normalize_embeddings=True).tolist()[0]


def embed_query(query: str) -> List[float]:
    """Returns the embedding vector for a single question, using whichever
    provider EMBEDDING_PROVIDER selects."""
    if EMBEDDING_PROVIDER == "local":
        return _embed_with_local_model(query)
    if EMBEDDING_PROVIDER == "huggingface":
        return _embed_with_huggingface(query)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. Use 'huggingface' or 'local'."
    )


if __name__ == "__main__":
    vec = embed_query("how do agents learn to use tools")
    print(f"Provider   : {EMBEDDING_PROVIDER}")
    print(f"Dimensions : {len(vec)}")
    print(f"First 5    : {vec[:5]}")
