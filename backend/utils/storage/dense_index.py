"""
Builds the DENSE (embedding-based) index from data/processed/chunks.jsonl
(produced by chunker.py) and stores it in Chroma Cloud — captures meaning,
so a question can match a chunk even if it doesn't share exact words.

Embeddings are computed LOCALLY with sentence-transformers at build time
(this only runs when you re-run the pipeline, not on every deployed
request), and only the resulting vectors are uploaded to Chroma Cloud. The
deployed API never loads this model — see retrieval/query_embedder.py,
which calls a cloud inference API instead so the live service stays light
enough for a no-VM / serverless-style deploy.

CHROMA CLOUD SETUP (one-time):
  1. Sign up / log in at https://trychroma.com.
  2. Create a database and copy its API key, tenant ID, and database name.
  3. Add them to your .env:
       CHROMA_API_KEY=...
       CHROMA_TENANT=...
       CHROMA_DATABASE=...

Expected input:  data/processed/chunks.jsonl   (from chunker.py)
Output:          a collection in your Chroma Cloud database
"""
import time
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

from .chroma_client import get_chroma_client
from .config import CHROMA_COLLECTION_NAME, CHROMA_DATABASE, CHROMA_TENANT, EMBEDDING_MODEL_NAME


@dataclass
class IndexedChunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    metadata: dict


def build_dense_index(chunks: List) -> None:
    """Embeds every chunk locally and uploads the vectors to a Chroma Cloud collection."""
    print("=" * 70)
    print("BUILDING DENSE (CHROMA CLOUD) INDEX")
    print("=" * 70)

    start = time.perf_counter()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(chunks)} chunks locally (this may take a couple of minutes)...", flush=True)
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    print("\nWriting vectors to Chroma Cloud...", flush=True)
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Chroma metadata values must be str/int/float/bool — flatten nested metadata.
    flat_metadatas = []
    for c in chunks:
        flat = {"paper_id": c.paper_id, "section": c.section}
        for k, v in c.metadata.items():
            flat[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        flat_metadatas.append(flat)

    # Batch to stay under Chroma's per-call upsert limit and keep each write small.
    BATCH = 250
    for i in range(0, len(chunks), BATCH):
        collection.upsert(
            ids=[c.chunk_id for c in chunks[i:i + BATCH]],
            embeddings=embeddings[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=flat_metadatas[i:i + BATCH],
        )

    elapsed = time.perf_counter() - start
    print(
        f"Dense index built: {collection.count()} vectors stored in Chroma Cloud "
        f"(tenant={CHROMA_TENANT}, database={CHROMA_DATABASE})"
    )
    print(f"Time taken: {elapsed:.1f} seconds")


def run_dense_index(chunks: List) -> None:
    """Entry point used by pipeline.py."""
    build_dense_index(chunks)


if __name__ == "__main__":
    import json

    from .config import CHUNKS_PATH

    loaded = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded.append(IndexedChunk(**json.loads(line)))
    run_dense_index(loaded)
