"""
Builds BOTH search indexes from data/processed/chunks.jsonl (produced by chunker.py):

  1. A DENSE (embedding-based) index in a LOCAL, PERSISTED ChromaDB — captures
     meaning, so a question can match a chunk even if it doesn't share exact words.
  2. A SPARSE (keyword-based) index using BM25 — captures exact terms, acronyms, and
     model/dataset names that embeddings sometimes blur together.

These two indexes are what a later retrieval step combines (hybrid search) and
reranks. This script only BUILDS the indexes — it doesn't do any searching itself.

Expected input:  data/processed/chunks.jsonl        (from chunker.py)
Outputs:
  - data/processed/chroma_db/                        (Chroma's persisted vector store, on disk)
  - data/processed/bm25_index.pkl                     (BM25 index + chunk lookup)

WHY LOCAL CHROMA INSTEAD OF CHROMA CLOUD: Chroma Cloud is free to start ($5 in
credits, no minimum) but is metered usage-based after that (per-GiB written/
stored/queried). PersistentClient runs the exact same Chroma engine locally,
writes to a folder on disk, and costs nothing at any scale — no account, no API
key, no usage cap. For a project this size (a few thousand chunks) local mode is
simpler and genuinely free forever. If you later need multi-machine access or
managed hosting, swapping back to CloudClient is a two-line change (see the
commented-out block below).

No OCR, no LLM calls here — just an embedding model (sentence-transformers) and a
classic keyword-ranking algorithm (BM25). Both run locally, no API key required.
"""
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from codebase.backend.utils.storage.ingestion1.loader import PROCESSED_DIR

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small, fast, strong quality for its size; free, runs locally

CHROMA_COLLECTION_NAME = "paper_chunks"

# Local, on-disk persisted Chroma store — no account, no API key, no cost.
CHROMA_PERSIST_DIR = PROCESSED_DIR / "chroma_db"

# --- If you ever want to switch back to Chroma Cloud, replace the client
# creation in build_dense_index()/dense_retriever.py with:
#
#   CHROMA_API_KEY = os.environ["CHROMA_API_KEY"]
#   CHROMA_TENANT = os.environ["CHROMA_TENANT"]
#   CHROMA_DATABASE = os.environ["CHROMA_DATABASE"]
#   client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
# ---


@dataclass
class IndexedChunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    metadata: dict


def load_chunks(path: Path = CHUNKS_PATH) -> List[IndexedChunk]:
    """Reads the chunks produced by chunker.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No chunks found at {path}. Run chunker.py first "
            f"(python -m src.ingestion.chunker)."
        )

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                chunks.append(IndexedChunk(**record))
    return chunks


def get_chroma_client() -> "chromadb.ClientAPI":
    """Local, persisted Chroma client — data lives at CHROMA_PERSIST_DIR on disk."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def build_dense_index(chunks: List[IndexedChunk]) -> None:
    """Embeds every chunk and stores the vectors in a persisted local Chroma collection."""
    print("=" * 70)
    print("BUILDING DENSE (CHROMA, LOCAL) INDEX")
    print("=" * 70)

    start = time.perf_counter()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(chunks)} chunks (this may take a couple of minutes)...", flush=True)
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    print(f"\nWriting vectors to local Chroma store at {CHROMA_PERSIST_DIR} ...", flush=True)
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Chroma metadata values must be str/int/float/bool — flatten nested metadata.
    flat_metadatas = []
    for c in chunks:
        flat = {"paper_id": c.paper_id, "section": c.section}
        for k, v in c.metadata.items():
            flat[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        flat_metadatas.append(flat)

    # Chroma has a per-call upsert limit on some versions; batch to be safe.
    BATCH = 250
    for i in range(0, len(chunks), BATCH):
        collection.upsert(
            ids=[c.chunk_id for c in chunks[i:i + BATCH]],
            embeddings=embeddings[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=flat_metadatas[i:i + BATCH],
        )

    elapsed = time.perf_counter() - start
    print(f"Dense index built: {collection.count()} vectors stored locally "
          f"at {CHROMA_PERSIST_DIR}")
    print(f"Time taken: {elapsed:.1f} seconds")


def build_sparse_index(chunks: List[IndexedChunk]) -> None:
    """Builds a BM25 keyword index over the same chunks and saves it to disk."""
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
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunk_records,
            },
            f,
        )

    elapsed = time.perf_counter() - start
    print(f"Sparse index built: {len(chunks)} chunks indexed -> {BM25_INDEX_PATH}")
    print(f"Time taken: {elapsed:.1f} seconds")


def main() -> None:
    try:
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}\n")

        build_dense_index(chunks)
        build_sparse_index(chunks)

        print("\n" + "=" * 70)
        print("INDEXING COMPLETE")
        print("=" * 70)
        print(f"Dense index (Chroma, local) : {CHROMA_PERSIST_DIR}")
        print(f"Sparse index (BM25)         : {BM25_INDEX_PATH}")

    except Exception as error:
        print("\nINDEXING STOPPED DUE TO A FATAL ERROR")
        print(f"Error: {error}")
        raise


if __name__ == "__main__":
    main()