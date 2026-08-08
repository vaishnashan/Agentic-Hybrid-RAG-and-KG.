"""
Run the ENTIRE storage pipeline end to end:

    1. load    — read PDFs + metadata from data/raw/, extract text
    2. chunk   — split extracted text into retrieval-ready chunks
    3. store   — build all three indexes FROM those chunks/papers:
                   a. dense index  -> Chroma Cloud   (dense_index.py)
                   b. sparse index -> local BM25 pickle (bm25_index.py)
                   c. graph index  -> Neo4j Aura     (graph_index.py, via GPT-OSS)

Usage:
    python -m utils.storage.pipeline                  # run everything
    python -m utils.storage.pipeline --skip-load       # reuse existing papers.jsonl
    python -m utils.storage.pipeline --skip-load --skip-chunk   # reuse existing chunks.jsonl
    python -m utils.storage.pipeline --only dense,bm25  # skip the graph step (e.g. no GROQ_API_KEY yet)

Run this whenever the raw PDF corpus changes (new papers added, or
chunking/embedding parameters changed) — this is the ONE command that
rebuilds everything the retrieval/ and agent/ packages read at query time.
"""
import argparse
import json
import time

from .bm25_index import run_bm25_index
from .chunker import load_processed_papers, run_chunker
from .config import CHUNKS_PATH, PAPERS_PATH
from .dense_index import run_dense_index
from .graph_index import run_graph_index
from .loader import run_loader

STEP_NAMES = ("dense", "bm25", "graph")


def _load_papers_dicts():
    papers = []
    with PAPERS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    return papers


def run_pipeline(skip_load: bool = False, skip_chunk: bool = False, only: list = None) -> None:
    only = set(only) if only else set(STEP_NAMES)
    unknown = only - set(STEP_NAMES)
    if unknown:
        raise ValueError(f"Unknown step(s) in --only: {unknown}. Valid steps: {STEP_NAMES}")

    overall_start = time.perf_counter()

    print("#" * 70)
    print("# STORAGE PIPELINE STARTED")
    print("#" * 70)

    # ── Step 1: load ─────────────────────────────────────────────────────
    if skip_load:
        print("\n[1/3] LOAD — skipped (--skip-load), reusing existing papers.jsonl")
        if not PAPERS_PATH.exists():
            raise FileNotFoundError(
                f"--skip-load was set but {PAPERS_PATH} doesn't exist. Run without "
                f"--skip-load at least once first."
            )
    else:
        print("\n[1/3] LOAD — extracting PDFs + metadata")
        run_loader()

    # ── Step 2: chunk ────────────────────────────────────────────────────
    if skip_chunk:
        print("\n[2/3] CHUNK — skipped (--skip-chunk), reusing existing chunks.jsonl")
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"--skip-chunk was set but {CHUNKS_PATH} doesn't exist. Run without "
                f"--skip-chunk at least once first."
            )
        chunks = None
    else:
        print("\n[2/3] CHUNK — splitting extracted text into retrieval-ready chunks")
        chunks = run_chunker()

    # dense_index.py / bm25_index.py both need in-memory Chunk objects — load
    # from disk if the chunk step above was skipped.
    if chunks is None and ("dense" in only or "bm25" in only):
        chunks = load_processed_papers  # placeholder, replaced below
        from .chunker import Chunk
        chunks = []
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

    # graph_index.py needs the raw paper dicts (full_text), not chunks.
    papers = _load_papers_dicts() if "graph" in only else None

    # ── Step 3: store (dense + bm25 + graph) ────────────────────────────
    print(f"\n[3/3] STORE — building: {', '.join(sorted(only))}")

    if "dense" in only:
        print("\n--- [3a] Dense index (Chroma Cloud) ---")
        run_dense_index(chunks)

    if "bm25" in only:
        print("\n--- [3b] Sparse index (BM25, local disk) ---")
        run_bm25_index(chunks)

    if "graph" in only:
        print("\n--- [3c] Graph index (Neo4j Aura, via GPT-OSS) ---")
        run_graph_index(papers)

    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "#" * 70)
    print("# STORAGE PIPELINE COMPLETE")
    print("#" * 70)
    print(f"Total time: {total_elapsed:.1f} seconds")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the storage pipeline (load -> chunk -> dense+bm25+graph).")
    parser.add_argument("--skip-load", action="store_true", help="Reuse existing data/processed/papers.jsonl")
    parser.add_argument("--skip-chunk", action="store_true", help="Reuse existing data/processed/chunks.jsonl")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated subset of storing steps to run: dense,bm25,graph (default: all three)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only_steps = [s.strip() for s in args.only.split(",")] if args.only else None
    run_pipeline(skip_load=args.skip_load, skip_chunk=args.skip_chunk, only=only_steps)
