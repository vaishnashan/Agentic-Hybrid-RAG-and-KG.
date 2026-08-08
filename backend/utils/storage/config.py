"""
Single source of truth for every path and environment-variable name the
storage pipeline (and retrieval/, which reads what storage/ writes) depends
on. Nothing in storage/ or retrieval/ should hardcode a path or an env var
name outside of this file — change it here once, it's correct everywhere.

Previously these constants were scattered across loader.py/chunker.py/
embed_and_index.py and one of them (loader.py) hardcoded an absolute Windows
path (E:\\4.Project 1\\codebase\\...) that only worked on one machine. Every
path here is derived from this file's own location instead, so the project
runs the same way on Windows, Linux, in Docker, or on any lightweight cloud
host.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project paths ────────────────────────────────────────────────────────
# This file lives at: backend/utils/storage/config.py
#   parents[0] = storage/   parents[1] = utils/   parents[2] = backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Override with RAW_DATA_DIR / PROCESSED_DATA_DIR in .env if you want to
# point the pipeline at a different corpus folder without touching code —
# e.g. to load a fresh batch of PDFs from somewhere else.
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BACKEND_ROOT / "data" / "raw")))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(BACKEND_ROOT / "data" / "processed")))

METADATA_PATH = RAW_DATA_DIR / "paper_metadata.jsonl"
PAPERS_PATH = PROCESSED_DATA_DIR / "papers.jsonl"          # loader.py output
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"           # chunker.py output
BM25_INDEX_PATH = PROCESSED_DATA_DIR / "bm25_index.pkl"     # bm25_index.py output

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
CHARS_PER_WORD = 6  # rough average for English technical text, incl. spaces

# ── Dense index (embeddings + Chroma Cloud) ─────────────────────────────
# The SAME model name is used at both index time (dense_index.py, local
# sentence-transformers) and query time (retrieval/query_embedder.py, cloud
# inference API) — the two vector spaces must match or search quality
# collapses. If you change this, you MUST rebuild the dense index.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSIONS = 384  # bge-small-en-v1.5's output size — used to sanity-check API responses

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "paper_chunks")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")

# ── Query-time embeddings (cloud API, no local model load in the API process) ──
# "huggingface" calls the HF Inference API for the exact model above, so the
# query vector lands in the same space as the chunks embedded locally during
# the pipeline run. "local" loads sentence-transformers in-process instead —
# useful for local dev/testing, but defeats the point of a lightweight
# deploy (pulls in torch + the model weights at runtime).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_FEATURE_EXTRACTION_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"
)

# ── Sparse index (BM25) ──────────────────────────────────────────────────
# Always local disk — the index is a small pickle file, so it ships inside
# the deploy image itself (see Dockerfile) rather than needing cloud storage.

# ── Knowledge graph (Neo4j Aura) ─────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # Aura's default database name

# ── Groq (LLM calls) ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Triple extraction (storage/graph_index.py): a small, fast, cheap model is
# plenty for structured extraction — GPT-OSS-20B is Groq's current
# recommendation for this kind of task.
GROQ_EXTRACTION_MODEL = os.getenv("GROQ_EXTRACTION_MODEL", "openai/gpt-oss-20b")

# Agent reasoning (planner.py / reasoner.py): Llama-3.3-70B-Versatile, as
# requested. NOTE (as of Aug 2026): Groq announced on 2026-06-17 that this
# model is deprecated, with traffic being phased out by ~August 2026 — see
# https://console.groq.com/docs/deprecations. If calls to it start failing
# with a "model_decommissioned" error, set GROQ_AGENT_MODEL in your .env to
# Groq's suggested replacement, openai/gpt-oss-120b (closest current
# equivalent in capability/size), with no other code changes needed.
GROQ_AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
