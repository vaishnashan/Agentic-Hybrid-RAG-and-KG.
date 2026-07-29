# API service — FastAPI backend.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# sentence-transformers depends on torch, and a plain `pip install torch` on
# Linux defaults to the GPU-enabled build — which drags in ~1.2GB of NVIDIA
# CUDA libraries (cudnn, nccl, cusparselt, etc.) that are completely unused on
# Render's (or most cloud hosts') CPU-only containers. Installing the
# CPU-only wheel FIRST satisfies sentence-transformers' torch requirement
# before pip ever considers the default GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models INTO THE IMAGE at build time.
# Without this, dense_retriever.py / reranker.py lazy-load them on the first
# real request instead — meaning the first query after every container start
# pays the full HF Hub download cost (slower still since it's unauthenticated),
# often blowing past the UI's request timeout. Baking them in means every
# container starts with both models already on disk, cold or not.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copies utils/ (agent4, api7, retrieval2, ingestion1, knowledge_graph3,
# observability5, evaluation6) AND data/processed/bm25_index.pkl —
# sparse_retriever.py reads that pickle file directly from disk at query
# time, so it must ship inside the image, not just the code. See
# .dockerignore — it deliberately does NOT exclude data/processed.
COPY . .

# Run as non-root — no reason this process needs root inside the container.
# chown BEFORE switching user: COPY above runs as root, so without this
# appuser can't write to data/processed/ (this is what was causing the
# "Failed to write local trace log: Permission denied" errors) or read the
# HF cache baked in above.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT','8000') + '/health')" || exit 1

# Shell form (not exec-array form) so $PORT actually gets expanded — Render
# assigns its own PORT at runtime and expects the app to listen on THAT, not
# on a hardcoded port. Falls back to 8000 so local docker-compose (which maps
# a fixed 8000:8000) keeps working unchanged.
CMD ["sh", "-c", "uvicorn utils.api7.main:app --host 0.0.0.0 --port ${PORT:-8000}"]