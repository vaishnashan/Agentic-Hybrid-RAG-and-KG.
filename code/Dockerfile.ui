# UI service — Streamlit demo frontend, talks to the API service over HTTP.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Uses requirements-ui.txt (streamlit, requests, python-dotenv), NOT the full
# requirements.txt — the UI never touches torch/chromadb/neo4j/langgraph/
# ragas/etc. This is what actually keeps this build fast: three small
# packages instead of the entire backend stack, including its CUDA downloads.
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY ui/ ui/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT','8501') + '/_stcore/health')" || exit 1

# Shell form so $PORT expands — same reasoning as Dockerfile.api. Falls back
# to 8501 so local docker-compose keeps working unchanged.
CMD ["sh", "-c", "streamlit run ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]