# UI service — Streamlit demo frontend, talks to the API service over HTTP.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Uses requirements-ui.txt (streamlit, requests, python-dotenv), NOT the
# backend's requirements.txt — the UI never touches torch/chromadb/neo4j/
# langgraph/etc. This is what keeps this build fast and the image small:
# three small packages instead of the entire backend stack.
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

# IMPORTANT: .streamlit/config.toml must live at the WORKDIR ROOT (/app),
# not inside ui/. Streamlit resolves .streamlit/config.toml relative to the
# directory `streamlit run` is INVOKED FROM (the working directory), not
# the entrypoint script's own folder — confirmed against Streamlit's own
# docs/issue tracker. Since CMD below runs `streamlit run ui/app.py` from
# WORKDIR /app, the config must be at /app/.streamlit/config.toml or the
# theme silently never applies.
COPY .streamlit/ .streamlit/
COPY ui/ ui/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT','8501') + '/_stcore/health')" || exit 1

# Shell form so $PORT expands — Render assigns its own PORT at runtime.
# Falls back to 8501 so local docker-compose keeps working unchanged.
CMD ["sh", "-c", "streamlit run ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
