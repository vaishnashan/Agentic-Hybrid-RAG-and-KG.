# NOVA — Streamlit Frontend

Professional Streamlit UI for the NOVA agentic RAG + knowledge-graph backend.
The frontend is a separate service and communicates with FastAPI through `/ask`.

## Project structure

```text
frontend/
├── Dockerfile
├── requirements-ui.txt
├── .env.example
├── .streamlit/
│   └── config.toml
└── ui/
    ├── __init__.py
    └── app.py
```

## Local run

```bash
pip install -r requirements-ui.txt
cp .env.example .env
# Set API_BASE_URL and API_KEY in .env
streamlit run ui/app.py
```

On Windows PowerShell, you can create `.env` manually instead of using `cp`.

## Docker

```bash
docker build -t nova-frontend .
docker run --env-file .env -p 8501:8501 nova-frontend
```

Then open `http://localhost:8501`.

## Render

Create a Web Service from this frontend folder and set:

- `API_BASE_URL` — deployed NOVA backend URL
- `API_KEY` — same API key configured in the backend

The Dockerfile uses Render's `PORT` automatically.

## UI stability fix

The UI no longer depends on a local `hero_art.jpg` asset. The visual graph pattern
is embedded in CSS, and the Streamlit theme is explicitly dark. The CSS also targets
both `.stApp` and Streamlit's current `data-testid` containers so a Streamlit theme
or version change cannot silently turn the page white while leaving light text.
