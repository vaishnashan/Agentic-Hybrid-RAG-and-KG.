# NOVA — Frontend (Streamlit UI)

Talks to the backend's `/ask` endpoint over plain HTTP — no direct imports
from the backend, no shared Python environment. Deploy it as a completely
separate service from the backend.

```
frontend/
├── Dockerfile
├── requirements-ui.txt      # streamlit, requests, python-dotenv — nothing else
├── .env.example
├── .streamlit/
│   └── config.toml          # theme — MUST stay at this path, see note below
└── ui/
    ├── __init__.py
    └── app.py                # the entire UI
```

### Why `.streamlit/config.toml` lives at the repo root, not inside `ui/`
Streamlit resolves `.streamlit/config.toml` relative to the **working
directory `streamlit run` is invoked from** — not the entrypoint script's own
folder. Since the Dockerfile runs `streamlit run ui/app.py` from `WORKDIR
/app`, the config must be at `/app/.streamlit/config.toml`, i.e. this repo's
root `.streamlit/`, or the theme silently never applies. Keep this file here
if you move things around later.

## Running locally

```bash
pip install -r requirements-ui.txt
cp .env.example .env   # fill in API_BASE_URL / API_KEY
streamlit run ui/app.py
```

## Deploying on Render

1. Push this `frontend/` folder to its own repo (or a subfolder of a repo —
   Render lets you point a service at a subdirectory).
2. Render → New → Web Service → connect the repo, it auto-detects the
   `Dockerfile`.
3. Set environment variables in Render's dashboard:
   - `API_BASE_URL` = your deployed backend's Render URL (e.g.
     `https://your-backend-name.onrender.com`)
   - `API_KEY` = must match the backend's `API_KEY` exactly
4. Deploy.

### Notes on Render's free tier
- Free web services spin down after 15 minutes idle; the next request takes
  roughly 30-60 seconds to wake back up. If both frontend and backend are on
  free instances, a cold start on both can stack — the first question after
  a period of inactivity may take noticeably longer than usual. This is
  normal, not a bug.
- If you ever see the frontend itself time out waiting for the backend on a
  multi-hop question (several LLM calls chained together), that's a proxy/
  request timeout setting, not the 180s timeout already set in `app.py`'s
  `requests.post(..., timeout=180)` — check Render's service-level timeout
  settings if this happens consistently.

### CORS
The backend's `main.py` currently allows all origins (`allow_origins=["*"]`).
That's fine for getting this running, but once both services have stable
Render URLs, consider tightening the backend's CORS to just this frontend's
exact URL.
