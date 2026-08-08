"""
Chroma Cloud client construction — split out from dense_index.py on purpose.

dense_index.py (the BUILD-time script) imports sentence-transformers at
module level, since it embeds chunks locally. utils/retrieval/dense_retriever.py
(the QUERY-time module, imported by every /ask request) only needs a Chroma
client — it must NOT transitively import sentence-transformers just by
importing this. Keeping get_chroma_client() here, with no heavy imports,
means the deployed API's import graph never touches sentence-transformers'
embedding model — only chromadb, which is lightweight.
"""
import chromadb

from .config import CHROMA_API_KEY, CHROMA_DATABASE, CHROMA_TENANT


def get_chroma_client() -> "chromadb.ClientAPI":
    """Chroma Cloud client — data lives in your Chroma Cloud database, not locally."""
    if not (CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE):
        raise ValueError(
            "CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE not set. Add them "
            "to your .env using the values from your Chroma Cloud database."
        )
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )
