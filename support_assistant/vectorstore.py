"""Shared embedding + ChromaDB helpers (Task 1 storage, used by ingest & graph).

Embeddings are generated locally with the open-source all-MiniLM-L6-v2 model
(sentence-transformers) and stored in a persistent ChromaDB collection configured
for COSINE similarity. No API key and no LLM network call are involved here -
this layer runs identically in both mock and real-LLM modes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
CHROMA_PATH = str(HERE / "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load (once) the local sentence-transformers embedding model."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Persistent ChromaDB client rooted at ./chroma_db."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(create: bool = False):
    """Return the policy collection (cosine space)."""
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return client.get_collection(name=COLLECTION_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors (list of floats)."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Embed the query and return the top-k most similar chunks by cosine similarity.

    Returns a list of {id, document, distance, similarity} dicts, best first.
    This runs FOR REAL in both mock and real-LLM modes.
    """
    coll = get_collection()
    q_emb = embed([query])
    res = coll.query(query_embeddings=q_emb, n_results=k)
    ids = res["ids"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]
    return [
        {"id": i, "document": d, "distance": dist, "similarity": 1.0 - dist}
        for i, d, dist in zip(ids, docs, dists)
    ]
