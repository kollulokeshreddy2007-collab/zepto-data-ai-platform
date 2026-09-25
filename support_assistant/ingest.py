"""Task 1 - ingestion: load the 8 policy docs, chunk, embed, store in ChromaDB.

Chunking scheme: one chunk per document. The corpus documents are short (a few
sentences each), so a per-document chunk keeps each retrievable unit a complete,
self-contained policy - which also makes the returned `sources` (document IDs)
clean and unambiguous.

Run:
    python support_assistant/ingest.py
"""

from __future__ import annotations

from pathlib import Path

from vectorstore import COLLECTION_NAME, embed, get_client

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"


def load_documents() -> tuple[list[str], list[str]]:
    """Return (ids, texts) for the 8 policy documents, sorted by filename."""
    ids, texts = [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        ids.append(path.stem)                       # e.g. "doc_01"
        texts.append(path.read_text(encoding="utf-8").strip())
    return ids, texts


def main() -> None:
    ids, texts = load_documents()
    if len(ids) != 8:
        raise SystemExit(f"Expected 8 documents, found {len(ids)}.")

    client = get_client()
    # Rebuild the collection from scratch for a deterministic index.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    embeddings = embed(texts)
    coll.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": i} for i in ids],
    )
    print(f"Ingested {coll.count()} chunks into ChromaDB collection "
          f"'{COLLECTION_NAME}' (cosine).")

    # Smoke-test retrieval.
    from vectorstore import retrieve
    hits = retrieve("How long do I have to return a damaged item?", k=3)
    print("Sample query -> top hit:", hits[0]["id"],
          f"(similarity {hits[0]['similarity']:.3f})")


if __name__ == "__main__":
    main()
