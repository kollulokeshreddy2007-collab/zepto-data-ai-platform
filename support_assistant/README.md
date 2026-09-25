# Module 3 - Support Assistant (`/support_assistant`)

A small but complete GenAI service for Zepto: a policy-document corpus embedded and
indexed in ChromaDB, a LangGraph-orchestrated intent router that retrieves grounded
context, a Pydantic structured-output guarantee, and a FastAPI wrapper you run
locally. **The graded baseline is a fully offline, deterministic mock** - no
signup, no API key, no network call to any LLM provider.

## Files

| File | Role |
|------|------|
| `docs/doc_01.txt` .. `doc_08.txt` | The 8-document policy corpus. |
| `vectorstore.py` | Local `all-MiniLM-L6-v2` embeddings + persistent ChromaDB (cosine) helpers; `retrieve()`. |
| `ingest.py` | Task 1: loads, chunks (one chunk per doc), embeds, and stores the corpus. |
| `prompts.py` | Task 2: the structured role/context/task/format/length prompt (with negative constraint + few-shot). |
| `graph.py` | Tasks 3/4: LangGraph `StateGraph` (3 nodes, conditional edge), `MOCK_LLM` branching, Pydantic schema, retry logic. |
| `app.py` | Task 5: FastAPI `POST /ask`. |
| `Dockerfile` | Task 6: builds and runs the app locally (offline mock). |

## Setup and run (graded offline mock baseline)

From the repository root with the project venv active:

```bash
pip install -r support_assistant/requirements.txt
cd support_assistant
python ingest.py                     # build the ChromaDB index (downloads MiniLM once)
uvicorn app:app --host 127.0.0.1 --port 8000
```

`MOCK_LLM` is unset/`1` by default, so the whole pipeline is deterministic and
offline. Then POST to `/ask`:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d "{\"query\": \"How much is the delivery fee for a small order?\"}"
```

### With Docker (also the graded baseline)

```bash
cd support_assistant
docker build -t zepto-support .
docker run -p 8000:8000 zepto-support
```

The Dockerfile installs CPU-only PyTorch, bakes the ChromaDB index at build time,
and serves `/ask` on port 8000 with `MOCK_LLM=1`. (Docker is not installed on the
authoring machine, so the image was not built here; the Dockerfile is written to be
buildable/runnable locally as above.)

## Example calls (recorded with MOCK_LLM at its default)

Run live via `uvicorn app:app` and queried over HTTP:

**Call 1 - a policy question (routes to `retrieve_and_answer`):**
```json
// POST /ask  {"query": "How much is the delivery fee for a small order?"}
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": ["doc_01", "doc_05", "doc_02"],
  "confidence": 1.0
}
```

**Call 2 - a general question (routes to `direct_answer`):**
```json
// POST /ask  {"query": "Who won the world cup in 2018?"}
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Call 1 contains the keyword "delivery", so `classify_intent` routes it to retrieval;
the top retrieved chunk is `doc_01` (Delivery Policy) - the correct source. Call 2
has no policy keyword, so it routes to the canned direct answer with empty sources.

## RAG pipeline architecture (Task 7)

Data flows through four stages, **ingestion -> embedding -> retrieval -> generation**:

```
                          ingest.py                         graph.py
  docs/doc_01..08.txt  ->  load + chunk  ->  embed (MiniLM)  ->  ChromaDB
                          (1 chunk/doc)      vectorstore.embed()   'zepto_policies'
                                                                    (cosine)
                                                                        |
  POST /ask {query}                                                     | retrieve top-3
     (app.py)                                                           v
        |                                                     vectorstore.retrieve()
        v                                                               |
  run_query() -> LangGraph:                                             |
     classify_intent --(policy_question)--> retrieve_and_answer --------+--> generate
                     \-(general_question)-> direct_answer                    (mock/real)
                                                                                |
                                             AnswerResponse {answer, sources, confidence}
```

- **Ingestion** - `ingest.py::load_documents` reads the 8 `.txt` files and chunks
  them one-chunk-per-document (they are short, so each chunk is a complete policy).
- **Embedding** - `vectorstore.py::embed` encodes each chunk with the local
  `all-MiniLM-L6-v2` model; `ingest.py::main` stores the vectors in the ChromaDB
  collection **`zepto_policies`** (configured `hnsw:space=cosine`) at `./chroma_db`.
- **Retrieval** - the `retrieve_and_answer` node calls `vectorstore.py::retrieve`,
  which embeds the query and pulls the **top-3** chunks by cosine similarity.
- **Generation** - `retrieve_and_answer` (policy) or `direct_answer` (general)
  produces the final text, which `run_query` wraps in the Pydantic `AnswerResponse`.

**Where `MOCK_LLM` branches.** Only the **generation** step inside the two answer
nodes branches on `MOCK_LLM`. Intent routing (`classify_intent`) and retrieval
(`vectorstore.retrieve`) always run for real, because embeddings and ChromaDB need
no API key or network.

| Stage | Default (mock, graded) | Optional real (`MOCK_LLM=0`) |
|-------|------------------------|------------------------------|
| classify_intent | keyword heuristic | LLM classification |
| retrieve | real cosine top-3 (unchanged) | real cosine top-3 (unchanged) |
| retrieve_and_answer generation | canned `"Based on the retrieved context: ..."` | LLM grounded in the retrieved chunks (structured prompt) |
| direct_answer generation | fixed canned string | direct LLM answer |
| schema | populated deterministically (sources = retrieved ids, confidence = 1.0) | validated with up to 2 corrective retries |

## Structured output schema (Task 4)

Every response is validated against the Pydantic model
`AnswerResponse{answer: str, sources: list[str], confidence: float 0-1}`. In mock
mode this is populated deterministically (sources = the retrieved chunk IDs for a
policy question, `[]` for a general question; confidence = 1.0), so there is no LLM
output that could fail validation. On the optional real path, invalid model output
triggers up to 2 corrective retries before a clearly-marked error response.

## MOCK_LLM toggle

| `MOCK_LLM` | Behavior |
|------------|----------|
| unset or `1` (**graded default**) | Fully deterministic, offline. No LLM call. |
| `0` (optional, ungraded) | Calls a real LLM (Groq free tier by default; set `GROQ_API_KEY`). |

## Notes

- **Optional extensions** (not attempted as live deployments): the `MOCK_LLM=0`
  real-LLM path (Groq free tier) is implemented in code but ungraded; a live
  Hugging Face Spaces deployment is not included (the Dockerfile is the graded
  baseline and is locally buildable).
- **doc_08 note:** the source text for doc_08 (Support Hours) was truncated at
  "Average in-app chat response time is under" in the provided brief; its final
  sentences were completed in the same style so the document is self-contained.
