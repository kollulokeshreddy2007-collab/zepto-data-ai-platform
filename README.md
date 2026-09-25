# Zepto Data & AI Platform - Capstone

One repository, three internally-linked modules that read as a single story: a
data-engineering pipeline turns raw scraped data into a clean relational store,
an analytics pipeline profiles and models a customer-style dataset end to end,
and a GenAI support assistant answers policy questions grounded in Zepto's own
documents.

| Module | Path | Marks | What it does |
|--------|------|-------|--------------|
| 1. Data Pipeline | [`/data_pipeline`](data_pipeline/) | 25 | Scrape -> clean -> convert -> SQLite -> SQL & pandas queries. |
| 2. Analytics | [`/analytics`](analytics/) | 50 | EDA + cleaning + three classifiers + tuning + regression on the Titanic dataset. |
| 3. Support Assistant | [`/support_assistant`](support_assistant/) | 25 | RAG over Zepto policy docs: ChromaDB + LangGraph + FastAPI, offline-mock baseline. |

## Setup

Requires **Python 3.11 or 3.12** (recommended for ChromaDB / PyTorch
compatibility in Module 3). Each module has its own `requirements.txt`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Running each module

**Module 1 - Data Pipeline**
```bash
python data_pipeline/scrape.py
python data_pipeline/build_db.py
```

**Module 2 - Analytics** (see [`analytics/README.md`](analytics/README.md))
```bash
# run the notebooks / scripts in order (EDA first, then modeling)
```

**Module 3 - Support Assistant** (see [`support_assistant/README.md`](support_assistant/README.md))
```bash
python support_assistant/ingest.py       # build the ChromaDB index
uvicorn support_assistant.app:app --reload
```

## Design decisions (summary)

- **Data Pipeline:** scrape 5 categories (107 books) so the >= 60 row / >= 3
  category bar is met with margin; drop unparseable rows (source is highly
  regular); fixed baseline **1 GBP = 105.50 INR**; two-table PK/FK schema.
- **Analytics:** single load of the Titanic dataset (cached to `titanic.csv`);
  percentage-threshold missing-value rule; leakage-safe `ColumnTransformer` +
  `Pipeline` fit on train only; three classifiers + imbalance comparison +
  `GridSearchCV`; regression side-task; full fitted pipeline saved via joblib.
- **Support Assistant:** deterministic offline **mock** LLM path is the graded
  baseline (`MOCK_LLM` toggle); local `all-MiniLM-L6-v2` embeddings in ChromaDB
  (cosine); LangGraph router with 3 nodes; Pydantic-validated `answer/sources/
  confidence`; FastAPI `/ask` + Dockerfile.

## Git workflow

The commit history shows at least one feature branch created, committed to twice,
and merged back into `main` (visible via `git log --graph --all`).

## Notes

- No paid services are required anywhere. Every external dependency has a free,
  keyless path (or a stated free tier for optional extensions).
- All deliverables are textual (code + Markdown). Saved chart PNGs are supporting
  artifacts only; every required interpretation lives as Markdown text.
