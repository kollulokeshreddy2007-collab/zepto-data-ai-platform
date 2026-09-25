# Module 1 - Data Pipeline (`/data_pipeline`)

A raw-to-relational pipeline: scrape live product data from
[books.toscrape.com](https://books.toscrape.com), clean it, convert currency,
load it into a normalized SQLite database, and query it with both SQL and pandas.

## Files

| File | Purpose |
|------|---------|
| `scrape.py` | Step 1. Scrapes books across 5 categories with `requests` + `BeautifulSoup`, writes `books_raw.csv`. |
| `build_db.py` | Steps 2-6. Cleans/types the data, applies the fixed-rate INR conversion, builds the SQLite schema, loads it, runs the SQL queries, and does the pandas read-back / `pd.merge` equivalence check. |
| `books_raw.csv` | Raw scraped output (regenerable via `scrape.py`). |
| `books.db` | The SQLite database (regenerable via `build_db.py`). |
| `query_output.txt` | Every executed query string with its printed output. |

## How to run

From the repository root, with the project virtual environment active:

```bash
python data_pipeline/scrape.py
python data_pipeline/build_db.py
```

`scrape.py` must run first (it produces `books_raw.csv`); `build_db.py` reads that
CSV and regenerates `books.db` and `query_output.txt` from scratch each run.

## Dataset scope

Scrapes **5 categories** (Travel, Mystery, Historical Fiction, Classics, Poetry)
for **107 books total**, comfortably clearing the >= 60 books / >= 3 categories bar.
Each category listing is fully paginated (the scraper follows `next` links).

## Fields captured and cleaning decisions

| Raw field | Cleaned column | Rule |
|-----------|----------------|------|
| `£51.77` | `price_gbp` (float) | Strip every non-numeric character, parse to float. |
| `"Three"` | `rating` (int 1-5) | Map the English word One..Five to 1..5. |
| `"In stock"` | `in_stock` (bool) | `True` if the text contains "in stock" (case-insensitive). |
| `price_gbp` | `price_inr` (float) | `price_gbp * 105.50`, rounded to 2 dp. |

**Currency conversion:** `price_inr` uses the required fixed project baseline
**1 GBP = 105.50 INR**. This is an artificial, project-defined constant, not a
live or historical market rate, so it needs no API call, no network access, and
no date reference.

**Handling unparseable rows:** rows where `price_gbp` or `rating` fail to parse
are **dropped** (not median-imputed). Justification: the source fields are highly
regular, so a parse failure signals genuinely malformed data whose true value
cannot be trusted; with a clean 100+ row dataset, dropping the rare bad row is
safer than inventing a median value. (Median imputation of numeric fields is the
documented alternative.) In practice this run dropped **0** rows.

## Database schema (two tables, PK/FK)

```
categories(category_id INTEGER PRIMARY KEY,
           category_name TEXT UNIQUE NOT NULL)

books(book_id     INTEGER PRIMARY KEY,
      title       TEXT NOT NULL,
      price_gbp   REAL NOT NULL,
      price_inr   REAL NOT NULL,
      rating      INTEGER NOT NULL,
      in_stock    INTEGER NOT NULL,           -- 0/1
      category_id INTEGER NOT NULL REFERENCES categories(category_id))
```

`books.category_id` is a foreign key into `categories.category_id`
(one category -> many books).

## SQL queries (Step 5)

Five queries collectively cover **SELECT/WHERE, ORDER BY, LIMIT, DISTINCT,
IN, BETWEEN, and a JOIN**. Full text and output are in `query_output.txt`.

| Query | Clauses demonstrated |
|-------|----------------------|
| Q1 | `SELECT` / `WHERE` (5-star, in stock) |
| Q2 | `ORDER BY` + `LIMIT` (10 most expensive) |
| Q3 | `DISTINCT` (rating values present) |
| Q4 | `BETWEEN` + `IN` + `JOIN` (mid-priced in selected categories) |
| Q5 | `JOIN` + aggregation (avg price and count per category) |

## Pandas read-back and merge equivalence (Step 6)

`build_db.py` reads two query results back with `pd.read_sql`, then reproduces
the join query using `pd.merge` on the in-memory `books` and `categories`
DataFrames (no SQL join). The script asserts and prints that the SQL-join and
`pd.merge` results are **identical** (`True`).
