"""Module 1 - Steps 2-6: clean, convert, load, query.

Reads data_pipeline/books_raw.csv (produced by scrape.py) and:
  2. Cleans fields into proper types (price_gbp float, rating int 1-5,
     in_stock bool), dropping rows that fail to parse.
  3. Converts price_gbp -> price_inr using the fixed baseline 1 GBP = 105.50 INR.
  4. Builds a normalized SQLite schema (categories 1--* books, PK/FK).
  5. Loads the data and runs >= 5 SQL queries covering SELECT/WHERE, ORDER BY,
     LIMIT, DISTINCT, IN/BETWEEN, plus a JOIN. Each query + output is saved.
  6. Reads two results back with pd.read_sql, and reproduces the JOIN query
     with pd.merge on in-memory DataFrames, showing the two agree.

Outputs:
  data_pipeline/books.db          - the SQLite database
  data_pipeline/query_output.txt  - every query string with its printed output

Run (after scrape.py):
    python data_pipeline/build_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW_CSV = HERE / "books_raw.csv"
DB_PATH = HERE / "books.db"
QUERY_OUT = HERE / "query_output.txt"

# Fixed, project-defined baseline conversion rate. Not a live/market rate.
GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# Collected report text is written to query_output.txt at the end.
_report: list[str] = []


def log(text: str = "") -> None:
    print(text)
    _report.append(text)


# ---------------------------------------------------------------------------
# Step 2 + 3: clean and convert
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw string columns into typed columns; drop rows that fail to parse.

    Cleaning decisions (documented in README):
      * price_gbp: strip the leading currency symbol, parse to float. A stray
        non-numeric price is unparseable and the row is dropped.
      * rating: map the English word (One..Five) to an int 1-5. Unknown word -> drop.
      * in_stock: True if the availability text contains "in stock" (case-insensitive).
      * We DROP unparseable rows rather than median-impute, because on this source
        the raw fields are highly regular, so a parse failure signals genuinely
        malformed data whose true value we cannot trust; with a clean 100+ row
        dataset, dropping the rare bad row is safer than inventing a median price
        or rating. (Median imputation would be the alternative for numeric fields.)
    """
    n_before = len(df)

    # price_gbp: "£51.77" -> 51.77 . Keep only digits and dot after stripping symbol.
    price_str = df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
    df["price_gbp"] = pd.to_numeric(price_str, errors="coerce")

    # rating word -> int
    df["rating"] = df["rating"].map(RATING_WORDS)

    # availability -> bool
    df["in_stock"] = df["availability"].astype(str).str.contains(
        "in stock", case=False, na=False
    )

    # Drop rows where a required numeric field failed to parse.
    bad = df["price_gbp"].isna() | df["rating"].isna()
    n_dropped = int(bad.sum())
    df = df.loc[~bad].copy()
    df["rating"] = df["rating"].astype(int)

    # Step 3: fixed-rate conversion.
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    log(f"Cleaning: started with {n_before} rows, dropped {n_dropped} unparseable, "
        f"kept {len(df)} rows.")
    log(f"Applied fixed baseline conversion 1 GBP = {GBP_TO_INR} INR.")
    log("")
    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


# ---------------------------------------------------------------------------
# Step 4 + 5: schema, load
# ---------------------------------------------------------------------------
SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,   -- stored as 0/1
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def build_database(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)

    # Populate categories table (unique names -> ids).
    categories = sorted(df["category"].unique())
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    conn.executemany(
        "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
        [(i, name) for name, i in cat_id.items()],
    )

    # Populate books table with the FK.
    rows = [
        (
            r.title,
            float(r.price_gbp),
            float(r.price_inr),
            int(r.rating),
            int(bool(r.in_stock)),
            cat_id[r.category],
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    log(f"Loaded {len(categories)} categories and {len(rows)} books into {DB_PATH.name}.")
    log("")
    return conn


# ---------------------------------------------------------------------------
# Step 5: five SQL queries covering every required clause + a JOIN
# ---------------------------------------------------------------------------
QUERIES: list[tuple[str, str]] = [
    (
        "Q1 SELECT/WHERE - books rated 5 that are in stock",
        """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating = 5 AND in_stock = 1;
        """,
    ),
    (
        "Q2 ORDER BY + LIMIT - 10 most expensive books (INR)",
        """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
        """,
    ),
    (
        "Q3 DISTINCT - the distinct rating values present",
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
        """,
    ),
    (
        "Q4 BETWEEN + IN - mid-priced books in selected categories",
        """
        SELECT b.title, b.price_gbp, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.price_gbp BETWEEN 20 AND 40
          AND c.category_name IN ('Travel', 'Poetry', 'Classics')
        ORDER BY b.price_gbp;
        """,
    ),
    (
        "Q5 JOIN + aggregation - avg price and book count per category",
        """
        SELECT c.category_name,
               COUNT(*)              AS n_books,
               ROUND(AVG(b.price_inr), 2) AS avg_price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY avg_price_inr DESC;
        """,
    ),
]

# The JOIN query reproduced later with pd.merge (Step 6).
JOIN_QUERY = """
SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC
LIMIT 10;
"""


def run_queries(conn: sqlite3.Connection) -> None:
    log("=" * 70)
    log("STEP 5: SQL QUERIES")
    log("=" * 70)
    for label, sql in QUERIES:
        result = pd.read_sql(sql, conn)
        log(f"\n--- {label} ---")
        log(sql.strip())
        log(result.to_string(index=False))
    log("")


# ---------------------------------------------------------------------------
# Step 6: pandas read-back + pd.merge equivalence
# ---------------------------------------------------------------------------
def pandas_readback(conn: sqlite3.Connection) -> None:
    log("=" * 70)
    log("STEP 6: PANDAS READ-BACK AND pd.merge EQUIVALENCE")
    log("=" * 70)

    # Read two earlier query results back into DataFrames via pd.read_sql.
    df_q3 = pd.read_sql(QUERIES[2][1], conn)
    df_q5 = pd.read_sql(QUERIES[4][1], conn)
    log("\npd.read_sql of Q3 (distinct ratings):")
    log(df_q3.to_string(index=False))
    log("\npd.read_sql of Q5 (avg price per category):")
    log(df_q5.to_string(index=False))

    # Reproduce the JOIN query using pd.merge on in-memory DataFrames (no SQL join).
    books = pd.read_sql("SELECT * FROM books", conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)

    sql_join = pd.read_sql(JOIN_QUERY, conn)

    merged = (
        pd.merge(books, categories, on="category_id")
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .loc[:, ["category_name", "title", "rating", "price_inr"]]
        .head(10)
        .reset_index(drop=True)
    )

    log("\n--- JOIN via SQL (pd.read_sql) ---")
    log(sql_join.to_string(index=False))
    log("\n--- Same JOIN via pd.merge (no SQL) ---")
    log(merged.to_string(index=False))

    equal = sql_join.reset_index(drop=True).equals(merged)
    log(f"\nSQL JOIN result equals pd.merge result: {equal}")


def main() -> None:
    df_raw = pd.read_csv(RAW_CSV)
    df = clean(df_raw)
    conn = build_database(df)
    try:
        run_queries(conn)
        pandas_readback(conn)
    finally:
        conn.close()
    QUERY_OUT.write_text("\n".join(_report), encoding="utf-8")
    log(f"\nSaved full query log -> {QUERY_OUT.name}")


if __name__ == "__main__":
    main()
