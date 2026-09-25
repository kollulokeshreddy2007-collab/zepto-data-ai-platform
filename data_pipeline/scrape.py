"""Module 1 - Step 1: Scrape live product data from books.toscrape.com.

Scrapes every book listed across a set of categories (>= 3 categories,
>= 60 books total) using requests + BeautifulSoup, and writes the raw,
UNCLEANED fields to data_pipeline/books_raw.csv.

Fields captured per book (all as raw listed text, cleaning happens later):
    title        - book title
    price        - price string as listed, in GBP (e.g. "£51.77")
    rating       - star rating as text word (e.g. "Three")
    availability - availability text (e.g. "In stock")
    category     - category name the book was listed under

Run:
    python data_pipeline/scrape.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
OUT_CSV = Path(__file__).with_name("books_raw.csv")

# Categories chosen to comfortably clear the >= 60 books / >= 3 categories bar.
# Each of these categories has enough titles that the three together exceed 60.
CATEGORIES = ["Travel", "Mystery", "Historical Fiction", "Classics", "Poetry"]

# Be polite to the practice site.
REQUEST_DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "zepto-capstone-scraper/1.0 (educational use)"}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed HTML, raising on any non-200 status."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser")


def discover_category_links() -> dict[str, str]:
    """Return {category_name: absolute_listing_url} from the homepage sidebar."""
    soup = get_soup(BASE_URL)
    links: dict[str, str] = {}
    side = soup.select_one("div.side_categories ul li ul")
    for a in side.select("li a"):
        name = a.get_text(strip=True)
        links[name] = urljoin(BASE_URL, a["href"])
    return links


def parse_book_pod(pod, category: str) -> dict[str, str]:
    """Extract the raw fields from a single book <article class='product_pod'>."""
    title = pod.h3.a["title"]
    price = pod.select_one("p.price_color").get_text(strip=True)
    # Rating word is the 2nd class, e.g. class="star-rating Three".
    rating_classes = pod.select_one("p.star-rating")["class"]
    rating = next((c for c in rating_classes if c != "star-rating"), "")
    availability = pod.select_one("p.instock.availability").get_text(strip=True)
    return {
        "title": title,
        "price": price,
        "rating": rating,
        "availability": availability,
        "category": category,
    }


def scrape_category(name: str, listing_url: str) -> list[dict[str, str]]:
    """Scrape all books in one category, following pagination ('next' links)."""
    rows: list[dict[str, str]] = []
    url = listing_url
    while url:
        soup = get_soup(url)
        for pod in soup.select("article.product_pod"):
            rows.append(parse_book_pod(pod, name))
        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None
    print(f"  {name}: {len(rows)} books")
    return rows


def main() -> None:
    all_links = discover_category_links()
    all_rows: list[dict[str, str]] = []
    print("Scraping categories:")
    for name in CATEGORIES:
        if name not in all_links:
            print(f"  WARNING: category '{name}' not found on site, skipping")
            continue
        all_rows.extend(scrape_category(name, all_links[name]))

    fieldnames = ["title", "price", "rating", "availability", "category"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_categories = len({r["category"] for r in all_rows})
    print(f"\nWrote {len(all_rows)} books across {n_categories} categories -> {OUT_CSV.name}")
    if len(all_rows) < 60 or n_categories < 3:
        raise SystemExit("Acceptance check FAILED: need >= 60 books across >= 3 categories.")
    print("Acceptance check OK: >= 60 books across >= 3 categories.")


if __name__ == "__main__":
    main()
