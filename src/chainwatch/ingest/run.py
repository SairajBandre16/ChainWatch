"""CLI: fetch news into the local store and optionally refresh the committed sample.

python -m chainwatch.ingest.run                 # fetch (uses cache where present)
python -m chainwatch.ingest.run --refresh       # force re-download
python -m chainwatch.ingest.run --offline       # cache only, no network
python -m chainwatch.ingest.run --write-sample  # also write data/sample/news_sample.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chainwatch.config import RAW_DIR, SAMPLE_DIR
from chainwatch.ingest.http_cache import CachedHttp
from chainwatch.ingest.models import NewsItem, dedupe
from chainwatch.ingest.sources import fetch_gdelt_doc, fetch_rss

STORE_PATH = RAW_DIR / "news_store.jsonl"
SAMPLE_PATH = SAMPLE_DIR / "news_sample.jsonl"


def load_items(path: Path) -> list[NewsItem]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [NewsItem.model_validate_json(line) for line in lines if line.strip()]


def save_items(items: list[NewsItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")


def ingest(offline: bool = False, refresh: bool = False, use_gdelt: bool = True) -> list[NewsItem]:
    """Fetch all sources, merge with the existing store, dedupe, save, return the store."""
    rss_http = CachedHttp(offline=offline)
    gdelt_http = CachedHttp(offline=offline, min_interval=6.0)
    fresh = fetch_rss(rss_http, refresh=refresh)
    if use_gdelt:
        fresh += fetch_gdelt_doc(gdelt_http, refresh=refresh)
    store = dedupe(fresh + load_items(STORE_PATH))
    save_items(store, STORE_PATH)
    return store


def make_sample(items: list[NewsItem], per_source: int = 12, total: int = 72) -> list[NewsItem]:
    """Round-robin across sources so the sample is not dominated by one feed."""
    by_source: dict[str, list[NewsItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    picked: list[NewsItem] = []
    for rank in range(per_source):
        for source_items in by_source.values():
            if rank < len(source_items) and len(picked) < total:
                picked.append(source_items[rank])
    return dedupe(picked)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-gdelt", action="store_true")
    parser.add_argument("--write-sample", action="store_true")
    args = parser.parse_args()

    store = ingest(offline=args.offline, refresh=args.refresh, use_gdelt=not args.no_gdelt)
    counts: dict[str, int] = {}
    for item in store:
        counts[item.origin] = counts.get(item.origin, 0) + 1
    print(f"Store: {len(store)} items {counts} -> {STORE_PATH}")
    if args.write_sample:
        sample = make_sample(store)
        save_items(sample, SAMPLE_PATH)
        print(f"Sample: {len(sample)} items -> {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
