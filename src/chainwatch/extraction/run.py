"""CLI: run extraction over a NewsItem JSONL file.

python -m chainwatch.extraction.run                                   # fake LLM, sample data
python -m chainwatch.extraction.run --provider ollama --model qwen2.5:3b
python -m chainwatch.extraction.run --provider ollama --limit 20 --prompt extract_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from chainwatch.config import PROCESSED_DIR, SAMPLE_DIR
from chainwatch.extraction.baseline import keyword_responder
from chainwatch.extraction.pipeline import (
    DEFAULT_PROMPT,
    run_extraction,
    save_records,
    summarize,
)
from chainwatch.ingest.run import load_items
from chainwatch.llm import FakeLLMClient, LLMClient, get_llm_client


def make_client(provider: str, model: str | None) -> LLMClient:
    if provider == "fake":
        # The fake client answers with the keyword baseline, in the same shape as a real model.
        return FakeLLMClient(keyword_responder, model="keyword-baseline")
    return get_llm_client(provider=provider, model=model)


def output_path(model: str, prompt: str) -> Path:
    safe = model.replace(":", "-").replace("/", "-")
    return PROCESSED_DIR / "extractions" / f"{safe}__{prompt}.jsonl"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE_DIR / "news_sample.jsonl")
    parser.add_argument("--provider", default="fake")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.input)[: args.limit]
    llm = make_client(args.provider, args.model)
    start = time.perf_counter()
    records = run_extraction(items, llm, prompt_name=args.prompt)
    elapsed = time.perf_counter() - start
    out = output_path(llm.model, args.prompt)
    save_records(records, out)
    stats = summarize(records) | {"seconds": round(elapsed, 1), "model": llm.model}
    print(json.dumps(stats, indent=1))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
