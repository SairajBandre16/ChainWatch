"""Extraction pipeline: NewsItem -> prompt -> LLM -> validated DisruptionEvents.

Failures (invalid JSON after retries, network errors) become `status="failed"` records instead of
crashing the run, so one bad article never loses a whole batch and failure rates can be measured.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from chainwatch.extraction.prompts import PromptTemplate, load_prompt
from chainwatch.extraction.schemas import (
    DisruptionEvent,
    EventType,
    ExtractionOutput,
    ExtractionRecord,
)
from chainwatch.ingest.models import NewsItem
from chainwatch.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "extract_v1"
TEXT_MAX_CHARS = 2000  # keep prompts short for small local models


def build_prompt(item: NewsItem, template: PromptTemplate) -> tuple[str, str]:
    return template.render(
        event_types=", ".join(f'"{t.value}"' for t in EventType),
        published=item.published.date().isoformat(),
        title=item.title,
        source=item.source,
        text=(item.summary or item.title)[:TEXT_MAX_CHARS],
    )


def _event_id(news_id: str, index: int, model: str, version: str) -> str:
    raw = f"{news_id}|{index}|{model}|{version}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def extract_item(
    item: NewsItem, llm: LLMClient, template: PromptTemplate, max_retries: int = 2
) -> ExtractionRecord:
    system, user = build_prompt(item, template)
    base = {"news_id": item.id, "model": llm.model, "prompt_version": template.version}
    try:
        output = llm.generate_structured(
            user, ExtractionOutput, system=system, max_retries=max_retries
        )
    except LLMError as exc:
        logger.warning("Extraction failed for %s: %s", item.id, exc)
        return ExtractionRecord(**base, status="failed", error=str(exc)[:500])

    events = []
    # A model sometimes says is_disruption=false but still lists events; trust the flag.
    if output.is_disruption:
        for idx, ev in enumerate(output.events):
            events.append(
                DisruptionEvent(
                    **ev.model_dump(),
                    event_id=_event_id(item.id, idx, llm.model, template.version),
                    news_id=item.id,
                    source_url=item.url,
                    source_name=item.source,
                    published=item.published,
                    model=llm.model,
                    prompt_version=template.version,
                )
            )
    return ExtractionRecord(**base, status="ok", is_disruption=output.is_disruption, events=events)


def run_extraction(
    items: Iterable[NewsItem], llm: LLMClient, prompt_name: str = DEFAULT_PROMPT
) -> list[ExtractionRecord]:
    template = load_prompt(prompt_name)
    return [extract_item(item, llm, template) for item in items]


def save_records(records: list[ExtractionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def load_records(path: Path) -> list[ExtractionRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ExtractionRecord.model_validate_json(line) for line in lines if line.strip()]


def summarize(records: list[ExtractionRecord]) -> dict[str, float]:
    n = len(records) or 1
    ok = [r for r in records if r.status == "ok"]
    return {
        "items": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "failure_rate": round((len(records) - len(ok)) / n, 3),
        "flagged_disruption": sum(r.is_disruption for r in ok),
        "events": sum(len(r.events) for r in ok),
    }
