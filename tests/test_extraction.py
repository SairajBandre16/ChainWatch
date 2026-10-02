import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from chainwatch.config import SAMPLE_DIR
from chainwatch.extraction.baseline import keyword_extract, keyword_responder
from chainwatch.extraction.pipeline import (
    build_prompt,
    extract_item,
    load_records,
    run_extraction,
    save_records,
    summarize,
)
from chainwatch.extraction.prompts import load_prompt
from chainwatch.extraction.schemas import EventType, ExtractedEvent
from chainwatch.ingest.models import NewsItem
from chainwatch.ingest.run import load_items
from chainwatch.llm import FakeLLMClient

ITEM = NewsItem(
    id="n1",
    title="Houthi missile attack forces ships to avoid Red Sea",
    summary="Carriers reroute via the Cape of Good Hope for weeks.",
    url="https://ex.com/a",
    source="Test",
    published=datetime(2024, 1, 12, tzinfo=UTC),
    origin="manual",
)

GOOD = {
    "is_disruption": True,
    "events": [
        {
            "event_type": "conflict_or_attack",
            "location": "Red Sea",
            "country_code": None,
            "port_code": None,
            "severity": 5,
            "start_date": "2024-01-12",
            "end_date": None,
            "industries": ["Retail", "automotive"],
            "summary": "Attacks force rerouting.",
            "confidence": 0.9,
        }
    ],
}


def test_prompt_renders_all_placeholders() -> None:
    system, user = build_prompt(ITEM, load_prompt("extract_v1"))
    assert "supply chain risk analyst" in system
    assert "Title: Houthi missile attack" in user
    assert "2024-01-12" in user and '"labor_strike"' in user
    assert "{title}" not in user and "{event_types}" not in user


def test_extract_item_attaches_provenance() -> None:
    llm = FakeLLMClient([json.dumps(GOOD)])
    record = extract_item(ITEM, llm, load_prompt("extract_v1"))
    assert record.status == "ok" and record.is_disruption
    event = record.events[0]
    assert event.event_type is EventType.CONFLICT_OR_ATTACK
    assert event.source_url == ITEM.url and event.news_id == "n1"
    assert event.prompt_version == "extract_v1"
    assert event.industries == ["automotive", "retail"]


def test_invalid_output_is_retried_then_recorded_as_failure() -> None:
    llm = FakeLLMClient(["not json", '{"is_disruption": true, "events": [{"severity": 9}]}'])
    record = extract_item(ITEM, llm, load_prompt("extract_v1"), max_retries=1)
    assert record.status == "failed" and record.error
    assert len(llm.calls) == 2


def test_retry_recovers_from_bad_first_answer() -> None:
    llm = FakeLLMClient(["{oops", json.dumps(GOOD)])
    record = extract_item(ITEM, llm, load_prompt("extract_v1"))
    assert record.status == "ok" and len(record.events) == 1


def test_not_disruption_drops_stray_events() -> None:
    llm = FakeLLMClient([json.dumps(GOOD | {"is_disruption": False})])
    record = extract_item(ITEM, llm, load_prompt("extract_v1"))
    assert record.events == [] and not record.is_disruption


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Strike", EventType.LABOR_STRIKE), ("port congestion", EventType.PORT_CONGESTION)],
)
def test_event_type_aliases(raw, expected) -> None:
    ev = ExtractedEvent(event_type=raw, location="x", severity=1, confidence=0.5)
    assert ev.event_type is expected


def test_schema_rejects_bad_codes_and_ranges() -> None:
    base = {"event_type": "other", "location": "x", "severity": 3, "confidence": 0.5}
    with pytest.raises(ValidationError):
        ExtractedEvent(**base | {"country_code": "IND"})
    with pytest.raises(ValidationError):
        ExtractedEvent(**base | {"port_code": "ROTTERDAM"})
    with pytest.raises(ValidationError):
        ExtractedEvent(**base | {"severity": 6})
    ok = ExtractedEvent(
        **base | {"country_code": "in", "port_code": "IN NSA", "start_date": "null"}
    )
    assert (ok.country_code, ok.port_code, ok.start_date) == ("IN", "INNSA", None)


def test_keyword_baseline() -> None:
    out = keyword_extract("Dockworkers strike shuts Port of Rotterdam for days")
    assert out["is_disruption"]
    assert out["events"][0]["event_type"] == "labor_strike"
    assert out["events"][0]["port_code"] == "NLRTM"
    assert keyword_extract("Quarterly earnings beat expectations")["is_disruption"] is False


def test_pipeline_end_to_end_on_sample_with_fake_llm(tmp_path) -> None:
    items = load_items(SAMPLE_DIR / "news_sample.jsonl")
    llm = FakeLLMClient(keyword_responder, model="keyword-baseline")
    records = run_extraction(items, llm)
    stats = summarize(records)
    assert stats["items"] == len(items) and stats["failed"] == 0
    path = tmp_path / "out.jsonl"
    save_records(records, path)
    assert load_records(path) == records
