import json
from datetime import UTC, datetime

import httpx
import pytest

from chainwatch.agent.store import EventStore
from chainwatch.agent.tools import TOOLS, ToolContext, call_tool
from chainwatch.agent.weather import DayWeather, classify, get_weather_risk
from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.forecast import data as fd
from chainwatch.forecast import train as ft
from chainwatch.ingest.http_cache import CachedHttp

AS_OF = datetime(2024, 1, 20, tzinfo=UTC)


def make_event(event_id: str, location: str, day: int, severity: int = 4, **kw) -> DisruptionEvent:
    return DisruptionEvent(
        event_type=kw.pop("event_type", "conflict_or_attack"),
        location=location,
        severity=severity,
        confidence=kw.pop("confidence", 1.0),
        summary=kw.pop("summary", f"Disruption at {location}"),
        event_id=event_id,
        news_id=f"n-{event_id}",
        source_url=f"https://ex.com/{event_id}",
        source_name="test",
        published=datetime(2024, 1, day, tzinfo=UTC),
        model="m",
        prompt_version="v",
        **kw,
    )


@pytest.fixture(scope="module")
def forecaster():
    split = fd.build_dataset(fd.synthetic_orders(2500, seed=7), "2016-07-01", "2017-01-01")
    return ft.train_all(split)


@pytest.fixture
def ctx(forecaster) -> ToolContext:
    store = EventStore(
        [
            make_event("rs1", "Red Sea", 15, severity=5),
            make_event("dub", "Dublin Port", 18, severity=2, event_type="labor_strike"),
            make_event("old", "Strait of Hormuz", 1),  # outside the 14-day lookback
            make_event("future", "Suez Canal", 25),  # after as_of: must stay invisible
            make_event("nowhere", "somewhere vague", 19),
        ]
    )
    return ToolContext(store=store, forecaster=forecaster, as_of=AS_OF, lookback_days=14)


def test_registry_has_schemas() -> None:
    for tool in TOOLS.values():
        schema = tool.schema()
        assert schema["name"] and schema["description"] and "properties" in schema["parameters"]


def test_search_events_respects_time_window(ctx) -> None:
    ids = {e["event_id"] for e in call_tool(ctx, "search_events", {})["events"]}
    assert ids == {"rs1", "dub", "nowhere"}
    hits = call_tool(ctx, "search_events", {"text": "red sea", "min_severity": 5})["events"]
    assert [e["event_id"] for e in hits] == ["rs1"]


def test_get_event_hides_future_and_unknown(ctx) -> None:
    assert call_tool(ctx, "get_event", {"event_id": "rs1"})["location"] == "Red Sea"
    assert "error" in call_tool(ctx, "get_event", {"event_id": "future"})
    assert "error" in call_tool(ctx, "get_event", {"event_id": "nope"})


def test_exposed_lanes_tool(ctx) -> None:
    out = call_tool(ctx, "exposed_lanes", {"event_id": "rs1"})
    lanes = {x["lane_id"] for x in out["lanes"]}
    assert out["linked_nodes"] == ["RED_SEA"]
    assert {"INNSA-NLRTM", "INMUN-IEDUB"} <= lanes
    vague = call_tool(ctx, "exposed_lanes", {"event_id": "nowhere"})
    assert vague["lanes"] == [] and "could not be linked" in vague["note"]


def test_lane_risk_uses_only_visible_events(ctx) -> None:
    out = call_tool(ctx, "lane_risk", {"lane_id": "INNSA-IEDUB"})
    contributing = {c["event_id"] for c in out["contributions"]}
    assert contributing == {"rs1", "dub"}  # not 'old' (too early) nor 'future'
    assert 0 < out["exposure_score"] <= 1
    assert "SUEZ_CANAL" in out["route_chokepoints"]


def test_alternate_routes_tool(ctx) -> None:
    out = call_tool(ctx, "alternate_routes", {"lane_id": "INNSA-NLRTM", "avoid": ["RED_SEA"]})
    assert "CAPE_OF_GOOD_HOPE" in out["alternatives"][0]["chokepoints"]
    assert out["alternatives"][0]["extra_days"] > 0
    assert "error" in call_tool(ctx, "alternate_routes", {"lane_id": "INNSA-NLRTM", "avoid": ["X"]})


def test_delay_risk_tool(ctx) -> None:
    out = call_tool(ctx, "delay_risk", {"lane_id": "INNSA-IEDUB", "shipping_mode": "First Class"})
    assert 0 < out["late_probability"] < 1
    assert out["explanation"].startswith("Late-delivery risk")
    no_model = ToolContext(store=ctx.store, as_of=AS_OF)
    assert "error" in call_tool(no_model, "delay_risk", {"lane_id": "INNSA-IEDUB"})


def test_bad_arguments_and_unknown_tool(ctx) -> None:
    assert "Invalid arguments" in call_tool(ctx, "lane_risk", {})["error"]
    assert "Unknown tool" in call_tool(ctx, "launch_rockets", {})["error"]
    assert "Unknown lane_id" in call_tool(ctx, "lane_risk", {"lane_id": "X-Y"})["error"]


def test_weather_tool_with_stubbed_http(ctx, tmp_path, monkeypatch) -> None:
    body = {"daily": {"time": ["2024-01-20", "2024-01-21"], "wind_speed_10m_max": [30.0, 70.0],
                      "wind_gusts_10m_max": [50.0, 95.0], "precipitation_sum": [1.0, 4.0]}}  # fmt: skip

    def fake_get(url, **kwargs):
        assert "_day" not in (kwargs.get("params") or {})
        return httpx.Response(200, content=json.dumps(body).encode(),
                              request=httpx.Request("GET", url))  # fmt: skip

    monkeypatch.setattr(httpx, "get", fake_get)
    ctx.http = CachedHttp(cache_dir=tmp_path)
    out = call_tool(ctx, "weather", {"node_id": "NLRTM"})
    assert out["level"] == "high" and "gale" in out["reason"]
    risk = get_weather_risk(51.9, 4.1, http=ctx.http)  # served from cache now
    assert len(risk.days) == 2


def test_weather_classification_thresholds() -> None:
    calm = [DayWeather(date="d", wind_max_kmh=20, gust_max_kmh=30, precipitation_mm=2)]
    windy = [DayWeather(date="d", wind_max_kmh=45, gust_max_kmh=60, precipitation_mm=2)]
    wet = [DayWeather(date="d", wind_max_kmh=10, gust_max_kmh=20, precipitation_mm=80)]
    assert classify(calm)[0] == "low"
    assert classify(windy)[0] == "elevated"
    assert classify(wet)[0] == "high"


def test_event_store_roundtrip(tmp_path) -> None:
    store = EventStore([make_event("a", "Red Sea", 2), make_event("b", "Cork", 3)])
    path = tmp_path / "s.jsonl"
    store.save(path)
    loaded = EventStore.load(path)
    assert len(loaded) == 2 and loaded.has("a") and not loaded.has("zzz")
    assert [e.event_id for e in loaded.all()] == ["b", "a"]  # newest first


def test_list_lanes_focus_all_means_no_filter(ctx) -> None:
    every = call_tool(ctx, "list_lanes", {})["lanes"]
    assert call_tool(ctx, "list_lanes", {"focus": "all"})["lanes"] == every
    assert len(call_tool(ctx, "list_lanes", {"focus": "india_ireland"})["lanes"]) < len(every)
