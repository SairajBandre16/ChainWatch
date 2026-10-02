import pytest
from fastapi.testclient import TestClient

from chainwatch.agent.store import EventStore
from chainwatch.api import main as api
from tests.test_agent_tools import make_event


@pytest.fixture
def client():
    store = EventStore(
        [
            make_event("rs1", "Red Sea", 15, severity=5, confidence=0.8),
            make_event("dub", "Dublin Port", 18, severity=2, event_type="labor_strike"),
        ]
    )
    api.app.dependency_overrides[api.get_store] = lambda: store
    api.app.dependency_overrides[api.get_forecaster] = lambda: None
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_events_endpoint_filters_by_as_of(client) -> None:
    body = client.get("/events", params={"as_of": "2024-01-20"}).json()
    assert {e["event_id"] for e in body["events"]} == {"rs1", "dub"}
    early = client.get("/events", params={"as_of": "2024-01-16"}).json()
    assert [e["event_id"] for e in early["events"]] == ["rs1"]
    assert client.get("/events", params={"as_of": "20-01-2024"}).status_code == 422


def test_risk_endpoint(client) -> None:
    body = client.get("/risk", params={"as_of": "2024-01-20", "focus": "india_ireland"}).json()
    scores = {r["lane_id"]: r["exposure_score"] for r in body["lanes"]}
    assert scores["INNSA-IEDUB"] > scores.get("INNSA-IEORK", 0) > 0
    one = client.get("/risk", params={"as_of": "2024-01-20", "lane_id": "INNSA-NLRTM"}).json()
    assert len(one["lanes"]) == 1
    assert client.get("/risk", params={"lane_id": "NOPE"}).status_code == 404


def test_brief_endpoint_deterministic(client) -> None:
    body = client.get("/brief", params={"as_of": "2024-01-20", "focus": "india_ireland"}).json()
    assert body["risk_level"] == "high"
    assert set(body["cited_event_ids"]) == {"rs1", "dub"}
    assert body["generated_by"] == "deterministic"


def test_brief_with_unavailable_llm_returns_503(client, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert client.get("/brief", params={"provider": "groq"}).status_code == 503


def test_lanes_and_backtest_endpoints(client) -> None:
    assert len(client.get("/lanes").json()["lanes"]) >= 20
    assert "runs" in client.get("/backtest").json()
