import json
from datetime import UTC, datetime

import pytest

from chainwatch.agent.brief import brief_to_markdown, build_brief_deterministic, ground_brief
from chainwatch.agent.loop import run_agent
from chainwatch.agent.store import EventStore
from chainwatch.agent.tools import ToolContext
from chainwatch.llm import FakeLLMClient
from tests.test_agent_tools import make_event

AS_OF = datetime(2024, 1, 20, tzinfo=UTC)


@pytest.fixture
def ctx() -> ToolContext:
    store = EventStore(
        [
            make_event("rs1", "Red Sea", 15, severity=5, confidence=0.9),
            make_event("dub", "Dublin Port", 18, severity=2, event_type="labor_strike"),
            make_event("future", "Suez Canal", 25),
        ]
    )
    return ToolContext(store=store, as_of=AS_OF, lookback_days=14)


def test_deterministic_brief_finds_red_sea_lanes_and_cape(ctx) -> None:
    brief = ground_brief(build_brief_deterministic(ctx, focus="india_ireland"), ctx)
    lanes = {a.lane_id for a in brief.affected_lanes}
    assert lanes == {"INNSA-IEDUB", "INNSA-IEORK", "INMUN-IEDUB", "INMAA-IEDUB", "INMAA-IEORK"}
    assert set(brief.cited_event_ids) == {"rs1", "dub"}
    reroutes = [m for m in brief.mitigations if m.action == "reroute"]
    assert reroutes and all("CAPE_OF_GOOD_HOPE" in m.detail for m in reroutes)
    assert any(m.action == "divert_port" and "IEORK" in m.detail for m in brief.mitigations)
    assert brief.risk_level == "high" and brief.guardrail_notes == []
    assert "# Disruption brief" in brief_to_markdown(brief)


def test_llm_agent_with_scripted_fake_llm(ctx) -> None:
    script = [
        {"action": "call_tool", "tool": "search_events", "args": {"min_severity": 3}},
        {"action": "call_tool", "tool": "lane_risk", "args": {"lane_id": "INNSA-IEDUB"}},
        {"action": "call_tool", "tool": "alternate_routes",
         "args": {"lane_id": "INNSA-IEDUB", "avoid": ["RED_SEA"]}},
        {"action": "final", "brief": {
            "summary": "Red Sea attacks threaten India to Ireland lanes.",
            "affected_lanes": [{"lane_id": "INNSA-IEDUB", "exposure_score": 0.5, "why": "Red Sea"},
                               {"lane_id": "MADE-UP", "exposure_score": 0.9}],
            "drivers": ["attack in Red Sea [rs1]"],
            "mitigations": [{"lane_id": "INNSA-IEDUB", "action": "reroute", "detail": "Cape"},
                            {"lane_id": "MADE-UP", "action": "reroute"}],
            "cited_event_ids": ["rs1", "fake-123", "future"],
        }},
    ]  # fmt: skip
    llm = FakeLLMClient([json.dumps(s) for s in script], model="scripted")
    result = run_agent(llm, ctx, focus="india_ireland")
    brief = result.brief
    assert not result.used_fallback
    assert [s.tool for s in result.steps] == ["search_events", "lane_risk", "alternate_routes"]
    # Guard rails: only real, visible events survive; made-up lane removed; score recomputed;
    # exposed lanes the LLM skipped are added with their own mitigations.
    assert brief.cited_event_ids[0] == "rs1"
    assert "fake-123" not in brief.cited_event_ids and "future" not in brief.cited_event_ids
    lanes = {a.lane_id: a for a in brief.affected_lanes}
    assert "MADE-UP" not in lanes and lanes["INNSA-IEDUB"].exposure_score != 0.5
    assert {"INNSA-IEORK", "INMAA-IEDUB"} <= set(lanes)
    assert all(m.lane_id in lanes for m in brief.mitigations)
    assert [m.detail for m in brief.mitigations if m.lane_id == "INNSA-IEDUB"] == ["Cape"]
    notes = " ".join(brief.guardrail_notes)
    assert "fake-123" in notes and "future" in notes and "replaced by computed" in notes
    assert brief.generated_by == "llm:scripted"
    # The transcript of earlier tool results is shown to the model on later turns.
    assert "INNSA-IEDUB" in llm.calls[-1]["prompt"]


def test_agent_falls_back_when_llm_never_finishes(ctx) -> None:
    llm = FakeLLMClient(["not json", '{"action": "dance"}', '{"action": "final", "brief": {}}'])
    result = run_agent(llm, ctx, focus="india_ireland", max_steps=3)
    assert result.used_fallback
    assert result.brief.generated_by.startswith("fallback")
    assert result.brief.affected_lanes  # deterministic brief still delivered
    assert all(ctx.store.has(e) for e in result.brief.cited_event_ids)


def test_tool_call_budget_is_enforced(ctx) -> None:
    loop_forever = json.dumps({"action": "call_tool", "tool": "list_lanes", "args": {}})
    result = run_agent(FakeLLMClient([loop_forever]), ctx, max_steps=2)
    assert sum(s.action == "call_tool" for s in result.steps) == 2
    assert result.used_fallback


def test_guardrail_adds_lanes_the_llm_missed(ctx) -> None:
    lazy = {"action": "final", "brief": {"summary": "No disruptions found.", "cited_event_ids": []}}
    result = run_agent(FakeLLMClient([json.dumps(lazy)]), ctx, focus="india_ireland")
    lanes = {a.lane_id for a in result.brief.affected_lanes}
    assert {"INNSA-IEDUB", "INMAA-IEORK"} <= lanes
    assert result.brief.risk_level == "high"
    assert "rs1" in result.brief.cited_event_ids
    assert any(n.startswith("added missing lane") for n in result.brief.guardrail_notes)


def test_search_matches_any_keyword(ctx) -> None:
    assert [e.event_id for e in ctx.store.search("sea trade disruption red")] == ["rs1"]
    assert {e.event_id for e in ctx.store.search("sea trade disruption")} == {
        "future",
        "dub",
        "rs1",
    }


def test_tool_name_as_action_is_accepted(ctx) -> None:
    replies = [
        {"action": "lane_risk", "args": {"lane_id": "INNSA-IEDUB"}},
        {"action": "lane_risk", "lane_id": "INNSA-IEORK"},  # args inline, no "args" key
        {"action": "final", "brief": {"summary": "s", "cited_event_ids": ["rs1"]}},
    ]
    result = run_agent(FakeLLMClient([json.dumps(r) for r in replies]), ctx, focus="india_ireland")
    assert [s.tool for s in result.steps] == ["lane_risk", "lane_risk"]
    assert result.steps[1].observation["lane_id"] == "INNSA-IEORK"
    assert not result.used_fallback


def test_rubric_scores_raw_draft(ctx) -> None:
    from chainwatch.agent.rubric import score_result

    draft = {"action": "final", "brief": {
        "summary": "s",
        "affected_lanes": [{"lane_id": "INNSA-IEDUB", "exposure_score": 0.9},
                           {"lane_id": "AEJEA-INNSA", "exposure_score": 0.2}],
        "mitigations": [{"lane_id": "INNSA-IEDUB", "action": "reroute"}],
        "cited_event_ids": ["rs1", "ghost"],
    }}  # fmt: skip
    result = run_agent(FakeLLMClient([json.dumps(draft)]), ctx, focus=None)
    score = score_result(result, ctx, focus=None)
    assert score.lane_precision == 0.5  # AEJEA-INNSA is not exposed
    assert 0 < score.lane_recall < 1  # many Red Sea lanes were missed
    assert score.citation_validity == 0.5 and score.citation_relevance == 1.0
    assert score.reroute_grounded is False  # reroute suggested without checking alternate_routes
    assert score.completed


def test_guardrail_drops_unexposed_lanes(ctx) -> None:
    draft = {"action": "final", "brief": {
        "summary": "s", "affected_lanes": [{"lane_id": "AEJEA-INNSA", "exposure_score": 0.7},
                                           {"lane_id": "INNSA-IEDUB", "exposure_score": 0.7},
                                           {"lane_id": "INNSA-IEDUB", "exposure_score": 0.7}],
        "mitigations": [{"lane_id": "AEJEA-INNSA", "action": "monitor"}]}}  # fmt: skip
    result = run_agent(FakeLLMClient([json.dumps(draft)]), ctx, focus=None)
    assert "AEJEA-INNSA" not in {a.lane_id for a in result.brief.affected_lanes}
    assert all(m.lane_id != "AEJEA-INNSA" for m in result.brief.mitigations)
    assert any("no visible event" in n for n in result.brief.guardrail_notes)
    assert [a.lane_id for a in result.brief.affected_lanes].count("INNSA-IEDUB") == 1


def test_guardrail_removes_impossible_reroute() -> None:
    store = EventStore([make_event("h", "Strait of Hormuz", 15, severity=5)])
    hctx = ToolContext(store=store, as_of=AS_OF, lookback_days=14)
    draft = {"action": "final", "brief": {
        "summary": "s", "affected_lanes": [{"lane_id": "AEJEA-NLRTM", "exposure_score": 1.0}],
        "mitigations": [{"lane_id": "AEJEA-NLRTM", "action": "alternate_routes",
                         "detail": "Alternate route takes 2 extra days"}]}}  # fmt: skip
    result = run_agent(FakeLLMClient([json.dumps(draft)]), hctx, focus=None)
    assert all("2 extra days" not in m.detail for m in result.brief.mitigations)
    assert any("no sea route avoids HORMUZ" in n for n in result.brief.guardrail_notes)
    assert {(m.lane_id, m.action) for m in result.brief.mitigations} == {
        ("AEJEA-NLRTM", "hold_or_air"),
        ("AEJEA-INNSA", "hold_or_air"),
    }
