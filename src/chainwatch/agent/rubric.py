"""Rubric for agent briefs: did it find the right lanes, and did it cite only real, relevant events?

Ground truth comes from the graph, not from the LLM: the expected lanes are the focus lanes with
exposure > 0 for the events visible at `as_of`. Scored on the LLM's raw draft (before guard rails)
so the numbers show what the model did on its own; the final brief passes guard rails by construction.

Metrics:
  lane_precision / lane_recall   raw draft's affected lanes vs expected lanes
  citation_validity              share of cited ids that exist in the store and are visible at as_of
  citation_relevance             share of valid cited ids that actually drive an expected lane
  reroute_grounded               every 'reroute' mitigation came after an alternate_routes call
  completed                      the model produced a valid brief without the fallback
"""

from __future__ import annotations

from pydantic import BaseModel

from chainwatch.agent.brief import REROUTE_ACTIONS
from chainwatch.agent.loop import AgentResult
from chainwatch.agent.tools import ToolContext, call_tool


class RubricScore(BaseModel):
    expected_lanes: list[str]
    proposed_lanes: list[str]
    lane_precision: float
    lane_recall: float
    citation_validity: float
    citation_relevance: float
    reroute_grounded: bool
    completed: bool


def expected_lanes(ctx: ToolContext, focus: str | None) -> tuple[set[str], set[str]]:
    """(lanes with exposure > 0, event ids that contribute to them)."""
    lanes, drivers = set(), set()
    for lane in call_tool(ctx, "list_lanes", {"focus": focus})["lanes"]:
        risk = call_tool(ctx, "lane_risk", {"lane_id": lane["lane_id"]})
        if risk["exposure_score"] > 0:
            lanes.add(lane["lane_id"])
            drivers |= {c["event_id"] for c in risk["contributions"]}
    return lanes, drivers


def score_result(result: AgentResult, ctx: ToolContext, focus: str | None) -> RubricScore:
    expected, drivers = expected_lanes(ctx, focus)
    draft = result.raw_brief
    proposed = {a.lane_id for a in draft.affected_lanes} if draft else set()
    cited = list(dict.fromkeys(draft.cited_event_ids)) if draft else []
    valid = [e for e in cited if ctx.store.has(e) and ctx.store.get(e).published <= ctx.as_of]
    hit = len(proposed & expected)
    called_alt = any(s.tool == "alternate_routes" for s in result.steps)
    reroutes = (
        [m for m in draft.mitigations if m.action.lower() in REROUTE_ACTIONS] if draft else []
    )

    def ratio(a: int, b: int, empty: float) -> float:
        return round(a / b, 3) if b else empty

    return RubricScore(
        expected_lanes=sorted(expected),
        proposed_lanes=sorted(proposed),
        lane_precision=ratio(hit, len(proposed), 1.0 if not expected else 0.0),
        lane_recall=ratio(hit, len(expected), 1.0),
        citation_validity=ratio(len(valid), len(cited), 1.0),
        citation_relevance=ratio(len([e for e in valid if e in drivers]), len(valid), 1.0),
        reroute_grounded=not reroutes or called_alt,
        completed=not result.used_fallback,
    )
