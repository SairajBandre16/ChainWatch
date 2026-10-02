"""The mitigation brief: schema, a deterministic builder, and the guard rails applied to any brief.

Two ways to produce a brief:
  - `build_brief_deterministic`: pure tool calls + rules. No LLM. Used offline, as the fallback when the
    LLM agent fails, and as the reference the rubric compares against.
  - the LLM agent (`agent.loop`), which must return the same `Brief` schema.

Guard rails (`ground_brief`) run on every brief, whoever wrote it:
  - citations: any event id not in the store (or published after `as_of`) is removed and reported;
  - numbers: each affected lane's exposure score is recomputed from the graph and overwritten,
    and lanes with zero computed exposure are removed;
  - coverage: lanes the graph finds exposed but the writer left out are added (and their events cited);
  - feasibility: a reroute claim is removed if no sea route avoids the lane's disrupted nodes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from chainwatch.agent.tools import ToolContext, call_tool
from chainwatch.graph.build import haversine_nm

RiskLevel = Literal["none", "low", "medium", "high"]


def risk_level(score: float) -> RiskLevel:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    if score > 0:
        return "low"
    return "none"


class AffectedLane(BaseModel):
    lane_id: str
    exposure_score: float = Field(ge=0, le=1)
    why: str = ""


class Mitigation(BaseModel):
    lane_id: str
    action: str
    detail: str = ""


class Brief(BaseModel):
    as_of: str
    focus: str | None = None
    summary: str
    risk_score: float = Field(ge=0, le=1, description="Highest exposure among affected lanes")
    risk_level: RiskLevel
    affected_lanes: list[AffectedLane] = Field(default_factory=list)
    drivers: list[str] = Field(default_factory=list)
    mitigations: list[Mitigation] = Field(default_factory=list)
    cited_event_ids: list[str] = Field(default_factory=list)
    baseline_delay_risk: dict[str, float] = Field(
        default_factory=dict, description="lane_id -> ML late-delivery probability"
    )
    generated_by: str = "deterministic"
    guardrail_notes: list[str] = Field(default_factory=list)


def _nearest_alternate_port(ctx: ToolContext, port: str) -> str | None:
    """Closest other port for 'divert to another port' mitigations: same country first (no new
    customs border, e.g. Dublin -> Cork rather than Liverpool), else same region."""
    ports = ctx.graph.ref.ports
    me = ports[port]
    others = [p for p in ports.values() if p.country_code == me.country_code and p.locode != port]
    others = others or [p for p in ports.values() if p.region == me.region and p.locode != port]
    if not others:
        return None
    best = min(others, key=lambda p: haversine_nm(me.lat, me.lon, p.lat, p.lon))
    return best.locode


def _mitigations_for(ctx: ToolContext, lane_id: str, contributions: list[dict]) -> list[Mitigation]:
    lane = ctx.graph.ref.lanes[lane_id]
    out: list[Mitigation] = []
    avoid = sorted({c["via"] for c in contributions if not c["via"].startswith("CTRY:")}
                   - {lane.origin, lane.destination})  # fmt: skip
    if avoid:
        alt = call_tool(ctx, "alternate_routes", {"lane_id": lane_id, "avoid": avoid})
        if alt.get("alternatives"):
            best = alt["alternatives"][0]
            via = ", ".join(best["chokepoints"]) or "open sea"
            out.append(Mitigation(
                lane_id=lane_id, action="reroute",
                detail=f"Avoid {', '.join(avoid)}: route via {via}, about "
                       f"{best['extra_days']:+.1f} days ({best['extra_nm']:+,.0f} nm) vs base.",
            ))  # fmt: skip
            if best["extra_days"] >= 3:
                out.append(Mitigation(
                    lane_id=lane_id, action="buffer_stock",
                    detail=f"Hold about {round(best['extra_days'])} extra days of safety stock at "
                           f"{lane.destination} while the detour applies.",
                ))  # fmt: skip
        else:
            out.append(Mitigation(lane_id=lane_id, action="hold_or_air",
                                  detail=f"No sea route avoids {', '.join(avoid)}; consider "
                                         "delaying bookings or moving urgent cargo by air."))  # fmt: skip
    for end in {lane.origin, lane.destination} & {c["via"] for c in contributions}:
        alt_port = _nearest_alternate_port(ctx, end)
        if alt_port:
            out.append(Mitigation(lane_id=lane_id, action="divert_port",
                                  detail=f"Disruption at {end}: consider {alt_port} "
                                         "(nearest alternative port)."))  # fmt: skip
    return out


def build_brief_deterministic(
    ctx: ToolContext, focus: str | None = None, with_delay_risk: bool = True
) -> Brief:
    lanes = call_tool(ctx, "list_lanes", {"focus": focus})["lanes"]
    affected: list[AffectedLane] = []
    drivers: list[str] = []
    mitigations: list[Mitigation] = []
    cited: list[str] = []
    for lane in lanes:
        risk = call_tool(ctx, "lane_risk", {"lane_id": lane["lane_id"]})
        if risk["exposure_score"] <= 0:
            continue
        contributions = risk["contributions"]
        top = contributions[0]
        affected.append(AffectedLane(lane_id=lane["lane_id"], exposure_score=risk["exposure_score"],
                                     why=top["reason"]))  # fmt: skip
        mitigations.extend(_mitigations_for(ctx, lane["lane_id"], contributions))
        for c in contributions:
            if c["event_id"] not in cited:
                cited.append(c["event_id"])
    for event_id in cited:
        ev = ctx.store.get(event_id)
        drivers.append(f"{ev.event_type.value.replace('_', ' ')} at {ev.location} "
                       f"(severity {ev.severity}, confidence {ev.confidence:.1f}, "
                       f"{ev.published.date()}) [{event_id}]")  # fmt: skip
    affected.sort(key=lambda a: -a.exposure_score)
    score = affected[0].exposure_score if affected else 0.0
    delay = {}
    if with_delay_risk and ctx.forecaster is not None:
        for a in affected[:5]:
            out = call_tool(ctx, "delay_risk", {"lane_id": a.lane_id})
            if "late_probability" in out:
                delay[a.lane_id] = out["late_probability"]
    if affected:
        summary = (
            f"{len(affected)} of {len(lanes)} lanes exposed as of {ctx.as_of.date()}; "
            f"highest exposure {score:.2f} on {affected[0].lane_id} ({affected[0].why})."
        )
    else:
        summary = f"No exposure on {len(lanes)} lanes from events visible as of {ctx.as_of.date()}."  # fmt: skip
    return Brief(as_of=ctx.as_of.date().isoformat(), focus=focus, summary=summary,
                 risk_score=score, risk_level=risk_level(score), affected_lanes=affected,
                 drivers=drivers, mitigations=mitigations, cited_event_ids=cited,
                 baseline_delay_risk=delay)  # fmt: skip


REROUTE_ACTIONS = {"reroute", "alternate_routes", "alternate_route", "divert", "reroute_via_cape"}


def _reroute_is_possible(ctx: ToolContext, m: Mitigation, notes: list[str]) -> bool:
    """A reroute claim survives only if the graph has a route avoiding the lane's event nodes."""
    if m.action.lower() not in REROUTE_ACTIONS:
        return True
    lane = ctx.graph.ref.lanes[m.lane_id]
    risk = call_tool(ctx, "lane_risk", {"lane_id": m.lane_id})
    avoid = sorted({c["via"] for c in risk["contributions"] if not c["via"].startswith("CTRY:")}
                   - {lane.origin, lane.destination})  # fmt: skip
    if not avoid:
        return True
    alt = call_tool(ctx, "alternate_routes", {"lane_id": m.lane_id, "avoid": avoid})
    if alt.get("alternatives"):
        return True
    notes.append(f"removed reroute for {m.lane_id}: no sea route avoids {', '.join(avoid)}")
    return False


def ground_brief(brief: Brief, ctx: ToolContext) -> Brief:
    """Apply guard rails in place of trusting the writer: real citations, recomputed scores."""
    notes = list(brief.guardrail_notes)
    valid: list[str] = []
    for event_id in brief.cited_event_ids:
        ev = ctx.store.get(event_id)
        if ev is None or ev.published > ctx.as_of:
            notes.append(f"removed citation {event_id!r}: not in event store as of {brief.as_of}")
        elif event_id not in valid:
            valid.append(event_id)
    lanes: list[AffectedLane] = []
    seen: set[str] = set()
    for lane in brief.affected_lanes:
        if lane.lane_id in seen:
            notes.append(f"removed duplicate lane {lane.lane_id}")
            continue
        seen.add(lane.lane_id)
        risk = call_tool(ctx, "lane_risk", {"lane_id": lane.lane_id})
        if "error" in risk:
            notes.append(f"removed lane {lane.lane_id!r}: unknown lane")
            continue
        if risk["exposure_score"] <= 0:
            notes.append(f"removed lane {lane.lane_id}: no visible event touches it")
            continue
        if abs(risk["exposure_score"] - lane.exposure_score) > 1e-6:
            notes.append(f"{lane.lane_id}: exposure {lane.exposure_score} replaced by computed "
                         f"{risk['exposure_score']}")  # fmt: skip
        lanes.append(lane.model_copy(update={"exposure_score": risk["exposure_score"]}))
    # Coverage: every lane the graph says is exposed must appear, whatever the writer concluded.
    listed = {lane.lane_id for lane in lanes}
    added_mitigations: list[Mitigation] = []
    focus_lanes = call_tool(ctx, "list_lanes", {"focus": brief.focus})["lanes"]
    for lane in focus_lanes:
        if lane["lane_id"] in listed:
            continue
        risk = call_tool(ctx, "lane_risk", {"lane_id": lane["lane_id"]})
        if risk["exposure_score"] > 0:
            top = risk["contributions"][0]
            lanes.append(AffectedLane(lane_id=lane["lane_id"], exposure_score=risk["exposure_score"],
                                      why=top["reason"]))  # fmt: skip
            notes.append(
                f"added missing lane {lane['lane_id']} (exposure {risk['exposure_score']})"
            )
            added_mitigations.extend(_mitigations_for(ctx, lane["lane_id"], risk["contributions"]))
            for c in risk["contributions"]:
                if c["event_id"] not in valid:
                    valid.append(c["event_id"])
    known_lanes = {lane.lane_id for lane in lanes}
    mitigations = [m for m in brief.mitigations if m.lane_id in known_lanes]
    if len(mitigations) < len(brief.mitigations):
        notes.append(
            f"removed {len(brief.mitigations) - len(mitigations)} mitigations for unknown lanes"
        )
    mitigations = [m for m in mitigations if _reroute_is_possible(ctx, m, notes)]
    mitigations += added_mitigations
    # Every affected lane gets at least one feasible mitigation, computed from the graph if needed.
    covered = {m.lane_id for m in mitigations}
    for lane in lanes:
        if lane.lane_id not in covered:
            risk = call_tool(ctx, "lane_risk", {"lane_id": lane.lane_id})
            mitigations += _mitigations_for(ctx, lane.lane_id, risk["contributions"])
            notes.append(f"added computed mitigations for {lane.lane_id}")
    lanes.sort(key=lambda a: -a.exposure_score)
    score = lanes[0].exposure_score if lanes else 0.0
    return brief.model_copy(update={
        "cited_event_ids": valid, "affected_lanes": lanes, "mitigations": mitigations,
        "risk_score": score, "risk_level": risk_level(score), "guardrail_notes": notes,
    })  # fmt: skip


def brief_to_markdown(brief: Brief) -> str:
    lines = [f"# Disruption brief ({brief.as_of})", "",
             f"**Risk:** {brief.risk_level} ({brief.risk_score:.2f}). Generated by: {brief.generated_by}.",
             "", brief.summary, "", "## Affected lanes"]  # fmt: skip
    lines += [
        f"- {a.lane_id}: exposure {a.exposure_score:.2f} ({a.why})" for a in brief.affected_lanes
    ]
    lines += ["", "## Drivers"] + [f"- {d}" for d in brief.drivers]
    lines += ["", "## Mitigations"]
    lines += [f"- [{m.lane_id}] {m.action}: {m.detail}" for m in brief.mitigations]
    if brief.baseline_delay_risk:
        lines += ["", "## Baseline late-delivery risk (ML, DataCo history)"]
        lines += [f"- {k}: {v:.0%}" for k, v in brief.baseline_delay_risk.items()]
    lines += ["", "## Cited events"] + [f"- {e}" for e in brief.cited_event_ids]
    if brief.guardrail_notes:
        lines += ["", "## Guard-rail notes"] + [f"- {n}" for n in brief.guardrail_notes]
    return "\n".join(lines) + "\n"
