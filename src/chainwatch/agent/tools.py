"""Agent tools: small, typed functions the agent can call by name with JSON arguments.

Every tool:
  - validates its arguments with a Pydantic model (bad arguments -> a clear error, not a crash),
  - returns plain JSON-serializable data,
  - only sees events published at or before `ctx.as_of`, so a backtest replay cannot peek ahead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from chainwatch.agent.store import EventStore
from chainwatch.agent.weather import get_weather_risk
from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.forecast.predict import lane_delay_risk
from chainwatch.forecast.train import TrainedModels
from chainwatch.graph.build import TradeGraph
from chainwatch.graph.linking import link_event, to_graph_event
from chainwatch.ingest.http_cache import CachedHttp


@dataclass
class ToolContext:
    store: EventStore
    graph: TradeGraph = field(default_factory=TradeGraph)
    forecaster: TrainedModels | None = None
    http: CachedHttp | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    lookback_days: int = 30
    _linked: bool = False

    def visible_events(self) -> list[DisruptionEvent]:
        """Events inside the lookback window ending at `as_of` (never after it)."""
        since = self.as_of - timedelta(days=self.lookback_days)
        return self.store.search(since=since, until=self.as_of, limit=10_000)

    def ensure_linked(self) -> None:
        """Add visible events to the graph once, so exposure scores reflect them."""
        if self._linked:
            return
        for ev in self.visible_events():
            gev = to_graph_event(ev, link_event(ev, self.graph.ref))
            if gev is not None:
                self.graph.add_event(gev)
        self._linked = True


def _event_card(ev: DisruptionEvent) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "event_type": ev.event_type.value,
        "location": ev.location,
        "severity": ev.severity,
        "confidence": ev.confidence,
        "published": ev.published.date().isoformat(),
        "summary": ev.summary,
        "source_url": ev.source_url,
    }


# --- argument models --------------------------------------------------------------------------


class SearchEventsArgs(BaseModel):
    text: str | None = Field(default=None, description="Words to match in location/summary/type")
    min_severity: int = Field(default=1, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=50)


class EventIdArgs(BaseModel):
    event_id: str


class LaneIdArgs(BaseModel):
    lane_id: str


class AlternateRoutesArgs(BaseModel):
    lane_id: str
    avoid: list[str] = Field(description="Node ids to avoid, e.g. ['RED_SEA']")


class DelayRiskArgs(BaseModel):
    lane_id: str
    shipping_mode: str = "Standard Class"


class WeatherArgs(BaseModel):
    node_id: str = Field(description="Port LOCODE or waypoint id, e.g. 'NLRTM' or 'BAB_EL_MANDEB'")


class ListLanesArgs(BaseModel):
    focus: str | None = Field(default=None, description="e.g. 'india_ireland'")


# --- tool functions ---------------------------------------------------------------------------


def search_events(ctx: ToolContext, args: SearchEventsArgs) -> dict:
    since = ctx.as_of - timedelta(days=ctx.lookback_days)
    hits = ctx.store.search(args.text, since=since, until=ctx.as_of,
                            min_severity=args.min_severity, limit=args.limit)  # fmt: skip
    return {"as_of": ctx.as_of.date().isoformat(), "events": [_event_card(e) for e in hits]}


def get_event(ctx: ToolContext, args: EventIdArgs) -> dict:
    ev = ctx.store.get(args.event_id)
    if ev is None or ev.published > ctx.as_of:
        return {"error": f"Unknown event_id {args.event_id}"}
    return _event_card(ev)


def exposed_lanes(ctx: ToolContext, args: EventIdArgs) -> dict:
    ev = ctx.store.get(args.event_id)
    if ev is None or ev.published > ctx.as_of:
        return {"error": f"Unknown event_id {args.event_id}"}
    link = link_event(ev, ctx.graph.ref)
    gev = to_graph_event(ev, link)
    if gev is None:
        return {"event_id": ev.event_id, "linked_nodes": [], "lanes": [],
                "note": f"Location {ev.location!r} could not be linked to the graph"}  # fmt: skip
    lanes = ctx.graph.exposed_lanes(gev)
    return {
        "event_id": ev.event_id,
        "linked_nodes": link.node_ids,
        "link_method": link.method,
        "lanes": [x.model_dump() for x in sorted(lanes, key=lambda x: -x.weight)],
    }


def lane_risk(ctx: ToolContext, args: LaneIdArgs) -> dict:
    if args.lane_id not in ctx.graph.ref.lanes:
        return {"error": f"Unknown lane_id {args.lane_id}"}
    ctx.ensure_linked()
    score, hits = ctx.graph.lane_exposure_score(args.lane_id)
    route = ctx.graph.lane_route(args.lane_id)
    return {
        "lane_id": args.lane_id,
        "exposure_score": score,
        "route_chokepoints": route.chokepoints,
        "route_days": route.days,
        "contributions": [h.model_dump() for h in hits],
    }


def alternate_routes(ctx: ToolContext, args: AlternateRoutesArgs) -> dict:
    if args.lane_id not in ctx.graph.ref.lanes:
        return {"error": f"Unknown lane_id {args.lane_id}"}
    unknown = [n for n in args.avoid if n not in ctx.graph.g]
    if unknown:
        return {"error": f"Unknown node ids {unknown}"}
    routes = ctx.graph.alternate_routes(args.lane_id, set(args.avoid))
    base = ctx.graph.lane_route(args.lane_id)
    return {
        "lane_id": args.lane_id,
        "base": {
            "distance_nm": base.distance_nm,
            "days": base.days,
            "chokepoints": base.chokepoints,
        },
        "alternatives": [r.model_dump(exclude={"path"}) | {"via": r.path[1:-1]} for r in routes],
    }


def delay_risk(ctx: ToolContext, args: DelayRiskArgs) -> dict:
    if ctx.forecaster is None:
        return {"error": "Forecaster not loaded (run `python -m chainwatch.forecast.train`)"}
    lane = ctx.graph.ref.lanes.get(args.lane_id)
    if lane is None:
        return {"error": f"Unknown lane_id {args.lane_id}"}
    dest_iso = ctx.graph.ref.ports[lane.destination].country_code
    risk = lane_delay_risk(ctx.forecaster, dest_iso, args.shipping_mode)
    return {
        "lane_id": args.lane_id,
        "late_probability": risk.probability,
        "explanation": risk.explanation.text,
        "note": risk.note,
    }


def weather(ctx: ToolContext, args: WeatherArgs) -> dict:
    node = ctx.graph.g.nodes.get(args.node_id)
    if node is None or "lat" not in node:
        return {"error": f"Unknown node_id {args.node_id}"}
    try:
        risk = get_weather_risk(node["lat"], node["lon"], http=ctx.http)
    except Exception as exc:  # noqa: BLE001 - weather is optional context, never fatal
        return {"error": f"Weather unavailable: {exc}"}
    return {"node_id": args.node_id, "level": risk.level, "reason": risk.reason}


def list_lanes(ctx: ToolContext, args: ListLanesArgs) -> dict:
    focus = None if (args.focus or "").lower() in {"", "all", "any", "none"} else args.focus
    lanes = [
        {"lane_id": lane.lane_id, "origin": lane.origin, "destination": lane.destination,
         "focus": lane.focus}
        for lane in ctx.graph.ref.lanes.values()
        if focus is None or lane.focus == focus
    ]  # fmt: skip
    return {"lanes": lanes}


# --- registry ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], dict]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.args_model.model_json_schema()}  # fmt: skip

    def __call__(self, ctx: ToolContext, raw_args: dict | None) -> dict:
        try:
            args = self.args_model.model_validate(raw_args or {})
        except ValidationError as exc:
            return {"error": f"Invalid arguments for {self.name}: {exc.errors()[:3]}"}
        return self.fn(ctx, args)


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("search_events", "Find recent disruption events by text and minimum severity.",
             SearchEventsArgs, search_events),
        Tool("get_event", "Get one event by event_id.", EventIdArgs, get_event),
        Tool("exposed_lanes", "List trade lanes exposed to an event, with weights.",
             EventIdArgs, exposed_lanes),
        Tool("lane_risk", "Disruption exposure score (0-1) for a lane and the events behind it.",
             LaneIdArgs, lane_risk),
        Tool("alternate_routes", "Routes for a lane that avoid given nodes, with extra days.",
             AlternateRoutesArgs, alternate_routes),
        Tool("delay_risk", "Baseline late-delivery probability for a lane's destination (ML model).",
             DelayRiskArgs, delay_risk),
        Tool("weather", "7-day weather risk at a port or waypoint (Open-Meteo).",
             WeatherArgs, weather),
        Tool("list_lanes", "List known trade lanes, optionally filtered by focus.",
             ListLanesArgs, list_lanes),
    ]
}  # fmt: skip


def call_tool(ctx: ToolContext, name: str, args: dict | None) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool {name!r}. Available: {sorted(TOOLS)}"}
    return tool(ctx, args)
