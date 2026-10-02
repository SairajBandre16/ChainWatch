"""Trade-lane knowledge graph on NetworkX, plus the queries the agent and dashboard use.

Node kinds:  port | chokepoint | sea_region | country | event
Edge kinds:  sea (routable, has distance_nm) | in_country (port -> country) | affects (event -> node)

Routing only ever walks `sea` edges, so country and event nodes never create shortcuts.
"""

from __future__ import annotations

import math
from itertools import islice

import networkx as nx
from pydantic import BaseModel, Field

from chainwatch.graph.reference import ReferenceData, default_reference

KNOTS = 14.0  # typical slow-steaming container ship speed, used to turn distance into days
EARTH_RADIUS_NM = 3440.065


def country_node(code: str) -> str:
    return f"CTRY:{code}"


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))


class GraphEvent(BaseModel):
    """A disruption already linked to graph nodes (see `graph.linking`)."""

    event_id: str
    node_ids: list[str] = Field(description="Ports, waypoints or CTRY:XX nodes the event touches")
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(default=1.0, ge=0, le=1)
    event_type: str = "other"
    label: str = ""


class Route(BaseModel):
    path: list[str]
    distance_nm: float
    days: float
    extra_nm: float = 0.0
    extra_days: float = 0.0
    chokepoints: list[str] = Field(default_factory=list)


class LaneExposure(BaseModel):
    lane_id: str
    event_id: str
    via: str
    reason: str
    weight: float = Field(description="severity/5 x confidence x match weight, in [0, 1]")


# How strongly a match counts: an event on the route or at an endpoint is direct exposure;
# a country-level event (e.g. a national strike) only touches lanes that start or end there.
MATCH_WEIGHTS = {"on_route": 1.0, "endpoint": 1.0, "country": 0.5}


class TradeGraph:
    def __init__(self, ref: ReferenceData | None = None) -> None:
        self.ref = ref or default_reference()
        self.g = nx.Graph()
        self._build()
        self._route_cache: dict[str, Route] = {}

    # --- construction ---------------------------------------------------------------------

    def _build(self) -> None:
        for port in self.ref.ports.values():
            self.g.add_node(port.locode, kind="port", name=port.name, lat=port.lat, lon=port.lon,
                            country=port.country_code, region=port.region)  # fmt: skip
            cnode = country_node(port.country_code)
            self.g.add_node(cnode, kind="country", name=port.country_code)
            self.g.add_edge(port.locode, cnode, kind="in_country")
        for wp in self.ref.waypoints.values():
            self.g.add_node(wp.node_id, kind=wp.kind, name=wp.name, lat=wp.lat, lon=wp.lon,
                            country=wp.country_code)  # fmt: skip
        for leg in self.ref.legs:
            a, b = self.g.nodes[leg.from_node], self.g.nodes[leg.to_node]
            dist = haversine_nm(a["lat"], a["lon"], b["lat"], b["lon"])
            self.g.add_edge(leg.from_node, leg.to_node, kind="sea", distance_nm=round(dist, 1))

    def add_event(self, event: GraphEvent) -> None:
        node = f"EVT:{event.event_id}"
        self.g.add_node(node, kind="event", **event.model_dump(exclude={"node_ids"}))
        for target in event.node_ids:
            if target not in self.g:
                raise KeyError(f"Event {event.event_id} links to unknown node {target}")
            self.g.add_edge(node, target, kind="affects")

    def events(self) -> list[GraphEvent]:
        out = []
        for node, data in self.g.nodes(data=True):
            if data.get("kind") == "event":
                targets = [n for n in self.g.neighbors(node)]
                fields = {k: v for k, v in data.items() if k != "kind"}
                out.append(GraphEvent(node_ids=targets, **fields))
        return out

    # --- routing --------------------------------------------------------------------------

    def sea_view(self, avoid: set[str] | frozenset[str] = frozenset()) -> nx.Graph:
        """Read-only view with only sea edges and without the avoided nodes."""

        def keep_node(n: str) -> bool:
            return n not in avoid and self.g.nodes[n].get("kind") in {
                "port",
                "chokepoint",
                "sea_region",
            }

        def keep_edge(u: str, v: str) -> bool:
            return self.g.edges[u, v].get("kind") == "sea"

        return nx.subgraph_view(self.g, filter_node=keep_node, filter_edge=keep_edge)

    def _route(self, path: list[str]) -> Route:
        dist = sum(self.g.edges[a, b]["distance_nm"] for a, b in zip(path, path[1:], strict=False))
        chokes = [n for n in path if self.g.nodes[n]["kind"] == "chokepoint"]
        return Route(path=path, distance_nm=round(dist, 1), days=round(dist / KNOTS / 24, 1),
                     chokepoints=chokes)  # fmt: skip

    def lane_route(self, lane_id: str) -> Route:
        """Base (shortest) sea route for a lane. Cached because lanes are fixed."""
        if lane_id not in self._route_cache:
            lane = self.ref.lanes[lane_id]
            path = nx.shortest_path(
                self.sea_view(), lane.origin, lane.destination, weight="distance_nm"
            )
            self._route_cache[lane_id] = self._route(path)
        return self._route_cache[lane_id]

    def alternate_routes(self, lane_id: str, avoid: set[str], k: int = 3) -> list[Route]:
        """Up to k shortest routes that avoid the given nodes, with the detour cost vs the base."""
        lane = self.ref.lanes[lane_id]
        base = self.lane_route(lane_id)
        view = self.sea_view(frozenset(avoid) - {lane.origin, lane.destination})
        try:
            paths = nx.shortest_simple_paths(
                view, lane.origin, lane.destination, weight="distance_nm"
            )
            routes = [self._route(p) for p in islice(paths, k)]
        except nx.NetworkXNoPath:
            return []
        for r in routes:
            r.extra_nm = round(r.distance_nm - base.distance_nm, 1)
            r.extra_days = round(r.days - base.days, 1)
        return routes

    # --- exposure -------------------------------------------------------------------------

    def exposed_lanes(self, event: GraphEvent) -> list[LaneExposure]:
        """Lanes whose base route passes through, starts or ends at, a node the event touches."""
        out: list[LaneExposure] = []
        base_weight = event.severity / 5 * event.confidence
        for lane_id, lane in self.ref.lanes.items():
            route = self.lane_route(lane_id)
            match = None
            for node in event.node_ids:
                if node in (lane.origin, lane.destination):
                    match = (node, "endpoint", f"event at lane endpoint {node}")
                elif node in route.path:
                    match = (node, "on_route", f"event on route at {node}")
                elif node.startswith("CTRY:"):
                    countries = {country_node(self.g.nodes[p]["country"])
                                 for p in (lane.origin, lane.destination)}  # fmt: skip
                    if node in countries:
                        match = (node, "country", f"event in endpoint country {node[5:]}")
                if match and match[1] != "country":
                    break  # direct match beats a country-level one
            if match:
                via, kind, reason = match
                weight = round(base_weight * MATCH_WEIGHTS[kind], 3)
                out.append(LaneExposure(lane_id=lane_id, event_id=event.event_id, via=via,
                                        reason=reason, weight=weight))  # fmt: skip
        return out

    def lane_exposure_score(
        self, lane_id: str, events: list[GraphEvent] | None = None
    ) -> tuple[float, list[LaneExposure]]:
        """Combine all events touching a lane into one score in [0, 1].

        score = 1 - prod(1 - w_i): independent "chances of disruption". One severe event gives a high
        score; several small ones add up but never exceed 1. Returns the contributions too, so the
        number can always be explained.
        """
        events = self.events() if events is None else events
        hits = [e for ev in events for e in self.exposed_lanes(ev) if e.lane_id == lane_id]
        survive = 1.0
        for hit in hits:
            survive *= 1 - hit.weight
        return round(1 - survive, 3), sorted(hits, key=lambda h: h.weight, reverse=True)
