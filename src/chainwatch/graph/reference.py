"""Curated reference data for the trade-lane graph, loaded from CSVs in data/processed/.

- ports.csv      ~60 major container ports (UN/LOCODE, country, coordinates, aliases)
- waypoints.csv  chokepoints (Suez, Bab-el-Mandeb, Hormuz, ...) and sea regions joining them
- sea_legs.csv   undirected connections between ports and waypoints (a schematic sea network)
- lanes.csv      named origin -> destination trade lanes, focused on India to Europe/Ireland

Coordinates are approximate (about 0.1 degree) and leg lengths are great-circle distances between
nodes, so this is a schematic network for exposure reasoning, not a navigation chart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from chainwatch.config import PROCESSED_DIR


def _split_aliases(value: object) -> object:
    if isinstance(value, str):
        return [a.strip() for a in value.split("|") if a.strip()]
    return value


class Port(BaseModel):
    locode: str = Field(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")
    name: str
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    region: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    aliases: list[str] = Field(default_factory=list)

    _aliases = field_validator("aliases", mode="before")(_split_aliases)

    @model_validator(mode="after")
    def _locode_matches_country(self) -> Port:
        # The first two letters of a UN/LOCODE are the ISO country code.
        if self.locode[:2] != self.country_code:
            raise ValueError(f"{self.locode}: LOCODE prefix != country {self.country_code}")
        return self


class Waypoint(BaseModel):
    node_id: str = Field(pattern=r"^[A-Z_]+$")
    name: str
    kind: Literal["chokepoint", "sea_region"]
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    country_code: str | None = None
    aliases: list[str] = Field(default_factory=list)

    _aliases = field_validator("aliases", mode="before")(_split_aliases)

    @field_validator("country_code", mode="before")
    @classmethod
    def _blank(cls, value: object) -> object:
        return value or None


class Leg(BaseModel):
    from_node: str
    to_node: str


class Lane(BaseModel):
    lane_id: str
    origin: str
    destination: str
    focus: str


@dataclass(frozen=True)
class ReferenceData:
    ports: dict[str, Port]
    waypoints: dict[str, Waypoint]
    legs: list[Leg]
    lanes: dict[str, Lane]

    def node_ids(self) -> set[str]:
        return set(self.ports) | set(self.waypoints)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def validate(ref: ReferenceData) -> None:
    """Cross-file checks that row-level Pydantic validation cannot do."""
    clash = set(ref.ports) & set(ref.waypoints)
    if clash:
        raise ValueError(f"Ids used by both ports and waypoints: {clash}")
    nodes = ref.node_ids()
    for leg in ref.legs:
        missing = {leg.from_node, leg.to_node} - nodes
        if missing:
            raise ValueError(f"Leg {leg.from_node}-{leg.to_node} uses unknown nodes {missing}")
        if leg.from_node == leg.to_node:
            raise ValueError(f"Self-loop leg at {leg.from_node}")
    for lane in ref.lanes.values():
        for end in (lane.origin, lane.destination):
            if end not in ref.ports:
                raise ValueError(f"Lane {lane.lane_id} endpoint {end} is not a known port")
    connected = {leg.from_node for leg in ref.legs} | {leg.to_node for leg in ref.legs}
    orphans = set(ref.ports) - connected
    if orphans:
        raise ValueError(f"Ports with no sea leg: {sorted(orphans)}")


def load_reference(directory: Path = PROCESSED_DIR) -> ReferenceData:
    def _unique(rows: list[BaseModel], key: str) -> dict:
        out: dict = {}
        for row in rows:
            k = getattr(row, key)
            if k in out:
                raise ValueError(f"Duplicate {key}: {k}")
            out[k] = row
        return out

    ref = ReferenceData(
        ports=_unique([Port(**r) for r in _read(directory / "ports.csv")], "locode"),
        waypoints=_unique([Waypoint(**r) for r in _read(directory / "waypoints.csv")], "node_id"),
        legs=[Leg(**r) for r in _read(directory / "sea_legs.csv")],
        lanes=_unique([Lane(**r) for r in _read(directory / "lanes.csv")], "lane_id"),
    )
    validate(ref)
    return ref


@lru_cache(maxsize=1)
def default_reference() -> ReferenceData:
    """Cached load of the committed reference data (read-only use)."""
    return load_reference()
