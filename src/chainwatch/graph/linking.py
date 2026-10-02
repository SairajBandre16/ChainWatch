"""Link extracted events to graph nodes: the bridge between free-text LLM output and the graph.

Matching order (first hit wins; each result records which rule matched, for the link-rate report):
  1. port_code    the event's UN/LOCODE is a known port
  2. alias_exact  the normalized location equals a known name or alias
  3. alias_within a known alias (4+ chars) appears inside the location as whole words; longest wins
  4. fuzzy        difflib similarity >= FUZZY_THRESHOLD against all aliases (typos, word order)
  5. country      fall back to the country node CTRY:XX, if that country has a port in the graph
  6. none         unlinked; the event is kept but cannot touch any lane

Rule 1 trusts the model's port code only when it is a real node, so a hallucinated code just falls
through to the name-based rules.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from chainwatch.config import PROCESSED_DIR
from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.graph.build import GraphEvent, TradeGraph, country_node
from chainwatch.graph.reference import ReferenceData, default_reference

FUZZY_THRESHOLD = 0.85
MIN_WITHIN_LEN = 4

LinkMethod = Literal["port_code", "alias_exact", "alias_within", "fuzzy", "country", "none"]


class LinkResult(BaseModel):
    node_ids: list[str]
    method: LinkMethod
    score: float = 1.0
    matched_alias: str | None = None


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, drop 'the' and 'port of'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bport of\b|\bthe\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=4)
def build_alias_table(ref: ReferenceData, extra_path: Path | None = None) -> dict[str, str]:
    """normalized alias -> node id, from port/waypoint names, their aliases and the extra CSV."""
    table: dict[str, str] = {}
    for port in ref.ports.values():
        for name in (port.name, port.locode, *port.aliases):
            table.setdefault(normalize(name), port.locode)
    for wp in ref.waypoints.values():
        for name in (wp.name, *wp.aliases):
            table.setdefault(normalize(name), wp.node_id)
    extra_path = extra_path or PROCESSED_DIR / "location_aliases.csv"
    if extra_path.exists():
        with extra_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                table[normalize(row["alias"])] = row["node_id"]
    table.pop("", None)
    return table


def link_location(
    location: str,
    country_code: str | None = None,
    port_code: str | None = None,
    ref: ReferenceData | None = None,
) -> LinkResult:
    ref = ref or default_reference()
    if port_code and port_code in ref.ports:
        return LinkResult(node_ids=[port_code], method="port_code")

    aliases = build_alias_table(ref)
    loc = normalize(location or "")
    if loc in aliases:
        return LinkResult(node_ids=[aliases[loc]], method="alias_exact", matched_alias=loc)

    if loc:
        padded = f" {loc} "
        within = [a for a in aliases if len(a) >= MIN_WITHIN_LEN and f" {a} " in padded]
        if within:
            best = max(within, key=len)
            return LinkResult(node_ids=[aliases[best]], method="alias_within", matched_alias=best)

        scored = [(SequenceMatcher(None, loc, a).ratio(), a) for a in aliases]
        score, best = max(scored)
        if score >= FUZZY_THRESHOLD:
            return LinkResult(node_ids=[aliases[best]], method="fuzzy", score=round(score, 3),
                              matched_alias=best)  # fmt: skip

    known_countries = {p.country_code for p in ref.ports.values()}
    if country_code and country_code in known_countries:
        return LinkResult(node_ids=[country_node(country_code)], method="country", score=0.5)
    return LinkResult(node_ids=[], method="none", score=0.0)


def link_event(event: DisruptionEvent, ref: ReferenceData | None = None) -> LinkResult:
    return link_location(event.location, event.country_code, event.port_code, ref=ref)


def to_graph_event(event: DisruptionEvent, link: LinkResult) -> GraphEvent | None:
    if not link.node_ids:
        return None
    return GraphEvent(
        event_id=event.event_id,
        node_ids=link.node_ids,
        severity=event.severity,
        confidence=event.confidence,
        event_type=event.event_type.value,
        label=event.summary or event.location,
    )


def link_events_into_graph(
    events: list[DisruptionEvent], graph: TradeGraph
) -> tuple[list[LinkResult], dict[str, float]]:
    """Link each event, add linked ones to the graph, and return per-event results plus a report."""
    results = []
    for ev in events:
        link = link_event(ev, graph.ref)
        results.append(link)
        gev = to_graph_event(ev, link)
        if gev is not None:
            graph.add_event(gev)
    return results, link_report(results)


def link_report(results: list[LinkResult]) -> dict[str, float]:
    """Link rate overall and by method. 'direct' excludes country-level fallbacks."""
    n = len(results)
    counts = Counter(r.method for r in results)
    linked = n - counts.get("none", 0)
    direct = linked - counts.get("country", 0)
    report: dict[str, float] = {
        "events": n,
        "link_rate": round(linked / n, 3) if n else 0.0,
        "direct_link_rate": round(direct / n, 3) if n else 0.0,
    }
    report.update({f"method_{m}": c for m, c in sorted(counts.items())})
    return report


def main() -> None:
    """Report link rates for every extraction output in data/processed/extractions/."""
    import json

    from chainwatch.extraction.pipeline import load_records

    for path in sorted((PROCESSED_DIR / "extractions").glob("*.jsonl")):
        events = [e for r in load_records(path) for e in r.events]
        results, report = link_events_into_graph(events, TradeGraph())
        print(path.name, json.dumps(report))
        for ev, res in zip(events, results, strict=True):
            print(f"   {ev.location!r:40} -> {res.node_ids} ({res.method})")


if __name__ == "__main__":
    main()
