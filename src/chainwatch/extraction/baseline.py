"""Keyword baseline extractor: no LLM, just rules.

Two jobs:
  1. A **baseline** for the eval: an LLM that cannot beat keyword rules is not worth its cost.
  2. The **fake LLM responder**: `keyword_responder` reads the rendered prompt and returns
     JSON in the same shape a real model would, so the full pipeline runs offline with no model.
"""

from __future__ import annotations

import json
import re

from chainwatch.extraction.schemas import EventType

# Ordered: the first matching rule wins, so specific types come before generic ones.
TYPE_RULES: list[tuple[EventType, tuple[str, ...]]] = [
    (EventType.CANAL_OR_STRAIT_BLOCKAGE, ("blocked", "blockage", "aground", "stuck in")),
    # Attacks before strikes: "drone strike" is an attack. Labor phrases avoid the bare word "strike".
    (EventType.CONFLICT_OR_ATTACK, ("attack", "missile", "houthi", "drone", "hijack", "seized")),
    (
        EventType.LABOR_STRIKE,
        (
            "workers strike",
            "on strike",
            "strike action",
            "walkout",
            "industrial action",
            "dockworkers",
            "union",
        ),
    ),  # fmt: skip
    (EventType.SEVERE_WEATHER, ("typhoon", "hurricane", "cyclone", "storm", "flood", "drought")),
    (EventType.PORT_CONGESTION, ("congestion", "backlog", "queue", "bottleneck")),
    (EventType.PORT_CLOSURE, ("closed", "closure", "shut", "suspend")),
    (EventType.ACCIDENT, ("collision", "fire", "explosion", "capsiz", "sank", "grounding")),
    (EventType.INFRASTRUCTURE_FAILURE, ("cyber", "outage", "crane collapse", "bridge collapse")),
    (EventType.SANCTIONS_OR_REGULATION, ("sanction", "embargo", "export ban", "tariff")),
]

# Small built-in gazetteer: name -> (country_code, port_code). The full reference data arrives in
# phase 2; this only needs to cover the most common places in shipping news.
GAZETTEER: dict[str, tuple[str | None, str | None]] = {
    "suez canal": ("EG", None),
    "red sea": (None, None),
    "bab-el-mandeb": (None, None),
    "bab el-mandeb": (None, None),
    "strait of hormuz": (None, None),
    "hormuz": (None, None),
    "strait of malacca": (None, None),
    "panama canal": ("PA", None),
    "cape of good hope": ("ZA", None),
    "black sea": (None, None),
    "rotterdam": ("NL", "NLRTM"),
    "antwerp": ("BE", "BEANR"),
    "hamburg": ("DE", "DEHAM"),
    "felixstowe": ("GB", "GBFXT"),
    "dublin": ("IE", "IEDUB"),
    "cork": ("IE", "IEORK"),
    "nhava sheva": ("IN", "INNSA"),
    "jawaharlal nehru port": ("IN", "INNSA"),
    "mundra": ("IN", "INMUN"),
    "chennai": ("IN", "INMAA"),
    "colombo": ("LK", "LKCMB"),
    "singapore": ("SG", "SGSIN"),
    "shanghai": ("CN", "CNSHA"),
    "jebel ali": ("AE", "AEJEA"),
    "los angeles": ("US", "USLAX"),
    "long beach": ("US", "USLGB"),
    "baltimore": ("US", "USBAL"),
    "yemen": ("YE", None),
    "india": ("IN", None),
    "china": ("CN", None),
    "taiwan": ("TW", None),
    "iran": ("IR", None),
    "egypt": ("EG", None),
}

SEVERITY_HINTS = {5: ("global", "worldwide"), 4: ("weeks", "rerout", "suspend"), 3: ("days",)}


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", prompt, flags=re.MULTILINE)
    return match.group(1) if match else ""


def keyword_extract(text: str) -> dict:
    """Rule-based extraction returning a dict in the `ExtractionOutput` shape."""
    lowered = text.lower()
    event_type = next(
        (etype for etype, words in TYPE_RULES if any(w in lowered for w in words)), None
    )
    place = next((name for name in GAZETTEER if name in lowered), None)
    # Need both a disruption keyword and a known place: precision over recall for a baseline.
    if event_type is None or place is None:
        return {"is_disruption": False, "events": []}
    country, port = GAZETTEER[place]
    severity = next(
        (s for s, words in SEVERITY_HINTS.items() if any(w in lowered for w in words)), 2
    )
    return {
        "is_disruption": True,
        "events": [
            {
                "event_type": event_type.value,
                "location": place.title(),
                "country_code": country,
                "port_code": port,
                "severity": severity,
                "start_date": None,
                "end_date": None,
                "industries": [],
                "summary": "",
                "confidence": 0.5,
            }
        ],
    }


def keyword_responder(prompt: str, system: str | None = None) -> str:
    """Fake-LLM responder: pull Title/Text out of the rendered prompt and run the rules."""
    text = f"{_field(prompt, 'Title')}\n{_field(prompt, 'Text')}"
    return json.dumps(keyword_extract(text))
