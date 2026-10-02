from datetime import UTC, datetime

import pytest

from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.graph.build import TradeGraph
from chainwatch.graph.linking import link_events_into_graph, link_location, link_report, normalize


@pytest.mark.parametrize(
    ("location", "country", "port", "expected", "method"),
    [
        ("Nhava Sheva", "IN", "INNSA", ["INNSA"], "port_code"),
        ("Somewhere", "IN", "INXXX", ["CTRY:IN"], "country"),  # hallucinated code falls through
        ("Yanbu port", "SA", "INNSA", ["CTRY:SA"], "country"),  # real code, wrong country
        ("Strait of Hormuz", None, None, ["HORMUZ"], "alias_exact"),
        ("Port of Rotterdam", "NL", None, ["NLRTM"], "alias_exact"),
        ("Port Said", "EG", None, ["EGPSD"], "alias_exact"),  # 'port' kept when not 'port of'
        ("Hormuz Strait", None, None, ["HORMUZ"], "alias_within"),
        ("southern Red Sea near Yemen", None, None, ["BAB_EL_MANDEB"], "alias_within"),
        ("Felixtowe", None, None, ["GBFXT"], "fuzzy"),
        ("Gdańsk", "PL", None, ["PLGDN"], "alias_exact"),
        ("Gujarat", "IN", None, ["CTRY:IN"], "country"),
        ("Port of Vancouver", "CA", None, [], "none"),  # Canada has no port in the graph
        ("", None, None, [], "none"),
    ],
)
def test_link_location(location, country, port, expected, method) -> None:
    result = link_location(location, country, port)
    assert (result.node_ids, result.method) == (expected, method)


def test_normalize() -> None:
    assert normalize("The Port of Cork!") == "cork"
    assert normalize("Bab-el-Mandeb") == "bab el mandeb"


def _event(event_id: str, location: str, cc: str | None = None) -> DisruptionEvent:
    return DisruptionEvent(
        event_type="conflict_or_attack",
        location=location,
        country_code=cc,
        severity=4,
        confidence=1.0,
        event_id=event_id,
        news_id="n",
        source_url="https://ex.com",
        source_name="t",
        published=datetime(2024, 1, 1, tzinfo=UTC),
        model="m",
        prompt_version="v",
    )


def test_link_events_into_graph_and_report() -> None:
    graph = TradeGraph()
    events = [
        _event("a", "Red Sea"),
        _event("b", "Hormuz Strait"),
        _event("c", "Gujarat", "IN"),
        _event("d", "most of the world"),
    ]
    results, report = link_events_into_graph(events, graph)
    assert [r.method for r in results] == ["alias_exact", "alias_within", "country", "none"]
    assert report["link_rate"] == 0.75 and report["direct_link_rate"] == 0.5
    assert {e.event_id for e in graph.events()} == {"a", "b", "c"}
    assert graph.lane_exposure_score("INNSA-NLRTM")[0] > 0  # via Red Sea and India country node


def test_empty_report() -> None:
    assert link_report([])["link_rate"] == 0.0
