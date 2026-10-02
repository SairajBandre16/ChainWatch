import pytest

from chainwatch.graph.build import GraphEvent, TradeGraph, haversine_nm


@pytest.fixture(scope="module")
def tg() -> TradeGraph:
    return TradeGraph()


def test_haversine_known_distance() -> None:
    # Dublin to Cork is about 120 nm in a straight line.
    assert 110 < haversine_nm(53.35, -6.20, 51.83, -8.32) < 130


def test_every_lane_has_a_route(tg) -> None:
    for lane_id in tg.ref.lanes:
        route = tg.lane_route(lane_id)
        assert route.path[0] == tg.ref.lanes[lane_id].origin
        assert route.distance_nm > 0


def test_india_europe_lanes_go_via_suez(tg) -> None:
    route = tg.lane_route("INNSA-NLRTM")
    assert {"BAB_EL_MANDEB", "SUEZ_CANAL"} <= set(route.chokepoints)
    assert 5500 < route.distance_nm < 7500  # real sailing distance is about 6,300 nm


def test_red_sea_event_flags_india_europe_lanes(tg) -> None:
    event = GraphEvent(event_id="rs", node_ids=["RED_SEA"], severity=5, confidence=0.9)
    exposed = {e.lane_id: e for e in tg.exposed_lanes(event)}
    for lane in ("INNSA-NLRTM", "INMUN-IEDUB", "INMAA-IEORK", "INNSA-GBFXT"):
        assert lane in exposed and exposed[lane].via == "RED_SEA"
    assert "AEJEA-INNSA" not in exposed  # Gulf to India never touches the Red Sea
    assert exposed["INNSA-NLRTM"].weight == 0.9


def test_red_sea_alternative_is_the_cape(tg) -> None:
    alts = tg.alternate_routes("INNSA-NLRTM", avoid={"RED_SEA"})
    best = alts[0]
    assert "CAPE_OF_GOOD_HOPE" in best.chokepoints
    assert "SUEZ_CANAL" not in best.path
    assert best.extra_nm > 3000 and best.extra_days > 8  # the Cape detour adds weeks, not hours


def test_hormuz_event_only_hits_gulf_lanes(tg) -> None:
    event = GraphEvent(event_id="h", node_ids=["HORMUZ"], severity=4)
    assert {e.lane_id for e in tg.exposed_lanes(event)} == {"AEJEA-NLRTM", "AEJEA-INNSA"}
    assert tg.alternate_routes("AEJEA-NLRTM", avoid={"HORMUZ"}) == []  # no way out of the Gulf


def test_port_and_country_level_events(tg) -> None:
    port_ev = GraphEvent(event_id="p", node_ids=["IEDUB"], severity=3)
    assert {e.lane_id for e in tg.exposed_lanes(port_ev)} == {
        "INNSA-IEDUB", "INMUN-IEDUB", "INMAA-IEDUB", "CNSHA-IEDUB",
    }  # fmt: skip
    country_ev = GraphEvent(event_id="c", node_ids=["CTRY:IE"], severity=5)
    hits = tg.exposed_lanes(country_ev)
    assert all(h.reason.startswith("event in endpoint country") for h in hits)
    assert all(h.weight == 0.5 for h in hits)  # country-level matches count half
    assert {h.lane_id for h in hits} >= {"INNSA-IEORK", "INNSA-IEDUB"}


def test_lane_exposure_score_combines_events() -> None:
    tg = TradeGraph()
    assert tg.lane_exposure_score("INNSA-NLRTM") == (0.0, [])
    tg.add_event(GraphEvent(event_id="a", node_ids=["RED_SEA"], severity=5, confidence=0.8))
    tg.add_event(GraphEvent(event_id="b", node_ids=["NLRTM"], severity=2, confidence=1.0))
    score, hits = tg.lane_exposure_score("INNSA-NLRTM")
    # 1 - (1 - 0.8) * (1 - 0.4) = 0.88
    assert score == 0.88
    assert [h.event_id for h in hits] == ["a", "b"]
    assert tg.lane_exposure_score("AEJEA-INNSA")[0] == 0.0


def test_event_nodes_do_not_create_shortcuts() -> None:
    tg = TradeGraph()
    before = tg.lane_route("INNSA-NLRTM").distance_nm
    tg._route_cache.clear()
    tg.add_event(GraphEvent(event_id="x", node_ids=["INNSA", "NLRTM"], severity=1))
    assert tg.lane_route("INNSA-NLRTM").distance_nm == before
    with pytest.raises(KeyError):
        tg.add_event(GraphEvent(event_id="y", node_ids=["NOWHERE"], severity=1))
