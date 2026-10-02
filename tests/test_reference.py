import shutil

import pytest
from pydantic import ValidationError

from chainwatch.config import PROCESSED_DIR
from chainwatch.graph.reference import Port, load_reference

FILES = ("ports.csv", "waypoints.csv", "sea_legs.csv", "lanes.csv")


def test_reference_data_loads_and_validates() -> None:
    ref = load_reference()
    assert len(ref.ports) >= 50
    chokepoints = {w.node_id for w in ref.waypoints.values() if w.kind == "chokepoint"}
    assert {"SUEZ_CANAL", "BAB_EL_MANDEB", "HORMUZ", "MALACCA", "PANAMA_CANAL"} <= chokepoints
    assert "CAPE_OF_GOOD_HOPE" in chokepoints
    assert len(ref.legs) >= 30
    # India to Europe/Ireland focus lanes are present.
    india_ports = {"INNSA", "INMUN", "INMAA"}
    europe_ports = {"NLRTM", "GBFXT", "IEDUB", "IEORK"}
    pairs = {(lane.origin, lane.destination) for lane in ref.lanes.values()}
    assert any(o in india_ports and d in europe_ports for o, d in pairs)
    assert ref.ports["IEDUB"].country_code == "IE"


def test_port_row_validation() -> None:
    row = {"locode": "IEDUB", "name": "Dublin", "country_code": "IE", "region": "x", "lat": 53.3}
    port = Port(**row, lon=-6.2, aliases="Dublin Port|Port of Dublin")
    assert port.aliases == ["Dublin Port", "Port of Dublin"]
    with pytest.raises(ValidationError):
        Port(**row | {"country_code": "GB"}, lon=-6.2)  # LOCODE prefix mismatch
    with pytest.raises(ValidationError):
        Port(**row, lon=-600)


def _copy_reference(tmp_path):
    for name in FILES:
        shutil.copy(PROCESSED_DIR / name, tmp_path / name)
    return tmp_path


def test_unknown_leg_node_is_rejected(tmp_path) -> None:
    d = _copy_reference(tmp_path)
    with (d / "sea_legs.csv").open("a", encoding="utf-8") as fh:
        fh.write("NLRTM,ATLANTIS\n")
    with pytest.raises(ValueError, match="unknown nodes"):
        load_reference(d)


def test_duplicate_port_is_rejected(tmp_path) -> None:
    d = _copy_reference(tmp_path)
    with (d / "ports.csv").open("a", encoding="utf-8") as fh:
        fh.write("NLRTM,Rotterdam again,NL,North Europe,51.9,4.1,\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_reference(d)


def test_lane_with_unknown_port_is_rejected(tmp_path) -> None:
    d = _copy_reference(tmp_path)
    with (d / "lanes.csv").open("a", encoding="utf-8") as fh:
        fh.write("X-Y,INNSA,ZZZZZ,test\n")
    with pytest.raises(ValueError, match="not a known port"):
        load_reference(d)
