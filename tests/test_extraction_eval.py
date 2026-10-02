import csv
from datetime import UTC, datetime
from pathlib import Path

from chainwatch.extraction import eval as ev
from chainwatch.extraction.schemas import DisruptionEvent, ExtractionRecord

FIXTURE = Path(__file__).parent / "fixtures" / "labels_tiny.jsonl"


def _event(news_id: str, etype: str, loc: str, cc, port, sev: int) -> DisruptionEvent:
    return DisruptionEvent(
        event_type=etype,
        location=loc,
        country_code=cc,
        port_code=port,
        severity=sev,
        confidence=0.8,
        event_id=f"{news_id}-0",
        news_id=news_id,
        source_url="https://ex.com",
        source_name="t",
        published=datetime(2024, 1, 1, tzinfo=UTC),
        model="m",
        prompt_version="v",
    )


def _rec(news_id: str, events: list, flag: bool = True, status: str = "ok") -> ExtractionRecord:
    return ExtractionRecord(
        news_id=news_id,
        model="m",
        prompt_version="v",
        status=status,
        is_disruption=flag,
        events=events,
    )


PREDICTIONS = [
    _rec("A", [_event("A", "conflict_or_attack", "the Red Sea", None, None, 4)]),
    _rec("B", [_event("B", "port_closure", "Felixstowe", "GB", None, 3)]),
    _rec("C", [_event("C", "other", "Dublin", "IE", "IEDUB", 1)]),
    _rec("D", [], flag=False, status="failed"),
]


def test_scorer_on_tiny_hand_made_set() -> None:
    result = ev.score(ev.load_gold(FIXTURE), PREDICTIONS)
    f = result["fields"]
    assert result["items"] == 4 and result["failed"] == 1
    assert (f["is_disruption"]["precision"], f["is_disruption"]["recall"]) == (0.667, 0.667)
    assert (f["event_type"]["precision"], f["event_type"]["recall"]) == (0.333, 0.333)
    assert (f["country_code"]["precision"], f["country_code"]["recall"]) == (0.5, 0.5)
    assert f["port_code"]["f1"] == 0.0 and f["port_code"]["support"] == 2
    assert f["location"]["f1"] == 0.667  # "Port of Felixstowe" matches "Felixstowe"
    assert result["severity"] == {"aligned_events": 2, "exact": 0.5, "within_1": 1.0, "mae": 0.5}


def test_perfect_predictions_score_one() -> None:
    gold = ev.load_gold(FIXTURE)
    preds = [
        _rec(g.news_id, [_event(g.news_id, e.event_type, e.location, e.country_code,
                                e.port_code, e.severity) for e in g.events], flag=g.is_disruption)
        for g in gold
    ]  # fmt: skip
    fields = ev.score(gold, preds)["fields"]
    assert all(fields[k]["f1"] == 1.0 for k in fields)


def test_missing_prediction_counts_as_negative() -> None:
    result = ev.score(ev.load_gold(FIXTURE), [])
    assert result["missing"] == 4
    assert result["fields"]["is_disruption"]["recall"] == 0.0


def test_template_and_csv_import_roundtrip(tmp_path, monkeypatch) -> None:
    tpl = tmp_path / "tpl.csv"
    n = ev.make_template(n=5, path=tpl)
    assert n == 5
    with tpl.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["is_disruption"] == "" for r in rows)  # nothing pre-filled
    # Simulate the owner labeling: first item has two events, second is negative, rest unlabeled.
    rows[0] |= {"is_disruption": "yes", "event_type": "labor_strike", "location": "Cork",
                "country_code": "ie", "port_code": "IEORK", "severity": "2"}  # fmt: skip
    extra = dict(rows[0]) | {"event_type": "port_congestion", "location": "Dublin", "port_code": ""}
    rows[1] |= {"is_disruption": "no"}
    with tpl.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ev.TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows([rows[0], extra, *rows[1:]])
    out = tmp_path / "labels.jsonl"
    assert ev.import_csv(tpl, out) == 2
    gold = ev.load_gold(out)
    assert len(gold[0].events) == 2 and gold[0].events[1].port_code is None
    assert gold[0].events[0].severity == 2
    assert gold[1].is_disruption is False and gold[1].events == []
