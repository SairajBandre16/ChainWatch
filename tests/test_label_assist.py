import json

import pytest

from chainwatch.extraction import eval as ev
from chainwatch.extraction import label_assist as la
from chainwatch.llm import FakeLLMClient

DRAFT_YES = json.dumps(
    {
        "is_disruption": True,
        "events": [
            {"event_type": "labor_strike", "location": "Port of Cork", "country_code": "IE",
             "port_code": "IEORK", "severity": 3, "confidence": 0.8}
        ],
    }
)  # fmt: skip
DRAFT_BAD = "not json at all"


def _row(news_id: str, **labels: str) -> dict[str, str]:
    row = dict.fromkeys(ev.TEMPLATE_COLUMNS, "")
    row |= {"news_id": news_id, "source": "t", "published": "2026-10-01",
            "title": f"title {news_id}", "text": f"text {news_id}", "url": f"https://ex.com/{news_id}"}  # fmt: skip
    return row | labels


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "tpl.csv"
    la.write_rows(path, [_row("A"), _row("B", is_disruption="no", label_source=""), _row("C")])
    return path


def _drafter(response: str = DRAFT_YES):
    return la.make_drafter(FakeLLMClient([response], model="fake-drafter"))


def _script(*answers: str):
    """Fake `input()` that replays answers in order and fails loudly if the tool asks for more."""
    it = iter(answers)

    def ask(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt!r}") from None

    return ask


def _by_id(path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in la.read_rows(path):
        out.setdefault(r["news_id"], []).append(r)
    return out


def test_accept_writes_draft_with_source(csv_path) -> None:
    counts = la.run_session(csv_path, _drafter(), "fake-drafter", _script("a", ""), print, limit=1)
    assert counts == {"accepted": 1}
    a = _by_id(csv_path)["A"][0]
    assert (a["is_disruption"], a["event_type"], a["port_code"], a["severity"]) == (
        "yes", "labor_strike", "IEORK", "3")  # fmt: skip
    assert (a["label_source"], a["draft_model"]) == ("accepted", "fake-drafter")
    assert _by_id(csv_path)["C"][0]["is_disruption"] == ""  # limit respected


def test_edit_changes_field_and_adds_second_event(csv_path) -> None:
    answers = ("e", "", "2",                       # keep yes, two events
               "", "", "", "", "4",               # event 1: keep all but severity -> 4
               "port_congestion", "Dublin", "ie", "", "2",  # event 2 typed in
               "checked url")  # fmt: skip
    la.run_session(csv_path, _drafter(), "fake-drafter", _script(*answers), print, limit=1)
    rows = _by_id(csv_path)["A"]
    assert len(rows) == 2 and all(r["label_source"] == "edited" for r in rows)
    assert rows[0]["severity"] == "4" and rows[1]["country_code"] == "IE"
    assert rows[0]["notes"] == "checked url"


def test_edit_without_changes_counts_as_accepted(csv_path) -> None:
    answers = ("e", "", "", "", "", "", "", "", "")
    counts = la.run_session(csv_path, _drafter(), "m", _script(*answers), print, limit=1)
    assert counts == {"accepted": 1}


def test_manual_ignores_draft_defaults(csv_path) -> None:
    answers = ("m", "n", "")
    counts = la.run_session(csv_path, _drafter(), "m", _script(*answers), print, limit=1)
    a = _by_id(csv_path)["A"][0]
    assert counts == {"manual": 1}
    assert (a["is_disruption"], a["event_type"], a["label_source"]) == ("no", "", "manual")


def test_invalid_input_is_reprompted(csv_path) -> None:
    answers = ("m", "maybe", "y", "1",
               "strike!", "labor_strike", "Cork", "IRL", "", "",  # bad type, then bad country
               "labor_strike", "Cork", "IE", "", "2", "")  # fmt: skip
    la.run_session(csv_path, _drafter(), "m", _script(*answers), print, limit=1)
    a = _by_id(csv_path)["A"][0]
    assert (a["country_code"], a["severity"]) == ("IE", "2")


def test_skip_and_quit_write_nothing(csv_path) -> None:
    before = csv_path.read_bytes()
    counts = la.run_session(csv_path, _drafter(), "m", _script("s", "q"), print)
    assert counts == {"skipped": 1}
    assert csv_path.read_bytes() == before


def test_resume_continues_after_last_saved_item(csv_path) -> None:
    la.run_session(csv_path, _drafter(), "m", _script("a", "", "q"), print)
    seen: list[str] = []
    la.run_session(csv_path, _drafter(), "m", _script("q"), seen.append)
    assert seen[0].startswith("2 items labeled, 1 to go")
    assert any("title C" in line for line in seen)


def test_never_overwrites_human_label(csv_path) -> None:
    label = ev.GoldItem(news_id="B", is_disruption=True)
    with pytest.raises(ValueError, match="already has a human label"):
        la.save_label(csv_path, "B", label, "accepted", "m")
    assert "B" not in la.pending_ids(la.read_rows(csv_path))


def test_label_added_meanwhile_is_not_overwritten(csv_path) -> None:
    """The owner labels A by hand in Excel while the tool shows A's draft: the tool must refuse."""

    def ask(prompt: str) -> str:
        rows = la.read_rows(csv_path)
        rows[0]["is_disruption"] = "no"
        la.write_rows(csv_path, rows)
        return "a"

    with pytest.raises(ValueError):
        la.run_session(csv_path, _drafter(), "m", ask, print, limit=1)
    assert _by_id(csv_path)["A"][0]["label_source"] == ""


def test_failed_draft_only_offers_manual(csv_path) -> None:
    answers = ("a", "e", "m", "n", "")  # a/e are not valid without a draft and are re-asked
    counts = la.run_session(csv_path, _drafter(DRAFT_BAD), "m", _script(*answers), print, limit=1)
    a = _by_id(csv_path)["A"][0]
    assert counts == {"manual": 1} and a["draft_model"] == ""


def test_dry_run_writes_preview_only_and_import_ignores_it(csv_path, tmp_path) -> None:
    before = csv_path.read_bytes()
    out = tmp_path / "preview.csv"
    assert la.dry_run(csv_path, out, _drafter(), "fake-drafter", limit=5) == 2
    assert csv_path.read_bytes() == before
    rows = la.read_rows(out)
    assert all(r["label_source"] == ev.DRAFT_ONLY and r["is_disruption"] == "yes" for r in rows)
    assert ev.import_csv(out, tmp_path / "labels.jsonl") == 0


def test_labels_import_and_stats(csv_path, tmp_path) -> None:
    la.run_session(csv_path, _drafter(), "m", _script("a", "", "m", "n", ""), print)
    assert la.stats(csv_path) == {"accepted": 1, "hand_csv": 1, "manual": 1}
    assert ev.import_csv(csv_path, tmp_path / "labels.jsonl") == 3
    sources = {g.news_id: g.label_source for g in ev.load_gold(tmp_path / "labels.jsonl")}
    assert sources == {"A": "accepted", "B": None, "C": "manual"}
