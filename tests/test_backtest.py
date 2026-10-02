from datetime import UTC, date, datetime, timedelta

import pytest

from chainwatch.backtest.engine import (
    BacktestSpec,
    Disruption,
    Params,
    Window,
    control_signals,
    load_spec,
    replay,
    score_controls,
    score_disruption,
)
from chainwatch.extraction.schemas import DisruptionEvent

PARAMS = Params(lookback_days=7, threshold=0.5, window_before_days=10, window_after_days=3)
ONSET = date(2024, 3, 20)
DISRUPTION = Disruption(id="synthetic", name="Synthetic canal block", onset=ONSET,
                        target_nodes=["SUEZ_CANAL"])  # fmt: skip


def ev(event_id: str, location: str, day: date, severity: int = 5, conf: float = 1.0):
    return DisruptionEvent(
        event_type="canal_or_strait_blockage", location=location, severity=severity,
        confidence=conf, event_id=event_id, news_id=event_id, source_url="https://x",
        source_name="t", published=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
        model="m", prompt_version="v",
    )  # fmt: skip


def window_days() -> list[date]:
    return Window(start=ONSET - timedelta(days=10), end=ONSET + timedelta(days=3)).days()


def test_spec_file_matches_backtest_spec_doc() -> None:
    spec = load_spec()
    assert spec.params.threshold == 0.5 and spec.params.lookback_days == 7
    ids = {d.id: d for d in spec.disruptions}
    assert ids["suez_2021"].onset == date(2021, 3, 23)
    assert ids["red_sea_2023"].onset == date(2023, 12, 15)
    assert ids["red_sea_2023"].secondary_reference.date == date(2023, 11, 19)
    assert spec.primary_lane == "INNSA-NLRTM"
    # Each window's lookback days are included in the days to fetch.
    assert date(2021, 2, 21) - timedelta(days=7) in spec.all_days()


def test_early_warning_gives_positive_lead_time() -> None:
    events = [ev("e1", "Suez Canal", ONSET - timedelta(days=4))]
    signals = replay(events, "INNSA-NLRTM", window_days(), PARAMS, {"SUEZ_CANAL"})
    result = score_disruption(DISRUPTION, "INNSA-NLRTM", signals)
    assert result.first_warning == ONSET - timedelta(days=4)
    assert result.lead_time_days == 4 and not result.missed


def test_lookback_expires_old_events() -> None:
    events = [ev("e1", "Suez Canal", ONSET - timedelta(days=10))]
    signals = replay(events, "INNSA-NLRTM", window_days(), PARAMS, {"SUEZ_CANAL"})
    flagged_days = [s.day for s in signals if s.flagged]
    # Visible on its publication day and the next 6 days (7-day window), then gone.
    assert flagged_days == [ONSET - timedelta(days=10) + timedelta(days=i) for i in range(7)]


def test_late_detection_gives_negative_lead_time() -> None:
    events = [ev("e1", "Suez Canal", ONSET + timedelta(days=2))]
    signals = replay(events, "INNSA-NLRTM", window_days(), PARAMS, {"SUEZ_CANAL"})
    assert score_disruption(DISRUPTION, "INNSA-NLRTM", signals).lead_time_days == -2


def test_off_target_flags_do_not_count_as_warning() -> None:
    events = [ev("e1", "Port of Rotterdam", ONSET - timedelta(days=5))]
    signals = replay(events, "INNSA-NLRTM", window_days(), PARAMS, {"SUEZ_CANAL"})
    result = score_disruption(DISRUPTION, "INNSA-NLRTM", signals)
    assert result.missed and result.lead_time_days is None
    assert result.pre_onset_off_target_flag_days == 5


def test_weak_events_must_accumulate_to_cross_threshold() -> None:
    # Each weak event weighs 2/5 x 0.5 = 0.2; three give 1 - 0.8^3 = 0.488 (< 0.5), four give 0.59.
    days = [ONSET - timedelta(days=k) for k in (6, 5, 4, 3)]
    events = [ev(f"w{i}", "Suez Canal", d, severity=2, conf=0.5) for i, d in enumerate(days)]
    signals = replay(events, "INNSA-NLRTM", window_days(), PARAMS, {"SUEZ_CANAL"})
    result = score_disruption(DISRUPTION, "INNSA-NLRTM", signals)
    assert result.first_warning == ONSET - timedelta(days=3)


def test_events_after_the_day_are_invisible() -> None:
    events = [ev("future", "Suez Canal", ONSET + timedelta(days=1))]
    signals = replay(events, "INNSA-NLRTM", [ONSET], PARAMS, {"SUEZ_CANAL"})
    assert signals[0].score == 0.0


def test_false_alarm_rate_and_episodes() -> None:
    spec = BacktestSpec(
        params=PARAMS, primary_lane="INNSA-NLRTM", secondary_lane_focus=[], disruptions=[],
        control_windows=[Window(start=date(2022, 9, 1), end=date(2022, 9, 10)),
                         Window(start=date(2019, 9, 1), end=date(2019, 9, 10))],
    )  # fmt: skip
    events = [ev("c1", "Red Sea", date(2022, 9, 9)), ev("c2", "Red Sea", date(2019, 9, 1))]
    signals = control_signals(events, spec, "INNSA-NLRTM")
    result = score_controls("INNSA-NLRTM", signals)
    # 2022: Sept 9-10 flagged (2 days); 2019: Sept 1-7 flagged (7 days) -> 9 of 20 days, 2 episodes.
    assert (result.days, result.flagged_days, result.episodes) == (20, 9, 2)
    assert result.false_alarm_rate == pytest.approx(0.45)


def test_secondary_reference_lead_time() -> None:
    d = load_spec().disruptions[1]
    first = d.onset - timedelta(days=20)
    events = [ev("a", "Red Sea", first)]
    days = Window(start=d.onset - timedelta(days=30), end=d.onset).days()
    signals = replay(events, "INNSA-NLRTM", days, PARAMS, set(d.target_nodes))
    result = score_disruption(d, "INNSA-NLRTM", signals)
    assert result.lead_time_days == 20
    assert result.lead_time_vs_secondary_days == (d.secondary_reference.date - first).days


def test_replay_score_matches_graph_exposure_score() -> None:
    from chainwatch.graph.build import TradeGraph
    from chainwatch.graph.linking import link_event, to_graph_event

    events = [ev("m1", "Suez Canal", ONSET, severity=3, conf=0.7),
              ev("m2", "Red Sea", ONSET, severity=4, conf=0.6),
              ev("m3", "Port of Rotterdam", ONSET, severity=2, conf=0.9)]  # fmt: skip
    signal = replay(events, "INNSA-NLRTM", [ONSET], PARAMS, {"SUEZ_CANAL"})[0]
    graph = TradeGraph()
    gevs = [to_graph_event(e, link_event(e)) for e in events]
    score, hits = graph.lane_exposure_score("INNSA-NLRTM", gevs)
    assert signal.score == score
    assert signal.contributing_events == [h.event_id for h in hits]
