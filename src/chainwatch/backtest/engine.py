"""Backtest replay engine and metrics, implementing docs/backtest-spec.md exactly.

Inputs are dated `DisruptionEvent`s (whatever extractor produced them). For each day the engine builds a
fresh graph with only the events published in the lookback window ending that day, computes the lane's
exposure score, and records whether it is flagged and whether the flag is on target.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from chainwatch.config import EVAL_DIR
from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.graph.build import LaneExposure, TradeGraph
from chainwatch.graph.linking import link_event, to_graph_event
from chainwatch.graph.reference import ReferenceData, default_reference

SPEC_PATH = EVAL_DIR / "backtest_events.json"


# --- spec -------------------------------------------------------------------------------------


class Params(BaseModel):
    lookback_days: int
    threshold: float
    window_before_days: int
    window_after_days: int


class SecondaryReference(BaseModel):
    label: str
    date: date


class Disruption(BaseModel):
    id: str
    name: str
    onset: date
    secondary_reference: SecondaryReference | None = None
    target_nodes: list[str]


class Window(BaseModel):
    start: date
    end: date

    def days(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]


class BacktestSpec(BaseModel):
    params: Params
    primary_lane: str
    secondary_lane_focus: list[str]
    disruptions: list[Disruption]
    control_windows: list[Window]

    def window_for(self, d: Disruption) -> Window:
        return Window(start=d.onset - timedelta(days=self.params.window_before_days),
                      end=d.onset + timedelta(days=self.params.window_after_days))  # fmt: skip

    def all_days(self) -> list[date]:
        """Every day whose news is needed, including lookback days before each window."""
        days: set[date] = set()
        windows = [self.window_for(d) for d in self.disruptions] + self.control_windows
        for w in windows:
            start = w.start - timedelta(days=self.params.lookback_days)
            days |= set(Window(start=start, end=w.end).days())
        return sorted(days)


def load_spec(path: Path = SPEC_PATH) -> BacktestSpec:
    return BacktestSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


# --- replay -----------------------------------------------------------------------------------


class DailySignal(BaseModel):
    day: date
    score: float
    flagged: bool
    on_target: bool
    contributing_events: list[str] = Field(default_factory=list)
    contributing_nodes: list[str] = Field(default_factory=list)


def _day_end(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=UTC)


# Per-event caches. An event's link and its exposure to each lane do not depend on the replay day,
# so they are computed once per event instead of once per day (same numbers, much faster).
_EXPOSURE_CACHE: dict[tuple, dict[str, LaneExposure]] = {}


def _exposures(ev: DisruptionEvent, graph: TradeGraph) -> dict[str, LaneExposure]:
    # Key on everything that affects linking and weights, not just the id (ids can repeat across
    # differently-built event sets, e.g. in tests).
    key = (id(graph.ref), ev.event_id, ev.location, ev.country_code, ev.port_code, ev.severity,
           ev.confidence)  # fmt: skip
    if key not in _EXPOSURE_CACHE:
        gev = to_graph_event(ev, link_event(ev, graph.ref))
        hits = graph.exposed_lanes(gev) if gev is not None else []
        _EXPOSURE_CACHE[key] = {h.lane_id: h for h in hits}
    return _EXPOSURE_CACHE[key]


def replay(
    events: list[DisruptionEvent],
    lane_id: str,
    days: list[date],
    params: Params,
    target_nodes: set[str] | None = None,
    ref: ReferenceData | None = None,
) -> list[DailySignal]:
    """One `DailySignal` per day for one lane.

    The score is the same as `TradeGraph.lane_exposure_score` over the events visible that day:
    1 - prod(1 - w_i), rounded to 3 decimals (checked by a test).
    """
    graph = TradeGraph(ref or default_reference())
    first, last = (
        _day_end(min(days)) - timedelta(days=params.lookback_days + 1),
        _day_end(max(days)),
    )
    relevant = []
    for ev in events:
        if first < ev.published <= last:
            hit = _exposures(ev, graph).get(lane_id)
            if hit is not None:
                relevant.append((ev.published, hit))
    signals = []
    for d in days:
        end = _day_end(d)
        start = end - timedelta(days=params.lookback_days)
        hits = sorted((h for published, h in relevant if start < published <= end),
                      key=lambda h: h.weight, reverse=True)  # fmt: skip
        survive = 1.0
        for h in hits:
            survive *= 1 - h.weight
        score = round(1 - survive, 3)
        nodes = sorted({h.via for h in hits})
        flagged = score >= params.threshold
        on_target = flagged and bool(target_nodes and set(nodes) & target_nodes)
        signals.append(DailySignal(day=d, score=score, flagged=flagged, on_target=on_target,
                                   contributing_events=[h.event_id for h in hits],
                                   contributing_nodes=nodes))  # fmt: skip
    return signals


# --- metrics ----------------------------------------------------------------------------------


class DisruptionResult(BaseModel):
    disruption_id: str
    lane_id: str
    onset: date
    first_warning: date | None
    lead_time_days: int | None  # positive = before onset; None = missed
    missed: bool
    secondary_reference: str | None = None
    lead_time_vs_secondary_days: int | None = None
    pre_onset_off_target_flag_days: int
    peak_score: float


class ControlResult(BaseModel):
    lane_id: str
    days: int
    flagged_days: int
    false_alarm_rate: float
    episodes: int


def score_disruption(d: Disruption, lane_id: str, signals: list[DailySignal]) -> DisruptionResult:
    first = next((s.day for s in signals if s.on_target), None)
    lead = (d.onset - first).days if first else None
    sec_label, sec_lead = None, None
    if d.secondary_reference is not None:
        sec_label = f"{d.secondary_reference.label} ({d.secondary_reference.date})"
        sec_lead = (d.secondary_reference.date - first).days if first else None
    off_target = sum(1 for s in signals if s.day < d.onset and s.flagged and not s.on_target)
    return DisruptionResult(
        disruption_id=d.id, lane_id=lane_id, onset=d.onset, first_warning=first,
        lead_time_days=lead, missed=first is None, secondary_reference=sec_label,
        lead_time_vs_secondary_days=sec_lead, pre_onset_off_target_flag_days=off_target,
        peak_score=max((s.score for s in signals), default=0.0),
    )  # fmt: skip


def score_controls(lane_id: str, windows: list[list[DailySignal]]) -> ControlResult:
    """False-alarm rate over all control windows. Episodes are counted per window, so a flag at the
    end of one window and the start of the next are two episodes, not one."""
    flagged_days = episodes = days = 0
    for signals in windows:
        flags = [s.flagged for s in signals]
        days += len(flags)
        flagged_days += sum(flags)
        episodes += sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))
    return ControlResult(lane_id=lane_id, days=days, flagged_days=flagged_days,
                         false_alarm_rate=round(flagged_days / days, 3) if days else 0.0,
                         episodes=episodes)  # fmt: skip


def control_signals(
    events: list[DisruptionEvent], spec: BacktestSpec, lane_id: str
) -> list[list[DailySignal]]:
    """Signals for each control window, kept separate."""
    return [replay(events, lane_id, w.days(), spec.params) for w in spec.control_windows]
