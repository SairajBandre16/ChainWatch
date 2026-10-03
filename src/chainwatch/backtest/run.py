"""CLI: run the backtest defined in docs/backtest-spec.md.

    python -m chainwatch.backtest.run                                  # keyword baseline, all events
    python -m chainwatch.backtest.run --event red_sea_2023
    python -m chainwatch.backtest.run --extractor ollama:qwen2.5:3b    # LLM run (slow; see --prefilter)
    python -m chainwatch.backtest.run --scoring v2                     # post-hoc scoring (spec v2)

Steps: GDELT daily files (cached) -> logistics NewsItems -> extraction (cached per extractor and day)
-> daily replay per lane -> lead time / false-alarm metrics -> docs/metrics/backtest_<extractor>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import UTC, date, datetime

from chainwatch.backtest.engine import (
    BacktestSpec,
    Scoring,
    control_signals,
    load_spec,
    replay,
    score_controls,
    score_disruption,
)
from chainwatch.config import DOCS_DIR, RAW_DIR
from chainwatch.extraction.baseline import keyword_responder
from chainwatch.extraction.pipeline import load_records, run_extraction, save_records
from chainwatch.extraction.schemas import DisruptionEvent
from chainwatch.graph.linking import build_alias_table
from chainwatch.graph.reference import default_reference
from chainwatch.ingest.http_cache import CachedHttp
from chainwatch.ingest.models import NewsItem
from chainwatch.ingest.sources import fetch_gdelt_events
from chainwatch.llm import FakeLLMClient, LLMClient, get_llm_client

logger = logging.getLogger(__name__)
CACHE_DIR = RAW_DIR / "backtest"
METRICS_DIR = DOCS_DIR / "metrics"


def make_extractor(name: str) -> LLMClient:
    if name == "keyword":
        return FakeLLMClient(keyword_responder, model="keyword-baseline")
    provider, _, model = name.partition(":")
    return get_llm_client(provider=provider, model=model or None)


def mentions_graph_location(item: NewsItem) -> bool:
    """Cheap pre-filter for slow extractors: the headline names a place the graph knows."""
    text = f" {item.title.lower()} "
    aliases = build_alias_table(default_reference())
    return any(f" {a} " in text for a in aliases if len(a) >= 4)


def events_for_day(
    day: date, llm: LLMClient, http: CachedHttp, prefilter: bool, offline: bool
) -> tuple[list[DisruptionEvent], dict]:
    safe = llm.model.replace(":", "-")
    tag = f"{safe}{'__prefilter' if prefilter else ''}"
    path = CACHE_DIR / tag / f"{day.isoformat()}.jsonl"
    if path.exists():
        records = load_records(path)
        n_items = len(records)
    else:
        if offline and not http.is_cached(
            f"https://data.gdeltproject.org/events/{day:%Y%m%d}.export.CSV.zip"
        ):
            return [], {"day": str(day), "status": "missing"}
        items = fetch_gdelt_events(http, datetime(day.year, day.month, day.day, tzinfo=UTC))
        if not items:
            return [], {"day": str(day), "status": "no_items"}
        if prefilter:
            items = [i for i in items if mentions_graph_location(i)]
        records = run_extraction(items, llm)
        save_records(records, path)
        n_items = len(items)
    events = [e for r in records if r.status == "ok" for e in r.events]
    failed = sum(r.status == "failed" for r in records)
    return events, {"day": str(day), "status": "ok", "items": n_items, "events": len(events),
                    "failed": failed}  # fmt: skip


def run_backtest(spec: BacktestSpec, extractor: str, only: str | None, prefilter: bool,
                 offline: bool = False, scoring: Scoring = "v1") -> dict:  # fmt: skip
    llm = make_extractor(extractor)
    http = CachedHttp(offline=offline)
    all_events: list[DisruptionEvent] = []
    day_log = []
    for day in spec.all_days():
        events, info = events_for_day(day, llm, http, prefilter, offline)
        all_events += events
        day_log.append(info)
    ref = default_reference()
    lanes = [spec.primary_lane] + sorted(
        lane.lane_id for lane in ref.lanes.values()
        if lane.focus in spec.secondary_lane_focus and lane.lane_id != spec.primary_lane
    )  # fmt: skip

    results: dict = {"extractor": llm.model, "prefilter": prefilter, "scoring": scoring,
                     "disruptions": {}, "controls": {}}  # fmt: skip
    for d in spec.disruptions:
        if only and d.id != only:
            continue
        window = spec.window_for(d).days()
        per_lane = []
        for lane in lanes:
            signals = replay(all_events, lane, window, spec.params, set(d.target_nodes), ref,
                             scoring)  # fmt: skip
            per_lane.append(score_disruption(d, lane, signals).model_dump(mode="json"))
            if lane == spec.primary_lane:
                results["disruptions"].setdefault(d.id, {})["primary_signals"] = [
                    s.model_dump(mode="json") for s in signals
                ]
        primary = per_lane[0]
        leads = [r["lead_time_days"] for r in per_lane if not r["missed"]]
        results["disruptions"][d.id] |= {
            "primary": primary,
            "secondary_lanes": per_lane[1:],
            "lanes_warned": f"{len(leads)}/{len(per_lane)}",
            "median_lead_time_days": statistics.median(leads) if leads else None,
        }
    for lane in lanes[:1] if only else lanes:
        res = score_controls(lane, control_signals(all_events, spec, lane, scoring))
        results["controls"][lane] = res.model_dump()
    results["days"] = {
        "total": len(day_log),
        "missing": [x["day"] for x in day_log if x["status"] != "ok"],
        "items": sum(x.get("items", 0) for x in day_log),
        "events": sum(x.get("events", 0) for x in day_log),
        "failed_extractions": sum(x.get("failed", 0) for x in day_log),
    }
    return results


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=None, help="one disruption id; default all")
    parser.add_argument("--extractor", default="keyword", help="keyword | ollama:<model> | ...")
    parser.add_argument("--prefilter", action="store_true",
                        help="only extract items whose headline names a graph location")  # fmt: skip
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--scoring", choices=["v1", "v2"], default="v1",
                        help="v1 = frozen spec; v2 = post-hoc rules in docs/backtest-spec-v2.md")  # fmt: skip
    args = parser.parse_args()

    spec = load_spec()
    results = run_backtest(spec, args.extractor, args.event, args.prefilter, args.offline,
                           args.scoring)  # fmt: skip
    name = results["extractor"].replace(":", "-") + ("__prefilter" if args.prefilter else "")
    if args.scoring != "v1":
        name += f"__{args.scoring}"  # v1 keeps its original file name
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"backtest_{name}.json"
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

    print(f"Extractor: {results['extractor']}  days={results['days']}")
    for did, r in results["disruptions"].items():
        p = r["primary"]
        lead = "MISSED" if p["missed"] else f"{p['lead_time_days']:+d} days"
        print(f"{did}: primary {p['lane_id']} first warning {p['first_warning']} lead {lead}; "
              f"secondary ref lead {p['lead_time_vs_secondary_days']}; lanes warned "
              f"{r['lanes_warned']}, median lead {r['median_lead_time_days']}; "
              f"pre-onset off-target flag days {p['pre_onset_off_target_flag_days']}")  # fmt: skip
    for lane, c in results["controls"].items():
        if lane == spec.primary_lane:
            print(f"controls {lane}: {c['flagged_days']}/{c['days']} days flagged "
                  f"(false-alarm rate {c['false_alarm_rate']}), {c['episodes']} episodes")  # fmt: skip
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
