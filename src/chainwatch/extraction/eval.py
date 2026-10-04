"""Extraction evaluation: per-field precision / recall / F1 against hand labels.

Label file: `data/eval/extraction_labels.jsonl`, one item per line:
    {"news_id": "...", "is_disruption": true,
     "events": [{"event_type": "...", "location": "...", "country_code": "YE",
                 "port_code": null, "severity": 4}]}

Scoring (micro-averaged over items):
  - `is_disruption`: binary classification per item.
  - `event_type`, `country_code`, `port_code`, `location`: for each item, compare the SET of values in
    gold events with the SET in predicted events. TP = in both, FP = predicted only, FN = gold only.
    Sets make the score independent of event order and of how events are split.
  - `severity`: events are aligned by location; report exact accuracy, within-1 accuracy and MAE.
Locations are normalized (lowercase, "port of"/"the" removed) before comparison.

CLI:
    python -m chainwatch.extraction.eval score --pred data/processed/extractions/<file>.jsonl
    python -m chainwatch.extraction.eval template --n 120      # CSV for the owner to hand-label
    python -m chainwatch.extraction.eval import-csv            # labeled CSV -> labels JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from pydantic import BaseModel, Field

from chainwatch.config import EVAL_DIR, RAW_DIR, SAMPLE_DIR
from chainwatch.extraction.pipeline import load_records
from chainwatch.extraction.schemas import EventType, ExtractionRecord
from chainwatch.ingest.models import NewsItem, dedupe
from chainwatch.ingest.run import load_items

LABELS_PATH = EVAL_DIR / "extraction_labels.jsonl"
TEMPLATE_PATH = EVAL_DIR / "labeling_template.csv"
SET_FIELDS = ("event_type", "country_code", "port_code", "location")


class GoldEvent(BaseModel):
    event_type: EventType
    location: str
    country_code: str | None = None
    port_code: str | None = None
    severity: int | None = Field(default=None, ge=1, le=5)


class GoldItem(BaseModel):
    news_id: str
    is_disruption: bool
    events: list[GoldEvent] = Field(default_factory=list)
    # How the label was made (assisted tool): accepted / edited / manual; None = hand-labeled CSV.
    label_source: str | None = None


def normalize_location(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(the|port of|port|of)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _values(events: list, field: str) -> set[str]:
    out = set()
    for ev in events:
        value = getattr(ev, field)
        if value is None:
            continue
        value = str(value.value if isinstance(value, EventType) else value)
        out.add(normalize_location(value) if field == "location" else value.upper())
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "support": tp + fn,
    }


def score(gold: list[GoldItem], records: list[ExtractionRecord]) -> dict:
    """Score predictions on the items that have labels. Missing/failed predictions count as empty."""
    by_id = {r.news_id: r for r in records}
    counts = {f: [0, 0, 0] for f in ("is_disruption", *SET_FIELDS)}
    sev_pairs: list[tuple[int, int]] = []
    failed = missing = 0

    for item in gold:
        rec = by_id.get(item.news_id)
        if rec is None:
            missing += 1
        elif rec.status == "failed":
            failed += 1
        pred_flag = bool(rec and rec.status == "ok" and rec.is_disruption)
        pred_events = rec.events if rec and rec.status == "ok" else []

        c = counts["is_disruption"]
        c[0] += pred_flag and item.is_disruption
        c[1] += pred_flag and not item.is_disruption
        c[2] += item.is_disruption and not pred_flag

        for field in SET_FIELDS:
            g, p = _values(item.events, field), _values(pred_events, field)
            counts[field][0] += len(g & p)
            counts[field][1] += len(p - g)
            counts[field][2] += len(g - p)

        pred_by_loc = {normalize_location(e.location): e.severity for e in pred_events}
        for ev in item.events:
            loc = normalize_location(ev.location)
            if ev.severity is not None and loc in pred_by_loc:
                sev_pairs.append((ev.severity, pred_by_loc[loc]))

    result = {"items": len(gold), "failed": failed, "missing": missing}
    result["fields"] = {f: _prf(*v) for f, v in counts.items()}
    if sev_pairs:
        diffs = [abs(g - p) for g, p in sev_pairs]
        result["severity"] = {
            "aligned_events": len(sev_pairs),
            "exact": round(sum(d == 0 for d in diffs) / len(diffs), 3),
            "within_1": round(sum(d <= 1 for d in diffs) / len(diffs), 3),
            "mae": round(sum(diffs) / len(diffs), 3),
        }
    return result


def load_gold(path: Path = LABELS_PATH) -> list[GoldItem]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldItem.model_validate_json(line) for line in lines if line.strip()]


# --- labeling workflow ------------------------------------------------------------------------

TEMPLATE_COLUMNS = [
    "news_id", "source", "published", "title", "text", "url",
    "is_disruption", "event_type", "location", "country_code", "port_code", "severity", "notes",
    "label_source", "draft_model",
]  # fmt: skip
# `label_source` / `draft_model` are filled by the assisted labeling tool (label_assist.py).
# Rows marked DRAFT_ONLY are unconfirmed model drafts from a dry run and are never imported.
DRAFT_ONLY = "draft_unconfirmed"


def label_pool() -> list[NewsItem]:
    """Items to label: the local news store if present, else the committed sample."""
    store = RAW_DIR / "news_store.jsonl"
    items = load_items(store) if store.exists() else []
    return dedupe(items + load_items(SAMPLE_DIR / "news_sample.jsonl"))


def make_template(n: int = 120, path: Path = TEMPLATE_PATH, seed: int = 42) -> int:
    """Write a CSV with one row per item and empty label columns. Shuffled so labeling order does
    not follow source or date. No model predictions are pre-filled, to avoid anchoring the labeler."""
    items = label_pool()
    random.Random(seed).shuffle(items)
    items = items[:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads UTF-8
        writer = csv.DictWriter(fh, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "news_id": item.id,
                    "source": item.source,
                    "published": item.published.date().isoformat(),
                    "title": item.title,
                    "text": item.summary,
                    "url": item.url,
                }
            )
    return len(items)


def _blank(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def import_csv(csv_path: Path = TEMPLATE_PATH, out_path: Path = LABELS_PATH) -> int:
    """Labeled CSV -> labels JSONL. Rows with an empty `is_disruption` are skipped (not labeled yet).
    For an item with several events, repeat the row (same news_id) once per extra event."""
    items: dict[str, GoldItem] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            flag = (row.get("is_disruption") or "").strip().lower()
            if not flag or (row.get("label_source") or "").strip() == DRAFT_ONLY:
                continue
            news_id = row["news_id"].strip()
            item = items.setdefault(
                news_id,
                GoldItem(
                    news_id=news_id,
                    is_disruption=flag in {"1", "y", "yes", "true"},
                    label_source=_blank(row.get("label_source")),
                ),
            )
            if item.is_disruption and _blank(row.get("event_type")):
                severity = _blank(row.get("severity"))
                item.events.append(
                    GoldEvent(
                        event_type=row["event_type"].strip(),
                        location=row.get("location", "").strip(),
                        country_code=_blank(row.get("country_code")),
                        port_code=_blank(row.get("port_code")),
                        severity=int(severity) if severity else None,
                    )
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items.values():
            fh.write(item.model_dump_json() + "\n")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_score = sub.add_parser("score")
    p_score.add_argument("--pred", type=Path, required=True)
    p_score.add_argument("--labels", type=Path, default=LABELS_PATH)
    p_tpl = sub.add_parser("template")
    p_tpl.add_argument("--n", type=int, default=120)
    sub.add_parser("import-csv")
    args = parser.parse_args()

    if args.cmd == "template":
        print(f"Wrote {make_template(args.n)} rows -> {TEMPLATE_PATH}")
    elif args.cmd == "import-csv":
        print(f"Imported {import_csv()} labeled items -> {LABELS_PATH}")
    else:
        gold = load_gold(args.labels)
        if len(gold) < 100:
            print(f"Only {len(gold)} labeled items (need 100+). Results are PENDING LABELS.")
            if not gold:
                return
        records = load_records(args.pred)
        print(json.dumps(score(gold, records), indent=1))
        # Drafted-and-accepted labels may flatter the drafting model (D16): also score the rest.
        unanchored = [g for g in gold if g.label_source != "accepted"]
        if len(unanchored) < len(gold):
            print(f"\nSubset without accepted drafts ({len(unanchored)} items):")
            print(json.dumps(score(unanchored, records), indent=1))


if __name__ == "__main__":
    main()
