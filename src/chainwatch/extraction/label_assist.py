"""Assisted labeling: an LLM drafts a label, the owner confirms, edits, or retypes it.

For each unlabeled row of `data/eval/labeling_template.csv` the tool shows the news text, runs the
normal extraction pipeline (same prompt and schema as step 1.3) to draft a label, and asks:

    [a]ccept  [e]dit  [m]anual (retype every field, no draft defaults)  [s]kip  [q]uit

Only what the owner confirms is written. Rules:
  - A row whose `is_disruption` is already filled is a human label and is never touched.
  - The CSV is saved after every confirmed item, so quitting and re-running resumes where you left off.
  - `label_source` records how each final label was made: `accepted` (final == draft), `edited`
    (owner changed the draft) or `manual` (owner retyped every field with no defaults). `draft_model`
    records which model's draft was on screen (blank only when no draft was shown, e.g. the
    extraction failed). The draft is visible before choosing, so `manual` is not blind. This lets the write-up disclose the process and lets the scorer
    report metrics on the subset the drafting model never touched.

Why disclose: a drafted label anchors the labeler toward the draft, which flatters the drafting model
when it is later scored against those labels (see D16).

CLI:
    python -m chainwatch.extraction.label_assist                       # qwen2.5:3b via Ollama
    python -m chainwatch.extraction.label_assist --limit 10
    python -m chainwatch.extraction.label_assist --dry-run --limit 5   # drafts to a preview CSV only
    python -m chainwatch.extraction.label_assist stats                 # label_source counts
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from chainwatch.config import EVAL_DIR
from chainwatch.extraction.eval import (
    DRAFT_ONLY,
    TEMPLATE_COLUMNS,
    TEMPLATE_PATH,
    GoldEvent,
    GoldItem,
)
from chainwatch.extraction.pipeline import DEFAULT_PROMPT, extract_item
from chainwatch.extraction.prompts import load_prompt
from chainwatch.extraction.schemas import EventType, ExtractionRecord
from chainwatch.ingest.models import NewsItem
from chainwatch.llm import LLMClient

PREVIEW_PATH = EVAL_DIR / "label_assist_preview.csv"
EVENT_COLUMNS = ("event_type", "location", "country_code", "port_code", "severity")
LABEL_SOURCES = ("accepted", "edited", "manual")

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
Drafter = Callable[[dict[str, str]], ExtractionRecord]


# --- CSV io -----------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Older templates lack the new columns; add them empty so every row has the same keys.
    return [{col: (row.get(col) or "") for col in TEMPLATE_COLUMNS} for row in rows]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write to a temp file then rename, so a crash mid-write never leaves a half-written CSV."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads UTF-8
        writer = csv.DictWriter(fh, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def is_labeled(row: dict[str, str]) -> bool:
    return bool(row.get("is_disruption", "").strip())


def pending_ids(rows: list[dict[str, str]]) -> list[str]:
    """news_ids in file order where no row carries a label yet."""
    labeled = {r["news_id"] for r in rows if is_labeled(r)}
    seen: list[str] = []
    for r in rows:
        if r["news_id"] not in labeled and r["news_id"] not in seen:
            seen.append(r["news_id"])
    return seen


def save_label(
    path: Path, news_id: str, label: GoldItem, source: str, draft_model: str, note: str = ""
) -> None:
    """Replace the unlabeled row(s) for `news_id` with one row per event, re-reading the file first
    so a label added elsewhere in the meantime is never overwritten."""
    rows = read_rows(path)
    targets = [i for i, r in enumerate(rows) if r["news_id"] == news_id]
    if not targets:
        raise KeyError(f"news_id {news_id} not in {path}")
    if any(is_labeled(rows[i]) for i in targets):
        raise ValueError(f"news_id {news_id} already has a human label; refusing to overwrite")
    base = rows[targets[0]]
    new_rows = label_to_rows(base, label, source, draft_model, note)
    rows = rows[: targets[0]] + new_rows + [r for i, r in enumerate(rows) if i > targets[0]
                                            and i not in targets]  # fmt: skip
    write_rows(path, rows)


def label_to_rows(
    base: dict[str, str], label: GoldItem, source: str, draft_model: str, note: str = ""
) -> list[dict[str, str]]:
    meta = {
        "is_disruption": "yes" if label.is_disruption else "no",
        "label_source": source,
        "draft_model": draft_model,
        "notes": note or base.get("notes", ""),
    }
    blank = dict.fromkeys(EVENT_COLUMNS, "")
    if not label.is_disruption or not label.events:
        return [base | blank | meta]
    return [base | _event_cells(ev) | meta for ev in label.events]


def _event_cells(ev: GoldEvent) -> dict[str, str]:
    return {
        "event_type": ev.event_type.value,
        "location": ev.location,
        "country_code": ev.country_code or "",
        "port_code": ev.port_code or "",
        "severity": str(ev.severity) if ev.severity is not None else "",
    }


# --- drafting ---------------------------------------------------------------------------------


def row_to_item(row: dict[str, str]) -> NewsItem:
    """Rebuild the NewsItem from the template row. Same title/text/date/source as the original
    item, so the prompt (and therefore the LLM cache key) matches the step 1.3 runs."""
    return NewsItem(
        id=row["news_id"],
        title=row["title"],
        summary=row["text"],
        url=row["url"] or f"urn:news:{row['news_id']}",
        source=row["source"],
        published=datetime.fromisoformat(row["published"]).replace(tzinfo=UTC),
        origin="manual",
    )


def make_drafter(llm: LLMClient, prompt_name: str = DEFAULT_PROMPT) -> Drafter:
    template = load_prompt(prompt_name)
    return lambda row: extract_item(row_to_item(row), llm, template)


def record_to_label(news_id: str, rec: ExtractionRecord) -> GoldItem | None:
    """A draft in the gold-label shape, or None if extraction failed."""
    if rec.status != "ok":
        return None
    events = [
        GoldEvent(
            event_type=ev.event_type,
            location=ev.location,
            country_code=ev.country_code,
            port_code=ev.port_code,
            severity=ev.severity,
        )
        for ev in rec.events
    ]
    return GoldItem(news_id=news_id, is_disruption=rec.is_disruption, events=events)


# --- prompting --------------------------------------------------------------------------------


def format_label(label: GoldItem) -> str:
    if not label.is_disruption:
        return "  is_disruption: no"
    lines = ["  is_disruption: yes"]
    for i, ev in enumerate(label.events, 1):
        cells = _event_cells(ev)
        lines.append(f"  event {i}: " + ", ".join(f"{k}={cells[k] or '-'}" for k in EVENT_COLUMNS))
    if not label.events:
        lines.append("  (no events listed)")
    return "\n".join(lines)


def _ask(prompt: str, default: str, ask: InputFn) -> str:
    """Prompt with an optional default; Enter keeps the default, '-' clears it."""
    shown = f" [{default}]" if default else ""
    answer = ask(f"{prompt}{shown}: ").strip()
    if answer == "-":
        return ""
    return answer or default


def _ask_valid(prompt: str, default: str, ask: InputFn, say: OutputFn, check) -> str:
    while True:
        value = _ask(prompt, default, ask)
        error = check(value)
        if error is None:
            return value
        say(f"    ! {error}")


def _check_flag(v: str) -> str | None:
    return None if v.lower() in {"y", "yes", "n", "no"} else "answer y or n"


def _check_type(v: str) -> str | None:
    return None if v in EventType._value2member_map_ else "one of: " + ", ".join(EventType)


def _check_count(v: str) -> str | None:
    return None if v.isdigit() and 1 <= int(v) <= 9 else "a number 1-9"


def prompt_label(news_id: str, draft: GoldItem | None, ask: InputFn, say: OutputFn) -> GoldItem:
    """Ask for every field, using the draft (if any) as defaults. Re-asks until the event validates."""
    d_flag = "" if draft is None else ("yes" if draft.is_disruption else "no")
    flag = _ask_valid("  is_disruption (y/n)", d_flag, ask, say, _check_flag).lower()
    if flag.startswith("n"):
        return GoldItem(news_id=news_id, is_disruption=False)

    d_events = draft.events if draft else []
    n = int(_ask_valid("  number of events", str(max(len(d_events), 1)), ask, say, _check_count))
    events = []
    for i in range(n):
        d = _event_cells(d_events[i]) if i < len(d_events) else dict.fromkeys(EVENT_COLUMNS, "")
        say(f"  event {i + 1}:")
        while True:
            cells = {
                "event_type": _ask_valid("    event_type", d["event_type"], ask, say, _check_type),
                "location": _ask("    location", d["location"], ask),
                "country_code": _ask("    country_code", d["country_code"], ask).upper(),
                "port_code": _ask("    port_code", d["port_code"], ask).upper().replace(" ", ""),
                "severity": _ask("    severity 1-5", d["severity"], ask),
            }
            try:
                events.append(_cells_to_event(cells))
                break
            except (ValidationError, ValueError) as exc:
                say(f"    ! invalid event, please re-enter: {_short_error(exc)}")
                d = cells
    return GoldItem(news_id=news_id, is_disruption=True, events=events)


def _cells_to_event(cells: dict[str, str]) -> GoldEvent:
    if not cells["location"]:
        raise ValueError("location is required")
    cc, port, sev = cells["country_code"], cells["port_code"], cells["severity"]
    if cc and (len(cc) != 2 or not cc.isalpha()):
        raise ValueError("country_code must be 2 letters")
    if port and (len(port) != 5 or not port.isalnum()):
        raise ValueError("port_code must be a 5-character UN/LOCODE")
    return GoldEvent(
        event_type=cells["event_type"],
        location=cells["location"],
        country_code=cc or None,
        port_code=port or None,
        severity=int(sev) if sev else None,
    )


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())
    return str(exc)


def _same(a: GoldItem, b: GoldItem | None) -> bool:
    return b is not None and a.model_dump(exclude={"news_id"}) == b.model_dump(exclude={"news_id"})


# --- session ----------------------------------------------------------------------------------


def run_session(
    path: Path,
    drafter: Drafter,
    draft_model: str,
    ask: InputFn = input,
    say: OutputFn = print,
    limit: int | None = None,
) -> Counter:
    """Interactive loop. Returns counts of outcomes (accepted/edited/manual/skipped)."""
    rows = read_rows(path)
    todo = pending_ids(rows)
    total_labeled = len({r["news_id"] for r in rows if is_labeled(r)})
    say(f"{total_labeled} items labeled, {len(todo)} to go. Draft model: {draft_model}")
    counts: Counter = Counter()
    by_id = {r["news_id"]: r for r in reversed(rows)}  # first row per id

    for news_id in todo[:limit] if limit else todo:
        row = by_id[news_id]
        say("\n" + "=" * 78)
        say(f"[{row['source']} | {row['published']}] {row['title']}")
        say(row["url"])
        say("-" * 78)
        say(row["text"] or "(no text)")
        say("-" * 78)

        draft = record_to_label(news_id, drafter(row))
        say("Draft:" if draft else "Draft: extraction failed (no draft for this item)")
        if draft:
            say(format_label(draft))
        choices = "[a]ccept [e]dit [m]anual [s]kip [q]uit" if draft else "[m]anual [s]kip [q]uit"
        while True:
            choice = ask(f"{choices}: ").strip().lower()[:1]
            if choice in ("e", "a") and draft is None:
                choice = ""
            if choice in {"a", "e", "m", "s", "q"}:
                break

        if choice == "q":
            break
        if choice == "s":
            counts["skipped"] += 1
            continue
        if choice == "a":
            label, source = draft, "accepted"
        elif choice == "e":
            label = prompt_label(news_id, draft, ask, say)
            # An "edit" that changes nothing is an acceptance; record it honestly as such.
            source = "accepted" if _same(label, draft) else "edited"
        else:
            label, source = prompt_label(news_id, None, ask, say), "manual"
        note = ask("  note (optional): ").strip()
        save_label(path, news_id, label, source, draft_model if draft else "", note)
        counts[source] += 1
        say(f"  saved ({source})")
    return counts


def dry_run(path: Path, out: Path, drafter: Drafter, draft_model: str, limit: int = 5) -> int:
    """Draft labels for the first `limit` pending items into a SEPARATE preview CSV. Nothing is
    confirmed, the real template is untouched, and import-csv ignores `draft_unconfirmed` rows."""
    rows = read_rows(path)
    by_id = {r["news_id"]: r for r in reversed(rows)}
    preview: list[dict[str, str]] = []
    for news_id in pending_ids(rows)[:limit]:
        draft = record_to_label(news_id, drafter(by_id[news_id]))
        if draft is None:
            preview.append(by_id[news_id] | {"label_source": DRAFT_ONLY, "notes": "draft failed"})
        else:
            preview += label_to_rows(by_id[news_id], draft, DRAFT_ONLY, draft_model)
    write_rows(out, preview)
    return len(preview)


def stats(path: Path) -> dict[str, int]:
    """Labeled items per label_source (blank = labeled by hand in the CSV, without this tool)."""
    first: dict[str, str] = {}
    for r in read_rows(path):
        if is_labeled(r):
            first.setdefault(r["news_id"], r["label_source"] or "hand_csv")
    return dict(Counter(first.values()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("cmd", nargs="?", default="label", choices=["label", "stats"])
    parser.add_argument("--csv", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help=f"write drafts to {PREVIEW_PATH}")
    args = parser.parse_args()

    if args.cmd == "stats":
        print(stats(args.csv))
        return

    from chainwatch.extraction.run import make_client  # lazy: keeps imports light for `stats`

    llm = make_client(args.provider, args.model)
    drafter = make_drafter(llm, args.prompt)
    if args.dry_run:
        n = dry_run(args.csv, PREVIEW_PATH, drafter, llm.model, limit=args.limit or 5)
        print(f"Wrote {n} unconfirmed draft rows -> {PREVIEW_PATH} (template untouched)")
        return
    try:
        counts = run_session(args.csv, drafter, llm.model, limit=args.limit)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Everything confirmed so far is saved; re-run to resume.")
        return
    print(f"\nThis session: {dict(counts)}. Totals: {stats(args.csv)}")


if __name__ == "__main__":
    main()
