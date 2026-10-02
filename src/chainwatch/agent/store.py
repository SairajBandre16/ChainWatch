"""Event store: the single source of truth for which disruption events exist.

The agent may only cite events that are in this store; `has()` is what the citation check uses.
Backed by a JSONL file of `DisruptionEvent`s (or built in memory for tests).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from chainwatch.config import PROCESSED_DIR, SAMPLE_DIR
from chainwatch.extraction.pipeline import load_records
from chainwatch.extraction.schemas import DisruptionEvent

STORE_PATH = PROCESSED_DIR / "event_store.jsonl"
SAMPLE_STORE_PATH = SAMPLE_DIR / "event_store_sample.jsonl"

# Words too generic to filter on: almost every event "disrupts" "trade" at "sea".
_FILLER = {"the", "and", "for", "with", "sea", "trade", "disruption", "disruptions", "event",
           "events", "recent", "news", "risk", "shipping", "lane", "lanes"}  # fmt: skip


class EventStore:
    def __init__(self, events: Iterable[DisruptionEvent] = ()) -> None:
        self._events: dict[str, DisruptionEvent] = {}
        self.add_many(events)

    # --- write ----------------------------------------------------------------------------

    def add_many(self, events: Iterable[DisruptionEvent]) -> None:
        for ev in events:
            self._events[ev.event_id] = ev

    def save(self, path: Path = STORE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for ev in self.all():
                fh.write(ev.model_dump_json() + "\n")

    # --- read -----------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> EventStore:
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return cls(DisruptionEvent.model_validate_json(line) for line in lines if line.strip())

    @classmethod
    def from_extractions(cls, path: Path) -> EventStore:
        return cls(ev for rec in load_records(path) if rec.status == "ok" for ev in rec.events)

    @classmethod
    def default(cls) -> EventStore:
        """Local store if one has been built, else the committed sample."""
        return cls.load(STORE_PATH if STORE_PATH.exists() else SAMPLE_STORE_PATH)

    def __len__(self) -> int:
        return len(self._events)

    def has(self, event_id: str) -> bool:
        return event_id in self._events

    def get(self, event_id: str) -> DisruptionEvent | None:
        return self._events.get(event_id)

    def all(self) -> list[DisruptionEvent]:
        """Newest first, then by id for a stable order."""
        return sorted(self._events.values(), key=lambda e: (-e.published.timestamp(), e.event_id))

    def search(
        self,
        text: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_severity: int = 1,
        limit: int = 20,
    ) -> list[DisruptionEvent]:
        """Filter by free text (location/summary/type), publication window and severity.

        Text matches if ANY keyword (3+ letters, minus filler words) appears. LLM agents tend to search
        whole phrases like "sea trade disruption"; exact-phrase matching returned nothing for those.
        """
        words = [w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in _FILLER]
        out = []
        for ev in self.all():
            if since and ev.published < since:
                continue
            if until and ev.published > until:
                continue
            if ev.severity < min_severity:
                continue
            haystack = f"{ev.location} {ev.summary} {ev.event_type.value}".lower()
            if words and not any(w in haystack for w in words):
                continue
            out.append(ev)
        return out[:limit]


def main() -> None:
    """Build the event store from an extraction output file.

    python -m chainwatch.agent.store data/processed/extractions/qwen2.5-3b__extract_v1.jsonl [--sample]
    """
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    store = EventStore.from_extractions(Path(args[0]))
    target = SAMPLE_STORE_PATH if "--sample" in sys.argv else STORE_PATH
    store.save(target)
    print(f"{len(store)} events -> {target}")


if __name__ == "__main__":
    main()
