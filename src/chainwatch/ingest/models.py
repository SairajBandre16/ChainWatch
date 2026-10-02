"""`NewsItem`: the normalized shape every news source is converted into."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

NewsOrigin = Literal["rss", "gdelt_doc", "gdelt_event", "manual"]


def item_id_for(url: str) -> str:
    """Stable 16-hex id from the URL, so the same article from two feeds dedupes."""
    return hashlib.sha1(url.strip().lower().encode("utf-8")).hexdigest()[:16]


class NewsItem(BaseModel):
    id: str
    title: str
    summary: str = ""
    url: str
    source: str = Field(description="Publisher or feed name, e.g. 'gCaptain' or a domain")
    published: datetime = Field(description="Publication time, always timezone-aware UTC")
    origin: NewsOrigin
    language: str = "en"
    source_country: str | None = None

    @field_validator("published")
    @classmethod
    def _force_utc(cls, value: datetime) -> datetime:
        # Naive timestamps are assumed UTC; mixing naive and aware datetimes breaks sorting.
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @property
    def text(self) -> str:
        """What the extractor reads: title plus summary."""
        return f"{self.title}\n\n{self.summary}".strip()


def dedupe(items: list[NewsItem]) -> list[NewsItem]:
    """Drop repeat ids (same URL) keeping the first, then sort newest first."""
    seen: dict[str, NewsItem] = {}
    for item in items:
        seen.setdefault(item.id, item)
    return sorted(seen.values(), key=lambda i: i.published, reverse=True)
