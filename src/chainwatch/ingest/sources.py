"""News fetchers: supply-chain RSS feeds and the GDELT DOC 2.0 API.

Both return `NewsItem`s and go through `CachedHttp`, so they work offline once fetched.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import UTC, datetime
from time import struct_time
from urllib.parse import urlparse

import feedparser

from chainwatch.ingest.http_cache import CachedHttp
from chainwatch.ingest.models import NewsItem, item_id_for

logger = logging.getLogger(__name__)

# Free, public feeds focused on shipping, ports and logistics.
RSS_FEEDS: dict[str, str] = {
    "gCaptain": "https://gcaptain.com/feed/",
    "The Loadstar": "https://theloadstar.com/feed/",
    "Splash247": "https://splash247.com/feed/",
    "Supply Chain Dive": "https://www.supplychaindive.com/feeds/news/",
    "Hellenic Shipping News": "https://www.hellenicshippingnews.com/feed/",
    "FreightWaves": "https://www.freightwaves.com/news/feed",
}

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT boolean query: a logistics term AND a disruption term, English sources only.
GDELT_DEFAULT_QUERY = (
    '(shipping OR port OR freight OR "supply chain" OR container OR vessel) '
    "(disruption OR strike OR closure OR blockage OR congestion OR attack OR storm OR delay) "
    "sourcelang:english"
)

SUMMARY_MAX_CHARS = 600


def clean_html(text: str) -> str:
    """Strip tags and entities from feed summaries and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _struct_to_dt(value: struct_time | None) -> datetime | None:
    if not value:
        return None
    return datetime(*value[:6], tzinfo=UTC)


def parse_rss(content: bytes, source: str) -> list[NewsItem]:
    """Normalize one RSS/Atom document. Entries without a link or date are skipped."""
    feed = feedparser.parse(content)
    items: list[NewsItem] = []
    for entry in feed.entries:
        link = entry.get("link")
        published = _struct_to_dt(entry.get("published_parsed") or entry.get("updated_parsed"))
        if not link or not published:
            continue
        summary = clean_html(entry.get("summary", ""))[:SUMMARY_MAX_CHARS]
        items.append(
            NewsItem(
                id=item_id_for(link),
                title=clean_html(entry.get("title", "")),
                summary=summary,
                url=link,
                source=source,
                published=published,
                origin="rss",
                language=(feed.feed.get("language") or "en")[:2],
            )
        )
    return items


def fetch_rss(http: CachedHttp, feeds: dict[str, str] = RSS_FEEDS, refresh: bool = False):
    """Fetch every feed; one broken feed is logged and skipped, not fatal."""
    items: list[NewsItem] = []
    for name, url in feeds.items():
        try:
            items.extend(parse_rss(http.get(url, refresh=refresh), source=name))
        except Exception as exc:  # noqa: BLE001 - keep other feeds going
            logger.warning("RSS feed %s failed: %s", name, exc)
    return items


def parse_gdelt_doc(content: bytes) -> list[NewsItem]:
    """Normalize a GDELT DOC API `mode=artlist&format=json` response."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text.startswith("{"):
        # GDELT answers query errors with a plain-text message, not JSON.
        logger.warning("GDELT returned non-JSON: %s", text[:200])
        return []
    articles = json.loads(text).get("articles", [])
    items = []
    for art in articles:
        url = art.get("url")
        seen = art.get("seendate")  # e.g. 20240115T120000Z
        if not url or not seen:
            continue
        items.append(
            NewsItem(
                id=item_id_for(url),
                title=clean_html(art.get("title", "")),
                url=url,
                source=art.get("domain") or urlparse(url).netloc,
                published=datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC),
                origin="gdelt_doc",
                language=(art.get("language") or "English")[:2].lower(),
                source_country=art.get("sourcecountry") or None,
            )
        )
    return items


def gdelt_doc_params(
    query: str = GDELT_DEFAULT_QUERY,
    timespan: str | None = "7d",
    start: datetime | None = None,
    end: datetime | None = None,
    max_records: int = 250,
) -> dict[str, str]:
    """Build DOC API params. Use either a rolling `timespan` or explicit start/end (UTC)."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "datedesc",
    }
    if start and end:
        params["startdatetime"] = start.strftime("%Y%m%d%H%M%S")
        params["enddatetime"] = end.strftime("%Y%m%d%H%M%S")
    elif timespan:
        params["timespan"] = timespan
    return params


def fetch_gdelt_doc(http: CachedHttp, refresh: bool = False, **param_kwargs) -> list[NewsItem]:
    try:
        content = http.get(GDELT_DOC_URL, params=gdelt_doc_params(**param_kwargs), refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - GDELT is flaky; RSS still works
        logger.warning("GDELT DOC fetch failed: %s", exc)
        return []
    return parse_gdelt_doc(content)


# --- GDELT 1.0 daily event files (historical, back to 2013; used by the backtest) -------------

GDELT_EVENTS_URL = "https://data.gdeltproject.org/events/{day}.export.CSV.zip"

# Column positions in the 58-column GDELT 1.0 event export (tab separated, no header).
_COL = {"date": 1, "event_code": 26, "geo_name": 50, "geo_country": 51, "source_url": 57}

# A URL slug must contain one of these words to count as logistics news. Slugs are real headline
# text written by the publisher, so this filter is cheap. Whole-word matching avoids hits like
# "report" or "support" for "port".
LOGISTICS_WORDS = frozenset(
    {
        "ship",
        "ships",
        "shipping",
        "shipper",
        "shippers",
        "shipment",
        "shipments",
        "port",
        "ports",
        "canal",
        "vessel",
        "vessels",
        "freight",
        "cargo",
        "container",
        "containers",
        "containership",
        "tanker",
        "tankers",
        "maritime",
        "suez",
        "houthi",
        "houthis",
        "hormuz",
        "malacca",
        "panama",
        "logistics",
        "dock",
        "docks",
        "dockworkers",
        "strait",
        "seafarers",
        "maersk",
        "msc",
        "hapag",
        "cma",
        "cgm",
        "evergreen",
        "boxship",
    }
)
LOGISTICS_PHRASES = ("red-sea", "supply-chain", "bab-el-mandeb", "ever-given", "cape-of-good-hope")

_SLUG_STOPWORDS = {"news", "article", "story", "index", "html", "htm", "amp", "www"}


def headline_from_url(url: str) -> str:
    """Turn the longest URL path segment into words: '/mega-ship-blocks-suez' -> 'mega ship ...'.

    Only the publisher's own slug is used; nothing is invented.
    """
    segments = [s for s in urlparse(url).path.split("/") if s]
    if not segments:
        return ""
    best = max(segments, key=lambda s: s.count("-") + s.count("_"))
    best = re.sub(r"\.(s?html?|php|aspx?)$", "", best)
    words = [w for w in re.split(r"[-_+]", best) if w and not w.isdigit()]
    words = [w for w in words if w.lower() not in _SLUG_STOPWORDS and len(w) < 25]
    return " ".join(words)


def is_logistics_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(phrase in path for phrase in LOGISTICS_PHRASES):
        return True
    return any(word in LOGISTICS_WORDS for word in re.split(r"[^a-z0-9]+", path))


def parse_gdelt_events(zip_bytes: bytes, day: datetime) -> list[NewsItem]:
    """Read one daily event file; keep one item per logistics-related source URL."""
    import csv
    import io
    import zipfile

    items: dict[str, NewsItem] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                                delimiter="\t")  # fmt: skip
            for row in reader:
                if len(row) < 58:
                    continue
                url = row[_COL["source_url"]]
                if not url.startswith("http") or not is_logistics_url(url):
                    continue
                item_id = item_id_for(url)
                if item_id in items:
                    continue
                title = headline_from_url(url)
                if len(title.split()) < 3:
                    continue
                geo = row[_COL["geo_name"]]
                items[item_id] = NewsItem(
                    id=item_id,
                    title=title,
                    summary=f"GDELT event location: {geo}" if geo else "",
                    url=url,
                    source=urlparse(url).netloc.removeprefix("www."),
                    published=day,
                    origin="gdelt_event",
                    source_country=row[_COL["geo_country"]] or None,
                )
    return list(items.values())


def fetch_gdelt_events(http: CachedHttp, day: datetime) -> list[NewsItem]:
    """One UTC day of GDELT events (~5-15 MB zip, cached under data/raw/http/)."""
    url = GDELT_EVENTS_URL.format(day=day.strftime("%Y%m%d"))
    try:
        content = http.get(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDELT events %s failed: %s", day.date(), exc)
        return []
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return parse_gdelt_events(content, start)
