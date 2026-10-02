import io
import json
import zipfile
from datetime import UTC, datetime

import httpx
import pytest

from chainwatch.config import SAMPLE_DIR
from chainwatch.ingest.http_cache import CachedHttp, OfflineCacheMiss
from chainwatch.ingest.models import NewsItem, dedupe, item_id_for
from chainwatch.ingest.run import load_items
from chainwatch.ingest.sources import (
    clean_html,
    fetch_gdelt_doc,
    fetch_rss,
    headline_from_url,
    is_logistics_url,
    parse_gdelt_doc,
    parse_gdelt_events,
)

RSS_XML = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test Feed</title><language>en-US</language>
<item><title>Port strike at Rotterdam</title><link>https://ex.com/a</link>
<pubDate>Fri, 02 Oct 2026 19:02:06 +0000</pubDate>
<description>&lt;p&gt;Dock workers &amp;amp; unions walk out.&lt;/p&gt;</description></item>
<item><title>No date item</title><link>https://ex.com/b</link></item>
</channel></rss>"""

GDELT_DOC_JSON = json.dumps(
    {
        "articles": [
            {
                "url": "https://news.example/red-sea-attack",
                "title": "Ship attacked in Red Sea",
                "seendate": "20240115T120000Z",
                "domain": "news.example",
                "language": "English",
                "sourcecountry": "United Kingdom",
            }
        ]
    }
).encode()


def _http_returning(monkeypatch, content: bytes, status: int = 200) -> list[str]:
    calls: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None, follow_redirects=None):
        calls.append(url)
        return httpx.Response(status, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_cached_http_downloads_once_then_works_offline(tmp_path, monkeypatch) -> None:
    calls = _http_returning(monkeypatch, RSS_XML)
    online = CachedHttp(cache_dir=tmp_path)
    assert online.get("https://ex.com/feed") == RSS_XML
    assert online.get("https://ex.com/feed") == RSS_XML
    assert len(calls) == 1

    def no_network(*a, **k):
        raise AssertionError("network used in offline mode")

    monkeypatch.setattr(httpx, "get", no_network)
    offline = CachedHttp(cache_dir=tmp_path, offline=True)
    assert offline.get("https://ex.com/feed") == RSS_XML
    with pytest.raises(OfflineCacheMiss):
        offline.get("https://ex.com/other")


def test_cached_http_retries_on_429(tmp_path, monkeypatch) -> None:
    responses = iter([429, 429, 200])
    monkeypatch.setattr("time.sleep", lambda s: None)

    def fake_get(url, **kwargs):
        return httpx.Response(next(responses), content=b"ok", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert CachedHttp(cache_dir=tmp_path).get("https://ex.com") == b"ok"


def test_rss_parsing_normalizes_items(tmp_path, monkeypatch) -> None:
    _http_returning(monkeypatch, RSS_XML)
    items = fetch_rss(CachedHttp(cache_dir=tmp_path), feeds={"Test": "https://ex.com/feed"})
    assert len(items) == 1  # the undated entry is skipped
    item = items[0]
    assert item.title == "Port strike at Rotterdam"
    assert item.summary == "Dock workers & unions walk out."
    assert item.published == datetime(2026, 10, 2, 19, 2, 6, tzinfo=UTC)
    assert item.source == "Test" and item.origin == "rss"


def test_broken_feed_does_not_crash(tmp_path, monkeypatch) -> None:
    _http_returning(monkeypatch, b"", status=404)
    assert fetch_rss(CachedHttp(cache_dir=tmp_path), feeds={"Bad": "https://ex.com/x"}) == []


def test_gdelt_doc_parsing(tmp_path, monkeypatch) -> None:
    _http_returning(monkeypatch, GDELT_DOC_JSON)
    items = fetch_gdelt_doc(CachedHttp(cache_dir=tmp_path))
    assert len(items) == 1
    assert items[0].published == datetime(2024, 1, 15, 12, tzinfo=UTC)
    assert items[0].origin == "gdelt_doc"
    # GDELT's rate-limit message is plain text, not JSON: no crash, no items.
    assert parse_gdelt_doc(b"Please limit requests to one every 5 seconds") == []


def _gdelt_events_zip(urls: list[str]) -> bytes:
    rows = []
    for url in urls:
        row = [""] * 58
        row[1], row[50], row[51], row[57] = "20210323", "Suez, Egypt", "EG", url
        rows.append("\t".join(row))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20210323.export.CSV", "\n".join(rows))
    return buf.getvalue()


def test_gdelt_events_filters_logistics_and_dedupes() -> None:
    urls = [
        "https://a.com/2021/03/23/mega-container-ship-hard-aground-in-suez-canal/",
        "https://a.com/2021/03/23/mega-container-ship-hard-aground-in-suez-canal/",
        "https://b.com/sports/local-team-wins-the-cup",
        "https://c.com/news/annual-report-shows-strong-support-for-mayor",
    ]
    items = parse_gdelt_events(_gdelt_events_zip(urls), datetime(2021, 3, 23, tzinfo=UTC))
    assert [i.title for i in items] == ["mega container ship hard aground in suez canal"]
    assert items[0].source_country == "EG"


def test_url_helpers() -> None:
    assert headline_from_url("https://x.com/news/ever-given-stuck.html") == "ever given stuck"
    assert is_logistics_url("https://x.com/red-sea-crisis-deepens")
    assert not is_logistics_url("https://x.com/report-on-sports-support")
    assert clean_html("<b>A&amp;B</b>  c") == "A&B c"


def test_dedupe_by_url_and_sort() -> None:
    def mk(url: str, day: int) -> NewsItem:
        return NewsItem(
            id=item_id_for(url),
            title=url,
            url=url,
            source="s",
            published=datetime(2024, 1, day),
            origin="manual",
        )

    items = dedupe([mk("https://x/1", 1), mk("https://x/2", 3), mk("https://X/1", 2)])
    assert [i.url for i in items] == ["https://x/2", "https://x/1"]
    assert items[0].published.tzinfo is not None


def test_committed_sample_has_50_plus_items() -> None:
    items = load_items(SAMPLE_DIR / "news_sample.jsonl")
    assert len(items) >= 50
    assert len({i.source for i in items}) >= 3
