"""HTTP GET with a disk cache under data/raw/http/, so every fetcher can run offline.

Modes:
  - online (default): use the cache if present (unless refresh=True), else download and store.
  - offline: cache only; a miss raises `OfflineCacheMiss` instead of touching the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from chainwatch.config import RAW_DIR
from chainwatch.llm.cache import make_key

USER_AGENT = "ChainWatch/0.1 (research project; contact via GitHub)"


class OfflineCacheMiss(RuntimeError):
    """Offline mode was requested but the URL has never been fetched."""


class CachedHttp:
    def __init__(
        self,
        cache_dir: Path = RAW_DIR / "http",
        offline: bool = False,
        min_interval: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        # GDELT asks for at most one request every 5 seconds; other hosts can use 0.
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request = 0.0

    def _paths(
        self, url: str, params: dict | None, cache_extra: dict | None = None
    ) -> tuple[Path, Path]:
        # cache_extra changes the key without being sent (e.g. a date to cache a forecast per day).
        key = make_key(
            url=url, params=params or {}, **({"extra": cache_extra} if cache_extra else {})
        )
        return self.cache_dir / f"{key}.body", self.cache_dir / f"{key}.meta.json"

    def is_cached(self, url: str, params: dict | None = None) -> bool:
        return self._paths(url, params)[0].exists()

    def get(
        self,
        url: str,
        params: dict | None = None,
        refresh: bool = False,
        cache_extra: dict | None = None,
    ) -> bytes:
        body_path, meta_path = self._paths(url, params, cache_extra)
        if body_path.exists() and not refresh:
            return body_path.read_bytes()
        if self.offline:
            raise OfflineCacheMiss(f"Not cached (offline mode): {url} {params or ''}")

        content = self._download(url, params)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        body_path.write_bytes(content)
        meta = {"url": url, "params": params, "fetched_at": time.time(), "bytes": len(content)}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return content

    def _download(self, url: str, params: dict | None) -> bytes:
        delay = max(self.min_interval, 2.0)
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            resp = httpx.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
            )
            # 429 = rate limited, 5xx = transient; back off exponentially and retry.
            transient = resp.status_code == 429 or resp.status_code >= 500
            if transient and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError("unreachable")  # pragma: no cover
