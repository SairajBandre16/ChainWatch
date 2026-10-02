"""Tiny disk cache for LLM responses.

Why: free-tier quotas are small and local models are slow; caching also makes runs reproducible.
The key hashes everything that changes the answer (provider, model, prompts, options).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_key(**parts: Any) -> str:
    """Stable SHA-256 key from keyword parts (sorted, JSON-encoded)."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per entry, sharded by the first two hex chars to keep folders small."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def set(self, key: str, response: str, meta: dict[str, Any] | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"response": response, "meta": meta or {}}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
