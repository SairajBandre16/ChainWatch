"""The one LLM interface every module uses, so models and providers are swappable.

Subclasses implement only `_complete()`. This base class adds disk caching, JSON parsing,
Pydantic validation, and a retry loop that feeds validation errors back to the model.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from chainwatch.llm.cache import DiskCache, make_key

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid answer after all retries."""


def extract_json(text: str) -> object:
    """Parse JSON from a model reply, tolerating ```json fences and chatter around the object."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in model output: {text[:200]!r}")


class LLMClient(ABC):
    """Base class: `generate()` for text, `generate_structured()` for validated Pydantic output."""

    provider: str = "base"

    def __init__(
        self, model: str, cache: DiskCache | None = None, temperature: float = 0.0
    ) -> None:
        self.model = model
        self.cache = cache
        # Temperature 0 by default: extraction should be as deterministic as the backend allows.
        self.temperature = temperature

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str:
        """Call the backend once and return raw text. No caching here."""

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        key = make_key(
            provider=self.provider,
            model=self.model,
            system=system,
            prompt=prompt,
            json_mode=json_mode,
            temperature=self.temperature,
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        text = self._complete(prompt, system, json_mode)
        if self.cache is not None:
            self.cache.set(key, text, meta={"provider": self.provider, "model": self.model})
        return text

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        max_retries: int = 2,
    ) -> T:
        """Ask for JSON matching `schema`; on parse/validation failure, retry with the error shown.

        The retry prompt differs from the original, so retries are cached separately and a bad
        first answer in the cache does not trap us in a loop.
        """
        current_prompt = prompt
        last_error = ""
        for attempt in range(max_retries + 1):
            raw = self.generate(current_prompt, system=system, json_mode=True)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("Invalid structured output (attempt %d): %s", attempt + 1, exc)
                current_prompt = (
                    f"{prompt}\n\nYour previous answer was invalid:\n{last_error[:800]}\n"
                    "Reply again with ONLY a JSON object that fixes these errors."
                )
        raise LLMError(f"No valid {schema.__name__} after {max_retries + 1} attempts: {last_error}")
