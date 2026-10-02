"""Concrete LLM backends: Ollama (local default), Groq and Gemini (optional free tiers), a fake.

Network clients use plain httpx: no heavy SDK dependency, and the request shape stays visible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import httpx

from chainwatch.llm.base import LLMClient, LLMError
from chainwatch.llm.cache import DiskCache

TIMEOUT = httpx.Timeout(180.0, connect=5.0)
SEED = 42


class OllamaClient(LLMClient):
    """Local models served by Ollama (https://ollama.com). Free and offline."""

    provider = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")

    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": SEED},
        }
        if json_mode:
            body["format"] = "json"  # Ollama constrains decoding to valid JSON
        try:
            resp = httpx.post(f"{self.host}/api/chat", json=body, timeout=TIMEOUT)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed ({self.host}, {self.model}): {exc}") from exc
        return resp.json()["message"]["content"]

    def is_available(self) -> bool:
        """True if the server is up and the model is pulled."""
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=3.0).json()
        except (httpx.HTTPError, ValueError):
            return False
        return any(m.get("name") == self.model for m in tags.get("models", []))


class GroqClient(LLMClient):
    """Groq free tier, OpenAI-compatible chat API. Needs GROQ_API_KEY."""

    provider = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, api_key: str, **kwargs) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        super().__init__(model, **kwargs)
        self.api_key = api_key

    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": SEED,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.post(self.url, json=body, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        return resp.json()["choices"][0]["message"]["content"]


class GeminiClient(LLMClient):
    """Google Gemini free tier via the REST API. Needs GEMINI_API_KEY."""

    provider = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, api_key: str, **kwargs) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        super().__init__(model, **kwargs)
        self.api_key = api_key

    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        url = f"{self.base_url}/{self.model}:generateContent"
        try:
            resp = httpx.post(
                url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=TIMEOUT
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


Responder = Callable[[str, str | None], str]


class FakeLLMClient(LLMClient):
    """Deterministic stand-in for tests and offline runs. Never touches the network.

    `responses` can be:
      - a list of strings, returned in order (the last one repeats once the list runs out), or
      - a function (prompt, system) -> str, for rule-based canned answers.
    Every call is recorded in `self.calls` so tests can assert on prompts and call counts.
    """

    provider = "fake"

    def __init__(
        self,
        responses: Sequence[str] | Responder | None = None,
        model: str = "fake-model",
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(model, cache=cache)
        self._responses = responses if responses is not None else ["{}"]
        self.calls: list[dict] = []

    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if callable(self._responses):
            return self._responses(prompt, system)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]
