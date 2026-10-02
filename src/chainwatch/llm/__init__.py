"""Single entry point for LLM access. Use `get_llm_client()` everywhere."""

from __future__ import annotations

from chainwatch.config import CACHE_DIR, get_settings
from chainwatch.llm.base import LLMClient, LLMError, extract_json
from chainwatch.llm.cache import DiskCache
from chainwatch.llm.clients import FakeLLMClient, GeminiClient, GroqClient, OllamaClient

__all__ = [
    "DiskCache",
    "FakeLLMClient",
    "GeminiClient",
    "GroqClient",
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "extract_json",
    "get_llm_client",
]

DEFAULT_MODELS = {
    "fake": "fake-model",
    "ollama": "qwen2.5:3b",
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
}


def get_llm_client(
    provider: str | None = None, model: str | None = None, use_cache: bool = True
) -> LLMClient:
    """Build a client from args or settings. Cache lives in data/cache/llm/ (gitignored)."""
    settings = get_settings()
    provider = (provider or settings.llm_provider).lower()
    if model is None:
        # The configured model only applies to the configured provider.
        same = provider == settings.llm_provider
        model = settings.llm_model if same else DEFAULT_MODELS.get(provider, "")
    cache = DiskCache(CACHE_DIR / "llm") if use_cache else None

    if provider == "fake":
        return FakeLLMClient(model=model, cache=cache)
    if provider == "ollama":
        return OllamaClient(model, host=settings.ollama_host, cache=cache)
    if provider == "groq":
        return GroqClient(model, api_key=settings.groq_api_key, cache=cache)
    if provider == "gemini":
        return GeminiClient(model, api_key=settings.gemini_api_key, cache=cache)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
