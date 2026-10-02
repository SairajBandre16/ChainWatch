"""Central settings loaded from environment variables (and an optional `.env` file).

Everything has a safe default so the project runs offline with no configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels above src/chainwatch/
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DIR = DATA_DIR / "cache"
DOCS_DIR = ROOT_DIR / "docs"

# load_dotenv does not override variables already set in the real environment.
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    llm_provider: str = field(default_factory=lambda: _env("CHAINWATCH_LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: _env("CHAINWATCH_LLM_MODEL", "qwen2.5:3b"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))


def get_settings() -> Settings:
    """Read settings fresh each call so tests can monkeypatch the environment."""
    return Settings()
