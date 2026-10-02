from chainwatch import __version__
from chainwatch.config import ROOT_DIR, get_settings


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults_are_offline_safe(monkeypatch) -> None:
    monkeypatch.delenv("CHAINWATCH_LLM_PROVIDER", raising=False)
    assert get_settings().llm_provider == "fake"
    assert (ROOT_DIR / "pyproject.toml").exists()
