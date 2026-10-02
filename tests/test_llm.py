import httpx
import pytest
from pydantic import BaseModel

from chainwatch.llm import (
    DiskCache,
    FakeLLMClient,
    LLMError,
    OllamaClient,
    extract_json,
    get_llm_client,
)


class Point(BaseModel):
    x: int
    y: int


def test_fake_returns_canned_responses_in_order() -> None:
    llm = FakeLLMClient(["a", "b"])
    assert [llm.generate("p1"), llm.generate("p2"), llm.generate("p3")] == ["a", "b", "b"]
    assert len(llm.calls) == 3


def test_fake_accepts_responder_function() -> None:
    llm = FakeLLMClient(lambda prompt, system: prompt.upper())
    assert llm.generate("hello") == "HELLO"


def test_generate_structured_validates_schema() -> None:
    llm = FakeLLMClient(['```json\n{"x": 1, "y": 2}\n```'])
    assert llm.generate_structured("give point", Point) == Point(x=1, y=2)


def test_generate_structured_retries_on_invalid_json() -> None:
    llm = FakeLLMClient(["not json at all", '{"x": "oops"}', '{"x": 3, "y": 4}'])
    assert llm.generate_structured("give point", Point, max_retries=2) == Point(x=3, y=4)
    assert len(llm.calls) == 3
    # The retry prompt must show the model what went wrong.
    assert "previous answer was invalid" in llm.calls[1]["prompt"]


def test_generate_structured_raises_after_retries() -> None:
    llm = FakeLLMClient(["garbage"])
    with pytest.raises(LLMError):
        llm.generate_structured("give point", Point, max_retries=1)
    assert len(llm.calls) == 2


def test_cache_hit_skips_backend(tmp_path) -> None:
    llm = FakeLLMClient(["first", "second"], cache=DiskCache(tmp_path))
    assert llm.generate("same prompt") == "first"
    assert llm.generate("same prompt") == "first"  # served from disk
    assert len(llm.calls) == 1
    # A fresh client over the same cache directory also hits (persistence across runs).
    llm2 = FakeLLMClient(["never used"], cache=DiskCache(tmp_path))
    assert llm2.generate("same prompt") == "first"
    assert llm2.calls == []


def test_cache_key_depends_on_model_and_prompt(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    a = FakeLLMClient(["A"], model="m1", cache=cache)
    b = FakeLLMClient(["B"], model="m2", cache=cache)
    assert a.generate("p") == "A"
    assert b.generate("p") == "B"
    assert a.generate("other prompt") == "A"
    assert len(a.calls) == 2


def test_extract_json_handles_chatter() -> None:
    assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_factory_defaults_to_fake(monkeypatch) -> None:
    monkeypatch.setenv("CHAINWATCH_LLM_PROVIDER", "fake")
    assert isinstance(get_llm_client(use_cache=False), FakeLLMClient)
    with pytest.raises(ValueError):
        get_llm_client(provider="nope")


def test_cloud_clients_require_keys(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(LLMError):
        get_llm_client(provider="groq", use_cache=False)


def test_ollama_client_parses_response(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, body=json)
        return httpx.Response(
            200,
            json={"message": {"content": '{"x": 1, "y": 1}'}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaClient("qwen2.5:3b", host="http://ollama:11434")
    assert llm.generate_structured("p", Point) == Point(x=1, y=1)
    assert captured["url"] == "http://ollama:11434/api/chat"
    assert captured["body"]["format"] == "json"


def test_ollama_client_wraps_network_errors(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError):
        OllamaClient("m").generate("p")
