import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gpt_chat_uses_standard_openai_env_fallback(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT", "12")

    module = importlib.import_module("AgentPrune.llm.gpt_chat")
    module = importlib.reload(module)

    assert module.MINE_API_KEY == "standard-key"
    assert module.MINE_BASE_URL == "https://example.test/v1"
    assert module.OPENAI_REQUEST_TIMEOUT == 12.0


def test_extract_usage_tokens_accepts_raw_response_dict():
    module = importlib.import_module("AgentPrune.llm.gpt_chat")

    assert module._extract_usage_tokens(
        {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        }
    ) == (11, 7, 18)
