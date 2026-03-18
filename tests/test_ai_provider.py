from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_provider import get_chat_model, get_model_identity


class DummyChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_default_model_identity_uses_openrouter(monkeypatch):
    monkeypatch.delenv("RLM_AI_PROVIDER", raising=False)
    monkeypatch.delenv("RLM_AI_MODEL", raising=False)

    identity = get_model_identity()

    assert identity.provider == "openrouter"
    assert identity.model == "anthropic/claude-sonnet-4.5"


def test_openrouter_chat_model_honours_env_model(monkeypatch):
    monkeypatch.setenv("RLM_AI_PROVIDER", "openrouter")
    monkeypatch.setenv("RLM_AI_MODEL", "openrouter/auto")
    module = ModuleType("langchain_openrouter")
    module.ChatOpenRouter = DummyChatModel
    monkeypatch.setitem(sys.modules, "langchain_openrouter", module)

    model = get_chat_model()

    assert isinstance(model, DummyChatModel)
    assert model.kwargs["model"] == "openrouter/auto"
    assert model.kwargs["temperature"] == 0.0
