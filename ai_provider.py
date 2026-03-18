"""
Provider abstraction for RLM chat models.

Defaults to OpenRouter so the LangGraph nodes can stay standard LangChain chat
models while the concrete model remains env-configurable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_XAI_MODEL = "grok-beta"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str


def get_model_identity(provider: str | None = None, model: str | None = None) -> ModelIdentity:
    backend = (provider or os.environ.get("RLM_AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if backend == "xai":
        default_model = DEFAULT_XAI_MODEL
    else:
        backend = "openrouter"
        default_model = DEFAULT_OPENROUTER_MODEL

    model_name = (model or os.environ.get("RLM_AI_MODEL") or default_model).strip()
    return ModelIdentity(provider=backend, model=model_name or default_model)


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    identity = get_model_identity(provider=provider, model=model)
    temp = float(os.environ.get("RLM_AI_TEMPERATURE", "0") if temperature is None else temperature)
    max_retries = int(os.environ.get("RLM_AI_MAX_RETRIES", "2"))

    if identity.provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=identity.model, temperature=temp, max_retries=max_retries)

    if identity.provider == "xai":
        from langchain_xai import ChatXAI

        return ChatXAI(model=identity.model, temperature=temp, max_retries=max_retries)

    raise ValueError(f"Unsupported RLM AI provider: {identity.provider}")
