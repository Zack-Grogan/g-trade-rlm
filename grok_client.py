"""
X.ai Grok API client. OpenAI-compatible REST API (https://api.x.ai/v1).
Used for hypothesis generation, conclusion engine, and trade analysis. Advisory-only.
"""
from __future__ import annotations

import os
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-beta"


def get_api_key(api_key_env: str = "X_AI_API_KEY") -> str:
    return os.environ.get(api_key_env, "").strip()


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
) -> str | None:
    """
    Send chat completion request to xAI Grok. Returns content of first choice or None on failure.
    """
    key = api_key or get_api_key()
    if not key:
        logger.warning("X_AI_API_KEY not set; skipping Grok call")
        return None
    url = f"{XAI_BASE_URL}/chat/completions"
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(url, json=payload, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if choices and isinstance(choices[0].get("message"), dict):
                return (choices[0]["message"].get("content") or "").strip() or None
            return None
        except httpx.HTTPError as e:
            logger.exception("Grok API request failed: %s", e)
            return None


async def generate_hypothesis_text(
    regime_context: str,
    prior_conclusions_summary: str,
    meta_stats: dict[str, Any],
) -> str | None:
    """
    Ask Grok to generate one or more testable market hypotheses given regime and prior knowledge.
    Returns raw text; caller parses into structured hypotheses.
    """
    system = (
        "You are a quantitative research assistant. Produce testable, falsifiable market hypotheses "
        "based on the given regime context and prior conclusions. Output concise hypothesis claims "
        "suitable for backtest/replay experiments. One hypothesis per line or in a short numbered list."
    )
    user = (
        f"Regime context: {regime_context}\n\n"
        f"Prior conclusions (summary): {prior_conclusions_summary}\n\n"
        f"Meta-learner stats: {meta_stats}\n\n"
        "Generate 1-3 testable hypotheses."
    )
    return await chat_completion([{"role": "system", "content": system}, {"role": "user", "content": user}])


async def embed_text_for_similarity(text: str, dimension: int = 128) -> list[float] | None:
    """
    Produce a semantic vector for the given text using Grok (no dedicated embed API).
    Asks Grok to return a JSON array of `dimension` floats representing the text for similarity search.
    """
    key = get_api_key()
    if not key:
        logger.warning("X_AI_API_KEY not set; skipping embed")
        return None
    system = (
        f"You are an embedding model. Output ONLY a JSON array of exactly {dimension} floating-point numbers "
        "that semantically represent the following text for similarity search. No other text or explanation."
    )
    user = f"Text to embed:\n{text}"
    raw = await chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2048,
    )
    if not raw:
        return None
    import json
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list) and len(arr) >= dimension:
            return [float(x) for x in arr[:dimension]]
        if isinstance(arr, list):
            return [float(x) for x in arr] + [0.0] * (dimension - len(arr))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None
