from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app as rlm_app  # noqa: E402


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeModel:
    def invoke(self, prompt):  # noqa: ANN001 - test double
        return _FakeResponse("Advisory analysis")


def test_submit_conclusion_requires_auth(monkeypatch):
    monkeypatch.setattr(rlm_app, "RLM_AUTH_TOKEN", "test-token")
    with TestClient(rlm_app.app) as client:
        response = client.post("/conclusions/submit", json={"result_id": 1, "verdict": "supported"})

    assert response.status_code == 401


def test_submit_conclusion_uses_rlm_owned_write_path(monkeypatch):
    monkeypatch.setattr(rlm_app, "RLM_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        rlm_app,
        "_submit_conclusion_record",
        lambda **kwargs: {"id": 1, "hypothesis_id": "hyp-1", "verdict": kwargs["verdict"]},
    )

    with TestClient(rlm_app.app) as client:
        response = client.post(
            "/conclusions/submit",
            headers={"Authorization": "Bearer test-token"},
            json={"result_id": 1, "verdict": "supported"},
        )

    assert response.status_code == 200
    assert response.json()["knowledge_entry"]["hypothesis_id"] == "hyp-1"


def test_chat_analyze_returns_llm_content(monkeypatch):
    monkeypatch.setattr(rlm_app, "RLM_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        rlm_app,
        "build_report_context",
        lambda **kwargs: {
            "summary": {"run_count": 2},
            "regime_context": kwargs["regime_context"],
            "generation": kwargs["generation"],
            "recent_runs_text": "- run_id=run-1",
            "recent_trades_text": "- id=1",
            "recent_knowledge_text": "- hypothesis_id=h1",
        },
    )
    monkeypatch.setattr(rlm_app, "get_chat_model", lambda: _FakeModel())

    with TestClient(rlm_app.app) as client:
        response = client.post(
            "/chat/analyze",
            headers={"Authorization": "Bearer test-token"},
            json={"prompt": "Summarize the latest bridge issues", "regime_context": "ops"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Advisory analysis"
    assert body["context"]["regime_context"] == "ops"
