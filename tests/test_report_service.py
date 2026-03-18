from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report_service
from ai_provider import ModelIdentity
from report_service import ReportNarrative


def test_generate_report_bundle_batches_all_steps(monkeypatch):
    context = {
        "regime_context": "overnight review",
        "generation": 3,
        "prior_conclusions_summary": "prior context",
        "summary": {"run_count": 4, "live_run_count": 1, "event_count": 9, "trade_count": 2, "total_pnl": 12.5},
        "analysis_snapshot": {"summary": {"run_count": 4}},
        "recent_runs_text": "- run_id=run-1",
        "recent_trades_text": "- id=1",
        "recent_knowledge_text": "- hypothesis_id=h1",
        "recent_reports": [],
    }
    hypotheses = [{"hypothesis_id": "h1", "claim_text": "Momentum holds in high volume", "status": "pending"}]
    conclusion = {"verdict": "supported", "confidence_score": 0.9}
    narrative = ReportNarrative(
        report_title="Overnight Review",
        executive_summary="The batch run found stable momentum and controlled drawdown.",
        highlights=["Momentum stayed intact."],
        risks=["Thin liquidity late in the session."],
        next_actions=["Watch continuation setups."],
    )
    captured = {}

    def fake_context(**kwargs):
        return context

    def fake_hypotheses(**kwargs):
        captured["hypothesis_kwargs"] = kwargs
        return hypotheses

    def fake_conclusion(**kwargs):
        captured["conclusion_kwargs"] = kwargs
        return conclusion

    def fake_narrative(received_context, received_hypotheses, received_conclusion):
        captured["narrative_args"] = (received_context, received_hypotheses, received_conclusion)
        return narrative

    def fake_save(payload):
        captured["payload"] = payload
        return payload

    monkeypatch.setattr(report_service, "build_report_context", fake_context)
    monkeypatch.setattr(report_service, "generate_report_narrative", fake_narrative)
    monkeypatch.setattr(report_service, "save_report", fake_save)
    monkeypatch.setattr(report_service, "get_model_identity", lambda: ModelIdentity(provider="openrouter", model="anthropic/claude-sonnet-4.5"))

    graphs = ModuleType("graphs")
    graphs.invoke_hypothesis_graph = fake_hypotheses
    graphs.invoke_conclusion_graph = fake_conclusion
    monkeypatch.setitem(sys.modules, "graphs", graphs)

    result = report_service.generate_report_bundle(regime_context="overnight review", generation=3, lookback=8)

    assert result["report_id"].startswith("report-")
    assert captured["hypothesis_kwargs"]["generation"] == 3
    assert captured["conclusion_kwargs"]["analysis"] == context["analysis_snapshot"]
    assert captured["payload"]["model_provider"] == "openrouter"
    assert captured["payload"]["model_name"] == "anthropic/claude-sonnet-4.5"
    assert captured["payload"]["title"] == "Overnight Review"
    assert captured["payload"]["summary_text"] == narrative.executive_summary
    assert captured["payload"]["hypotheses"] == hypotheses


def test_compose_report_payload_keeps_model_identity(monkeypatch):
    context = {
        "regime_context": "batch",
        "generation": 1,
        "prior_conclusions_summary": "none",
        "summary": {"run_count": 1, "live_run_count": 0, "event_count": 2, "trade_count": 1, "total_pnl": 4.0},
        "analysis_snapshot": {"summary": {"run_count": 1}},
        "recent_runs_text": "- run_id=run-1",
        "recent_trades_text": "- id=1",
        "recent_knowledge_text": "- hypothesis_id=h1",
        "recent_reports": [],
    }
    narrative = ReportNarrative(
        report_title="Batch Review",
        executive_summary="Concise summary.",
        highlights=[],
        risks=[],
        next_actions=[],
    )

    monkeypatch.setattr(report_service, "get_model_identity", lambda: ModelIdentity(provider="openrouter", model="anthropic/claude-sonnet-4.5"))

    payload = report_service.compose_report_payload("report-123", context, [], {}, narrative)

    assert payload["report_id"] == "report-123"
    assert payload["model_provider"] == "openrouter"
    assert payload["model_name"] == "anthropic/claude-sonnet-4.5"
    assert payload["title"] == "Batch Review"
