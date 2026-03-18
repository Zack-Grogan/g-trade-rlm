"""
Generate and persist on-demand AI report bundles for the RLM service.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from psycopg2.extras import Json, RealDictCursor

from ai_provider import get_chat_model, get_model_identity
from db import db_conn


class ReportNarrative(BaseModel):
    report_title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    highlights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


def _rows_as_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines: list[str] = []
    for row in rows:
        parts = []
        for field in fields:
            value = row.get(field)
            if value not in (None, ""):
                parts.append(f"{field}={value}")
        if parts:
            lines.append("- " + ", ".join(parts))
    return "\n".join(lines) if lines else "None."


def build_report_context(lookback: int = 8, regime_context: str = "", generation: int = 1) -> dict[str, Any]:
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT COUNT(*) AS run_count,
                      COUNT(*) FILTER (WHERE data_mode = 'live') AS live_run_count
               FROM runs"""
        )
        run_counts = dict(cur.fetchone() or {})
        cur.execute("SELECT COUNT(*) AS event_count FROM events")
        event_count = int((cur.fetchone() or {}).get("event_count") or 0)
        cur.execute("SELECT COUNT(*) AS trade_count, COALESCE(SUM(pnl), 0) AS total_pnl FROM completed_trades")
        trade_counts = dict(cur.fetchone() or {})
        cur.execute("SELECT run_id, created_at, data_mode, symbol FROM runs ORDER BY created_at DESC LIMIT %s", (lookback,))
        recent_runs = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """SELECT id, run_id, entry_time, exit_time, pnl, zone, strategy, regime, source
               FROM completed_trades
               ORDER BY COALESCE(exit_time, inserted_at) DESC
               LIMIT %s""",
            (lookback,),
        )
        recent_trades = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """SELECT hypothesis_id, verdict, confidence_score, mutation_directive, created_at
               FROM knowledge_store
               ORDER BY created_at DESC
               LIMIT %s""",
            (lookback,),
        )
        recent_knowledge = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """SELECT id, run_id, status, result_summary, created_at
               FROM replay_runs
               ORDER BY created_at DESC
               LIMIT %s""",
            (lookback,),
        )
        recent_replays = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """SELECT report_id, title, summary_text, created_at
               FROM ai_reports
               ORDER BY created_at DESC
               LIMIT %s""",
            (lookback,),
        )
        recent_reports = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """SELECT hypothesis_id, verdict, confidence_score, mutation_directive, created_at
               FROM knowledge_store
               ORDER BY created_at DESC
               LIMIT %s""",
            (lookback,),
        )
        prior_conclusions = [dict(r) for r in cur.fetchall()]

    summary = {
        "run_count": int(run_counts.get("run_count") or 0),
        "live_run_count": int(run_counts.get("live_run_count") or 0),
        "event_count": event_count,
        "trade_count": int(trade_counts.get("trade_count") or 0),
        "total_pnl": float(trade_counts.get("total_pnl") or 0),
    }
    analysis_snapshot = {
        "summary": summary,
        "recent_runs": recent_runs,
        "recent_trades": recent_trades,
        "recent_replays": recent_replays,
    }
    return {
        "regime_context": regime_context or "overnight review",
        "generation": generation,
        "prior_conclusions_summary": _rows_as_text(
            prior_conclusions,
            ["hypothesis_id", "verdict", "confidence_score", "mutation_directive", "created_at"],
        ),
        "summary": summary,
        "analysis_snapshot": analysis_snapshot,
        "recent_runs_text": _rows_as_text(recent_runs, ["run_id", "created_at", "data_mode", "symbol"]),
        "recent_trades_text": _rows_as_text(
            recent_trades,
            ["id", "run_id", "pnl", "zone", "strategy", "regime", "source"],
        ),
        "recent_knowledge_text": _rows_as_text(
            recent_knowledge,
            ["hypothesis_id", "verdict", "confidence_score", "mutation_directive", "created_at"],
        ),
        "recent_reports": recent_reports,
    }


def generate_report_narrative(
    context: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    conclusion: dict[str, Any],
) -> ReportNarrative:
    llm = get_chat_model()
    structured = llm.with_structured_output(ReportNarrative)
    prompt = [
        (
            "system",
            "You write concise overnight trading reports for a single operator. "
            "No chat, no filler, no speculative advice. Summarize the latest batch of data, "
            "the generated hypotheses, and the conclusion verdict in a structured report.",
        ),
        (
            "user",
            "\n\n".join(
                [
                    f"Regime context: {context['regime_context']}",
                    f"Generation: {context['generation']}",
                    f"Summary metrics: {json.dumps(context['summary'], default=str)}",
                    f"Prior conclusions:\n{context['prior_conclusions_summary']}",
                    f"Recent runs:\n{context['recent_runs_text']}",
                    f"Recent trades:\n{context['recent_trades_text']}",
                    f"Recent knowledge:\n{context['recent_knowledge_text']}",
                    f"Hypotheses:\n{json.dumps(hypotheses, indent=2, default=str)}",
                    f"Conclusion:\n{json.dumps(conclusion, indent=2, default=str)}",
                ]
            ),
        ),
    ]
    return structured.invoke(prompt)


def compose_report_payload(
    report_id: str,
    context: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    conclusion: dict[str, Any],
    narrative: ReportNarrative,
) -> dict[str, Any]:
    model = get_model_identity()
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "report_id": report_id,
        "report_type": "on_demand",
        "title": narrative.report_title,
        "summary_text": narrative.executive_summary,
        "highlights": narrative.highlights,
        "risks": narrative.risks,
        "next_actions": narrative.next_actions,
        "model_provider": model.provider,
        "model_name": model.model,
        "regime_context": context["regime_context"],
        "generation": context["generation"],
        "summary": context["summary"],
        "analysis_snapshot": context["analysis_snapshot"],
        "prior_conclusions_summary": context["prior_conclusions_summary"],
        "hypotheses": hypotheses,
        "conclusion": conclusion,
        "generated_at": generated_at,
    }


def save_report(payload: dict[str, Any]) -> dict[str, Any]:
    report_id = payload["report_id"]
    model_provider = payload["model_provider"]
    model_name = payload["model_name"]
    title = payload["title"]
    summary_text = payload["summary_text"]
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """INSERT INTO ai_reports (
                   report_id, title, report_type, model_provider, model_name,
                   status, summary_text, report_json, created_at, completed_at
               )
               VALUES (%s, %s, %s, %s, %s, 'completed', %s, %s, NOW(), NOW())
               RETURNING report_id, title, report_type, model_provider, model_name, status,
                         summary_text, report_json, created_at, completed_at""",
            (
                report_id,
                title,
                payload["report_type"],
                model_provider,
                model_name,
                summary_text,
                Json(payload),
            ),
        )
        row = dict(cur.fetchone() or {})
        conn.commit()
    return row or payload


def generate_report_bundle(
    regime_context: str = "",
    generation: int = 1,
    report_type: str = "on_demand",
    lookback: int = 8,
) -> dict[str, Any]:
    from graphs import invoke_conclusion_graph, invoke_hypothesis_graph

    context = build_report_context(lookback=lookback, regime_context=regime_context, generation=generation)
    report_id = f"report-{uuid.uuid4()}"
    hypotheses = invoke_hypothesis_graph(
        regime_context=context["regime_context"],
        prior_conclusions_summary=context["prior_conclusions_summary"],
        generation=generation,
        parent_hypothesis_id=None,
        run_id=report_id,
    )
    conclusion = invoke_conclusion_graph(
        result_id=report_id,
        analysis=context["analysis_snapshot"],
        run_id=report_id,
    )
    narrative = generate_report_narrative(context, hypotheses, conclusion)
    payload = compose_report_payload(report_id, context, hypotheses, conclusion, narrative)
    payload["report_type"] = report_type
    return save_report(payload)


def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT report_id, title, report_type, model_provider, model_name, status,
                      summary_text, report_json, created_at, completed_at
               FROM ai_reports
               ORDER BY created_at DESC
               LIMIT %s""",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_report(report_id: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT report_id, title, report_type, model_provider, model_name, status,
                      summary_text, report_json, created_at, completed_at
               FROM ai_reports
               WHERE report_id = %s""",
            (report_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
