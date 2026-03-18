"""
g-trade-rlm: Advisory-only RLM service.
Produces reports, hypotheses, conclusions, and recommendations. Does NOT change execution config or strategy.
Auth: Bearer RLM_AUTH_TOKEN or GTRADE_INTERNAL_API_TOKEN. /health and /config are open.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Body, status
from fastapi.responses import JSONResponse
from psycopg2.extras import Json, RealDictCursor

from ai_provider import get_chat_model
from db import db_conn
from report_service import build_report_context, generate_report_bundle, get_report, list_reports

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config: dict | None = None

INTERNAL_API_TOKEN = (os.environ.get("GTRADE_INTERNAL_API_TOKEN") or "").strip()
RLM_AUTH_TOKEN = (os.environ.get("RLM_AUTH_TOKEN") or "").strip()


def load_config() -> dict:
    global _config
    if _config is None and CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    return _config or {}


def _bearer_ok(authorization: str | None) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization[7:].strip()
    allowed = [candidate for candidate in (RLM_AUTH_TOKEN, INTERNAL_API_TOKEN) if candidate]
    return bool(allowed) and token in allowed


def _require_auth(authorization: str | None) -> None:
    if not _bearer_ok(authorization):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing Bearer token")


def _submit_conclusion_record(
    *,
    result_id: int,
    verdict: str,
    confidence_score: float | None = None,
    mutation_directive: str | None = None,
    regime_tags: dict | None = None,
) -> dict | None:
    if verdict not in {"supported", "rejected", "inconclusive"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verdict")
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT hypothesis_id FROM experiment_results WHERE id = %s", (result_id,))
        row = cur.fetchone()
        if not row:
            return None
        hypothesis_id = row["hypothesis_id"]
        cur.execute(
            """INSERT INTO knowledge_store (
                   hypothesis_id, result_id, verdict, confidence_score, mutation_directive, regime_tags,
                   survival_count, rejection_count, last_validated_at
               )
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
               RETURNING id, hypothesis_id, result_id, verdict, confidence_score, mutation_directive, regime_tags,
                         survival_count, rejection_count, created_at, last_validated_at""",
            (
                hypothesis_id,
                result_id,
                verdict,
                confidence_score,
                mutation_directive,
                Json(regime_tags or {}),
                1 if verdict == "supported" else 0,
                1 if verdict == "rejected" else 0,
            ),
        )
        record = dict(cur.fetchone() or {})
        conn.commit()
        return record or None


app = FastAPI(
    title="g-trade-rlm",
    description="Advisory-only RLM: hypotheses, experiments, conclusions. No execution changes.",
)


def _ensure_schema() -> None:
    """Apply RLM schema (knowledge_store, trade_embeddings, replay_runs) if tables don't exist."""
    import psycopg2
    DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not DATABASE_URL:
        return
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with open(schema_path) as f:
            conn.cursor().execute(f.read())
        conn.commit()
        conn.close()
        logger.info("RLM schema applied")
    except Exception as e:
        logger.warning("Could not apply RLM schema: %s", e)


@app.on_event("startup")
def on_startup() -> None:
    _ensure_schema()


@app.exception_handler(RuntimeError)
def runtime_error_handler(request, exc: RuntimeError):
    """Return 503 when DATABASE_URL (or other required config) is missing instead of 500."""
    msg = str(exc) if exc else ""
    if "DATABASE_URL" in msg or "not set" in msg:
        return JSONResponse(
            status_code=503,
            content={"detail": msg or "DATABASE_URL not set", "service_unavailable": True},
        )
    raise exc


# --- Open endpoints (no auth required) ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "g-trade-rlm"}


@app.get("/config")
def config_summary():
    """Return non-secret config summary (env var names only, no values)."""
    c = load_config()
    return {
        "ai": c.get("ai", {}),
        "rlm": c.get("rlm", {}),
    }


# --- Auth-protected endpoints ---

@app.post("/feedback/cycle")
async def feedback_cycle(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Run one recursive feedback cycle: prior conclusions + meta stats → new hypotheses. Advisory-only."""
    _require_auth(authorization)
    from workflow_orchestrator import run_one_feedback_cycle
    hypotheses = run_one_feedback_cycle(
        regime_context=body.get("regime_context", ""),
        generation=body.get("generation", 1),
        parent_hypothesis_id=body.get("parent_hypothesis_id"),
    )
    return {"hypotheses": hypotheses}


@app.post("/hypotheses/generate")
async def generate_hypothesis(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Advisory-only: invoke LangGraph hypothesis flow; return first generated hypothesis (or list)."""
    _require_auth(authorization)
    from graphs import invoke_hypothesis_graph
    body = body or {}
    try:
        hypotheses = invoke_hypothesis_graph(
            regime_context=body.get("regime_context", ""),
            prior_conclusions_summary=body.get("prior_conclusions_summary", ""),
            generation=body.get("generation", 1),
            parent_hypothesis_id=body.get("parent_hypothesis_id"),
            run_id=body.get("run_id"),
        )
    except Exception as e:
        msg = str(e).lower()
        if "database_url" in msg or "api key" in msg or "not set" in msg or "connection" in msg:
            return JSONResponse(
                status_code=503,
                content={"detail": str(e), "service_unavailable": True},
            )
        raise
    if not hypotheses:
        return {"hypothesis": None, "hypotheses": []}
    return {"hypothesis": hypotheses[0], "hypotheses": hypotheses}


@app.post("/reports/generate")
async def generate_report(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Generate and persist one on-demand AI report bundle."""
    _require_auth(authorization)
    body = body or {}
    try:
        return generate_report_bundle(
            regime_context=body.get("regime_context", ""),
            generation=body.get("generation", 1),
            report_type=body.get("report_type", "on_demand"),
            lookback=body.get("lookback", 8),
        )
    except Exception as e:
        msg = str(e).lower()
        if "api key" in msg or "openrouter" in msg or "database_url" in msg or "not set" in msg:
            return JSONResponse(status_code=503, content={"detail": str(e), "service_unavailable": True})
        raise


@app.get("/reports")
async def reports(
    limit: int = Query(20, ge=1, le=100),
    authorization: str | None = Header(None),
):
    """List persisted AI report bundles."""
    _require_auth(authorization)
    return {"reports": list_reports(limit=limit)}


@app.get("/reports/{report_id}")
async def report_detail(
    report_id: str,
    authorization: str | None = Header(None),
):
    """Fetch a single AI report bundle."""
    _require_auth(authorization)
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@app.post("/conclusions/submit")
async def submit_conclusion(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Persist an advisory conclusion to the knowledge store. RLM owns this write path."""
    _require_auth(authorization)
    result_id = body.get("result_id")
    if result_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="result_id required")
    record = _submit_conclusion_record(
        result_id=int(result_id),
        verdict=str(body.get("verdict") or ""),
        confidence_score=body.get("confidence_score"),
        mutation_directive=body.get("mutation_directive"),
        regime_tags=body.get("regime_tags") if isinstance(body.get("regime_tags"), dict) else None,
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment result not found")
    return {"knowledge_entry": record}


@app.post("/chat/analyze")
async def chat_analyze(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Operator-facing advisory chat endpoint for investigation and report drafting."""
    _require_auth(authorization)
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="prompt required")
    context = build_report_context(
        lookback=int(body.get("lookback") or 8),
        regime_context=str(body.get("regime_context") or "operator investigation"),
        generation=int(body.get("generation") or 1),
    )
    llm = get_chat_model()
    response = llm.invoke(
        [
            (
                "system",
                "You are the G-Trade operator analysis assistant. Stay advisory-only. "
                "Use the supplied context to explain risks, anomalies, bridge/log issues, and research conclusions. "
                "Do not suggest live trade execution changes as commands.",
            ),
            (
                "user",
                "\n\n".join(
                    [
                        f"Prompt: {prompt}",
                        f"Regime context: {context['regime_context']}",
                        f"Summary: {context['summary']}",
                        f"Recent runs:\n{context['recent_runs_text']}",
                        f"Recent trades:\n{context['recent_trades_text']}",
                        f"Recent knowledge:\n{context['recent_knowledge_text']}",
                    ]
                ),
            ),
        ]
    )
    return {
        "message": getattr(response, "content", str(response)),
        "context": {
            "summary": context["summary"],
            "regime_context": context["regime_context"],
            "generation": context["generation"],
        },
    }


@app.get("/benchmark/checkpoints")
async def benchmark_checkpoints(
    limit: int = Query(400, ge=1, le=500),
    authorization: str | None = Header(None),
):
    """List replayable checkpoints for the 360-replay benchmark (valid sessions only)."""
    _require_auth(authorization)
    from benchmark import get_benchmark_checkpoints
    return {"checkpoints": get_benchmark_checkpoints(limit=limit)}


@app.post("/benchmark/run")
async def benchmark_run(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Run the full benchmark (replay each checkpoint). Optional body: checkpoint_run_ids list."""
    _require_auth(authorization)
    from benchmark import run_full_benchmark
    run_ids = body.get("checkpoint_run_ids")
    return run_full_benchmark(run_ids)


@app.get("/replay/checkpoints")
async def replay_checkpoints(
    limit: int = Query(100, ge=1, le=500),
    authorization: str | None = Header(None),
):
    """List replayable checkpoints (runs that have at least one completed trade)."""
    _require_auth(authorization)
    from replay_api import list_replayable_checkpoints
    return {"checkpoints": list_replayable_checkpoints(limit=limit)}


@app.post("/replay/trigger")
async def replay_trigger(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Trigger a replay job for run_id (enqueue to QStash). What-if config for sandbox only."""
    _require_auth(authorization)
    from replay_api import trigger_replay
    run_id = body.get("run_id")
    if not run_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="run_id required")
    ok = trigger_replay(run_id, body.get("what_if_config"))
    return {"enqueued": ok, "run_id": run_id}


@app.post("/replay/what-if")
async def replay_what_if(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Run what-if replay in Daytona sandbox (no push to main). Body: run_id, what_if_config."""
    _require_auth(authorization)
    from daytona_client import run_what_if_in_sandbox
    run_id = body.get("run_id")
    if not run_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="run_id required")
    result = run_what_if_in_sandbox(run_id, body.get("what_if_config") or {}, timeout_seconds=300)
    return result


@app.post("/replay/worker")
async def replay_worker_endpoint(
    body: dict = Body(default_factory=dict),
    authorization: str | None = Header(None),
):
    """Called by QStash to execute a replay job. Body: { run_id, what_if_config }."""
    _require_auth(authorization)
    from replay_worker import handle_replay_request
    return handle_replay_request(body)


@app.get("/similar_trades")
async def similar_trades(
    trade_id: int = Query(..., description="Completed trade id"),
    limit: int = Query(10, ge=1, le=50),
    failed_only: bool = Query(False, description="Only return trades with negative PnL"),
    authorization: str | None = Header(None),
):
    """Semantic search: trades similar to the given trade (optionally failed only)."""
    _require_auth(authorization)
    from embedding_service import find_similar_failed_trades, find_similar_trades
    if failed_only:
        items = await find_similar_failed_trades(trade_id, limit=limit)
    else:
        items = await find_similar_trades(trade_id, limit=limit, pnl_negative_only=False)
    return {"trade_id": trade_id, "similar": items}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
