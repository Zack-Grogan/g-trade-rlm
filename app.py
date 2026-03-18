"""
g-trade-rlm: Advisory-only RLM service.
Produces reports, hypotheses, conclusions, and recommendations. Does NOT change execution config or strategy.
Auth: Bearer RLM_AUTH_TOKEN (from env). /health and /config are open (same policy as other services).
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Body, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config: dict | None = None

RLM_AUTH_TOKEN = (os.environ.get("RLM_AUTH_TOKEN") or "").strip()


def load_config() -> dict:
    global _config
    if _config is None and CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    return _config or {}


def _bearer_ok(authorization: str | None) -> bool:
    if not RLM_AUTH_TOKEN:
        # If no token is configured, fail closed rather than open.
        return False
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return authorization[7:].strip() == RLM_AUTH_TOKEN


def _require_auth(authorization: str | None) -> None:
    if not _bearer_ok(authorization):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing Bearer token")


app = FastAPI(
    title="g-trade-rlm",
    description="Advisory-only RLM: hypotheses, experiments, conclusions. No execution changes.",
)


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
        "x_ai": {"model": c.get("x_ai", {}).get("model"), "api_key_env": c.get("x_ai", {}).get("api_key_env")},
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
