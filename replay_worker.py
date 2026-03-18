"""
Replay worker: executes replay from Postgres (state_snapshots, events, completed_trades).
No Topstep/ProjectX API. Logic-only execution; results stored in replay_runs.
Invoked by QStash when a replay job is enqueued.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import RealDictCursor, Json

from db import db_conn

logger = logging.getLogger(__name__)


def run_replay(run_id: str, what_if_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load run from Postgres and replay (logic-only; no live broker).
    Returns result_summary: event_count, trade_count, total_pnl, etc.
    """
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS n FROM events WHERE run_id = %s", (run_id,))
        event_count = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS total_pnl FROM completed_trades WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        trade_count = row["n"]
        total_pnl = float(row["total_pnl"] or 0)
        cur.execute("SELECT COUNT(*) AS n FROM state_snapshots WHERE run_id = %s", (run_id,))
        snapshot_count = cur.fetchone()["n"]
        return {
            "run_id": run_id,
            "event_count": event_count,
            "trade_count": trade_count,
            "total_pnl": total_pnl,
            "snapshot_count": snapshot_count,
            "what_if_config": what_if_config or {},
            "replayed_at": datetime.now(timezone.utc).isoformat(),
        }


def update_replay_status(replay_run_id: int, status: str, result_summary: dict[str, Any] | None = None, finished_at: datetime | None = None) -> None:
    """Update replay_runs row after worker completes."""
    with db_conn() as conn:
        cur = conn.cursor()
        if finished_at is None and status in ("completed", "failed"):
            finished_at = datetime.now(timezone.utc)
        cur.execute(
            """UPDATE replay_runs SET status = %s, result_summary = COALESCE(%s, result_summary), finished_at = COALESCE(%s, finished_at)
               WHERE id = %s""",
            (status, Json(result_summary) if result_summary else None, finished_at, replay_run_id),
        )
        conn.commit()


def get_or_create_replay_run(run_id: str, what_if_config: dict | None) -> int | None:
    """Get latest replay_runs id for this run_id with status pending, or create one. Returns id."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM replay_runs WHERE run_id = %s AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO replay_runs (run_id, status, what_if_config) VALUES (%s, 'pending', %s) RETURNING id",
            (run_id, Json(what_if_config or {})),
        )
        r = cur.fetchone()
        conn.commit()
        return r[0] if r else None


def handle_replay_request(body: dict[str, Any]) -> dict[str, Any]:
    """
    Handle incoming QStash replay job. body = { run_id, what_if_config }.
    Updates replay_runs to running then completed/failed with result_summary.
    """
    run_id = body.get("run_id")
    if not run_id:
        return {"ok": False, "error": "run_id required"}
    what_if = body.get("what_if_config") or {}
    replay_id = get_or_create_replay_run(run_id, what_if)
    if not replay_id:
        replay_id = 0
    try:
        update_replay_status(replay_id, "running")
        summary = run_replay(run_id, what_if)
        update_replay_status(replay_id, "completed", result_summary=summary)
        return {"ok": True, "result_summary": summary}
    except Exception as e:
        logger.exception("Replay failed: %s", e)
        update_replay_status(replay_id, "failed", result_summary={"error": str(e)})
        return {"ok": False, "error": str(e)}
