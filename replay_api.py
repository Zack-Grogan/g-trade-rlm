"""
Replay API: list replayable checkpoints (runs with trades), trigger replay jobs via QStash.
Replay runs from Postgres only; no Topstep/ProjectX API. What-if mode is sandbox-only.
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor, Json

from db import db_conn
from upstash_client import qstash_publish_json

logger = logging.getLogger(__name__)

import os
REPLAY_WORKER_URL = os.environ.get("RLM_REPLAY_WORKER_URL", "").rstrip("/")


def list_replayable_checkpoints(limit: int = 100) -> list[dict[str, Any]]:
    """List run_ids that have at least one completed trade (valid replayable sessions)."""
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT r.run_id, r.created_at, COUNT(t.id) AS trade_count
               FROM runs r
               INNER JOIN completed_trades t ON t.run_id = r.run_id
               GROUP BY r.run_id, r.created_at
               HAVING COUNT(t.id) > 0
               ORDER BY r.created_at DESC
               LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def trigger_replay(run_id: str, what_if_config: dict[str, Any] | None = None) -> bool:
    """Enqueue a replay job via QStash. Worker URL must be set (RLM_REPLAY_WORKER_URL)."""
    if not REPLAY_WORKER_URL:
        logger.warning("RLM_REPLAY_WORKER_URL not set; cannot trigger replay")
        return False
    ok = qstash_publish_json(
        REPLAY_WORKER_URL,
        {"run_id": run_id, "what_if_config": what_if_config or {}},
    )
    if ok:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO replay_runs (run_id, status, what_if_config) VALUES (%s, 'pending', %s)""",
                    (run_id, Json(what_if_config or {})),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to insert replay_runs row: %s", e)
    return ok
