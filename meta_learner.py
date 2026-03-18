"""
Meta-learner: tracks hypothesis class survival rates; informs hypothesis generation.
Reads from knowledge_store (and hypotheses). Advisory-only: no execution changes.
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from db import db_conn

logger = logging.getLogger(__name__)


def get_meta_stats() -> dict[str, Any]:
    """Return survival/rejection counts and per-verdict stats from knowledge_store."""
    try:
        with db_conn() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """SELECT COALESCE(SUM(survival_count), 0) AS survival_count, COALESCE(SUM(rejection_count), 0) AS rejection_count
                   FROM knowledge_store"""
            )
            row = cur.fetchone()
            survival = int(row["survival_count"] or 0)
            rejection = int(row["rejection_count"] or 0)
            cur.execute(
                """SELECT verdict, COUNT(*) AS cnt FROM knowledge_store GROUP BY verdict"""
            )
            by_verdict = {r["verdict"]: r["cnt"] for r in cur.fetchall()}
            return {
                "survival_count": survival,
                "rejection_count": rejection,
                "by_verdict": by_verdict,
            }
    except RuntimeError:
        # DATABASE_URL not configured — return empty stats rather than crashing the feedback loop.
        logger.warning("get_meta_stats: DATABASE_URL not set; returning empty stats")
        return {"survival_count": 0, "rejection_count": 0, "by_verdict": {}}
