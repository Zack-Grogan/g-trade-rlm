"""
Orchestrates RLM pipeline: knowledge_store → prior conclusions summary → hypothesis generation.
Recursive feedback loop (advisory only): conclusions inform next generation; no execution changes.
Can be triggered by Upstash Workflow or cron; one cycle = fetch conclusions, generate hypotheses.
"""
from __future__ import annotations

from typing import Any

from psycopg2.extras import RealDictCursor

from db import db_conn
from graphs import invoke_hypothesis_graph
from meta_learner import get_meta_stats


def get_prior_conclusions_summary(limit: int = 20) -> str:
    """Build a text summary of recent knowledge_store entries for hypothesis generation context."""
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT hypothesis_id, verdict, confidence_score, mutation_directive, created_at
               FROM knowledge_store ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()
        if not rows:
            return "No prior conclusions yet."
        lines = []
        for r in rows:
            lines.append(f"- {r['verdict']} (confidence={r.get('confidence_score')}); {r.get('mutation_directive') or ''}; at {r['created_at']}")
        return "\n".join(lines)


def run_one_feedback_cycle(
    regime_context: str = "",
    generation: int = 1,
    parent_hypothesis_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    One cycle of the recursive feedback loop: load prior conclusions and meta stats, generate new hypotheses.
    Advisory-only; does not change executor.
    """
    prior_summary = get_prior_conclusions_summary()
    meta_stats = get_meta_stats()
    hypotheses = invoke_hypothesis_graph(
        regime_context=regime_context,
        prior_conclusions_summary=prior_summary,
        generation=generation,
        parent_hypothesis_id=parent_hypothesis_id,
    )
    return hypotheses
