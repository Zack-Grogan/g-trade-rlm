"""
Trade embedding generation (Grok) and semantic search (Upstash Vector + Postgres trade_embeddings).
Used for similar-trade queries, e.g. similar failed trades in a regime. Advisory-only.
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from db import db_conn
from grok_client import embed_text_for_similarity
from upstash_client import vector_upsert, vector_query

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 128
EMBEDDING_MODEL = "grok-beta"


def _trade_to_text(trade: dict[str, Any]) -> str:
    """Build a text representation of a trade for embedding."""
    return (
        f"Zone: {trade.get('zone', '')} "
        f"Strategy: {trade.get('strategy', '')} "
        f"Regime: {trade.get('regime', '')} "
        f"PnL: ${trade.get('pnl', 0)} "
        f"Entry: {trade.get('entry_price')} @ {trade.get('entry_time')} "
        f"Exit: {trade.get('exit_price')} @ {trade.get('exit_time')} "
        f"Direction: {trade.get('direction')} Contracts: {trade.get('contracts')}"
    )


async def embed_trade(trade: dict[str, Any], trade_id: int) -> list[float] | None:
    """Generate embedding for a trade using Grok; store in Postgres and Upstash Vector."""
    text = _trade_to_text(trade)
    vec = await embed_text_for_similarity(text, dimension=EMBEDDING_DIMENSION)
    if not vec:
        return None
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO trade_embeddings (trade_id, embedding, embedding_model)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (trade_id) DO UPDATE SET embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model""",
                (trade_id, vec, EMBEDDING_MODEL),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to store trade_embeddings in Postgres: %s", e)
    vector_upsert(str(trade_id), vec, metadata={"pnl": trade.get("pnl"), "zone": trade.get("zone"), "regime": trade.get("regime")})
    return vec


def get_trade(trade_id: int) -> dict[str, Any] | None:
    """Fetch a single completed trade by id."""
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT id, run_id, entry_time, exit_time, direction, contracts, entry_price, exit_price, pnl, zone, strategy, regime, source, backfilled, payload_json
               FROM completed_trades WHERE id = %s""",
            (trade_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_embedding_from_postgres(trade_id: int) -> list[float] | None:
    """Get stored embedding for a trade from Postgres (trade_embeddings.embedding is double precision[])."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM trade_embeddings WHERE trade_id = %s", (trade_id,))
        row = cur.fetchone()
        if row and row[0] is not None:
            return list(row[0])
        return None


async def find_similar_trades(trade_id: int, limit: int = 10, pnl_negative_only: bool = False) -> list[dict[str, Any]]:
    """
    Find trades similar to the given trade_id. Uses embedding from Postgres or generates via Grok.
    If pnl_negative_only=True, filter to similar trades that lost money (post-query filter if Vector has no filter).
    """
    trade = get_trade(trade_id)
    if not trade:
        return []
    embedding = get_embedding_from_postgres(trade_id)
    if not embedding:
        embedding = await embed_trade(trade, trade_id)
    if not embedding:
        return []
    results = vector_query(embedding, top_k=limit + 20)
    out = []
    for vid, score, meta in results:
        if vid == str(trade_id):
            continue
        try:
            tid = int(vid)
        except (ValueError, TypeError):
            continue
        if pnl_negative_only and meta is not None and (meta.get("pnl") or 0) >= 0:
            continue
        t = get_trade(tid)
        if t:
            t["_score"] = score
            out.append(t)
        if len(out) >= limit:
            break
    return out


async def find_similar_failed_trades(trade_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """Find similar trades that had negative PnL."""
    return await find_similar_trades(trade_id, limit=limit, pnl_negative_only=True)
