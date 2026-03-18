"""
UpStash client wrapper: Redis, QStash, Workflow, Vector.
No Box; replay data lives in Postgres. Used for cache, queue, orchestration, and semantic search.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Optional: lazy init so missing env vars don't break import
_redis: Any = None
_qstash: Any = None
_vector_index: Any = None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- Redis (cache for analytics, regime snapshots, rate limiting) ---
def get_redis():
    """Return Upstash Redis async client. Uses UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN."""
    global _redis
    if _redis is None:
        try:
            from upstash_redis.asyncio import Redis
            _redis = Redis.from_env()
        except Exception as e:
            logger.warning("Upstash Redis not configured or import failed: %s", e)
    return _redis


async def redis_get(key: str) -> str | None:
    r = get_redis()
    if r is None:
        return None
    try:
        return await r.get(key)
    except Exception as e:
        logger.warning("Redis get failed: %s", e)
        return None


async def redis_set(key: str, value: str, ex_seconds: int | None = None) -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        if ex_seconds is not None:
            await r.setex(key, ex_seconds, value)
        else:
            await r.set(key, value)
        return True
    except Exception as e:
        logger.warning("Redis set failed: %s", e)
        return False


# --- QStash (async jobs: hypothesis generation, experiment runs, embeddings) ---
def get_qstash():
    """Return QStash client. Uses UPSTASH_QSTASH_TOKEN or QSTASH_TOKEN."""
    global _qstash
    if _qstash is None:
        try:
            from qstash import QStash
            token = _env("UPSTASH_QSTASH_TOKEN") or _env("QSTASH_TOKEN")
            if token:
                _qstash = QStash(token)
            else:
                logger.warning("QStash token not set")
        except Exception as e:
            logger.warning("QStash not configured or import failed: %s", e)
    return _qstash


def qstash_publish_json(url: str, body: dict[str, Any]) -> bool:
    """Publish JSON body to URL via QStash. Returns True on success."""
    q = get_qstash()
    if q is None:
        return False
    try:
        if hasattr(q, "message") and hasattr(q.message, "publish_json"):
            q.message.publish_json(url=url, body=body)
        else:
            q.publish_json(url=url, body=body)
        return True
    except Exception as e:
        logger.warning("QStash publish failed: %s", e)
        return False


# --- Vector (semantic search over trade embeddings) ---
def get_vector_index():
    """Return Upstash Vector Index. Uses UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN."""
    global _vector_index
    if _vector_index is None:
        try:
            from upstash_vector import Index
            _vector_index = Index.from_env()
        except Exception as e:
            logger.warning("Upstash Vector not configured or import failed: %s", e)
    return _vector_index


def vector_upsert(id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> bool:
    """Upsert one vector with optional metadata."""
    idx = get_vector_index()
    if idx is None:
        return False
    try:
        idx.upsert(vectors=[(id, vector, metadata or {})])
        return True
    except Exception as e:
        logger.warning("Vector upsert failed: %s", e)
        return False


def vector_query(vector: list[float], top_k: int = 10, metadata_filter: str | None = None) -> list[tuple[str, float, dict | None]]:
    """
    Query by vector. Returns list of (id, score, metadata).
    metadata_filter: optional filter string for Vector API (e.g. pnl < 0).
    """
    idx = get_vector_index()
    if idx is None:
        return []
    try:
        res = idx.query(data=vector, top_k=top_k, include_metadata=True)
        # API returns list of {id, score, metadata?}
        out = []
        for r in (res or []):
            id_ = r.get("id", "")
            score = float(r.get("score", 0))
            meta = r.get("metadata")
            out.append((id_, score, meta))
        return out
    except Exception as e:
        logger.warning("Vector query failed: %s", e)
        return []


# --- Workflow (orchestration: coordinate RLM pipeline stages, recursive feedback) ---
def get_workflow_client():
    """Return Upstash Workflow client if available. Uses UPSTASH_WORKFLOW_* env vars."""
    try:
        from upstash_workflow import Workflow
        url = _env("UPSTASH_WORKFLOW_URL")
        token = _env("UPSTASH_WORKFLOW_TOKEN")
        if url and token:
            return Workflow(base_url=url, token=token)
    except Exception as e:
        logger.warning("Upstash Workflow not configured or import failed: %s", e)
    return None
