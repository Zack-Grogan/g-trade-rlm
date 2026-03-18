"""
360-replay benchmark system: fixed set of replayable checkpoints (100–300+ scenarios).
Used to validate any proposed executor change (human-applied) before sign-off.
Replayable sessions must be points in time where the system actually traded.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from replay_api import list_replayable_checkpoints, trigger_replay
from replay_worker import handle_replay_request

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
BENCHMARK_SIZE_TARGET = 360  # 360 trade tests per plan
MIN_REPLAYABLE = 100


def _get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def get_benchmark_checkpoints(limit: int = 400) -> list[dict[str, Any]]:
    """
    Return replayable checkpoints suitable for the benchmark (runs with trades).
    Valid sessions only: point in time where the system actually traded.
    """
    return list_replayable_checkpoints(limit=limit)


def run_benchmark_replay_sync(run_id: str) -> dict[str, Any]:
    """Run a single replay synchronously (for benchmark runner). Returns result_summary or error."""
    return handle_replay_request({"run_id": run_id, "what_if_config": {}})


def run_full_benchmark(checkpoint_run_ids: list[str] | None = None) -> dict[str, Any]:
    """
    Run the benchmark: execute replay for each checkpoint. If checkpoint_run_ids not provided,
    use get_benchmark_checkpoints() up to BENCHMARK_SIZE_TARGET.
    Returns counts: total, passed, failed, errors.
    """
    if not checkpoint_run_ids:
        checkpoints = get_benchmark_checkpoints(limit=BENCHMARK_SIZE_TARGET)
        checkpoint_run_ids = [c["run_id"] for c in checkpoints]
    total = len(checkpoint_run_ids)
    passed = 0
    failed = 0
    errors = []
    for run_id in checkpoint_run_ids:
        try:
            result = run_benchmark_replay_sync(run_id)
            if result.get("ok"):
                passed += 1
            else:
                failed += 1
                errors.append({"run_id": run_id, "error": result.get("error", "unknown")})
        except Exception as e:
            failed += 1
            errors.append({"run_id": run_id, "error": str(e)})
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors[:20],
    }
