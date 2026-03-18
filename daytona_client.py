"""
Daytona sandbox integration for what-if RLM runs. Branched codebase, no push to main.
Runs replay/experiments in an isolated sandbox; results stay until human review.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_daytona_client():
    """
    Return a Daytona SDK client configured with the API key from DAYTONA_API_KEY.
    Uses the official Python SDK (daytona_sdk).
    """
    api_key = os.environ.get("DAYTONA_API_KEY", "").strip()
    if not api_key:
        logger.warning("DAYTONA_API_KEY not set; skipping sandbox run")
        return None, {"ok": False, "error": "DAYTONA_API_KEY not set"}
    try:
        from daytona_sdk import Daytona, DaytonaConfig
    except ImportError:
        logger.warning("daytona_sdk not installed; pip install daytona-sdk")
        return None, {"ok": False, "error": "daytona-sdk not installed"}
    config = DaytonaConfig(api_key=api_key)
    return Daytona(config), None


def run_what_if_in_sandbox(
    run_id: str,
    what_if_config: dict[str, Any],
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Run a what-if replay/experiment in a Daytona sandbox. No live broker; no push to main.
    Returns sandbox run result (stdout, stderr, artifacts) or error.
    """
    daytona, err = _get_daytona_client()
    if daytona is None:
        return err or {"ok": False, "error": "Daytona client not available"}
    try:
        sandbox = daytona.create()
        # TODO: Replace this simulation with a real replay call.
        # In a full implementation, this would load events/trades from Postgres,
        # run the engine logic with what_if_config overrides, and emit a real result summary.
        # For now it emits a stub result so the sandbox integration can be end-to-end tested.
        code = f"""
import json
# In a full implementation this would load replay from Postgres and run with what_if_config
# Here we only simulate: emit a result summary for the sandbox run
result = {{"run_id": "{run_id}", "what_if_config": {json.dumps(what_if_config)}, "simulated": True}}
print(json.dumps(result))
"""
        response = sandbox.process.code_run(code, timeout=timeout_seconds)
        stdout = getattr(response.artifacts, "stdout", "") or ""
        stderr = getattr(response.artifacts, "stderr", "") or ""
        try:
            out = json.loads(stdout.strip().split("\n")[-1] if stdout else "{}")
        except json.JSONDecodeError:
            out = {"stdout": stdout, "stderr": stderr}
        sandbox.delete(timeout=30)
        return {"ok": True, "result": out, "stderr": stderr}
    except Exception as e:
        logger.exception("Daytona sandbox run failed: %s", e)
        return {"ok": False, "error": str(e)}
