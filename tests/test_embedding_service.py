from __future__ import annotations

import asyncio
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import embedding_service  # noqa: E402


def test_find_similar_trades_uses_postgres_candidates(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "get_trade",
        lambda trade_id: {
            "id": trade_id,
            "run_id": "run-1",
            "entry_time": "2026-03-18T10:00:00+00:00",
            "exit_time": "2026-03-18T10:05:00+00:00",
            "direction": 1,
            "contracts": 1,
            "entry_price": 5000.0,
            "exit_price": 5001.0,
            "pnl": -10.0 if trade_id == 2 else 15.0,
            "zone": "open",
            "strategy": "test",
            "regime": "trend",
        },
    )
    monkeypatch.setattr(embedding_service, "get_embedding_from_postgres", lambda trade_id: [1.0, 0.0])
    monkeypatch.setattr(
        embedding_service,
        "get_candidate_embeddings",
        lambda exclude_trade_id: [
            {"trade_id": 2, "embedding": [0.9, 0.1], "pnl": -10.0},
            {"trade_id": 3, "embedding": [0.1, 0.9], "pnl": 15.0},
        ],
    )

    results = asyncio.run(embedding_service.find_similar_trades(1, limit=2))

    assert [row["id"] for row in results] == [2, 3]
    assert results[0]["_score"] > results[1]["_score"]
