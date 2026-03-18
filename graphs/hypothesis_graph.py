"""
Hypothesis generation as a LangGraph DAG: observation → hypothesis (Grok) → parse.
Persisted checkpoints, run_id via thread_id, step-level telemetry. Advisory-only.
"""
from __future__ import annotations

import re
import uuid
from typing import Any
from typing_extensions import TypedDict

from langchain_xai import ChatXAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from meta_learner import get_meta_stats


class HypothesisState(TypedDict, total=False):
    """State for the hypothesis generation graph."""
    run_id: str
    regime_context: str
    prior_conclusions_summary: str
    meta_stats: dict[str, Any]
    generation: int
    parent_hypothesis_id: str | None
    raw_claims: str
    hypotheses: list[dict[str, Any]]


def _get_grok():
    """Lazy ChatXAI for Grok (grok-beta)."""
    return ChatXAI(model="grok-beta", temperature=0)


def gather_observation(state: HypothesisState) -> dict[str, Any]:
    """Node: ensure observation context (regime, prior conclusions, meta stats) is set."""
    meta = state.get("meta_stats")
    if meta is None:
        meta = get_meta_stats()
    return {
        "regime_context": state.get("regime_context") or "No specific regime",
        "prior_conclusions_summary": state.get("prior_conclusions_summary") or "No prior conclusions yet.",
        "meta_stats": meta,
    }


def generate_claims(state: HypothesisState) -> dict[str, Any]:
    """Node: call Grok to generate testable hypothesis claims (raw text)."""
    llm = _get_grok()
    system = (
        "You are a quantitative research assistant. Produce testable, falsifiable market hypotheses "
        "based on the given regime context and prior conclusions. Output concise hypothesis claims "
        "suitable for backtest/replay experiments. One hypothesis per line or in a short numbered list."
    )
    user = (
        f"Regime context: {state.get('regime_context', '')}\n\n"
        f"Prior conclusions (summary): {state.get('prior_conclusions_summary', '')}\n\n"
        f"Meta-learner stats: {state.get('meta_stats', {})}\n\n"
        "Generate 1-3 testable hypotheses."
    )
    msg = llm.invoke([("system", system), ("human", user)])
    raw = (msg.content or "").strip()
    return {"raw_claims": raw}


def parse_hypotheses(state: HypothesisState) -> dict[str, Any]:
    """Node: parse raw_claims into structured hypothesis list."""
    raw = state.get("raw_claims") or ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    claims = []
    for ln in lines:
        m = re.match(r"^\d+[.)]\s*(.+)$", ln)
        claims.append(m.group(1) if m else ln)
    if not claims:
        claims = [raw.strip()] if raw else []
    generation = state.get("generation", 1)
    parent_id = state.get("parent_hypothesis_id")
    regime = state.get("regime_context")
    hypotheses = [
        {
            "hypothesis_id": str(uuid.uuid4()),
            "generation": generation,
            "parent_hypothesis_id": parent_id,
            "claim_text": claim,
            "regime_context": regime,
            "status": "pending",
            "independent_var": None,
            "dependent_vars": None,
            "control_vars": None,
            "test_window": None,
            "mutation_directive": None,
            "mutation_type": None,
        }
        for claim in claims
    ]
    return {"hypotheses": hypotheses}


def build_hypothesis_graph():
    """Build and compile the hypothesis generation StateGraph with checkpointer."""
    builder = StateGraph(HypothesisState)
    builder.add_node("gather_observation", gather_observation)
    builder.add_node("generate_claims", generate_claims)
    builder.add_node("parse_hypotheses", parse_hypotheses)
    builder.add_edge(START, "gather_observation")
    builder.add_edge("gather_observation", "generate_claims")
    builder.add_edge("generate_claims", "parse_hypotheses")
    builder.add_edge("parse_hypotheses", END)
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


_hypothesis_graph = None


def get_hypothesis_graph():
    """Return singleton compiled hypothesis graph (with in-memory checkpointer)."""
    global _hypothesis_graph
    if _hypothesis_graph is None:
        _hypothesis_graph = build_hypothesis_graph()
    return _hypothesis_graph


def invoke_hypothesis_graph(
    regime_context: str = "",
    prior_conclusions_summary: str = "",
    generation: int = 1,
    parent_hypothesis_id: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Invoke the hypothesis graph and return the list of structured hypotheses.
    Uses thread_id in config for checkpoint persistence (run_id).
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
    graph = get_hypothesis_graph()
    initial: HypothesisState = {
        "run_id": run_id,
        "regime_context": regime_context,
        "prior_conclusions_summary": prior_conclusions_summary,
        "generation": generation,
        "parent_hypothesis_id": parent_hypothesis_id,
    }
    final = graph.invoke(initial, config=config)
    return final.get("hypotheses") or []
