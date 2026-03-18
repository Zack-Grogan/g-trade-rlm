"""
Conclusion engine as a LangGraph DAG: analysis → interpret (Grok) → parse verdict.
Persisted checkpoints, run_id via thread_id. Advisory-only; no execution changes.
"""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langchain_xai import ChatXAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver


class ConclusionState(TypedDict, total=False):
    """State for the conclusion engine graph."""
    run_id: str
    result_id: str
    analysis: dict[str, Any]
    raw_conclusion: str
    verdict: str
    confidence_score: float
    mutation_directive: str | None


def _get_grok():
    """Lazy ChatXAI for Grok."""
    return ChatXAI(model="grok-beta", temperature=0)


def interpret_analysis(state: ConclusionState) -> dict[str, Any]:
    """Node: Grok interprets experiment analysis and produces verdict + directive."""
    llm = _get_grok()
    analysis = state.get("analysis") or {}
    system = (
        "You are a quantitative research reviewer. Given experiment analysis (Sharpe, win rate, p-value, etc.), "
        "output a verdict: one of SUPPORTED, REJECTED, or INCONCLUSIVE; a confidence score between 0 and 1; "
        "and an optional mutation_directive (one sentence) for the next hypothesis generation. "
        "Output in this exact format on separate lines: VERDICT=<supported|rejected|inconclusive> "
        "CONFIDENCE=<0.0-1.0> MUTATION_DIRECTIVE=<sentence or none>"
    )
    user = f"Analysis:\n{analysis}"
    msg = llm.invoke([("system", system), ("human", user)])
    raw = (msg.content or "").strip()
    return {"raw_conclusion": raw}


def parse_conclusion(state: ConclusionState) -> dict[str, Any]:
    """Node: parse raw_conclusion into verdict, confidence_score, mutation_directive."""
    raw = state.get("raw_conclusion") or ""
    verdict = "inconclusive"
    confidence_score = 0.0
    mutation_directive = None
    for line in raw.splitlines():
        # Match prefix case-insensitively but preserve the value's original casing.
        upper = line.strip().upper()
        value = line.strip()
        if upper.startswith("VERDICT="):
            v = value[8:].strip().lower()
            if v in ("supported", "rejected", "inconclusive"):
                verdict = v
        elif upper.startswith("CONFIDENCE="):
            try:
                confidence_score = float(value[11:].strip())
                confidence_score = max(0.0, min(1.0, confidence_score))
            except ValueError:
                pass
        elif upper.startswith("MUTATION_DIRECTIVE="):
            md = value[19:].strip()
            if md and md.lower() != "none":
                mutation_directive = md
    return {
        "verdict": verdict,
        "confidence_score": confidence_score,
        "mutation_directive": mutation_directive,
    }


def build_conclusion_graph():
    """Build and compile the conclusion engine StateGraph with checkpointer."""
    builder = StateGraph(ConclusionState)
    builder.add_node("interpret_analysis", interpret_analysis)
    builder.add_node("parse_conclusion", parse_conclusion)
    builder.add_edge(START, "interpret_analysis")
    builder.add_edge("interpret_analysis", "parse_conclusion")
    builder.add_edge("parse_conclusion", END)
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


_conclusion_graph = None


def get_conclusion_graph():
    """Return singleton compiled conclusion graph."""
    global _conclusion_graph
    if _conclusion_graph is None:
        _conclusion_graph = build_conclusion_graph()
    return _conclusion_graph


def invoke_conclusion_graph(
    result_id: str,
    analysis: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Invoke the conclusion graph and return verdict, confidence_score, mutation_directive.
    """
    if run_id is None:
        run_id = result_id
    config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
    graph = get_conclusion_graph()
    initial: ConclusionState = {"run_id": run_id, "result_id": result_id, "analysis": analysis}
    final = graph.invoke(initial, config=config)
    return {
        "verdict": final.get("verdict", "inconclusive"),
        "confidence_score": final.get("confidence_score", 0.0),
        "mutation_directive": final.get("mutation_directive"),
    }
