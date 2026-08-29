"""
planner/tools/search_evidence.py

LangGraph tool for retrieving document evidence.

The tool is intentionally thin.

Responsibilities:
    - Accept a search query and workspace.
    - Call the existing HybridRetriever.
    - Return bounded, serializable evidence.

It does NOT:
    - decide whether retrieval is needed
    - inspect whether evidence is sufficient
    - query SQLite
    - call the LLM
    - decide when the agent should stop
"""

from __future__ import annotations

from typing import Any

from retrieval.services.hybrid_retriever import (
    HybridRetriever,
)


_retriever = HybridRetriever()

MAX_EVIDENCE_ITEMS = 8
MAX_CHARS_PER_EVIDENCE = 6000
MAX_TOTAL_EVIDENCE_CHARS = 24000


def _bound_evidence(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Bound retrieved evidence before it enters AgentState.

    Metadata is preserved. Only the content body is truncated.
    """

    bounded: list[dict[str, Any]] = []
    total_chars = 0

    for item in evidence[:MAX_EVIDENCE_ITEMS]:

        if not isinstance(item, dict):
            continue

        bounded_item = dict(item)

        content = str(
            bounded_item.get(
                "content",
                "",
            )
        )

        remaining = (
            MAX_TOTAL_EVIDENCE_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        content_limit = min(
            MAX_CHARS_PER_EVIDENCE,
            remaining,
        )

        bounded_item["content"] = (
            content[:content_limit]
        )

        total_chars += len(
            bounded_item["content"]
        )

        bounded.append(
            bounded_item
        )

    return bounded


def search_evidence(
    *,
    query: str,
    workspace_id: int,
) -> dict[str, Any]:
    """
    Retrieve the strongest bounded document evidence.

    Args:
        query:
            Search query produced by the agent.

        workspace_id:
            Current workspace.

    Returns:
        Dictionary containing:
            - evidence
            - confidence
            - candidate_count

    Raises:
        ValueError:
            If query is empty or workspace_id is invalid.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "search_evidence query cannot be empty."
        )

    if workspace_id <= 0:
        raise ValueError(
            "workspace_id must be greater than zero."
        )

    result = _retriever.retrieve(
        query=query,
        workspace_id=workspace_id,
    )

    raw_evidence = [
        item.to_dict()
        for item in result.evidence
    ]

    evidence = _bound_evidence(
        raw_evidence
    )

    return {
        "evidence": evidence,
        "confidence": result.confidence,
        "candidate_count": result.candidate_count,
    }