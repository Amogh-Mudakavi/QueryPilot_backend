from __future__ import annotations

import logging
from typing import Any

from retrieval.services.hybrid_retriever import (
    HybridRetriever,
)


_retriever = HybridRetriever()
logger = logging.getLogger(__name__)

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
    allowed_document_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Retrieve bounded document evidence.

    If allowed_document_ids is supplied, retrieval is restricted
    to exactly those document IDs.

    An empty allow-list is treated as no allowed documents and
    therefore returns no evidence.
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

    logger.info(
        "search_evidence: workspace_id=%s allowed_document_ids=%s",
        workspace_id,
        allowed_document_ids,
    )

    result = _retriever.retrieve(
        query=query,
        workspace_id=workspace_id,
        allowed_document_ids=allowed_document_ids,
    )

    raw_evidence = [
        item.to_dict()
        for item in result.evidence
    ]

    evidence = _bound_evidence(
        raw_evidence
    )

    logger.info(
        "search_evidence: returning evidence=%d candidate_count=%s confidence=%.4f",
        len(evidence),
        result.candidate_count,
        result.confidence,
    )

    return {
        "evidence": evidence,
        "confidence": result.confidence,
        "candidate_count": result.candidate_count,
    }