"""Tools Search Evidence module for QueryPilot."""

from __future__ import annotations

import logging
from typing import Any

from retrieval.services.hybrid_retriever import (
    HybridRetriever,
    RetrievalError,
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

        remaining = MAX_TOTAL_EVIDENCE_CHARS - total_chars

        if remaining <= 0:
            break

        content_limit = min(
            MAX_CHARS_PER_EVIDENCE,
            remaining,
        )

        bounded_item["content"] = content[:content_limit]

        total_chars += len(bounded_item["content"])

        bounded.append(bounded_item)

    return bounded


def _error_result(
    *,
    error: RetrievalError,
) -> dict[str, Any]:
    """Return a stable structured retrieval error for the agent."""

    return {
        "error_code": error.code,
        "error": error.message,
        "evidence": [],
        "confidence": 0.0,
        "candidate_count": 0,
    }


def search_evidence(
    *,
    query: str,
    workspace_id: int,
    allowed_document_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Retrieve bounded text evidence from workspace documents.

    This tool is deliberately text-only. Table metadata is handled by
    the table/evidence inspection path and should not compete with
    ordinary document text retrieval here.

    If allowed_document_ids is supplied, retrieval is restricted
    to exactly those document IDs.

    An empty allow-list is treated as no allowed documents and
    therefore returns no evidence.

    Retrieval failures are returned as structured data so the agent
    can distinguish an empty result from an infrastructure/indexing
    failure without depending on raw Chroma exception strings.
    """

    query = query.strip()

    if not query:
        raise ValueError("search_evidence query cannot be empty.")

    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    logger.info(
        "search_evidence: workspace_id=%s allowed_document_ids=%s source_types=%s",
        workspace_id,
        allowed_document_ids,
        ["text"],
    )

    try:
        result = _retriever.retrieve(
            query=query,
            workspace_id=workspace_id,
            allowed_document_ids=allowed_document_ids,
            source_types=["text"],
        )
    except RetrievalError as exc:
        logger.error(
            "search_evidence: retrieval failed code=%s message=%s",
            exc.code,
            exc.message,
        )
        return _error_result(error=exc)

    raw_evidence = [item.to_dict() for item in result.evidence]

    evidence = _bound_evidence(raw_evidence)

    logger.info(
        "search_evidence: returning evidence=%d candidate_count=%s confidence=%.4f",
        len(evidence),
        result.candidate_count,
        result.confidence,
    )

    if result.dense_failed:
        logger.warning(
            "search_evidence: dense retrieval unavailable for "
            "allowed_document_ids=%s; result is lexical-only (degraded).",
            allowed_document_ids,
        )

    return {
        "error_code": None,
        "error": None,
        "evidence": evidence,
        "confidence": result.confidence,
        "candidate_count": result.candidate_count,
        "degraded": bool(result.dense_failed),
    }


def search_evidence_by_keys(
    *,
    keys: list[str],
    workspace_id: int,
    allowed_document_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Exact stable-key lookup in document text (embedding-independent).

    Same workspace / document scoping and same structured error contract as
    search_evidence. Used to fetch companion evidence with a key that was
    resolved from another source (PDF, spreadsheet row, ...).
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    logger.info(
        "search_evidence_by_keys: workspace_id=%s allowed_document_ids=%s keys=%s",
        workspace_id,
        allowed_document_ids,
        keys,
    )

    try:
        result = _retriever.lookup_by_keys(
            keys=keys,
            workspace_id=workspace_id,
            allowed_document_ids=allowed_document_ids,
            source_types=["text"],
        )
    except RetrievalError as exc:
        logger.error(
            "search_evidence_by_keys: lookup failed code=%s message=%s",
            exc.code,
            exc.message,
        )
        return {**_error_result(error=exc), "matched_keys": {}, "non_identifying": []}

    evidence = _bound_evidence([item.to_dict() for item in result.evidence])

    return {
        "error_code": None,
        "error": None,
        "evidence": evidence,
        "confidence": 1.0 if evidence else 0.0,
        "candidate_count": len(evidence),
        "matched_keys": result.matched_keys,
        "non_identifying": result.non_identifying,
    }


def fetch_document_chunks(
    *,
    workspace_id: int,
    document_ids: list[int],
) -> dict[str, Any]:
    """
    All text chunks of the given documents, in reading order.

    Workspace- and document-scoped like every other retrieval entry point;
    failures come back as the same structured error contract.
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    try:
        chunks = _retriever.fetch_document_chunks(
            workspace_id=workspace_id,
            document_ids=document_ids,
        )
    except RetrievalError as exc:
        logger.error(
            "fetch_document_chunks: failed code=%s message=%s",
            exc.code,
            exc.message,
        )
        return {**_error_result(error=exc), "chunks": []}

    return {
        "error_code": None,
        "error": None,
        "chunks": chunks,
    }
