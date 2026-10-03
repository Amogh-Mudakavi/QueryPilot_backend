"""
planner/services/sufficiency.py

Deterministic "did we actually get something that can answer this?" checks.

    retrieved something  !=  retrieved something that answers the question

No LLM calls, no Django. Everything here is a pure function so it can be
unit-tested without a database.

Retrieval confidence is only used when a threshold is configured
(`min_retrieval_confidence`). Do NOT guess that number: print real
`confidence` values for queries your documents answer and queries they
don't, then set it from data.
"""

from __future__ import annotations

from typing import Any

TABLE_NO_RESULT = "TABLE_NO_RESULT"
TABLE_ZERO_ROWS = "TABLE_ZERO_ROWS"
TABLE_ERROR = "TABLE_ERROR"
NO_DOCUMENT_EVIDENCE = "NO_DOCUMENT_EVIDENCE"
LOW_RETRIEVAL_CONFIDENCE = "LOW_RETRIEVAL_CONFIDENCE"

_DOCUMENT_REASONS = {NO_DOCUMENT_EVIDENCE, LOW_RETRIEVAL_CONFIDENCE}


def documents_insufficient(
    chunk_count: int,
    confidence: float,
    min_retrieval_confidence: float | None,
) -> bool:
    """True when a document search did not return usable evidence."""

    if chunk_count <= 0:
        return True

    if min_retrieval_confidence is not None and confidence < min_retrieval_confidence:
        return True

    return False


def assess_evidence(
    *,
    wants_table: bool,
    table_result_status: str | None,
    wants_documents: bool,
    document_chunk_count: int,
    retrieval_confidence: float,
    min_retrieval_confidence: float | None,
) -> dict[str, Any]:
    """
    Decide whether the collected evidence can answer the question.

    Returns {"sufficient": bool, "reasons": [str, ...]}.
    """

    reasons: list[str] = []

    if wants_table:
        if table_result_status is None:
            reasons.append(TABLE_NO_RESULT)
        elif table_result_status == "NOT_FOUND":
            reasons.append(TABLE_ZERO_ROWS)
        elif table_result_status == "RETRIEVAL_ERROR":
            reasons.append(TABLE_ERROR)

    if wants_documents:
        if document_chunk_count <= 0:
            reasons.append(NO_DOCUMENT_EVIDENCE)
        elif (
            min_retrieval_confidence is not None
            and retrieval_confidence < min_retrieval_confidence
        ):
            reasons.append(LOW_RETRIEVAL_CONFIDENCE)

    return {
        "sufficient": not reasons,
        "reasons": reasons,
    }


def insufficiency_note(reasons: list[str]) -> dict[str, Any] | None:
    """
    An evidence item the answer model already understands
    (INSUFFICIENT_EVIDENCE is an authoritative state in the answer prompt).

    Only produced for document-side problems; table-side problems are
    already carried by the table evidence item's own result_status.
    """

    document_reasons = [r for r in reasons if r in _DOCUMENT_REASONS]

    if not document_reasons:
        return None

    if NO_DOCUMENT_EVIDENCE in document_reasons:
        text = (
            "No relevant passages were found in the workspace documents "
            "for this question."
        )
    else:
        text = (
            "The retrieved document passages have low relevance to this "
            "question; they may not actually answer it."
        )

    return {
        "source_type": "retrieval_assessment",
        "result_status": "INSUFFICIENT_EVIDENCE",
        "content": text,
    }
