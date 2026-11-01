"""Small retrieval-quality predicate used by the retrieval fallback path.

This module does not decide whether a user question is answerable. The answer
model receives the collected evidence and reports any unsupported facts.
"""

from __future__ import annotations


def documents_insufficient(
    chunk_count: int,
    confidence: float,
    min_retrieval_confidence: float | None,
) -> bool:
    """Whether a text retrieval result is too weak to guide another search."""
    if chunk_count <= 0:
        return True
    return (
        min_retrieval_confidence is not None
        and confidence < min_retrieval_confidence
    )
