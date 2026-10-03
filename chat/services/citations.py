"""
Deterministic citation parsing and validation for QueryPilot.

This module never calls an LLM and never invents or reorders sources.
It only maps citation markers already present in the final answer to the
real, indexed sources produced from the agent's final evidence.

The citation marker [n] means evidence position n. A marker is accepted only
when that evidence position produced a real source row for the active
workspace.

Semantic claim-to-source correctness is intentionally not inferred here:
this module validates the citation indexes and their mapping, while the
Answer Generator prompt is responsible for choosing the sources that support
its claims.
"""

from __future__ import annotations

import re
from typing import Any

from chat.services.sources import build_sources

CITATION_MARKER_PATTERN = re.compile(r"\[(\d{1,3})\]")


def find_marker_numbers(answer: str) -> set[int]:
    """
    Return every distinct citation number explicitly present in ``answer``.

    The result includes both valid and invalid numbers. Validation against
    the real source list happens in ``extract_citations``.
    """
    if not answer:
        return set()

    return {int(match.group(1)) for match in CITATION_MARKER_PATTERN.finditer(answer)}


def extract_citations(
    *,
    answer: str,
    evidence: list[Any],
    workspace_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build the deterministic ``(sources, citations)`` pair for an answer.

    ``sources`` contains the complete indexed source list generated from the
    supplied evidence. ``citations`` contains one entry for each distinct,
    valid [n] marker actually present in the answer.

    Citation numbering is tied to evidence position, not document IDs,
    chunk indexes, page numbers, table names, or any other identifier.

    Invalid/out-of-range markers are never converted into citations.
    Repeated markers are allowed in the answer text but appear only once in
    the returned citation list.
    """
    sources = build_sources(evidence, workspace_id=workspace_id)

    if not answer or not sources:
        return sources, []

    valid_indices = {
        int(source["index"])
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int)
    }

    marker_numbers = find_marker_numbers(answer)

    citations = [
        {"index": marker, "source_index": marker}
        for marker in sorted(marker_numbers & valid_indices)
    ]

    return sources, citations
