"""
planner/services/document_coverage.py

Question-agnostic evidence assembly for requests that span 2+ documents.

Instead of "top-k of my query per document", the rule is:

  1. Load the document's chunks (they are not ranked, just read).
  2. If everything fits in the context budget, send EVERYTHING, in reading
     order. Nothing can be missed, so "why was Q1 better than Q2?" works
     without the question sharing words with the answer.
  3. If it does not fit, split the budget fairly between documents (a small
     document never starves a big one, and a big one never crowds out a
     small one), then fill each document's share with the most relevant
     chunks (BM25 over several query variants, boosted by the normal
     hybrid retrieval when available), their neighbours, and the opening
     chunk, and put them back into reading order.
  4. Tell the answer model exactly what it was given (full vs selected),
     so it says "not found in the provided passages" instead of guessing.

Nothing here depends on file format, entity names, IDs or the question.
Pure functions plus injected loaders, so it is unit-testable without
Django, Chroma or an LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable

from retrieval.services.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


DOCUMENT_CONTEXT_CHAR_BUDGET = 48_000

SEGMENT_CHAR_LIMIT = 8_000
MAX_QUERIES = 6
NEIGHBOUR_SEEDS = 3


def build_queries(question: str, understanding: dict[str, Any] | None) -> list[str]:
    """Several views of the question: raw text, then the planner's pieces."""
    understanding = understanding or {}
    parts: list[str] = [str(question or "").strip()]
    fields = [str(x) for x in understanding.get("requested_fields", []) or []]
    refs = [str(x) for x in understanding.get("references", []) or []]
    operation = str(understanding.get("operation") or "")

    if fields:
        parts.append(" ".join([operation, *fields]).strip())
    parts.extend(fields)
    parts.extend(refs)

    seen: set[str] = set()
    queries: list[str] = []
    for part in parts:
        key = " ".join(part.lower().split())
        if key and key not in seen:
            seen.add(key)
            queries.append(part)
    return queries[:MAX_QUERIES]


def allocate_budget(sizes: dict[int, int], total_budget: int) -> dict[int, int]:
    """
    Fair share with redistribution ("water-filling"): smallest documents are
    satisfied first and whatever they leave unused goes to the larger ones.
    """
    remaining = max(int(total_budget), 0)
    pending = sorted(sizes, key=lambda d: (sizes[d], d))
    allocation: dict[int, int] = {}
    for position, doc_id in enumerate(pending):
        share = remaining // (len(pending) - position)
        allocation[doc_id] = min(sizes[doc_id], share)
        remaining -= allocation[doc_id]
    return allocation


def select_chunks(
    chunks: list[dict[str, Any]],
    queries: list[str],
    budget: int,
    boosted: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """
    Choose the chunks of ONE document that fit `budget` characters.
    Returns them in reading order.
    """
    chunks = sorted(chunks, key=lambda c: c["chunk_index"])
    if sum(len(c["content"]) for c in chunks) <= budget:
        return chunks

    boosted = set(boosted)
    scores = [0.0] * len(chunks)
    if queries:
        bm25 = BM25Retriever([c["content"] for c in chunks])
        for query in queries:
            raw = bm25.get_scores(query)
            top = max(raw) if raw else 0.0
            if top > 0:
                for i, value in enumerate(raw):
                    scores[i] += value / top
    for i, chunk in enumerate(chunks):
        if chunk["chunk_index"] in boosted:
            scores[i] += 1.0

    by_score = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    positive = [i for i in by_score if scores[i] > 0]

    priority: list[int] = list(positive)
    for i in positive[:NEIGHBOUR_SEEDS]:
        priority.extend(j for j in (i - 1, i + 1) if 0 <= j < len(chunks))
    priority.append(0)
    priority.extend(by_score)

    chosen: list[int] = []
    used = 0
    seen: set[int] = set()
    for i in priority:
        if i in seen:
            continue
        seen.add(i)
        size = len(chunks[i]["content"])
        if used + size <= budget:
            chosen.append(i)
            used += size
    return [chunks[i] for i in sorted(chosen)]


def _pages(chunks: list[dict[str, Any]]) -> str:
    starts = [c["metadata"].get("page_start") for c in chunks]
    ends = [c["metadata"].get("page_end") for c in chunks]
    starts = [s for s in starts if isinstance(s, int)]
    ends = [e for e in ends if isinstance(e, int)]
    if not starts or not ends:
        return ""
    lo, hi = min(starts), max(ends)
    return f", page {lo}" if lo == hi else f", pages {lo}-{hi}"


def pack_segments(
    *,
    workspace_id: int,
    document_id: int,
    title: str,
    selected: list[dict[str, Any]],
    total: int,
    mode: str,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    size = 0
    for chunk in selected:
        length = len(chunk["content"])
        if not groups or size + length > SEGMENT_CHAR_LIMIT:
            groups.append([])
            size = 0
        groups[-1].append(chunk)
        size += length

    items: list[dict[str, Any]] = []
    for group in groups:
        first, last = group[0]["chunk_index"], group[-1]["chunk_index"]
        body: list[str] = []
        previous = None
        for chunk in group:
            if previous is not None and chunk["chunk_index"] != previous + 1:
                body.append("[... passages not included ...]")
            body.append(chunk["content"].strip())
            previous = chunk["chunk_index"]
        header = (
            f"[Source: {title} (document_id={document_id}) | "
            f"passages {first + 1}-{last + 1} of {total}{_pages(group)}]"
        )
        items.append(
            {
                "content": header + "\n" + "\n\n".join(body),
                "score": 1.0 if mode == "full" else 0.8,
                "source_type": "text_chunk",
                "workspace_id": workspace_id,
                "document_id": document_id,
                "document_name": title,
                "metadata": {
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "type": "text",
                    "chunk_indexes": [c["chunk_index"] for c in group],
                    "retrieval_method": (
                        "full_document" if mode == "full" else "fair_share_selection"
                    ),
                },
                "result_status": "SUCCESS",
            }
        )
    return items


def build_document_evidence(
    *,
    workspace_id: int,
    document_ids: Iterable[int],
    question: str,
    understanding: dict[str, Any] | None,
    titles: dict[int, str],
    fetch_chunks: Callable[..., dict[str, Any]],
    boost_lookup: Callable[[int], tuple[set[int], bool]] | None = None,
    budget: int = DOCUMENT_CONTEXT_CHAR_BUDGET,
) -> dict[str, Any] | None:
    """
    Returns {"evidence", "notes", "coverage", "confidence", "degraded"} or
    None when chunks could not be loaded (caller falls back to ranked search).
    """
    doc_ids = sorted({int(d) for d in document_ids})
    fetched = fetch_chunks(workspace_id=workspace_id, document_ids=doc_ids)
    if fetched.get("error_code"):
        logger.warning(
            "document coverage unavailable (%s); falling back to ranked search",
            fetched.get("error_code"),
        )
        return None

    by_doc: dict[int, list[dict[str, Any]]] = {d: [] for d in doc_ids}
    for chunk in fetched.get("chunks", []):
        if chunk["document_id"] in by_doc:
            by_doc[chunk["document_id"]].append(chunk)

    sizes = {d: sum(len(c["content"]) for c in cs) for d, cs in by_doc.items() if cs}
    allocation = allocate_budget(sizes, budget)
    queries = build_queries(question, understanding)

    evidence: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    degraded = False

    for doc_id in doc_ids:
        chunks = by_doc[doc_id]
        title = titles.get(doc_id) or f"document {doc_id}"
        if not chunks:
            coverage.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "mode": "none",
                    "included": 0,
                    "total": 0,
                }
            )
            continue

        full = allocation[doc_id] >= sizes[doc_id]
        boosted: set[int] = set()
        if not full and boost_lookup is not None:
            try:
                boosted, was_degraded = boost_lookup(doc_id)
                degraded = degraded or was_degraded
            except Exception:
                logger.exception("boost lookup failed for document %s", doc_id)

        selected = (
            sorted(chunks, key=lambda c: c["chunk_index"])
            if full
            else select_chunks(chunks, queries, allocation[doc_id], boosted)
        )
        mode = "full" if full else "selected"
        evidence.extend(
            pack_segments(
                workspace_id=workspace_id,
                document_id=doc_id,
                title=title,
                selected=selected,
                total=len(chunks),
                mode=mode,
            )
        )
        coverage.append(
            {
                "document_id": doc_id,
                "title": title,
                "mode": mode,
                "included": len(selected),
                "total": len(chunks),
            }
        )

    notes: list[dict[str, Any]] = []
    with_text = [c for c in coverage if c["mode"] != "none"]
    if with_text:
        lines = []
        for c in with_text:
            if c["mode"] == "full":
                lines.append(
                    f"'{c['title']}': the COMPLETE text ({c['total']} passages) is provided."
                )
            else:
                lines.append(
                    f"'{c['title']}': only {c['included']} of {c['total']} passages "
                    "are provided (selected for relevance because the document is large)."
                )
        partial = any(c["mode"] == "selected" for c in with_text)
        guidance = (
            "If the answer needs content that is not in the provided passages, "
            "say it was not found in the provided passages; do not guess."
            if partial
            else "Anything not present in this text is genuinely absent from "
            "these documents; say so rather than guessing."
        )
        notes.append(
            {
                "source_type": "evidence_note",
                "result_status": "SUCCESS",
                "content": "Document coverage. " + " ".join(lines) + " " + guidance,
            }
        )

    logger.info("document coverage: %s degraded=%s", coverage, degraded)
    return {
        "evidence": evidence,
        "notes": notes,
        "coverage": coverage,
        "confidence": 1.0 if all(c["mode"] != "selected" for c in coverage) else 0.6,
        "degraded": degraded,
    }
