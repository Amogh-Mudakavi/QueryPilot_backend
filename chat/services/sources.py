"""
chat/services/sources.py

Deterministic source extraction for QueryPilot.

This module builds an INDEXED source list from an agent's final
evidence. It performs no LLM calls and makes no decisions about what
the answer says -- it only describes, in a stable/reproducible way,
where each piece of evidence came from.

Indexing contract (this is the piece that makes citations work):

    Each source row's "index" is the 1-based POSITION of the evidence
    item it was extracted from, in the exact same `evidence` list/
    order that was supplied to answer generation.

    The answer prompt (planner/prompts/answer.py) instructs the model
    to cite a claim using [n] where n is that same 1-based evidence
    position. Because both sides read the same list in the same
    order, index N here always means the same thing as marker [N] in
    the answer text.

This intentionally does NOT deduplicate source rows *across* evidence
items, because collapsing them would break the 1:1 correspondence
between citation markers and evidence positions. Duplicate rows
produced *within* a single evidence item are still deduplicated.

Workspace scoping:

    Sources are resolved against the current workspace. A document can
    only appear in the returned source list when it belongs to that
    workspace. Evidence referring to a document outside the supplied
    workspace is ignored rather than returned with incomplete metadata.
"""

from __future__ import annotations

from typing import Any

from documents.models import Document


def build_sources(
    evidence: list[Any],
    *,
    workspace_id: int,
) -> list[dict[str, Any]]:
    """
    Build an indexed source list from final evidence.
    """

    if not evidence:
        return []

    document_ids: set[int] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}

        document_id = item.get("document_id") or metadata.get("document_id")

        if document_id is not None:
            try:
                document_ids.add(int(document_id))
            except (TypeError, ValueError):
                pass

        sources_list = item.get("sources") or metadata.get("sources")

        if isinstance(sources_list, list):
            for src in sources_list:
                if not isinstance(src, dict):
                    continue

                did = src.get("document_id")

                if did is None:
                    continue

                try:
                    document_ids.add(int(did))
                except (TypeError, ValueError):
                    continue

    document_names: dict[int, str] = {}

    if document_ids:
        documents = Document.objects.filter(
            id__in=document_ids,
            workspace_id=workspace_id,
        ).only(
            "id",
            "title",
        )

        document_names = {document.id: document.title for document in documents}

    valid_document_ids = set(document_names.keys())

    sources: list[dict[str, Any]] = []

    for position, item in enumerate(
        evidence,
        start=1,
    ):
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}

        rows_for_item: list[dict[str, Any]] = []

        if item.get("source_type") == "product_knowledge":
            rows_for_item.append(
                {
                    "document_id": None,
                    "document_name": "QueryPilot",
                    "source_type": "product",
                }
            )

        document_id = item.get("document_id") or metadata.get("document_id")

        if document_id is not None:
            try:
                document_id = int(document_id)
            except (TypeError, ValueError):
                document_id = None

        if document_id is not None and document_id in valid_document_ids:
            row: dict[str, Any] = {
                "document_id": document_id,
                "document_name": document_names[document_id],
                "source_type": (
                    item.get("source_type") or metadata.get("type") or "document"
                ),
            }

            chunk_index = item.get("chunk_index") or metadata.get("chunk_index")

            page_start = item.get("page_start") or metadata.get("page_start")

            page_end = item.get("page_end") or metadata.get("page_end")

            heading = item.get("heading") or metadata.get("heading")

            table_name = item.get("table_name") or metadata.get("table_name")

            if chunk_index is not None:
                row["chunk_index"] = chunk_index

            if page_start is not None:
                row["page_start"] = page_start

            if page_end is not None:
                row["page_end"] = page_end

            if heading:
                row["heading"] = heading

            if table_name:
                row["table_name"] = table_name

            rows_for_item.append(row)

        sources_list = item.get("sources") or metadata.get("sources")

        if isinstance(sources_list, list):
            for src in sources_list:
                if not isinstance(src, dict):
                    continue

                doc_id = src.get("document_id")

                if doc_id is None:
                    continue

                try:
                    doc_id = int(doc_id)
                except (TypeError, ValueError):
                    continue

                if doc_id not in valid_document_ids:
                    continue

                table_name = src.get("table_name")

                table_row: dict[str, Any] = {
                    "document_id": doc_id,
                    "document_name": document_names[doc_id],
                    "source_type": "table",
                }

                if table_name:
                    table_row["table_name"] = table_name

                rows_for_item.append(table_row)

        seen_row_keys: set[tuple[Any, ...]] = set()

        for row in rows_for_item:
            row_key = (
                row.get("document_id"),
                row.get("chunk_index"),
                row.get("page_start"),
                row.get("page_end"),
                row.get("heading"),
                row.get("table_name"),
            )

            if row_key in seen_row_keys:
                continue

            seen_row_keys.add(row_key)

            row["index"] = position
            sources.append(row)

    return sources
