"""Tools Source Metadata module for QueryPilot."""

from __future__ import annotations

import os
from typing import Any, Iterable

from planner.services.entity_resolution import normalize_text
from planner.services.schema_registry import get_workspace_catalog


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_type(title: str, source_type: Any) -> str | None:
    extension = os.path.splitext(str(title or ""))[1].lstrip(".").lower()
    return extension or (str(source_type).lower() if source_type else None)


def _title_stem(title: str) -> str:
    return os.path.splitext(str(title or ""))[0]


def _document_entry(
    document: dict[str, Any], *, include_tables: bool
) -> dict[str, Any]:
    """
    Document-first metadata record.

    `document_name` is the human-readable file name / title and is what a
    user means by "which documents are uploaded". `document_id` is an
    internal id. Table identifiers are internal storage names and are only
    exposed when the request concerns one specific source.
    """
    title = str(document.get("title") or "")
    tables = document.get("tables", []) or []
    entry: dict[str, Any] = {
        "document_id": document.get("document_id"),
        "document_name": title,
        "title": title,
        "file_type": _file_type(title, document.get("source_type")),
        "source_type": document.get("source_type"),
        "status": document.get("status"),
        "table_count": len(tables),
    }
    if include_tables:
        entry["tables"] = [
            {
                "table_name": table.get("table_name"),
                "columns": table.get("columns"),
                "row_count": table.get("row_count"),
            }
            for table in tables
        ]
    return entry


def _matches_reference(item: dict[str, Any], needle: str) -> bool:
    """
    A reference matches a document when it normalizes exactly to the
    document title, the title without its file extension
    ("Employee_Master" for "Employee_Master.pdf"), or one of its table
    names. Exact matching only: no fuzzy or substring guessing.
    """
    title = str(item.get("title", ""))
    if normalize_text(title) == needle or normalize_text(_title_stem(title)) == needle:
        return True
    return any(
        normalize_text(str(table.get("table_name", ""))) == needle
        for table in item.get("tables", []) or []
    )


def source_metadata(
    *,
    workspace_id: int,
    reference: str | None = None,
    document_id: int | None = None,
    allowed_document_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """
    Workspace document/table catalog metadata. Pure catalog lookup:
    no SQL over table rows, no chunk retrieval, no LLM tokens.

    Isolation:
        The catalog is always scoped to `workspace_id`. When the
        conversation restricts its documents, pass them as
        `allowed_document_ids`; only those documents can ever be
        listed, matched, or counted. None means "no conversation-level
        restriction" (workspace scope still applies). An EMPTY
        collection means "no documents allowed" and yields an empty
        (FOUND) catalog, never the full workspace.

    Three call modes (every parameter except workspace_id is optional):

    1. No reference, no document_id -> the FULL (allowed) catalog: one
       record per uploaded document (name, file type, status). Table
       identifiers are NOT included: they are storage details, not what
       "which documents are uploaded" asks for. Always FOUND (an empty
       workspace is a valid catalog).

    2. document_id -> only that document, including its tables (columns,
       row counts).

    3. reference -> only documents whose title (with or without its file
       extension) or one of their table names normalizes exactly to the
       reference, including tables.

    A specific filter that matches nothing returns status="NOT_FOUND" with
    an empty catalog, never the full catalog. document_id takes priority.
    """

    catalog = get_workspace_catalog(workspace_id)

    if allowed_document_ids is not None:
        allowed = {
            value
            for value in (_as_int(item) for item in allowed_document_ids)
            if value is not None
        }
        catalog = [
            item for item in catalog if _as_int(item.get("document_id")) in allowed
        ]

    filtered = False
    target_id = _as_int(document_id) if document_id is not None else None

    if target_id is not None:
        filtered = True
        catalog = [
            item for item in catalog if _as_int(item.get("document_id")) == target_id
        ]
    elif reference:
        filtered = True
        needle = normalize_text(reference)
        catalog = [item for item in catalog if _matches_reference(item, needle)]

    documents = [
        _document_entry(document, include_tables=filtered) for document in catalog
    ]

    sources = [
        {
            "document_id": entry["document_id"],
            "document_name": entry["document_name"],
        }
        for entry in documents
    ]

    return {
        "source_type": "workspace_metadata",
        "workspace_id": workspace_id,
        "status": "NOT_FOUND" if (filtered and not catalog) else "FOUND",
        "sources": sources,
        "content": {
            "evidence_role": "workspace_metadata",
            "document_count": len(documents),
            "document_names": [e["document_name"] for e in documents],
            "documents": documents,
        },
    }
