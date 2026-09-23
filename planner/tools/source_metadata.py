from __future__ import annotations

from typing import Any

from planner.services.entity_resolution import normalize_text
from planner.services.schema_registry import get_workspace_catalog


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_metadata(
    *,
    workspace_id: int,
    reference: str | None = None,
    document_id: int | None = None,
) -> dict[str, Any]:
    """
    Workspace document/table catalog metadata.

    Three call modes (every parameter except workspace_id is optional,
    so existing callers are unaffected):

    1. No reference, no document_id -> the FULL catalog. Used for
       generic catalog questions ("tell me about documents") and for
       building the available-documents hint. Always FOUND: an empty
       workspace is a valid catalog, not a failed lookup.

    2. document_id -> only that document. The resolved-match path from
       the planner (entity_resolution already proved the document or
       table exists). A question about one specific document is
       answered with that document's metadata, never the whole
       catalog.

    3. reference -> only documents whose title (or one of their table
       names) normalizes exactly to the reference. Fallback for
       resolved matches that carry no document_id.

    A specific filter (document_id or reference) that matches nothing
    returns status="NOT_FOUND" with an empty catalog - NOT the full
    catalog - so a question about a missing named source is never
    answered with unrelated documents.

    document_id takes priority when both are supplied.
    """

    catalog = get_workspace_catalog(workspace_id)

    filtered = False

    target_id = _as_int(document_id) if document_id is not None else None

    if target_id is not None:
        filtered = True
        catalog = [
            item for item in catalog
            if _as_int(item.get("document_id")) == target_id
        ]
    elif reference:
        filtered = True
        needle = normalize_text(reference)
        catalog = [
            item for item in catalog
            if normalize_text(str(item.get("title", ""))) == needle
            or any(
                normalize_text(str(table.get("table_name", ""))) == needle
                for table in item.get("tables", [])
            )
        ]

    sources = []
    for document in catalog:
        for table in document.get("tables", []):
            sources.append({
                "document_id": document.get("document_id"),
                "table_name": table.get("table_name"),
            })
        if not document.get("tables"):
            sources.append({
                "document_id": document.get("document_id"),
            })

    return {
        "source_type": "workspace_metadata",
        "workspace_id": workspace_id,
        "status": "NOT_FOUND" if (filtered and not catalog) else "FOUND",
        "sources": sources,
        "content": {
            "document_count": len(catalog),
            "documents": catalog,
        },
    }