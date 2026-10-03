"""Services Intent module for QueryPilot."""

from __future__ import annotations

from typing import Any

CAPABILITIES = {
    "SOURCE_METADATA",
    "DOCUMENT_SEARCH",
    "TABLE_QUERY",
    "CONVERSATION_MEMORY",
    "PRODUCT_KNOWLEDGE",
}

_STRUCTURE_FIELD_TOKENS = frozenset(
    {
        "column",
        "columns",
        "column names",
        "column headers",
        "field names",
        "fields",
        "schema",
        "structure",
        "format",
        "row count",
        "number of rows",
        "how many rows",
        "sheet names",
        "table names",
        "list of tables",
        "table list",
        "tables",
        "file names",
        "list of files",
        "file list",
        "file types",
        "uploaded files",
        "uploaded documents",
        "document list",
        "list of documents",
        "metadata",
        "data sources",
    }
)


_STRUCTURE_CONTAINER_WORDS = frozenset(
    {
        "file",
        "files",
        "document",
        "documents",
        "doc",
        "docs",
        "sheet",
        "sheets",
        "table",
        "tables",
        "workspace",
        "source",
        "sources",
        "column",
        "columns",
        "row",
        "rows",
        "field",
        "fields",
        "schema",
        "schemas",
        "dataset",
        "datasets",
        "pdf",
        "pdfs",
        "upload",
        "uploads",
        "uploaded",
    }
)
_STRUCTURE_ATTRIBUTE_WORDS = frozenset(
    {
        "name",
        "names",
        "title",
        "titles",
        "type",
        "types",
        "format",
        "formats",
        "status",
        "count",
        "number",
        "many",
        "size",
        "list",
        "header",
        "headers",
        "structure",
        "metadata",
        "total",
    }
)
_STRUCTURE_FILLER_WORDS = frozenset(
    {"of", "the", "a", "an", "in", "and", "are", "is", "how", "what", "all"}
)

ROW_RESULT_OPERATIONS = frozenset(
    {"calculate", "aggregate", "overview", "list", "compare"}
)


def _resolved_document_ids(references: list[dict[str, Any]]) -> set[int]:
    """Return distinct document IDs represented by resolved source matches."""
    document_ids: set[int] = set()
    for item in references:
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if not isinstance(match, dict):
            continue
        document_id = _as_int(match.get("document_id"))
        if document_id is not None and match.get("type") in {"document", "table"}:
            document_ids.add(document_id)
    return document_ids


def _documents_with_tables(
    resolved_document_ids: set[int],
    resolved_tables: list[dict[str, Any]],
    source_shapes: dict[int, bool],
) -> bool:
    if resolved_tables:
        return True
    return any(
        source_shapes.get(document_id, False) for document_id in resolved_document_ids
    )


def build_execution_plan(
    understanding: dict[str, Any],
    resolution: dict[str, Any],
    source_shapes: dict[int, bool] | None = None,
) -> dict[str, Any]:
    """Convert language-level extraction + deterministic resolution into a plan.

    Cross-document behavior is source-driven. The LLM's category_hint is only a
    semantic hint; it cannot suppress evidence from another resolved source.
    """
    if resolution.get("has_ambiguity"):
        matches = []
        for item in resolution.get("references", []):
            if item.get("status") == "AMBIGUOUS":
                matches.extend(item.get("matches", []))
        return {
            "status": "NEEDS_CLARIFICATION",
            "capabilities": [],
            "actions": [],
            "clarification_options": matches[:8],
        }

    category = str(understanding.get("category_hint") or "document").lower()
    references = resolution.get("references", [])
    shapes = source_shapes or {}

    if category == "conversation":
        return _plan("OK", ["CONVERSATION_MEMORY"], ["conversation_memory"])

    if category == "product":
        return _plan("OK", ["PRODUCT_KNOWLEDGE"], ["product_knowledge"])

    resolved_documents = [
        item
        for item in references
        if item.get("status") == "RESOLVED"
        and item.get("match", {}).get("type") == "document"
    ]
    resolved_tables = [
        item
        for item in references
        if item.get("status") == "RESOLVED"
        and item.get("match", {}).get("type") == "table"
    ]
    resolved_entities = [
        item
        for item in references
        if item.get("status") == "RESOLVED"
        and item.get("match", {}).get("type") == "entity"
    ]

    has_resolved_source = bool(resolved_documents or resolved_tables)
    wants_structure = _requests_workspace_structure(understanding)
    resolved_source_ids = _resolved_document_ids(references)
    cross_document = len(resolved_source_ids) >= 2
    has_structured_sources = _documents_with_tables(
        resolved_source_ids,
        resolved_tables,
        shapes,
    )

    if wants_structure and (has_resolved_source or not resolved_entities):
        return _plan("OK", ["SOURCE_METADATA"], ["source_metadata"])

    if cross_document:
        capabilities: list[str] = []
        actions: list[str] = []
        if has_structured_sources:
            capabilities.append("TABLE_QUERY")
            actions.append("table_query")

        capabilities.append("DOCUMENT_SEARCH")
        actions.append("document_search")
        return _plan(
            "OK",
            capabilities,
            actions,
            cross_document=True,
        )

    if category in {"metadata", "workspace"} and not has_resolved_source:
        return _plan("OK", ["SOURCE_METADATA"], ["source_metadata"])

    if category in {"metadata", "workspace"} and has_resolved_source:
        if has_structured_sources:
            return _plan("OK", ["TABLE_QUERY"], ["table_query"])
        return _plan("OK", ["DOCUMENT_SEARCH"], ["document_search"])

    if (
        category == "structured"
        and not resolved_tables
        and not resolved_entities
        and resolved_documents
        and not has_structured_sources
    ):
        return _plan("OK", ["DOCUMENT_SEARCH"], ["document_search"])

    if category == "structured" or resolved_tables or resolved_entities:
        capabilities = ["TABLE_QUERY"]
        actions = ["table_query"]
        if category == "mixed":
            capabilities.append("DOCUMENT_SEARCH")
            actions.append("document_search")
        return _plan("OK", capabilities, actions)

    if category == "mixed":
        return _plan(
            "OK",
            ["TABLE_QUERY", "DOCUMENT_SEARCH"],
            ["table_query", "document_search"],
        )

    return _plan("OK", ["DOCUMENT_SEARCH"], ["document_search"])


def _is_structure_field(field: Any) -> bool:
    normalized = " ".join(str(field or "").lower().split())
    if not normalized:
        return False
    if normalized in _STRUCTURE_FIELD_TOKENS:
        return True

    words = [word for word in normalized.replace("_", " ").split() if word.isalpha()]
    if not words or len(words) != len(normalized.replace("_", " ").split()):
        return False
    known = (
        _STRUCTURE_CONTAINER_WORDS
        | _STRUCTURE_ATTRIBUTE_WORDS
        | _STRUCTURE_FILLER_WORDS
    )
    return (
        all(word in known for word in words)
        and any(word in _STRUCTURE_CONTAINER_WORDS for word in words)
        and any(word in _STRUCTURE_ATTRIBUTE_WORDS for word in words)
    )


def _requests_workspace_structure(understanding: dict[str, Any]) -> bool:
    fields = understanding.get("requested_fields", []) or []
    return any(_is_structure_field(field) for field in fields)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _plan(
    status: str,
    capabilities: list[str],
    actions: list[str],
    *,
    cross_document: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "capabilities": [item for item in capabilities if item in CAPABILITIES],
        "actions": actions,
        "clarification_options": [],
        "cross_document": cross_document,
    }
