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

# Operations that read/describe a source as a whole. They are document-content
# operations: they never need a table lookup, and over several sources they are
# independent per-source retrievals, not a relationship between the sources.
OVERVIEW_OPERATIONS = frozenset({"overview"})

# Operations that, over several named sources, are just independent reads of
# each source (what does each one say / list / describe). They only become
# cross-source reasoning when an entity is involved or the planner marked the
# request "mixed" (a value that depends on a rule in another source).
INDEPENDENT_MULTI_SOURCE_OPERATIONS = frozenset(
    {"overview", "describe", "list", "answer"}
)

# Operations whose natural result is rows computed from a structured source.
# Only these can make a document-only request prefer TABLE_QUERY (and even then
# DOCUMENT_SEARCH is kept as a fallback). A plain lookup/explanation about a
# document is answered from the document's content.
TABLE_ROW_OPERATIONS = frozenset({"calculate", "aggregate", "filter", "list", "lookup"})

_TABLE_CONTENT_WORDS = frozenset(
    {
        "table",
        "tables",
        "row",
        "rows",
        "column",
        "columns",
        "sheet",
        "sheets",
        "cell",
        "cells",
    }
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


def normalized_operation(understanding: dict[str, Any] | None) -> str:
    """Planner operation as a lowercase string ("" when absent)."""
    return str((understanding or {}).get("operation") or "").strip().lower()


def _requests_table_content(understanding: dict[str, Any]) -> bool:
    """True when the extracted fields ask for table/sheet content itself."""
    for field in understanding.get("requested_fields", []) or []:
        words = {
            word
            for word in str(field or "").lower().replace("_", " ").split()
            if word.isalpha()
        }
        if words & _TABLE_CONTENT_WORDS:
            return True
    return False


def _decide_cross_document(
    *,
    source_ids: set[int],
    operation: str,
    category: str,
    has_entities: bool,
) -> tuple[bool, str]:
    """
    Cross-document REASONING is a property of the request, not of how many
    sources are in scope. Several sources in scope only means several sources
    to retrieve from; reasoning across them is needed when the task relates,
    compares, joins or calculates across them.
    """
    if len(source_ids) < 2:
        # A mixed request is explicitly defined as structured data plus rules
        # or formulas from document evidence. The companion source may not be
        # resolved by name on the first pass, so do not disable cross-source
        # reasoning solely because only the data source resolved initially.
        if category == "mixed" and source_ids:
            return True, "mixed request requires rules beyond the initially resolved source"
        return False, "fewer than two resolved sources"
    if (
        operation in INDEPENDENT_MULTI_SOURCE_OPERATIONS
        and category != "mixed"
        and not has_entities
    ):
        return False, (
            f"independent multi-source {operation}: each source is retrieved "
            "on its own, no relationship between sources required"
        )
    return True, (
        f"{len(source_ids)} resolved sources and operation="
        f"{operation or 'unspecified'} requires combining sources"
    )


def build_execution_plan(
    understanding: dict[str, Any],
    resolution: dict[str, Any],
    source_shapes: dict[int, bool] | None = None,
    pending_value_reference: bool = False,
) -> dict[str, Any]:
    """Convert language-level extraction + deterministic resolution into a plan.

    pending_value_reference: the caller found a non-source reference (a person,
    id, value...) that is not resolved yet. It still has to be looked up in
    structured data, so the request is not a pure document-content request.

    Three separate questions are answered here, in this order:

      1. SOURCE SCOPE   - which documents/tables were explicitly resolved.
      2. RETRIEVAL      - the minimum evidence capabilities the request needs.
      3. CROSS-DOCUMENT - whether answering needs a relationship/comparison/
                          calculation ACROSS sources (not merely >1 source).

    A source's physical shape (e.g. a PDF that has extracted tables) never
    overrides the semantic intent: asking what a document says is a
    DOCUMENT_SEARCH even when the document also owns tables. The LLM's
    category_hint is a semantic hint; the resolved sources are deterministic.
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
            "cross_document": False,
            "fallback_capabilities": [],
            "decision": {"reason": "ambiguous source reference"},
        }

    category = str(understanding.get("category_hint") or "document").lower()
    operation = normalized_operation(understanding)
    references = resolution.get("references", [])
    shapes = source_shapes or {}

    if category == "conversation":
        return _plan(
            "OK",
            ["CONVERSATION_MEMORY"],
            ["conversation_memory"],
            reason="question about this chat's history",
        )

    if category == "product":
        return _plan(
            "OK",
            ["PRODUCT_KNOWLEDGE"],
            ["product_knowledge"],
            reason="question about QueryPilot itself",
        )

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
    source_ids = sorted(resolved_source_ids)
    has_structured_sources = _documents_with_tables(
        resolved_source_ids,
        resolved_tables,
        shapes,
    )
    cross_document, cross_reason = _decide_cross_document(
        source_ids=resolved_source_ids,
        operation=operation,
        category=category,
        has_entities=bool(resolved_entities),
    )

    def plan(
        capabilities: list[str],
        actions: list[str],
        reason: str,
        *,
        fallback: list[str] | None = None,
        cross: bool = False,
    ) -> dict[str, Any]:
        return _plan(
            "OK",
            capabilities,
            actions,
            cross_document=cross,
            reason=reason,
            category=category,
            operation=operation,
            source_ids=source_ids,
            cross_reason=cross_reason,
            fallback_capabilities=fallback,
        )

    if wants_structure and (has_resolved_source or not resolved_entities):
        return plan(
            ["SOURCE_METADATA"], ["source_metadata"], "workspace structure question"
        )

    # Cross-document reasoning: only when the task relates the sources.
    if cross_document:
        capabilities: list[str] = []
        actions: list[str] = []
        # A mixed request is defined as needing table data plus document
        # text; it must schedule TABLE_QUERY even when the sources resolved
        # so far have no extracted tables (the data source is discovered).
        if has_structured_sources or category == "mixed":
            capabilities.append("TABLE_QUERY")
            actions.append("table_query")

        capabilities.append("DOCUMENT_SEARCH")
        actions.append("document_search")
        return plan(
            capabilities,
            actions,
            "relationship/comparison/calculation across resolved sources",
            cross=True,
        )

    if category in {"metadata", "workspace"} and not has_resolved_source:
        return plan(
            ["SOURCE_METADATA"], ["source_metadata"], "workspace metadata question"
        )

    # A request anchored only on named document(s): no entity value to look up
    # and no table named. The document's own content is the evidence.
    # DOCUMENT_SEARCH already reads the document's text AND its extracted
    # tables, so nothing is lost by not starting from TABLE_QUERY.
    # A "mixed" request (a value that depends on a rule/policy in ANOTHER
    # source) is not anchored on the named document alone, even when only one
    # source resolved: the missing companion must still be discovered.
    document_only = (
        bool(resolved_documents)
        and not resolved_tables
        and not resolved_entities
        and not pending_value_reference
        and category != "mixed"
    )
    if document_only:
        if len(source_ids) >= 2:
            multi_note = " (independent sources, retrieved separately)"
        else:
            multi_note = ""

        table_content_requested = _requests_table_content(understanding)
        row_operation = (
            category == "structured"
            and operation in TABLE_ROW_OPERATIONS
            and has_structured_sources
        )
        if operation not in OVERVIEW_OPERATIONS and (
            table_content_requested or row_operation
        ):
            return plan(
                ["TABLE_QUERY"],
                ["table_query"],
                "request targets the table/row content of the named document",
                fallback=["DOCUMENT_SEARCH"],
            )
        return plan(
            ["DOCUMENT_SEARCH"],
            ["document_search"],
            "document-content request on named document(s)" + multi_note,
        )

    if category in {"metadata", "workspace"} and has_resolved_source:
        if has_structured_sources and (
            resolved_tables or _requests_table_content(understanding)
        ):
            return plan(["TABLE_QUERY"], ["table_query"], "named table content request")
        return plan(
            ["DOCUMENT_SEARCH"], ["document_search"], "named source content request"
        )

    if category == "structured" or resolved_tables or resolved_entities:
        capabilities = ["TABLE_QUERY"]
        actions = ["table_query"]
        if category == "mixed":
            capabilities.append("DOCUMENT_SEARCH")
            actions.append("document_search")
        # Only a document (no entity, no named table) anchors this request:
        # if the table path finds nothing, the document's content is next.
        document_anchored = (
            bool(resolved_documents)
            and not resolved_tables
            and not resolved_entities
            and "DOCUMENT_SEARCH" not in capabilities
        )
        return plan(
            capabilities,
            actions,
            "structured lookup (named table or resolved entity)"
            if resolved_tables or resolved_entities
            else "structured request without a named document",
            fallback=["DOCUMENT_SEARCH"] if document_anchored else None,
        )

    if category == "mixed":
        return plan(
            ["TABLE_QUERY", "DOCUMENT_SEARCH"],
            ["table_query", "document_search"],
            "structured data plus document rules both required",
        )

    return plan(
        ["DOCUMENT_SEARCH"],
        ["document_search"],
        "document-content request (allowed corpus searched)",
    )


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
    reason: str = "",
    category: str = "",
    operation: str = "",
    source_ids: list[int] | None = None,
    cross_reason: str = "",
    fallback_capabilities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "capabilities": [item for item in capabilities if item in CAPABILITIES],
        "actions": actions,
        "clarification_options": [],
        "cross_document": cross_document,
        # Capabilities to run ONLY if the primary evidence path finds nothing.
        # Used when the plan started from a structured guess about a document.
        "fallback_capabilities": [
            item for item in (fallback_capabilities or []) if item in CAPABILITIES
        ],
        # Compact, log-friendly explanation of the decision. Never prompt text.
        "decision": {
            "category": category,
            "operation": operation,
            "resolved_document_ids": list(source_ids or []),
            "reason": reason,
            "cross_document_reason": cross_reason
            if cross_document
            else (cross_reason or "no cross-source reasoning required"),
        },
    }
