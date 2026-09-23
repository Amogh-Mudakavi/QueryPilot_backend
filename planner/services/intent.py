from __future__ import annotations

from typing import Any


CAPABILITIES = {
    "SOURCE_METADATA",
    "DOCUMENT_SEARCH",
    "TABLE_QUERY",
    "CONVERSATION_MEMORY",
    "PRODUCT_KNOWLEDGE",
}


def build_execution_plan(
    understanding: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Convert LLM extraction + deterministic resolution into a plan."""
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
    operation = str(understanding.get("operation") or "answer").lower()
    references = resolution.get("references", [])

    if category == "conversation":
        return _plan("OK", ["CONVERSATION_MEMORY"], ["conversation_memory"])

    if category == "product":
        return _plan("OK", ["PRODUCT_KNOWLEDGE"], ["product_knowledge"])

    resolved_documents = [
        item for item in references
        if item.get("status") == "RESOLVED" and item.get("match", {}).get("type") == "document"
    ]
    resolved_tables = [
        item for item in references
        if item.get("status") == "RESOLVED" and item.get("match", {}).get("type") == "table"
    ]
    resolved_entities = [
        item for item in references
        if item.get("status") == "RESOLVED" and item.get("match", {}).get("type") == "entity"
    ]

    describes_source = operation in {"describe", "inspect", "list", "overview"}

    if category in {"metadata", "workspace"} or (describes_source and (resolved_documents or resolved_tables)):
        return _plan("OK", ["SOURCE_METADATA"], ["source_metadata"])

    if category == "structured" or resolved_tables or resolved_entities:
        capabilities = ["TABLE_QUERY"]
        actions = ["table_query"]
        if category == "mixed":
            capabilities.append("DOCUMENT_SEARCH")
            actions.append("document_search")
        return _plan("OK", capabilities, actions)

    if category == "mixed":
        return _plan("OK", ["TABLE_QUERY", "DOCUMENT_SEARCH"], ["table_query", "document_search"])

    return _plan("OK", ["DOCUMENT_SEARCH"], ["document_search"])


def _plan(status: str, capabilities: list[str], actions: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "capabilities": [item for item in capabilities if item in CAPABILITIES],
        "actions": actions,
        "clarification_options": [],
    }
