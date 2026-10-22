"""Pure helpers for authorization-aware planner source selection.

These helpers deliberately keep three concepts separate:

* explicitly resolved sources (the planner's decision);
* request-scoped candidate sources (Scope Guard and entity resolution);
* the conversation's authorization allow-list.

A candidate can broaden discovery only inside the authorization boundary.  The
module has no Django/database/LLM dependencies, so these contracts are testable
in isolation.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def authorized_document_ids(state: Mapping[str, Any]) -> list[int] | None:
    """Return the conversation allow-list; ``None`` means no extra restriction."""
    raw = state.get("allowed_document_ids")
    if raw is None:
        return None
    return sorted({item for item in (_as_int(value) for value in raw) if item is not None})


def candidate_document_ids(
    state: Mapping[str, Any],
    table_index: Sequence[Mapping[str, Any]] | None = None,
) -> list[int]:
    """Union resolved, Scope-Guard-selected and entity-located source IDs.

    ``relevant_document_ids`` is an initial relevance proposal, not a final
    execution plan.  The planner must preserve it alongside resolved sources
    instead of allowing a non-empty (but incomplete) resolved list to mask it.
    Entity matches can identify a document directly or through a table name.
    """
    ids: set[int] = set()
    plan = state.get("execution_plan") or {}
    decision = plan.get("decision") or {}
    ids.update(
        item
        for item in (_as_int(value) for value in decision.get("resolved_document_ids", []) or [])
        if item is not None
    )
    # A document the user explicitly NAMED (resolved reference of type
    # "document") on a non-cross-document plan is the request's scope.  The
    # Scope Guard's loose relevance proposal must not widen it again.
    named_document_scope = (not plan.get("cross_document")) and any(
        isinstance(ref, Mapping)
        and ref.get("status") == "RESOLVED"
        and isinstance(ref.get("match"), Mapping)
        and ref["match"].get("type") == "document"
        and _as_int(ref["match"].get("document_id")) is not None
        for ref in (state.get("resolved_references") or {}).get("references", []) or []
    )
    if not named_document_scope:
        ids.update(
            item
            for item in (
                _as_int(value) for value in state.get("relevant_document_ids", []) or []
            )
            if item is not None
        )

    entity_table_names: set[str] = set()
    resolution = state.get("resolved_references") or {}
    for reference in resolution.get("references", []) or []:
        if not isinstance(reference, Mapping) or reference.get("status") != "RESOLVED":
            continue
        match = reference.get("match") or {}
        if not isinstance(match, Mapping):
            continue
        match_type = match.get("type")
        document_id = _as_int(match.get("document_id"))
        if match_type in {"document", "table", "entity"} and document_id is not None:
            ids.add(document_id)
        if match_type == "table" and match.get("table_name"):
            entity_table_names.add(str(match["table_name"]))
        for location in match.get("locations", []) or []:
            if not isinstance(location, Mapping):
                continue
            location_id = _as_int(location.get("document_id"))
            if location_id is not None:
                ids.add(location_id)
            table_name = location.get("table_name")
            if table_name:
                entity_table_names.add(str(table_name))

    for item in state.get("resolved_tables", []) or []:
        if isinstance(item, Mapping):
            document_id = _as_int(item.get("document_id"))
            if document_id is not None:
                ids.add(document_id)

    if entity_table_names and table_index:
        for table in table_index:
            if str(table.get("table_name") or "") not in entity_table_names:
                continue
            document_id = _as_int(table.get("document_id"))
            if document_id is not None:
                ids.add(document_id)

    # Authorization is applied after unioning every discovery signal, so a
    # resolved reference can never bypass the caller's allow-list.
    allowed = authorized_document_ids(state)
    if allowed is not None:
        allowed_set = set(allowed)
        ids = {document_id for document_id in ids if document_id in allowed_set}

    return sorted(ids)


_FOLLOWUP_INTENT = re.compile(
    r"\b(?:explain\w*|why|how\s+(?:is|was|did|does|do|can|could|would)|"
    r"break\s*(?:it|this|that)?\s*down|show\s+(?:me\s+)?(?:the\s+)?"
    r"(?:working|calculation|steps|reason)|elaborat\w*|clarif\w*)\b",
    re.IGNORECASE,
)
_FOLLOWUP_REFERENCE = re.compile(
    r"\b(?:that|this|it|those|these|same|above|previous|earlier|"
    r"your answer|the result|the amount|the number)\b",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"\b(?:that's wrong|that is wrong|wrong answer|incorrect answer|"
    r"correct the answer|recalculate|recheck|try again|redo it)\b",
    re.IGNORECASE,
)
_GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "thx"}


def effective_planning_query(state: Mapping[str, Any]) -> str:
    """Return the enriched task query for planning, retrieval and answering.

    For contextual follow-ups, this preserves the prior task alongside the
    current request so final answer generation does not receive an elliptical
    query without its subject. The original utterance remains in ``query``.
    """
    return str(state.get("planning_query") or state.get("query") or "").strip()


def contextual_planning_query(
    current_query: str,
    conversation_history: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Expand a clear referential follow-up for planning without changing output query.

    The caller uses this return value only in the planner's local state. The
    original current query remains in the graph state for answering the user.
    """
    current = str(current_query or "").strip()
    if not current or not conversation_history:
        return current

    normalized_current = " ".join(current.casefold().split())
    if normalized_current in _GREETINGS or _CORRECTION.search(current):
        return current

    contextual = bool(_FOLLOWUP_REFERENCE.search(current)) and bool(
        _FOLLOWUP_INTENT.search(current)
    )
    # Tiny follow-ups such as "why?" / "explain" are also contextual when a
    # prior task exists; they are not sufficient grounds to inherit long queries.
    contextual = contextual or (
        len(current.split()) <= 4 and bool(_FOLLOWUP_INTENT.search(current))
    )
    if not contextual:
        return current

    previous: str | None = None
    for message in reversed(conversation_history):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        candidate = str(message.get("content") or "").strip()
        if not candidate or " ".join(candidate.casefold().split()) == normalized_current:
            continue
        if candidate.casefold() in _GREETINGS or _CORRECTION.search(candidate):
            continue
        previous = candidate
        break

    if not previous:
        return current
    return f"{previous}\n\nContextual follow-up: {current}"


def requires_dependency_discovery(state: Mapping[str, Any]) -> bool:
    """Whether the plan needs an extra authorized search for governing rules."""
    plan = state.get("execution_plan") or {}
    capabilities = set(plan.get("capabilities") or [])
    understanding = state.get("query_understanding") or {}
    operation = str(understanding.get("operation") or "").strip().lower()
    category = str(understanding.get("category_hint") or "").strip().lower()
    return (
        "TABLE_QUERY" in capabilities
        and "DOCUMENT_SEARCH" in capabilities
        and (operation in {"calculate", "aggregate", "compare"} or category == "mixed")
    )


def requires_document_evidence(
    state: Mapping[str, Any],
    *,
    skip_document_search: bool,
    has_structured_document_evidence: bool,
) -> bool:
    """Do not let table rows suppress textual evidence needed for a mixed plan."""
    plan = state.get("execution_plan") or {}
    capabilities = set(plan.get("capabilities") or [])
    if "DOCUMENT_SEARCH" not in capabilities or skip_document_search:
        return False
    return (not has_structured_document_evidence) or requires_dependency_discovery(state)


def can_answer_from_evidence(
    state: Mapping[str, Any],
    *,
    has_structured_evidence: bool,
    documents_usable: bool,
    skip_document_search: bool = False,
    has_structured_document_evidence: bool = False,
    dependency_evidence_found: bool = False,
) -> bool:
    """Fail closed when a mixed calculation lacks its required rule evidence."""
    rules_required = requires_document_evidence(
        state,
        skip_document_search=skip_document_search,
        has_structured_document_evidence=has_structured_document_evidence,
    )
    dependencies_required = requires_dependency_discovery(state)
    return (has_structured_evidence or documents_usable) and not (
        rules_required and not documents_usable
    ) and not (dependencies_required and not dependency_evidence_found)


def dependency_discovery_query(state: Mapping[str, Any]) -> str:
    """Build a generic retrieval query for rules that govern a calculation."""
    query = effective_planning_query(state)
    understanding = state.get("query_understanding") or {}
    fields = [str(value).strip() for value in understanding.get("requested_fields", []) or [] if str(value).strip()]
    references = [str(value).strip() for value in understanding.get("references", []) or [] if str(value).strip()]
    entities = []
    for item in (state.get("resolved_references") or {}).get("references", []) or []:
        if not isinstance(item, Mapping) or item.get("status") != "RESOLVED":
            continue
        match = item.get("match") or {}
        if isinstance(match, Mapping) and match.get("type") == "entity" and match.get("value"):
            entities.append(str(match["value"]))
    subject = " ".join([query, *fields, *references, *entities])
    suffix = (
        " Find all applicable source rules for this result: definitions, formulas, "
        "eligibility conditions, exclusions, deductions, penalties, caps, limits, "
        "thresholds, exceptions, and rounding rules."
    )
    return (subject[:700] + suffix)[:1000]