"""Services State module for QueryPilot."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    # Planner/retrieval context for referential follow-ups; `query` remains
    # the user's current utterance for final answer generation.
    planning_query: str
    workspace_id: int

    conversation_history: list[dict[str, Any]]
    relevant_document_ids: list[int]
    # Conversation-level authorization boundary. None means the containing
    # workspace access policy is authoritative; [] means no documents allowed.
    allowed_document_ids: list[int] | None
    # Scope Guard's verdict for THIS request (DOCUMENT, PRODUCT, ...). Empty
    # when the guard is disabled. The planner honours PRODUCT; it never
    # re-derives the request's intent from scratch.
    scope_route: str

    query_understanding: dict[str, Any]
    resolved_references: dict[str, Any]
    execution_plan: dict[str, Any]

    evidence: list[Any]
    retrieval_confidence: float
    known_facts: list[str]
    unresolved_questions: list[str]

    pending_clarification: bool
    clarification_options: list[dict[str, Any]]
    error_code: str

    last_tool: str
    tool_history: list[dict[str, Any]]
    iteration_count: int
    total_tool_calls: int

    evidence_sufficient: bool
    answerable: bool
    dependency_evidence_found: bool
    insufficient_reasons: list[str]
    recovery_attempts: int
    deadline_monotonic: float
    final_evidence: list[Any]
    answer: str
    grounding_passed: Optional[bool]
    route: str
    error: str
    tool_failed: bool

    # Request-scoped structured-resource metadata shared by agent/dispatch.
    resource_context: Any

    _pending_action: dict[str, Any]