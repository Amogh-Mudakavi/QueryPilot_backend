from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    workspace_id: int

    conversation_history: list[dict[str, Any]]
    relevant_document_ids: list[int]

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
    insufficient_reasons: list[str]
    recovery_attempts: int
    deadline_monotonic: float
    final_evidence: list[Any]
    answer: str
    grounding_passed: Optional[bool]
    route: str
    error: str
    tool_failed: bool

    _pending_action: dict[str, Any]