from __future__ import annotations

from typing import Any, TypedDict, Optional


class AgentState(TypedDict, total=False):
    query: str
    workspace_id: int

    conversation_history: list[dict[str, Any]]
    relevant_document_ids: list[int]

    evidence: list[Any]
    retrieval_confidence: float
    known_facts: list[str]
    unresolved_questions: list[str]

    last_tool: str
    tool_history: list[dict[str, Any]]
    iteration_count: int
    total_tool_calls: int

    evidence_sufficient: bool
    final_evidence: list[Any]

    answer: str
    # Grounding LLM was removed; keep this field for API
    # compatibility but allow None to indicate no grounding
    # check was performed.
    grounding_passed: Optional[bool]
    route: str

    error: str
    tool_failed: bool

    _pending_action: dict[str, Any]