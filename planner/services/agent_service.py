from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from llm.exceptions import InvalidResponseError
from llm.router import router
from planner.prompts.agent import ACTION_SCHEMA, SYSTEM_PROMPT
from planner.services.answer import generate_answer
from planner.services.entity_resolution import resolve_references
from planner.services.history import bound_history_for_planner
from planner.services.intent import ROW_RESULT_OPERATIONS, build_execution_plan
from planner.services.product_knowledge import get_product_knowledge
from planner.services.schema_registry import (
    get_workspace_catalog,
    get_workspace_schema,
)
from planner.services.sql_generation import generate_sql
from planner.services.state import AgentState
from planner.services.sufficiency import (
    assess_evidence,
    documents_insufficient,
    insufficiency_note,
)
from planner.tools.conversation_memory import conversation_memory
from planner.tools.query_tables import query_tables
from planner.tools.search_evidence import search_evidence
from planner.tools.source_metadata import source_metadata

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 8

PLANNER_MAX_COMPLETION_TOKENS = 1000

# Exactly ONE recovery attempt per request, shared across all tools.
MAX_RECOVERY_ATTEMPTS = 1

# Set from data, not by guessing. None = threshold check off.
MIN_RETRIEVAL_CONFIDENCE: float | None = None

DEFAULT_FAILURE_MESSAGE = (
    "I could not complete that request because the required data "
    "operation failed."
)

ERROR_MESSAGES = {
    "REQUEST_TIMEOUT": (
        "This request took longer than expected, so I stopped it. "
        "Please try again, or try a simpler question."
    ),
    "DATABASE_ERROR": (
        "I couldn't reach the workspace data just now. Please try again "
        "in a moment."
    ),
    "ANSWER_GENERATION_ERROR": (
        "I found the relevant information but couldn't write up the "
        "answer just now. Please try again in a moment."
    ),
}

CHITCHAT_REPLIES = {
    "hi": "Hi! Ask me anything about the documents and tables in this workspace.",
    "hello": "Hello! Ask me anything about the documents and tables in this workspace.",
    "hey": "Hey! Ask me anything about the documents and tables in this workspace.",
    "thanks": "You're welcome! Let me know if you'd like to look at anything else.",
    "thank you": "You're welcome! Let me know if you'd like to look at anything else.",
    "thx": "You're welcome! Let me know if you'd like to look at anything else.",
}

# Cross-document evidence bound: maximum tables pulled from resolved
# documents per request. Deterministic schema order, inside MAX_EVIDENCE.
MAX_DOCUMENT_TABLES = 3


class AgentRuntimeError(Exception):
    def __init__(self, message: str, *, code: str = "EXECUTION_ERROR") -> None:
        super().__init__(message)
        self.code = code


class _RecoveryBudget:
    """Shared, single-use recovery allowance for one request."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def try_consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"dispatch": "dispatch", "finish": "finish"},
    )
    graph.add_edge("dispatch", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def _emit_event(
    *,
    event: str,
    status: str,
    message: str,
    data: Any = None,
) -> None:
    payload = {
        "event": event,
        "status": status,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    get_stream_writer()(payload)


# ---------------------------------------------------------------------------
# Deadline / small helpers
# ---------------------------------------------------------------------------


def _past_deadline(state: AgentState) -> bool:
    deadline = state.get("deadline_monotonic")
    return bool(deadline) and time.monotonic() > float(deadline)


def _check_deadline(state: AgentState, stage: str) -> None:
    if _past_deadline(state):
        logger.warning("Request budget exceeded before stage=%s", stage)
        raise AgentRuntimeError(
            "Request time budget exceeded.",
            code="REQUEST_TIMEOUT",
        )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chitchat_reply(query: str) -> str | None:
    """Deterministic greeting/thanks handling: no LLM call, no retrieval."""
    normalized = re.sub(r"[^\w\s]", "", str(query or "").lower()).strip()
    return CHITCHAT_REPLIES.get(normalized)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def agent_node(state: AgentState) -> dict[str, Any]:
    reply = _chitchat_reply(state.get("query", ""))
    if reply is not None:
        return {
            "query_understanding": {"category_hint": "chitchat"},
            "resolved_references": {},
            "execution_plan": {
                "status": "OK",
                "capabilities": [],
                "actions": [],
                "clarification_options": [],
            },
            "pending_clarification": False,
            "clarification_options": [],
            "answer": reply,
            "evidence": [],
            "final_evidence": [],
            "evidence_sufficient": False,
            "tool_failed": False,
            "error": "",
            "error_code": "",
            "route": "CHITCHAT",
        }

    _emit_event(
        event="planning",
        status="started",
        message="Understanding the request.",
    )

    try:
        response = router.generate(
            task="planner",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_context(state)},
            ],
            temperature=0.0,
            max_tokens=PLANNER_MAX_COMPLETION_TOKENS,
            response_format=ACTION_SCHEMA,
            include_reasoning=False,
        )

        understanding = _parse_understanding(response)
        resolution = resolve_references(
            workspace_id=state["workspace_id"],
            references=understanding.get("references", []),
            category_hint=understanding.get("category_hint"),
        )

        # Deterministic contextual resolution ("the second table",
        # "that document", "the previous one", "it"). Runs AFTER name
        # resolution and BEFORE planning; the LLM never picks the
        # actual resource. Failure is non-fatal.
        try:
            resolution = _resolve_contextual_references(state, resolution)
        except Exception:
            logger.exception(
                "Contextual reference resolution failed; continuing"
            )

        # Deterministic source shapes (which resolved documents actually
        # expose tables) so planning picks the evidence path from what
        # the source IS, never from name guessing.
        plan = build_execution_plan(
            understanding,
            resolution,
            _resolved_source_shapes(state, understanding, resolution),
        )

    except AgentRuntimeError as exc:
        logger.exception("Query understanding failed")
        _emit_event(
            event="planning",
            status="failed",
            message="Query understanding failed.",
        )
        return {
            "query_understanding": {},
            "resolved_references": {},
            "execution_plan": {},
            "pending_clarification": False,
            "clarification_options": [],
            "tool_failed": True,
            "error": str(exc),
            "error_code": exc.code,
            "route": "ERROR",
        }
    except Exception as exc:
        logger.exception("Query understanding failed")
        _emit_event(
            event="planning",
            status="failed",
            message="Query understanding failed.",
        )
        return {
            "query_understanding": {},
            "resolved_references": {},
            "execution_plan": {},
            "pending_clarification": False,
            "clarification_options": [],
            "tool_failed": True,
            "error": str(exc),
            "error_code": "PLANNING_ERROR",
            "route": "ERROR",
        }

    clarification = plan.get("status") == "NEEDS_CLARIFICATION"
    message = (
        "Clarification required."
        if clarification
        else "Request understood."
    )

    _emit_event(
        event="planning",
        status="completed",
        message=message,
        data={
            "category_hint": understanding.get("category_hint"),
            "operation": understanding.get("operation"),
            "capabilities": plan.get("capabilities", []),
            "pending_clarification": clarification,
        },
    )

    return {
        "query_understanding": understanding,
        "resolved_references": resolution,
        "execution_plan": plan,
        "pending_clarification": clarification,
        "clarification_options": plan.get("clarification_options", []),
        "tool_history": list(state.get("tool_history", []))
        + [
            {
                "stage": "query_understanding",
                "understanding": understanding,
                "resolution": resolution,
                "plan": plan,
            }
        ],
    }


def dispatch_node(state: AgentState) -> dict[str, Any]:
    plan = state.get("execution_plan", {})

    if plan.get("status") == "NEEDS_CLARIFICATION":
        options = state.get("clarification_options", [])
        labels: list[str] = []

        for item in options:
            label = (
                item.get("value")
                or item.get("title")
                or item.get("document_title")
                or item.get("table_name")
            )
            if label and label not in labels:
                labels.append(str(label))

        if labels:
            answer = (
                "I found multiple matches: "
                + ", ".join(labels[:6])
                + ". Which one do you mean?"
            )
        else:
            answer = "I found multiple possible matches. Which one do you mean?"

        return {
            "answer": answer,
            "evidence": [],
            "final_evidence": [],
            "evidence_sufficient": False,
            "route": "NEEDS_CLARIFICATION",
            "error": "",
            "error_code": "NEEDS_CLARIFICATION",
            "tool_failed": False,
        }

    capabilities = plan.get("capabilities", [])
    budget = _RecoveryBudget(MAX_RECOVERY_ATTEMPTS)

    structured: list[Any] = []
    documents: list[Any] = []
    document_tables: list[Any] = []
    total_tool_calls = 0
    table_status: str | None = None
    table_result: dict[str, Any] | None = None
    retrieval_confidence = 0.0
    document_search_error = False

    try:
        _check_deadline(state, "dispatch")

        if "PRODUCT_KNOWLEDGE" in capabilities:
            product = get_product_knowledge()
            product["result_status"] = "SUCCESS"
            structured.append(product)
            total_tool_calls += 1

        if "SOURCE_METADATA" in capabilities:
            structured.append(_run_source_metadata(state))
            total_tool_calls += 1

        if "TABLE_QUERY" in capabilities:
            table_result = _run_table(state, budget)
            structured.extend(table_result["evidence"])
            table_status = table_result["status"]
            total_tool_calls += 1

        if "CONVERSATION_MEMORY" in capabilities:
            result = conversation_memory(
                query=state.get("query", ""),
                conversation_history=state.get("conversation_history", []),
            )
            memory_messages = result.get("messages", [])
            structured.append(
                {
                    "source_type": "conversation_memory",
                    "result_status": "SUCCESS" if memory_messages else "NOT_FOUND",
                    "content": memory_messages,
                    "metadata": {
                        "resolution": result.get("resolution", ""),
                    },
                }
            )
            total_tool_calls += 1

        table_not_found = table_status == "NOT_FOUND"

        # Bug 4 (document side): references that explicitly name a file
        # but did not resolve mean the source does not exist. Searching
        # OTHER documents in its place would be a random source choice.
        resolved_doc_ids = _resolved_document_ids(state)
        missing_named_sources = _named_missing_source_references(state)
        named_source_missing = bool(missing_named_sources) and not resolved_doc_ids

        # Honest status note whenever an explicitly named source is
        # missing - including partial misses - so the answer never
        # implies evidence was taken from a source that was not found.
        if missing_named_sources:
            structured.append(
                {
                    "source_type": "source_resolution",
                    "result_status": "NOT_FOUND",
                    "content": (
                        "The following explicitly named source(s) were not "
                        "found in this workspace: "
                        + ", ".join(missing_named_sources[:5])
                        + ". No evidence was taken from them."
                    ),
                }
            )

        skip_document_search = table_not_found or named_source_missing

        if "DOCUMENT_SEARCH" in capabilities and skip_document_search:
            logger.info(
                "Skipping document search: %s",
                "required table lookup matched no rows"
                if table_not_found
                else "named source(s) not found in workspace",
            )

        # Cross-document evidence: a document the user deterministically
        # resolved is evidence through every channel it has. Its text
        # arrives via DOCUMENT_SEARCH (text chunks only); its tabular
        # content lives only in SQLite and is otherwise reachable by
        # neither evidence path. Generically: no query keywords, no
        # table-name matching, no domain assumptions.
        if (
            "DOCUMENT_SEARCH" in capabilities
            and not skip_document_search
            and _resolved_document_ids(state)
        ):
            document_tables = _run_resolved_document_tables(
                state,
                _tables_used_by_table_query(table_result),
            )
            if document_tables:
                structured.extend(document_tables)
                total_tool_calls += 1

        if "DOCUMENT_SEARCH" in capabilities and not skip_document_search:
            docs = _run_documents(state, budget)
            total_tool_calls += 1
            retrieval_confidence = docs["confidence"]

            if docs["error"] is not None:
                document_search_error = True
                code, message = docs["error"]
                if not structured:
                    raise AgentRuntimeError(message, code=code)
                structured.append(
                    {
                        "source_type": "document_search",
                        "result_status": "RETRIEVAL_ERROR",
                        "content": (
                            "Document retrieval failed; document evidence "
                            "is unavailable."
                        ),
                    }
                )
            else:
                documents = docs["evidence"]

        assessment = assess_evidence(
            wants_table="TABLE_QUERY" in capabilities,
            table_result_status=table_status,
            wants_documents=(
                "DOCUMENT_SEARCH" in capabilities
                and not skip_document_search
            ),
            # Resolved documents' tables ARE document evidence; count
            # them so the sufficiency assessment reflects reality.
            document_chunk_count=len(documents) + len(document_tables),
            retrieval_confidence=retrieval_confidence,
            min_retrieval_confidence=MIN_RETRIEVAL_CONFIDENCE,
        )

        documents_usable = bool(documents) and not documents_insufficient(
            len(documents),
            retrieval_confidence,
            MIN_RETRIEVAL_CONFIDENCE,
        )
        answerable = bool(structured) or documents_usable

        notes: list[Any] = []
        note = insufficiency_note(assessment["reasons"])
        if note is not None and (structured or documents):
            notes.append(note)

        # Bug 2: a mixed request explicitly depends on policy/rule
        # evidence. When NEITHER document text NOR the resolved
        # documents' tabular content was retrieved, say so in the
        # evidence itself: the answer must not invent or re-derive
        # the policy.
        if (
            "TABLE_QUERY" in capabilities
            and "DOCUMENT_SEARCH" in capabilities
            and not skip_document_search
            and not document_search_error
            and not documents
            and not document_tables
            and structured
        ):
            notes.append(
                {
                    "source_type": "evidence_note",
                    "result_status": "NOT_FOUND",
                    "content": (
                        "Policy/rule evidence was required but no document "
                        "evidence was retrieved. Use the structured facts "
                        "only where they answer the question directly; do "
                        "NOT invent, assume or re-derive policy rules, "
                        "formulas, slabs or thresholds, and state clearly "
                        "that the policy document evidence could not be "
                        "retrieved."
                    ),
                }
            )

        room_for_documents = max(MAX_EVIDENCE - len(structured) - len(notes), 0)
        evidence = (
            structured
            + documents[:room_for_documents]
            + notes
        )[:MAX_EVIDENCE]

        return {
            "evidence": evidence,
            "final_evidence": evidence,
            "evidence_sufficient": assessment["sufficient"],
            "answerable": answerable,
            "insufficient_reasons": assessment["reasons"],
            "retrieval_confidence": retrieval_confidence,
            "recovery_attempts": budget.used,
            "total_tool_calls": total_tool_calls,
            "tool_failed": False,
            "error": "",
            "error_code": "",
            "route": "ANSWER",
        }

    except AgentRuntimeError as exc:
        logger.exception("Evidence execution failed")
        collected = structured + documents
        return {
            "evidence": collected,
            "final_evidence": collected,
            "evidence_sufficient": False,
            "recovery_attempts": budget.used,
            "total_tool_calls": total_tool_calls,
            "tool_failed": True,
            "error": str(exc),
            "error_code": exc.code,
            "route": "ERROR",
        }
    except Exception as exc:
        logger.exception("Evidence execution failed")
        collected = structured + documents
        return {
            "evidence": collected,
            "final_evidence": collected,
            "evidence_sufficient": False,
            "recovery_attempts": budget.used,
            "total_tool_calls": total_tool_calls,
            "tool_failed": True,
            "error": str(exc),
            "error_code": "EXECUTION_ERROR",
            "route": "ERROR",
        }


def finish_node(state: AgentState) -> dict[str, Any]:
    route = state.get("route", "")

    if state.get("answer") and route in {"NEEDS_CLARIFICATION", "CHITCHAT"}:
        return state

    if state.get("tool_failed"):
        code = state.get("error_code", "EXECUTION_ERROR")
        return {
            "final_evidence": state.get(
                "final_evidence",
                state.get("evidence", []),
            ),
            "evidence_sufficient": False,
            "answer": ERROR_MESSAGES.get(code, DEFAULT_FAILURE_MESSAGE),
            "grounding_passed": None,
            "error": state.get("error", ""),
            "error_code": code,
            "route": "ERROR",
        }

    evidence = state.get("evidence", [])
    if not evidence:
        return {
            "final_evidence": [],
            "evidence_sufficient": False,
            "answer": "I could not find enough evidence in the available workspace data.",
            "grounding_passed": None,
            "error": "",
            "error_code": "NO_EVIDENCE",
            "route": "NO_EVIDENCE",
        }

    if state.get("answerable") is False:
        return {
            "final_evidence": [],
            "evidence_sufficient": False,
            "answer": (
                "I found some passages, but they don't clearly answer your "
                "question, so I'd rather not guess. Try rephrasing it, or "
                "name the document you mean."
            ),
            "grounding_passed": None,
            "error": "",
            "error_code": "INSUFFICIENT_EVIDENCE",
            "route": "NO_EVIDENCE",
        }

    statuses = {
        str(item.get("result_status", "")).upper()
        for item in evidence
        if isinstance(item, dict)
    }

    if "RETRIEVAL_ERROR" in statuses and statuses.issubset(
        {"RETRIEVAL_ERROR", "NOT_FOUND", ""}
    ):
        return {
            "final_evidence": evidence,
            "evidence_sufficient": False,
            "answer": (
                "I couldn't retrieve the data needed to answer that. "
                "Please try again in a moment."
            ),
            "grounding_passed": None,
            "error": "",
            "error_code": "RETRIEVAL_ERROR",
            "route": "ERROR",
        }

    if statuses and statuses.issubset({"NOT_FOUND", ""}):
        top_level_route = "NOT_FOUND"
        top_level_error_code = "NOT_FOUND"
    else:
        top_level_route = "ANSWER"
        top_level_error_code = ""

    if _past_deadline(state):
        return {
            "final_evidence": evidence,
            "evidence_sufficient": False,
            "answer": ERROR_MESSAGES["REQUEST_TIMEOUT"],
            "grounding_passed": None,
            "error": "Request time budget exceeded.",
            "error_code": "REQUEST_TIMEOUT",
            "route": "ERROR",
        }

    _emit_event(
        event="generating_answer",
        status="started",
        message="Generating the final answer.",
    )
    logger.info(
        "FINAL ANSWER EVIDENCE:\n%s",
        json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    try:
        answer = generate_answer(
            query=state.get("query", ""),
            evidence=evidence,
            conversation_history=state.get("conversation_history", []),
        )
    except Exception as exc:
        logger.exception("Answer generation failed")
        _emit_event(
            event="generating_answer",
            status="failed",
            message="Answer generation failed.",
        )
        return {
            "final_evidence": evidence,
            "evidence_sufficient": bool(state.get("evidence_sufficient")),
            "answer": ERROR_MESSAGES["ANSWER_GENERATION_ERROR"],
            "grounding_passed": None,
            "error": str(exc),
            "error_code": "ANSWER_GENERATION_ERROR",
            "route": "ERROR",
        }

    _emit_event(
        event="generating_answer",
        status="completed",
        message="Final answer generated.",
    )

    return {
        "final_evidence": evidence,
        "evidence_sufficient": bool(state.get("evidence_sufficient")),
        "answer": answer,
        "grounding_passed": None,
        "error": "",
        "error_code": top_level_error_code,
        "route": top_level_route,
    }


def route_after_agent(state: AgentState) -> str:
    if state.get("error") or state.get("route") == "CHITCHAT":
        return "finish"
    return "dispatch"


# ---------------------------------------------------------------------------
# Planning context
# ---------------------------------------------------------------------------


def _build_context(state: AgentState) -> str:
    workspace_id = state.get("workspace_id")
    catalog: list[dict[str, Any]] = []

    if workspace_id:
        try:
            raw_catalog = get_workspace_catalog(int(workspace_id))
            # The planner needs titles/types/table names for resolving phrases
            # such as "the two PDFs". It must never receive table row data here.
            for document in raw_catalog[:50]:
                catalog.append(
                    {
                        "document_id": document.get("document_id"),
                        "title": document.get("title"),
                        "source_type": document.get("source_type"),
                        "status": document.get("status"),
                        "tables": [
                            {
                                "table_name": table.get("table_name"),
                                "row_count": table.get("row_count"),
                            }
                            for table in document.get("tables", [])[:10]
                        ],
                    }
                )
        except Exception:
            logger.exception("Could not load planner workspace catalog; continuing without it")

    return json.dumps(
        {
            "user_query": state.get("query", ""),
            "workspace_id": workspace_id,
            "relevant_document_ids": state.get("relevant_document_ids", []),
            "workspace_catalog": catalog,
            "conversation_history": bound_history_for_planner(
                state.get("conversation_history", []),
            ),
        },
        ensure_ascii=False,
        default=str,
    )


def _parse_understanding(response: Any) -> dict[str, Any]:
    try:
        content = response.choices[0].message.content
        data = json.loads(content)
    except (AttributeError, IndexError, json.JSONDecodeError) as exc:
        raise AgentRuntimeError(
            "Query understanding returned invalid JSON.",
            code="PLANNING_ERROR",
        ) from exc

    if not isinstance(data, dict):
        raise AgentRuntimeError(
            "Query understanding returned an invalid object.",
            code="PLANNING_ERROR",
        )

    return data


# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------


def _first_resolved_match(
    state: AgentState,
    *types: str,
) -> dict[str, Any] | None:
    for item in state.get("resolved_references", {}).get("references", []):
        match = item.get("match", {})
        if item.get("status") != "RESOLVED":
            continue
        if match.get("type") in types:
            return match
    return None


_GENERIC_CATALOG_REFERENCES = {
    "all documents",
    "all tables",
    "attached documents",
    "attached files",
    "catalog",
    "data sources",
    "datasets",
    "doc",
    "docs",
    "document",
    "document catalog",
    "documents",
    "file",
    "files",
    "knowledge base",
    "knowledge bases",
    "pdf",
    "pdfs",
    "sources",
    "structured tables",
    "tables",
    "the attached documents",
    "the attached files",
    "the datasets",
    "the document",
    "the documents",
    "the files",
    "the knowledge base",
    "the pdfs",
    "the tables",
    "the uploaded documents",
    "the uploaded files",
    "the workspace documents",
    "these documents",
    "this document",
    "uploaded documents",
    "uploaded files",
    "workspace documents",
    "workspace tables",
}


def _is_generic_catalog_reference(reference: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", str(reference or "").lower())
    normalized = " ".join(normalized.split())
    return normalized in _GENERIC_CATALOG_REFERENCES


def _run_source_metadata(state: AgentState) -> dict[str, Any]:
    """
    Preserve the tool's FOUND / NOT_FOUND state, pass the resolved
    document_id when we have one, and never answer a question about a
    specific missing source with the whole catalog.
    """

    workspace_id = state["workspace_id"]
    references = state.get("resolved_references", {}).get("references", [])
    match = _first_resolved_match(state, "document", "table")

    if match is not None:
        document_id = _coerce_int(match.get("document_id"))
        reference = None
        if document_id is None:
            reference = (
                match.get("title")
                or match.get("document_title")
                or match.get("table_name")
            )
        metadata = source_metadata(
            workspace_id=workspace_id,
            reference=reference,
            document_id=document_id,
        )
    else:
        unresolved = [
            str(item.get("reference"))
            for item in references
            if item.get("status") == "NOT_FOUND" and item.get("reference")
        ]

        unresolved_specific = [
            reference
            for reference in unresolved
            if not _is_generic_catalog_reference(reference)
        ]

        if unresolved_specific:
            catalog = source_metadata(workspace_id=workspace_id)
            titles = [
                str(doc.get("title"))
                for doc in catalog.get("content", {}).get("documents", [])
                if doc.get("title")
            ][:20]
            return {
                "source_type": "workspace_metadata",
                "workspace_id": workspace_id,
                "status": "NOT_FOUND",
                "result_status": "NOT_FOUND",
                "requested_reference": unresolved_specific,
                "sources": [],
                "content": {
                    "document_count": 0,
                    "documents": [],
                    "available_documents": titles,
                },
            }

        metadata = source_metadata(workspace_id=workspace_id)

    metadata = dict(metadata)
    metadata["result_status"] = (
        "NOT_FOUND" if metadata.get("status") == "NOT_FOUND" else "SUCCESS"
    )
    return metadata


# ---------------------------------------------------------------------------
# Deterministic contextual reference resolution (Bug 7) and
# named-source-missing guards (Bug 4)
# ---------------------------------------------------------------------------

_TABLE_REFERENCE_WORDS = {"table", "sheet", "record"}
_DOCUMENT_REFERENCE_WORDS = {"document", "file", "doc", "pdf"}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_CONTEXTUAL_TOKENS = (
    set(_ORDINALS)
    | {"last", "one", "it", "that", "this", "previous", "next"}
    | _TABLE_REFERENCE_WORDS
    | _DOCUMENT_REFERENCE_WORDS
)

_FILE_EXTENSION_RE = re.compile(
    r"\b[\w .-]+\.(xlsx|xls|csv|tsv|pdf|docx|doc|txt|json|md)\b",
    re.IGNORECASE,
)


def _plan_is_mixed(state: AgentState) -> bool:
    capabilities = state.get("execution_plan", {}).get("capabilities", [])
    return "TABLE_QUERY" in capabilities and "DOCUMENT_SEARCH" in capabilities


def _parse_contextual_reference(reference: str) -> dict[str, Any] | None:
    """
    Recognize purely contextual/positional source references:
    "the second table", "the last document", "that table",
    "this document", "the previous one", "it", "the second one".

    Returns {"thing": "table"|"document"|None, "ordinal": int|"last"|None,
    "deictic": bool} or None. Only recognizes the PHRASE; the actual
    workspace resource is chosen deterministically later — the LLM
    never picks it.
    """
    text = re.sub(r"[^\w\s]", "", str(reference or "").lower())
    words = [w for w in text.split() if w not in {"the", "a", "an"}]
    if not words or not all(w in _CONTEXTUAL_TOKENS for w in words):
        return None

    thing = None
    for word in words:
        if word in _TABLE_REFERENCE_WORDS:
            thing = "table"
            break
        if word in _DOCUMENT_REFERENCE_WORDS:
            thing = "document"
            break

    ordinal: int | str | None = None
    for word in words:
        if word in _ORDINALS:
            ordinal = _ORDINALS[word]
            break
    if ordinal is None and "last" in words:
        ordinal = "last"

    deictic = any(w in words for w in {"that", "this", "previous", "it", "next"})

    if thing is None:
        # "the previous one" / "it" (deictic) AND "the second one" /
        # "the last one" (positional) are contextual even without a
        # thing word: the thing is whatever was established.
        if deictic or ordinal is not None:
            return {"thing": None, "ordinal": ordinal, "deictic": deictic}
        return None

    if ordinal is None and not deictic:
        return None  # a bare thing word ("table") is not contextual

    return {"thing": thing, "ordinal": ordinal, "deictic": deictic}


def _catalog_tables(
    catalog: list[dict[str, Any]],
    document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    documents = [document] if document is not None else catalog
    tables: list[dict[str, Any]] = []
    for doc in documents:
        for table in doc.get("tables", []):
            tables.append(
                {
                    "table_name": table.get("table_name"),
                    "document_id": doc.get("document_id"),
                    "title": doc.get("title"),
                }
            )
    return tables


def _catalog_documents(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"document_id": doc.get("document_id"), "title": doc.get("title")}
        for doc in catalog
    ]


def _table_match(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "table",
        "table_name": candidate.get("table_name"),
        "document_id": candidate.get("document_id"),
        "title": candidate.get("title"),
        "document_title": candidate.get("title"),
    }


def _document_match(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "document",
        "document_id": candidate.get("document_id"),
        "title": candidate.get("title"),
        "document_title": candidate.get("title"),
    }


def _previously_established_resource(state: AgentState) -> dict[str, Any] | None:
    """
    Most recent RESOLVED document/table from EARLIER turns
    (tool_history), preferring tables (more specific). Read from what
    deterministic resolution actually established.
    """
    for entry in reversed(list(state.get("tool_history", []))):
        if not isinstance(entry, dict):
            continue
        references = entry.get("resolution", {}).get("references", [])
        resolved = [
            item.get("match", {})
            for item in references
            if item.get("status") == "RESOLVED"
            and isinstance(item.get("match"), dict)
            and item["match"].get("type") in {"document", "table"}
        ]
        for match in resolved:
            if match.get("type") == "table":
                return {"type": "table", "match": match}
        if resolved:
            return {"type": "document", "match": resolved[0]}
    return None


def _established_resource(
    current_resolved: list[dict[str, Any]],
    state: AgentState,
) -> dict[str, Any] | None:
    # Current-turn named sources take priority as the base scope
    # ("the second table of employee_leaves.xlsx").
    for item in current_resolved:
        if item["match"].get("type") == "table":
            return {"type": "table", "match": item["match"]}
    for item in current_resolved:
        if item["match"].get("type") == "document":
            return {"type": "document", "match": item["match"]}
    return _previously_established_resource(state)


def _contextual_table_candidates(
    catalog: list[dict[str, Any]],
    established: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Tables a positional table reference can refer to, in deterministic
    order: the tables of the established document when there is one,
    otherwise every table in workspace catalog order.
    """
    if established:
        document_id = _coerce_int(established["match"].get("document_id"))
        if document_id is not None:
            for doc in catalog:
                if _coerce_int(doc.get("document_id")) == document_id:
                    return _catalog_tables(catalog, doc)
    return _catalog_tables(catalog)


def _resolve_one_contextual(
    parsed: dict[str, Any],
    reference: str,
    catalog: list[dict[str, Any]],
    established: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Resolve one parsed contextual reference against the deterministic
    catalog. Returns {"match": ...} (+ optional "note"), or
    {"status": "AMBIGUOUS", "matches": [...]}, or None (leave as-is).
    """
    thing = parsed["thing"]
    ordinal = parsed["ordinal"]

    # --- deictic without ordinal: "that table" / "this document" ----
    if parsed["deictic"] and ordinal is None:
        if thing == "table":
            if established and established["type"] == "table":
                return {"match": established["match"]}
            candidates = _contextual_table_candidates(catalog, established)
        elif thing == "document":
            if established and established["type"] == "document":
                return {"match": established["match"]}
            if established and established["type"] == "table":
                document_id = _coerce_int(established["match"].get("document_id"))
                for doc in catalog:
                    if _coerce_int(doc.get("document_id")) == document_id:
                        return {"match": _document_match(doc)}
            candidates = _catalog_documents(catalog)
        else:
            if established:
                return {"match": established["match"]}
            candidates = _catalog_documents(catalog)

        if len(candidates) == 1:
            if thing == "table":
                return {"match": _table_match(candidates[0])}
            return {"match": _document_match(candidates[0])}
        if len(candidates) > 1:
            return {
                "status": "AMBIGUOUS",
                "matches": [
                    _table_match(c) if thing == "table" else _document_match(c)
                    for c in candidates[:8]
                ],
            }
        return None

    # --- ordinal: "the second table" / "the last document" ----------
    if thing == "table":
        candidates = _contextual_table_candidates(catalog, established)
    elif thing == "document":
        candidates = _catalog_documents(catalog)
    else:
        if established:
            thing = established["type"]
            candidates = (
                _contextual_table_candidates(catalog, established)
                if thing == "table"
                else _catalog_documents(catalog)
            )
        else:
            thing = "document"
            candidates = _catalog_documents(catalog)

    if not candidates:
        return None

    index = len(candidates) if ordinal == "last" else int(ordinal or 0)
    if index < 1 or index > len(candidates):
        return None  # out of range: leave NOT_FOUND (honest)

    candidate = candidates[index - 1]
    if thing == "table":
        match = _table_match(candidate)
        note = (
            f"The contextual reference '{reference}' was resolved "
            f"deterministically to table '{match['table_name']}' of "
            f"document '{match['title']}'. The positional phrase selects "
            "WHICH table to use; the user wants that table's CONTENT, "
            "not a positional row inside it."
        )
        return {"match": match, "note": note}
    return {"match": _document_match(candidate)}


def _resolve_contextual_references(
    state: AgentState,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic resolution for contextual references that name
    resolution could not match. Order of authority:
    1. resources resolved BY NAME in the current request,
    2. the resource established in an earlier turn (tool_history),
    3. the workspace catalog in deterministic order.
    One candidate resolves it; several candidates make it AMBIGUOUS;
    an out-of-range ordinal stays NOT_FOUND.
    """
    references = resolution.get("references", [])
    if not references:
        return resolution

    has_candidates = any(
        item.get("status") == "NOT_FOUND"
        and _parse_contextual_reference(str(item.get("reference") or "")) is not None
        for item in references
    )
    if not has_candidates:
        return resolution

    catalog = get_workspace_catalog(state["workspace_id"])

    current_resolved = [
        item
        for item in references
        if item.get("status") == "RESOLVED"
        and isinstance(item.get("match"), dict)
        and item["match"].get("type") in {"document", "table"}
    ]
    established = _established_resource(current_resolved, state)

    notes: list[str] = []
    updated = False
    for item in references:
        if item.get("status") != "NOT_FOUND":
            continue
        reference = str(item.get("reference") or "")
        parsed = _parse_contextual_reference(reference)
        if parsed is None:
            continue
        outcome = _resolve_one_contextual(parsed, reference, catalog, established)
        if outcome is None:
            continue
        updated = True
        if outcome.get("status") == "AMBIGUOUS":
            item["status"] = "AMBIGUOUS"
            item["matches"] = outcome.get("matches", [])
            resolution["has_ambiguity"] = True
            continue
        item["status"] = "RESOLVED"
        item["match"] = outcome["match"]
        if outcome.get("note"):
            notes.append(outcome["note"])

    if updated and notes:
        resolution["contextual_notes"] = notes
    return resolution


def _named_missing_source_references(state: AgentState) -> list[str]:
    """
    References that explicitly name a file (they carry a file
    extension) but did NOT resolve. Per the deterministic resolution
    contract (0 matches -> NOT_FOUND) these sources do not exist in
    the workspace. Generic catalog phrases and contextual references
    are excluded.
    """
    missing: list[str] = []
    for item in state.get("resolved_references", {}).get("references", []):
        if item.get("status") != "NOT_FOUND":
            continue
        reference = str(item.get("reference") or "")
        if not reference or _is_generic_catalog_reference(reference):
            continue
        if _parse_contextual_reference(reference) is not None:
            continue
        if _FILE_EXTENSION_RE.search(reference):
            missing.append(reference)
    return missing


def _resolved_source_shapes(
    state: AgentState,
    understanding: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[int, bool]:
    """
    Deterministic workspace truth for planning: which resolved
    documents expose structured tables, {document_id: bool}. Computed
    from the actual catalog only when a plan could depend on it. The
    LLM never supplies this.
    """
    # Source shape is workspace truth, not a category hint. Cross-document
    # planning needs it even when the LLM classified the request as mixed or
    # document.
    document_ids: list[int] = []
    for item in resolution.get("references", []):
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if match.get("type") == "document":
            document_id = _coerce_int(match.get("document_id"))
            if document_id is not None:
                document_ids.append(document_id)
    if not document_ids:
        return {}

    has_tables = {
        _coerce_int(doc.get("document_id")): bool(doc.get("tables"))
        for doc in get_workspace_catalog(state["workspace_id"])
    }
    return {
        document_id: bool(has_tables.get(document_id))
        for document_id in document_ids
    }


# ---------------------------------------------------------------------------
# Document search (with one broadened retry)
# ---------------------------------------------------------------------------


def _document_query(state: AgentState) -> str:
    understanding = state.get("query_understanding", {})
    return " ".join(
        [
            str(understanding.get("operation", "answer")),
            " ".join(str(x) for x in understanding.get("requested_fields", [])),
            " ".join(str(x) for x in understanding.get("references", [])),
            state.get("query", ""),
        ]
    ).strip()


def _resolved_document_ids(state: AgentState) -> list[int]:
    """
    Documents the user EXPLICITLY pointed at (a named document or
    table). Entity matches are deliberately excluded: an entity's
    location says where a VALUE lives, not which text document the
    question is about.
    """
    resolved_ids: list[int] = []

    for item in state.get("resolved_references", {}).get("references", []):
        match = item.get("match", {})
        if item.get("status") != "RESOLVED":
            continue
        if match.get("type") in {"document", "table"}:
            document_id = _coerce_int(match.get("document_id"))
            if document_id is not None:
                resolved_ids.append(document_id)

    return sorted(set(resolved_ids))


def _document_ids_for_search(state: AgentState) -> list[int]:
    return _resolved_document_ids(state) or list(
        state.get("relevant_document_ids", [])
    )


def _mark_success(chunks: list[Any]) -> list[Any]:
    for item in chunks:
        if isinstance(item, dict):
            item.setdefault("result_status", "SUCCESS")
    return chunks


def _run_documents(
    state: AgentState,
    budget: _RecoveryBudget,
) -> dict[str, Any]:
    primary_query = _document_query(state)
    primary_ids = _document_ids_for_search(state)
    cross_document = len(_resolved_document_ids(state)) >= 2

    if cross_document:
        # A single top-k retrieval over all documents can return every hit from
        # one document and silently omit another source. Search each resolved
        # document independently so cross-document reasoning has source
        # coverage before ranking/answer generation.
        per_document = _resolved_document_ids(state)
        collected: list[Any] = []
        confidences: list[float] = []
        errors: list[tuple[str, str]] = []

        for document_id in per_document:
            if _past_deadline(state):
                break
            try:
                result = search_evidence(
                    query=primary_query,
                    workspace_id=state["workspace_id"],
                    allowed_document_ids=[document_id],
                )
            except Exception as exc:
                logger.exception(
                    "Cross-document search failed for document_id=%s",
                    document_id,
                )
                errors.append(("RETRIEVAL_ERROR", str(exc)))
                continue

            error_code = result.get("error_code")
            if error_code:
                errors.append(
                    (
                        str(error_code),
                        result.get("error") or "Document retrieval failed.",
                    )
                )
                continue

            chunks = _mark_success(list(result.get("evidence", [])))
            if chunks:
                collected.extend(chunks[:4])
            confidences.append(float(result.get("confidence") or 0.0))

        # Keep deterministic source coverage first, then enforce the normal
        # evidence bound at the final assembly stage.
        if errors and not collected:
            return {
                "evidence": [],
                "confidence": 0.0,
                "error": errors[0],
            }

        confidence = min(confidences) if confidences else 0.0
        return {
            "evidence": collected,
            "confidence": confidence,
            "error": errors[0] if errors and not collected else None,
        }

    result = search_evidence(
        query=primary_query,
        workspace_id=state["workspace_id"],
        allowed_document_ids=primary_ids,
    )

    error_code = result.get("error_code")
    if error_code:
        return {
            "evidence": [],
            "confidence": 0.0,
            "error": (
                str(error_code),
                result.get("error") or "Document retrieval failed.",
            ),
        }

    chunks = _mark_success(list(result.get("evidence", [])))
    confidence = float(result.get("confidence") or 0.0)

    if documents_insufficient(len(chunks), confidence, MIN_RETRIEVAL_CONFIDENCE):
        raw_query = str(state.get("query", "")).strip()

        scoped_to_resolved = bool(_resolved_document_ids(state))
        all_ids = list(state.get("relevant_document_ids", []))
        retry_ids = primary_ids if scoped_to_resolved else (all_ids or primary_ids)

        if scoped_to_resolved and not chunks and _plan_is_mixed(state):
            widened = all_ids or [
                _coerce_int(doc.get("document_id"))
                for doc in get_workspace_catalog(state["workspace_id"])
                if _coerce_int(doc.get("document_id")) is not None
            ]
            if widened:
                retry_ids = widened

        different = bool(raw_query) and (
            raw_query != primary_query
            or sorted(retry_ids) != sorted(primary_ids)
        )

        if different and not _past_deadline(state) and budget.try_consume():
            logger.info(
                "document_search recovery: chunks=%d confidence=%.4f",
                len(chunks),
                confidence,
            )
            try:
                retry = search_evidence(
                    query=raw_query,
                    workspace_id=state["workspace_id"],
                    allowed_document_ids=retry_ids,
                )
            except Exception:
                logger.exception("document_search recovery failed")
                retry = None

            if retry is not None and not retry.get("error_code"):
                retry_chunks = _mark_success(list(retry.get("evidence", [])))
                retry_confidence = float(retry.get("confidence") or 0.0)
                if (len(retry_chunks), retry_confidence) > (
                    len(chunks),
                    confidence,
                ):
                    chunks, confidence = retry_chunks, retry_confidence

    return {"evidence": chunks, "confidence": confidence, "error": None}


# ---------------------------------------------------------------------------
# Cross-document table evidence
# ---------------------------------------------------------------------------


def _tables_used_by_table_query(table: dict[str, Any] | None) -> set[str]:
    """Table names the executed business SQL actually read."""
    if not table:
        return set()
    names: set[str] = set()
    for item in table.get("evidence", []):
        if not isinstance(item, dict):
            continue
        for name in item.get("content", {}).get("tables_used", []) or []:
            names.add(str(name))
    return names


def _run_resolved_document_tables(
    state: AgentState,
    tables_used: set[str],
) -> list[dict[str, Any]]:
    """
    Cross-document evidence: tables extracted from documents the user
    deterministically resolved, that the business SQL did not read.

    A resolved document's content is evidence through every channel it
    has. Its text arrives via DOCUMENT_SEARCH (text chunks only); its
    tabular content lives only in SQLite and is otherwise reachable by
    neither evidence path. Generically: no query keywords, no
    table-name matching, no domain assumptions.

    Deterministic (no ranking), read-only, allowlisted per table.
    source_type stays "table_query" so downstream consumers are
    unchanged; evidence_role marks the semantic difference.
    """
    resolved_ids = set(_resolved_document_ids(state))
    if not resolved_ids:
        return []

    schema = get_workspace_schema(state["workspace_id"])
    candidates = [
        item
        for item in schema
        if _coerce_int(item.get("document_id")) in resolved_ids
        and str(item.get("table_name")) not in tables_used
    ][:MAX_DOCUMENT_TABLES]

    evidence: list[dict[str, Any]] = []
    for item in candidates:
        table_name = str(item["table_name"])
        try:
            result = query_tables(
                workspace_id=state["workspace_id"],
                sql=f'SELECT * FROM "{table_name}"',
                parameters=[],
                allowed_table_names={table_name},
            )
        except ValueError:
            logger.exception("document table query failed: %s", table_name)
            continue

        row_count = int(result.get("row_count", 0) or 0)
        evidence.append(
            {
                "source_type": "table_query",
                "workspace_id": state["workspace_id"],
                "sources": [
                    {
                        "table_name": table_name,
                        "document_id": item.get("document_id"),
                    }
                ],
                "content": {
                    **result,
                    "evidence_role": "document_table",
                    "document_title": item.get("document_title"),
                    "interpretation": (
                        "Rows of a table extracted from a document the "
                        "user referenced. This is that document's own "
                        "tabular content — its rules, rates, thresholds, "
                        "or facts. Treat it as authoritative document "
                        "evidence, alongside the document's text "
                        "passages."
                    ),
                },
                "result_status": "SUCCESS" if row_count > 0 else "NOT_FOUND",
            }
        )
    return evidence


# ---------------------------------------------------------------------------
# Table query (with one SQL repair pass)
# ---------------------------------------------------------------------------


def _resolved_table_names(state: AgentState) -> set[str]:
    names: set[str] = set()
    for item in state.get("resolved_references", {}).get("references", []):
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if match.get("type") == "table" and match.get("table_name"):
            names.add(str(match["table_name"]))
    return names


def _resolved_entity_values(state: AgentState) -> list[dict[str, Any]]:
    """
    Extract the authoritative resolved entity values (not table/
    document matches) for handing to SQL generation. Generic: reads
    whatever resolve_references actually found.
    """
    entities: list[dict[str, Any]] = []
    for item in state.get("resolved_references", {}).get("references", []):
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if match.get("type") != "entity":
            continue
        value = match.get("value")
        locations = match.get("locations", [])
        if value and locations:
            entities.append({"value": value, "locations": locations})
    return entities


_SQL_OUTPUT_FAILURE = (
    "The previous response was not valid or was cut off before the JSON was "
    "complete. Return only the JSON object with the shortest valid query."
)

_SQL_DECLINED = (
    "The previous response declined to write SQL (needs_sql=false), but this "
    "request requires a table query. Set needs_sql=true and return the query."
)

_INFRASTRUCTURE_ERRORS = (
    "database not found",
    "no tables are available",
    "no tables are allowed",
)


def _is_infrastructure_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _INFRASTRUCTURE_ERRORS)


def _zero_rows_repair_guidance(
    schema_for_sql: list[dict[str, Any]],
    is_mixed: bool,
) -> str:
    """
    Generic, evidence-driven repair guidance for a zero-row result
    that contradicts the actual source data. Keyed on table sizes from
    the authoritative schema and on the plan — never on query wording.
    """
    sizes = ", ".join(
        f"{item.get('table_name')}: {int(item.get('row_count') or 0)} rows"
        for item in schema_for_sql[:5]
    )
    guidance = (
        "The query returned zero rows although the source table(s) contain "
        f"rows ({sizes}), so the SQL conditions are wrong - the data is not "
        "missing. Remove conditions the question does not require, verify "
        "column names, CASTs and exact stored values, and remove any JOIN "
        "that is not required by the question."
    )
    if is_mixed:
        guidance += (
            " This request also depends on a rule/policy from a document: if "
            "the SQL joins a table extracted from a policy/rules document, "
            "or encodes its tiers, thresholds or formula (CASE expressions, "
            "range conditions), remove that entirely. Rule logic is applied "
            "later from document evidence, never in SQL. Return the business "
            "table's rows as inputs."
        )
    return guidance


def _attempt_sql(
    *,
    state: AgentState,
    schema_for_sql: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    allowed: set[str],
    previous_attempt: dict[str, Any] | None,
    planner_notes: list[str] | None = None,
) -> dict[str, Any]:
    """
    One generate-SQL + run-SQL attempt. Never raises for problems the
    SQL generator or SQLite can cause; those come back as `error` so
    the caller can decide whether to repair. Infrastructure problems
    raise.
    """

    outcome: dict[str, Any] = {
        "needs_sql": True,
        "sql": None,
        "parameters": [],
        "result": None,
        "error": None,
    }

    try:
        decision = generate_sql(
            query=state.get("query", ""),
            schema=schema_for_sql,
            resolved_entities=entities,
            previous_attempt=previous_attempt,
            sql_required=True,
            planner_notes=planner_notes or None,
        )
        if not isinstance(decision, dict):
            raise TypeError("SQL generator returned a non-object.")
        needs_sql = bool(decision.get("needs_sql"))
    except (
        InvalidResponseError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
    ):
        logger.exception("SQL generation failed or returned invalid output")
        outcome["error"] = _SQL_OUTPUT_FAILURE
        return outcome

    if not needs_sql:
        logger.warning(
            "SQL generator returned needs_sql=false for a planned TABLE_QUERY"
        )
        outcome["error"] = _SQL_DECLINED
        return outcome

    sql = decision.get("sql")
    parameters = decision.get("parameters", [])
    outcome["sql"] = sql
    outcome["parameters"] = parameters

    if not sql:
        outcome["error"] = "SQL generator returned no SQL."
        return outcome

    logger.info(
        "table_query attempt: repair=%s sql_preview=%r parameter_count=%d",
        previous_attempt is not None,
        str(sql)[:300],
        len(parameters) if isinstance(parameters, (list, tuple)) else 0,
    )

    try:
        outcome["result"] = query_tables(
            workspace_id=state["workspace_id"],
            sql=sql,
            parameters=parameters,
            allowed_table_names=allowed,
        )
    except ValueError as exc:
        message = str(exc)
        if _is_infrastructure_error(message):
            raise AgentRuntimeError(message, code="DATABASE_ERROR") from exc
        logger.warning("Table query failed: %s", message)
        outcome["error"] = message

    if outcome["result"] is not None:
        logger.info(
            "table_query result: rows=%s tables_used=%s truncated=%s",
            outcome["result"].get("row_count"),
            outcome["result"].get("tables_used", []),
            outcome["result"].get("truncated", False),
        )

    return outcome


def _outcome_rank(outcome: dict[str, Any]) -> int:
    """SUCCESS (rows) > NOT_FOUND (zero rows) > ERROR."""
    result = outcome.get("result")
    if result is None:
        return 0
    return 2 if result.get("row_count", 0) > 0 else 1


def _source_row_counts(
    schema_by_table: dict[str, dict[str, Any]],
    tables_used: list[str],
) -> dict[str, int]:
    """Size of each SOURCE table the query read, keyed by document title."""
    counts: dict[str, int] = {}
    for name in tables_used:
        item = schema_by_table.get(name)
        if not item:
            continue
        label = str(item.get("document_title") or name)
        if label in counts:
            label = f"{label} ({name})"
        counts[label] = int(item.get("row_count") or 0)
    return counts


def _describe_table_result(
    *,
    rows_returned: int,
    truncated: bool,
    source_counts: dict[str, int],
) -> str:
    size = ", ".join(f"{label}: {n} rows" for label, n in source_counts.items())

    if rows_returned == 0:
        text = "The query ran successfully and no rows matched the conditions."
        if size:
            text += f" The source data exists ({size})."
        text += (
            " Report this as no matching records, not as missing or "
            "inaccessible data."
        )
        return text

    text = (
        f"This query returned {rows_returned} row(s). That is the size of "
        "THIS QUERY'S RESULT only. It is NOT the size of the source table: "
        "a filtered, ordered or limited query (for example the 'first', "
        "'third' or 'last' employee) legitimately returns a single row."
    )
    if size:
        text += f" Source table size: {size}."
    if truncated:
        text += " Results were truncated at the row limit."
    text += " Never conclude the source contains only that many records."
    return text


def _query_semantics(
    *,
    sql: str | None,
    operation: str,
) -> dict[str, Any]:
    """
    Describe what the executed SQL result means, using the actual SQL
    and planner operation rather than domain-specific assumptions.
    """
    normalized_sql = re.sub(r"\s+", " ", str(sql or "")).strip()
    upper_sql = normalized_sql.upper()
    normalized_operation = str(operation or "answer").lower()

    if re.search(r"ORDER BY\s+ROWID\s+DESC\s+LIMIT\s+1\b", upper_sql):
        return {
            "type": "positional",
            "selection": "last",
            "basis": "ORDER BY rowid DESC LIMIT 1",
            "meaning": (
                "The query deliberately selected the last row of the whole "
                "unfiltered source table. The returned row is the requested "
                "last record; it is not evidence that the source has one row."
            ),
        }

    positional_match = re.search(
        r"ORDER BY\s+ROWID(?:\s+ASC)?\s+LIMIT\s+1\s+OFFSET\s+(\d+)\b",
        upper_sql,
    )
    if positional_match:
        offset = int(positional_match.group(1))
        return {
            "type": "positional",
            "selection": offset + 1,
            "basis": f"ORDER BY rowid LIMIT 1 OFFSET {offset}",
            "meaning": (
                f"The query deliberately selected row position {offset + 1} "
                "of the whole unfiltered source table. The returned row is "
                "the requested positional record; it is not evidence that "
                "the source has one row."
            ),
        }

    if re.search(r"\bORDER BY\b.+\bLIMIT\s+1\b", upper_sql):
        direction = "DESC" if re.search(r"\bDESC\b\s+LIMIT\s+1\b", upper_sql) else "ASC"
        return {
            "type": "ranking",
            "direction": direction,
            "operation": normalized_operation,
            "basis": "ORDER BY ... LIMIT 1",
            "meaning": (
                "The query deliberately selected the single highest/lowest "
                "ranked result according to its ORDER BY expression. The "
                "returned row is the ranking result, not an arbitrary sample."
            ),
        }

    if re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(", upper_sql):
        return {
            "type": "aggregate",
            "operation": normalized_operation,
            "basis": "aggregate SQL function",
            "meaning": (
                "The returned scalar or aggregate row is the result of the "
                "aggregate query, not a sample row from the source table."
            ),
        }

    return {
        "type": "table_result",
        "operation": normalized_operation,
        "meaning": (
            "Interpret the returned rows according to the executed SQL and "
            "the query operation; do not infer source-table size from the "
            "number of returned rows."
        ),
    }


def _table_content(
    result: dict[str, Any],
    schema_by_table: dict[str, dict[str, Any]],
    *,
    sql: str | None,
    operation: str,
) -> dict[str, Any]:
    """
    Table evidence contract: `row_count` is what THIS query returned.
    Source table size and an explicit interpretation travel with it.
    """
    content = dict(result)
    rows_returned = int(result.get("row_count", 0) or 0)
    source_counts = _source_row_counts(
        schema_by_table, list(result.get("tables_used", []))
    )
    content["rows_returned"] = rows_returned
    content["source_table_row_counts"] = source_counts
    content["interpretation"] = _describe_table_result(
        rows_returned=rows_returned,
        truncated=bool(result.get("truncated")),
        source_counts=source_counts,
    )
    content["query_semantics"] = _query_semantics(
        sql=sql,
        operation=operation,
    )
    return content


def _cross_document_source_groups(
    state: AgentState,
    schema: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Build deterministic per-document SQL scopes for cross-document requests.

    Each source is queried independently. This preserves cross-document
    reasoning while making it impossible for SQL generation to create a
    Cartesian product across different documents.
    """
    references = state.get("resolved_references", {}).get("references", [])
    resolved_document_ids: list[int] = []
    resolved_table_names: set[str] = set()

    for item in references:
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if not isinstance(match, dict):
            continue
        match_type = match.get("type")
        document_id = _coerce_int(match.get("document_id"))
        if match_type == "document" and document_id is not None:
            resolved_document_ids.append(document_id)
        elif match_type == "table" and match.get("table_name"):
            resolved_table_names.add(str(match["table_name"]))
            if document_id is not None:
                resolved_document_ids.append(document_id)

    entities = _resolved_entity_values(state)
    entity_table_names = {
        str(location.get("table_name"))
        for entity in entities
        for location in entity.get("locations", [])
        if location.get("table_name")
    }

    document_ids = list(dict.fromkeys(resolved_document_ids))
    groups: list[list[dict[str, Any]]] = []
    assigned_tables: set[str] = set()

    for document_id in document_ids:
        group = [
            item
            for item in schema
            if _coerce_int(item.get("document_id")) == document_id
            and (
                not resolved_table_names
                or str(item.get("table_name")) in resolved_table_names
                or document_id in document_ids
            )
        ]
        if group:
            groups.append(group)
            assigned_tables.update(str(item.get("table_name")) for item in group)

    # Entity resolution is authoritative even when the entity's business
    # table was not explicitly named as a document/table reference.
    entity_group = [
        item
        for item in schema
        if str(item.get("table_name")) in entity_table_names
        and str(item.get("table_name")) not in assigned_tables
    ]
    if entity_group:
        by_document: dict[int, list[dict[str, Any]]] = {}
        for item in entity_group:
            document_id = _coerce_int(item.get("document_id"))
            if document_id is None:
                continue
            by_document.setdefault(document_id, []).append(item)
        for document_id in sorted(by_document):
            groups.append(by_document[document_id])

    return groups


def _run_cross_document_tables(
    state: AgentState,
    budget: _RecoveryBudget,
    schema: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retrieve resolved cross-document tables independently, without LLM SQL generation.

    A cross-document request has already identified the exact source documents.
    At this stage we need the source contents, not a model-generated join. Each
    table is therefore read directly with a single allowlisted SELECT * query.
    This guarantees that tables from different documents can never be joined
    or produce a Cartesian product, and it also removes the failure mode where
    the SQL model incorrectly returns ``needs_sql=false`` for a request that
    the planner has already classified as requiring structured evidence.
    """
    del budget  # Cross-document retrieval is deterministic; recovery is not needed here.

    groups = _cross_document_source_groups(state, schema)
    if len(groups) < 2:
        return {"evidence": [], "status": "NOT_FOUND"}

    evidence: list[dict[str, Any]] = []
    any_success = False
    any_error = False
    operation = str(
        state.get("query_understanding", {}).get("operation") or "answer"
    ).lower()

    for group in groups:
        if _past_deadline(state):
            break

        # Keep each resolved document independent. MAX_DOCUMENT_TABLES is
        # applied per document so one source cannot crowd out another.
        for item in group[:MAX_DOCUMENT_TABLES]:
            if _past_deadline(state):
                break

            table_name = str(item.get("table_name") or "")
            if not table_name:
                continue

            schema_by_table = {table_name: item}

            try:
                result = query_tables(
                    workspace_id=state["workspace_id"],
                    sql=f'SELECT * FROM "{table_name}"',
                    parameters=[],
                    allowed_table_names={table_name},
                )
            except ValueError as exc:
                message = str(exc)
                if _is_infrastructure_error(message):
                    raise AgentRuntimeError(
                        message,
                        code="DATABASE_ERROR",
                    ) from exc
                any_error = True
                logger.exception(
                    "Cross-document direct table query failed: %s",
                    table_name,
                )
                evidence.append(
                    {
                        "source_type": "table_query",
                        "workspace_id": state["workspace_id"],
                        "sources": [
                            {
                                "table_name": table_name,
                                "document_id": _coerce_int(item.get("document_id")),
                            }
                        ],
                        "content": {
                            "message": (
                                "Structured evidence from this resolved source "
                                "could not be retrieved."
                            ),
                            "document_id": _coerce_int(item.get("document_id")),
                            "document_title": item.get("document_title"),
                            "error": message,
                        },
                        "result_status": "RETRIEVAL_ERROR",
                    }
                )
                continue

            row_count = int(result.get("row_count", 0) or 0)
            success = row_count > 0
            any_success = any_success or success

            evidence.append(
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [
                        {
                            "table_name": table_name,
                            "document_id": _coerce_int(item.get("document_id")),
                        }
                    ],
                    "content": {
                        **_table_content(
                            result,
                            schema_by_table,
                            sql=f'SELECT * FROM "{table_name}"',
                            operation=operation,
                        ),
                        "evidence_role": "document_table",
                        "document_title": item.get("document_title"),
                        "interpretation": (
                            "Rows extracted directly from a table belonging to a "
                            "document the user explicitly referenced. This is source "
                            "content for cross-document reasoning. It was retrieved "
                            "independently and was never joined with another document."
                        ),
                    },
                    "result_status": "SUCCESS" if success else "NOT_FOUND",
                }
            )

    if any_success:
        overall = "SUCCESS"
    elif any_error:
        overall = "RETRIEVAL_ERROR"
    else:
        overall = "NOT_FOUND"

    return {"evidence": evidence, "status": overall}


def _run_table(
    state: AgentState,
    budget: _RecoveryBudget,
) -> dict[str, Any]:
    schema = get_workspace_schema(
        state["workspace_id"],
        allowed_document_ids=None,
    )
    if not schema:
        logger.info(
            "table_query: no structured tables exist in workspace %s; "
            "returning NOT_FOUND evidence",
            state["workspace_id"],
        )
        return {
            "evidence": [
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [],
                    "content": {
                        "message": (
                            "No structured tables are available in this "
                            "workspace, so the requested record could not "
                            "be looked up."
                        ),
                    },
                    "result_status": "NOT_FOUND",
                }
            ],
            "status": "NOT_FOUND",
        }

    resolved_table_names = _resolved_table_names(state)
    resolved_document_ids = _resolved_document_ids(state)
    entities = _resolved_entity_values(state)
    is_mixed = _plan_is_mixed(state)

    # Multiple resolved documents are always queried independently. This is
    # the hard boundary that prevents cross-document SQL Cartesian products.
    # The answer layer performs the actual cross-document reasoning later.
    if len(set(resolved_document_ids)) >= 2 or (
        resolved_table_names
        and len({
            _coerce_int(item.get("document_id"))
            for item in schema
            if str(item.get("table_name")) in resolved_table_names
            and _coerce_int(item.get("document_id")) is not None
        }) >= 2
    ):
        return _run_cross_document_tables(state, budget, schema)

    # Tables that deterministic entity resolution PROVED contain the
    # referenced entity. Workspace truth: must stay in scope regardless
    # of which documents the user named.
    entity_location_tables = {
        str(location.get("table_name"))
        for entity in entities
        for location in entity.get("locations", [])
        if location.get("table_name")
    }

    missing_named_sources = _named_missing_source_references(state)
    if (
        missing_named_sources
        and not resolved_table_names
        and not resolved_document_ids
        and not entities
    ):
        logger.info(
            "table_query: named source(s) %s not found in workspace %s; "
            "returning NOT_FOUND evidence",
            missing_named_sources,
            state["workspace_id"],
        )
        available = sorted(
            {
                str(item.get("document_title") or item.get("table_name"))
                for item in schema
                if item.get("document_title") or item.get("table_name")
            }
        )
        return {
            "evidence": [
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [],
                    "content": {
                        "message": (
                            "The requested source(s) "
                            + ", ".join(missing_named_sources[:5])
                            + " were not found in this workspace."
                            + (
                                f" Available sources: {', '.join(available[:10])}."
                                if available
                                else ""
                            )
                        ),
                    },
                    "result_status": "NOT_FOUND",
                }
            ],
            "status": "NOT_FOUND",
        }

    if resolved_table_names:
        filtered_schema = [
            item
            for item in schema
            if item.get("table_name") in resolved_table_names
        ]

        if not filtered_schema:
            logger.info(
                "table_query: resolved table(s) %s no longer exist in workspace %s; "
                "returning NOT_FOUND evidence",
                sorted(resolved_table_names),
                state["workspace_id"],
            )
            return {
                "evidence": [
                    {
                        "source_type": "table_query",
                        "workspace_id": state["workspace_id"],
                        "sources": [],
                        "content": {
                            "message": (
                                "The requested structured table could not be found "
                                "in this workspace."
                            ),
                        },
                        "result_status": "NOT_FOUND",
                    }
                ],
                "status": "NOT_FOUND",
            }

        schema_for_sql = filtered_schema
    elif resolved_document_ids:
        # A resolved document is the scope for its own structured evidence.
        # Mixed requests are still kept source-scoped; cross-document requests
        # were handled above and each source gets its own SQL call.
        filtered_schema = [
            item
            for item in schema
            if _coerce_int(item.get("document_id")) in resolved_document_ids
            or item.get("table_name") in entity_location_tables
        ]
        if not filtered_schema:
            logger.info(
                "table_query: resolved document(s) %s expose no tables; "
                "returning NOT_FOUND evidence",
                sorted(resolved_document_ids),
            )
            return {
                "evidence": [
                    {
                        "source_type": "table_query",
                        "workspace_id": state["workspace_id"],
                        "sources": [],
                        "content": {
                            "message": (
                                "The resolved source does not expose "
                                "structured tables in this workspace."
                            ),
                        },
                        "result_status": "NOT_FOUND",
                    }
                ],
                "status": "NOT_FOUND",
            }
        schema_for_sql = filtered_schema
    else:
        schema_for_sql = schema

    allowed = {item["table_name"] for item in schema_for_sql}
    schema_by_table = {item["table_name"]: item for item in schema_for_sql}

    planner_notes = list(
        state.get("resolved_references", {}).get("contextual_notes", []) or []
    )

    first = _attempt_sql(
        state=state,
        schema_for_sql=schema_for_sql,
        entities=entities,
        allowed=allowed,
        previous_attempt=None,
        planner_notes=planner_notes,
    )

    chosen = first

    failed = first["result"] is None
    zero_rows_for_known_entity = (
        first["result"] is not None
        and first["result"].get("row_count", 0) == 0
        and bool(entities)
    )
    operation = str(
        state.get("query_understanding", {}).get("operation") or "answer"
    ).lower()
    # Row-producing operations whose zero-row result contradicts a
    # populated source mean the SQL is wrong, not the data.
    zero_rows_implies_bad_sql = (
        first["result"] is not None
        and first["result"].get("row_count", 0) == 0
        and operation in ROW_RESULT_OPERATIONS
        and any(int(item.get("row_count") or 0) > 0 for item in schema_for_sql)
    )

    if (
        failed
        or zero_rows_for_known_entity
        or zero_rows_implies_bad_sql
    ) and not _past_deadline(state):
        if budget.try_consume():
            previous: dict[str, Any] = {
                "sql": first["sql"],
                "parameters": first["parameters"],
            }
            if failed:
                previous["error"] = first["error"]
            else:
                previous["zero_rows"] = True
                if zero_rows_implies_bad_sql:
                    previous["error"] = _zero_rows_repair_guidance(
                        schema_for_sql, is_mixed
                    )

            logger.info(
                "table_query recovery: reason=%s",
                "error" if failed else "zero_rows",
            )
            try:
                second = _attempt_sql(
                    state=state,
                    schema_for_sql=schema_for_sql,
                    entities=entities,
                    allowed=allowed,
                    previous_attempt=previous,
                    planner_notes=planner_notes,
                )
            except AgentRuntimeError:
                raise
            except Exception:
                logger.exception("table_query recovery failed")
                second = None

            if (
                second is not None
                and _outcome_rank(second) >= _outcome_rank(first)
            ):
                chosen = second

    result = chosen["result"]

    if result is None:
        # Both attempts failed. Degrade instead of failing the whole
        # request. RETRIEVAL_ERROR matches the sufficiency module's
        # known table statuses (TABLE_ERROR).
        logger.error("Table query failed after recovery: %s", chosen["error"])
        return {
            "evidence": [
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [],
                    "content": {
                        "message": (
                            "The requested table data could not be "
                            "retrieved, so no structured facts are "
                            "available for this request."
                        ),
                    },
                    "result_status": "RETRIEVAL_ERROR",
                }
            ],
            "status": "RETRIEVAL_ERROR",
        }

    status = (
        "SUCCESS" if int(result.get("row_count", 0) or 0) > 0 else "NOT_FOUND"
    )
    return {
        "evidence": [
            {
                "source_type": "table_query",
                "workspace_id": state["workspace_id"],
                # document_id included so chat.services.sources can
                # build citable source rows for business tables.
                "sources": [
                    {
                        "table_name": name,
                        "document_id": (
                            schema_by_table[name].get("document_id")
                            if name in schema_by_table
                            else None
                        ),
                    }
                    for name in result.get("tables_used", [])
                ],
                "content": _table_content(
                    result,
                    schema_by_table,
                    sql=chosen["sql"],
                    operation=operation,
                ),
                "result_status": status,
            }
        ],
        "status": status,
    }


# Module-level compiled graph. chat.services.execution imports this
# symbol, so it must exist at import time.
agent_graph = build_agent_graph()