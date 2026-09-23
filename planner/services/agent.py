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
from planner.services.intent import build_execution_plan
from planner.services.product_knowledge import get_product_knowledge
from planner.services.schema_registry import get_workspace_schema
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

# Planner output is a small strict-JSON object (~150 tokens), but gpt-oss
# spends completion tokens on hidden reasoning first. 350 had the same
# failure mode as the SQL generator ("max completion tokens reached before
# generating a valid document"). This is a ceiling, not a target.
PLANNER_MAX_COMPLETION_TOKENS = 1000

# Exactly ONE recovery attempt per request, shared across all tools.
# Worst-case LLM calls per request are therefore bounded by construction:
#   planner (1) + SQL (1) + SQL repair (1, only if recovery unused)
#   + answer (1) = 4
MAX_RECOVERY_ATTEMPTS = 1

# Set this from data, not by guessing. Print `confidence` from
# search_evidence for queries your documents DO answer and ones they don't,
# then pick a value between the two groups. None = threshold check off
# (only "zero chunks returned" counts as insufficient).
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
        plan = build_execution_plan(understanding, resolution)

    except AgentRuntimeError as exc:
        logger.exception("Query understanding failed")
        # Log the real detail above; never forward raw internal
        # exception text to the user-facing SSE stream / persisted
        # thoughts. state["error"] below still carries the detail for
        # internal callers, but it is never emitted as a ChatEvent.
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

    # Structured / authoritative evidence (small) is collected apart from
    # document chunks (up to 8) so a full set of chunks can never push the
    # table result out of the evidence cap.
    structured: list[Any] = []
    documents: list[Any] = []
    total_tool_calls = 0
    table_status: str | None = None
    retrieval_confidence = 0.0

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
            table = _run_table(state, budget)
            structured.extend(table["evidence"])
            table_status = table["status"]
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

        # If the table lookup ran fine but matched nothing (missing employee,
        # unknown ID, no row meets the condition), a policy document cannot
        # make the request answerable. Skip document search: it would only
        # add worked examples the answer model might mistake for real data.
        table_not_found = table_status == "NOT_FOUND"
        if "DOCUMENT_SEARCH" in capabilities and table_not_found:
            logger.info(
                "Skipping document search: required table lookup matched no rows"
            )

        if "DOCUMENT_SEARCH" in capabilities and not table_not_found:
            docs = _run_documents(state, budget)
            total_tool_calls += 1
            retrieval_confidence = docs["confidence"]

            if docs["error"] is not None:
                code, message = docs["error"]
                if not structured:
                    # Nothing else to answer from: this is a real failure.
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
            wants_documents="DOCUMENT_SEARCH" in capabilities and not table_not_found,
            document_chunk_count=len(documents),
            retrieval_confidence=retrieval_confidence,
            min_retrieval_confidence=MIN_RETRIEVAL_CONFIDENCE,
        )

        # Can ANYTHING here support an answer? Structured evidence
        # (table result, metadata, memory, product info, even a clean
        # NOT_FOUND) is answerable. Document chunks only count when the
        # search was not insufficient. A pure document request whose
        # chunks are all weak is NOT answerable: finish_node stops there
        # instead of letting the answer model improvise from them.
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

        # Structured evidence first, then as many chunks as fit, then notes.
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
        # Only weak document evidence exists. Do not call the answer
        # model, and do not surface the weak chunks as sources.
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

    # Every tool that ran failed: say so directly instead of spending an
    # LLM call to phrase it.
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

    # A genuine "not found" (e.g. SQL ran cleanly and matched zero rows)
    # is distinguished from real evidence in the route/error_code.
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
    return json.dumps(
        {
            "user_query": state.get("query", ""),
            "workspace_id": state.get("workspace_id"),
            "relevant_document_ids": state.get("relevant_document_ids", []),
            # Rolling summary + newest 10 messages (bounded), NOT [-8:].
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

# Generic references to the document/table catalog itself ("tell me
# about documents", "what files are uploaded"). These are NOT specific
# source names: asking about them means the user wants the catalog,
# not a lookup of a missing named source. Without this check a plain
# catalog request falls into the NOT_FOUND branch below and produces
# "the requested documents were not found" even when documents exist.
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

        # Generic catalog references ("documents", "uploaded files",
        # "the tables") are not missing named sources: the user is
        # asking about the catalog itself. Serve the full catalog.
        unresolved_specific = [
            reference
            for reference in unresolved
            if not _is_generic_catalog_reference(reference)
        ]

        if unresolved_specific:
            # The user named something specific and none of it exists.
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
    Documents the user EXPLICITLY pointed at (a named document or table).

    Entity matches are deliberately excluded: an entity's location says
    where a VALUE lives (usually a spreadsheet), not which text document
    the question is about. Scoping text search to that spreadsheet would
    hide a policy PDF the question also depends on.
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
        # One broadened retry: the user's own words, over every workspace
        # document. Skipped when it would just repeat the first search.
        raw_query = str(state.get("query", "")).strip()

        # If the user pointed at a specific document, recovery must NOT
        # widen beyond it (a similar phrase in another file could then
        # end up in the answer). Broaden the QUERY only. Widen the
        # document set only when the original search covered the whole
        # workspace anyway.
        scoped_to_resolved = bool(_resolved_document_ids(state))
        all_ids = list(state.get("relevant_document_ids", []))
        retry_ids = primary_ids if scoped_to_resolved else (all_ids or primary_ids)

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
    whatever resolve_references actually found, with no reference to
    any specific person/table name.
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


# Sent to the SQL generator on the repair pass when the first attempt produced
# no usable SQL. Deliberately generic: raw provider text (Groq error bodies,
# failed_generation, etc.) is logged server-side and never forwarded.
_SQL_OUTPUT_FAILURE = (
    "The previous response was not valid or was cut off before the JSON was "
    "complete. Return only the JSON object with the shortest valid query."
)

# The planner already selected TABLE_QUERY, so a table query is REQUIRED.
# needs_sql=false from the generator is therefore an invalid decision, not
# "nothing to fetch". Sent (generic text only) on the repair pass.
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


def _attempt_sql(
    *,
    state: AgentState,
    schema_for_sql: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    allowed: set[str],
    previous_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    One generate-SQL + run-SQL attempt. Never raises for problems the SQL
    generator or SQLite can cause; those come back as `error` so the
    caller can decide whether to repair. Infrastructure problems raise.
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
        )
        if not isinstance(decision, dict):
            raise TypeError("SQL generator returned a non-object.")
        needs_sql = bool(decision.get("needs_sql"))
    except (
        InvalidResponseError,  # provider rejected/truncated the structured output
        json.JSONDecodeError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
    ):
        # A normal FAILED SQL attempt: it flows into the existing one-time,
        # shared-budget repair pass in _run_table. Other provider errors
        # (rate limits, timeouts, auth) are NOT caught here on purpose.
        logger.exception("SQL generation failed or returned invalid output")
        outcome["error"] = _SQL_OUTPUT_FAILURE
        return outcome

    if not needs_sql:
        # A planned TABLE_QUERY must produce SQL. Treat a refusal as a failed
        # attempt so it enters the existing one-time repair pass instead of
        # silently producing empty evidence (and NO_EVIDENCE downstream).
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
    Describe what the executed SQL result means, using the actual SQL and
    planner operation rather than domain-specific assumptions.

    This metadata is part of the evidence contract so the answer model can
    distinguish a deliberately selected one-row positional/ranking result
    from an arbitrary one-row sample.
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
    Source table size and an explicit interpretation travel with it so the
    answer model cannot read a one-row result as a one-row table.
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



def _run_table(
    state: AgentState,
    budget: _RecoveryBudget,
) -> dict[str, Any]:
    schema = get_workspace_schema(
        state["workspace_id"],
        allowed_document_ids=None,
    )
    if not schema:
        # No structured tables exist in this workspace. This is a
        # permanent condition, not a transient outage: return clean
        # NOT_FOUND evidence instead of raising DATABASE_ERROR, which
        # would tell the user to "try again" for something that can
        # never succeed. Downstream this reuses the KNOWN NOT_FOUND
        # semantics everywhere (assess_evidence, dispatch's
        # document-search skip, finish_node's route handling), so no
        # dependent module sees a new status value.
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
    if resolved_table_names:
        filtered_schema = [
            item
            for item in schema
            if item.get("table_name") in resolved_table_names
        ]

        # A resolved table is an authoritative scope constraint. If it has
        # disappeared since resolution, fail closed and report NOT_FOUND
        # rather than silently widening SQL generation to every workspace
        # table (or sending an empty schema to the SQL generator).
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
    else:
        schema_for_sql = schema

    entities = _resolved_entity_values(state)
    allowed = {item["table_name"] for item in schema_for_sql}

    first = _attempt_sql(
        state=state,
        schema_for_sql=schema_for_sql,
        entities=entities,
        allowed=allowed,
        previous_attempt=None,
    )

    chosen = first

    # Recover from a real failure, or from zero rows for an entity that the
    # resolver already proved exists (so the SQL, not the data, is wrong).
    failed = first["result"] is None
    zero_rows_for_known_entity = (
        first["result"] is not None
        and first["result"].get("row_count", 0) == 0
        and bool(entities)
    )

    if (failed or zero_rows_for_known_entity) and not _past_deadline(state):
        if budget.try_consume():
            previous: dict[str, Any] = {
                "sql": first["sql"],
                "parameters": first["parameters"],
            }
            if failed:
                previous["error"] = first["error"]
            else:
                previous["zero_rows"] = True

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
        # Both attempts failed. Degrade instead of failing the whole request;
        # internal error text is logged, not shown to the answer model.
        logger.error("Table query failed after recovery: %s", chosen["error"])
        return {
            "evidence": [
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [],
                    "content": {
                        "error": "The table query could not be completed."
                    },
                    "result_status": "RETRIEVAL_ERROR",
                }
            ],
            "status": "RETRIEVAL_ERROR",
        }

    result = dict(result)
    status = "NOT_FOUND" if result.get("row_count") == 0 else "SUCCESS"
    result["result_status"] = status

    schema_by_table = {
        item["table_name"]: item
        for item in schema
    }
    sources = [
        {
            "document_id": schema_by_table[name].get("document_id"),
            "table_name": name,
        }
        for name in result.get("tables_used", [])
        if name in schema_by_table
    ]

    return {
        "evidence": [
            {
                "source_type": "table_query",
                "workspace_id": state["workspace_id"],
                "sources": sources,
                "content": _table_content(
                    result,
                    schema_by_table,
                    sql=chosen.get("sql"),
                    operation=str(
                        state.get("query_understanding", {}).get("operation")
                        or "answer"
                    ),
                ),
                "result_status": status,
            }
        ],
        "status": status,
    }


    
agent_graph = build_agent_graph()