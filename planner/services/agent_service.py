"""Services Agent Service module for QueryPilot.

This module implements the agent runtime used by QueryPilot. It contains
the StateGraph nodes and helpers used during planning and dispatch.

Timing helpers
---------------
The module provides lightweight timing helpers used for diagnostics:

`_timed_call(stage, fn)` -- a decorator factory that wraps a function and
measures its execution time using the `timed(stage)` context manager. It
preserves the wrapped function's metadata with `functools.wraps`.

`_timed_node(name, fn)` -- a wrapper generator for StateGraph node
functions. The returned wrapper accepts a single `state` argument and
measures duration with `timed(f"node.{name}")` so node-level timing is
consistently labeled in diagnostics.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time

from jsonschema import Draft202012Validator
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from llm.exceptions import InvalidResponseError
from llm.router import router
from planner.prompts.agent import ACTION_SCHEMA, PLANNER_RESPONSE_FORMAT, SYSTEM_PROMPT
from planner.services.answer import (
    _narrow_spreadsheet_tables_v3,
    _parse_answer_response,
    generate_answer_stream,
)
from planner.services.cross_document_join import (
    build_joined_view,
    run_cross_document_join,
)
from planner.services.coverage import (
    document_coverage_note,
    prior_conclusions_note,
    requirement_coverage_note,
)
from planner.services.document_coverage import build_document_evidence
from planner.services.entity_resolution import (
    resolve_entities_from_query,
    resolve_references,
)
from planner.services.evidence import (
    assign_citation_indexes,
    finalize_structured,
    reconcile_coverage_notes,
)
from planner.services.history import (
    bound_history_for_planner,
    resolve_replay,
)
from planner.services.intent import (
    OVERVIEW_OPERATIONS,
    build_execution_plan,
    normalized_operation,
)
from planner.services.source_scope import (
    authorized_document_ids,
    candidate_document_ids,
    contextual_planning_query,
    dependency_discovery_query,
    effective_planning_query,
    requires_dependency_discovery,
    requires_document_evidence,
    requests_workspace_wide_cross_document_search,
)
from planner.services.product_knowledge import get_product_knowledge
from planner.services.schema_registry import (
    WorkspaceResources,
    get_workspace_catalog,
    get_workspace_schema,
)
from planner.services.sql_generation import generate_sql
from planner.services.state import AgentState
from planner.services.requirements import (
    ensure_query_fallback_requirement,
    normalize_requirements,
)
from planner.services.query_trace import QueryTrace, current_trace
from planner.services.sufficiency import documents_insufficient
from planner.services.table_selection import (
    select_tables,
    select_tables_detailed,
)
from planner.tools.conversation_memory import conversation_memory
from planner.tools.query_tables import MAX_RESULTS as QUERY_ROW_CAP
from planner.tools.query_tables import query_tables
from planner.tools.search_evidence import fetch_document_chunks, search_evidence
from planner.tools.source_metadata import source_metadata
from utils.timing import timed

logger = logging.getLogger(__name__)
_PLANNER_VALIDATOR = Draft202012Validator(ACTION_SCHEMA["json_schema"]["schema"])


def _timed_call(stage, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        try:
            with timed(stage):
                return fn(*args, **kwargs)
        finally:
            trace = current_trace()
            if trace is not None:
                trace.record_stage(stage, time.monotonic() - started)

    return wrapper


def _timed_node(name, fn):
    @functools.wraps(fn)
    def wrapper(state):
        trace = state.get("query_trace") or current_trace()
        if trace is None and name == "agent":
            trace = QueryTrace(
                query=str(state.get("query", "")),
                workspace_id=state.get("workspace_id"),
            )
            state = {**state, "query_trace": trace}

        bound_here = bool(trace is not None and trace.activate())
        started = time.monotonic()
        result = None
        failed = False
        try:
            with timed(f"node.{name}"):
                result = fn(state)
            active_trace = current_trace() or trace
            if active_trace is not None and isinstance(result, dict):
                result.setdefault("query_trace", active_trace)
            return result
        except BaseException:
            failed = True
            raise
        finally:
            active_trace = current_trace() or trace
            if active_trace is not None:
                active_trace.record_stage(f"node.{name}", time.monotonic() - started)
                if name == "finish":
                    result_route = (
                        (result or {}).get("route", state.get("route", "unknown"))
                        if isinstance(result, dict)
                        else state.get("route", "unknown")
                    )
                    result_code = (
                        (result or {}).get("error_code", state.get("error_code", ""))
                        if isinstance(result, dict)
                        else state.get("error_code", "")
                    )
                    active_trace.close(route=result_route, error_code=result_code)
                elif failed:
                    # LangGraph may retry a failed node. Record the failed
                    # attempt without closing the request trace prematurely.
                    active_trace.increment("node_errors")
                    logger.error(
                        "[QUERY_TRACE_NODE_ERROR] trace_id=%s node=%s",
                        active_trace.trace_id,
                        str(name)[:80],
                    )
            if bound_here and active_trace is not None:
                active_trace.deactivate()

    return wrapper


def _count_tool_call(current_count: int) -> int:
    next_count = int(current_count) + 1
    trace = current_trace()
    if trace is not None:
        trace.increment("tool_calls")
    if next_count > MAX_TOOL_CALLS_PER_QUERY:
        raise AgentRuntimeError(
            "The request reached its tool-call budget.",
            code="TOOL_BUDGET_EXCEEDED",
        )
    return next_count

resolve_references = _timed_call("resolve_references", resolve_references)
resolve_entities_from_query = _timed_call(
    "resolve_entities", resolve_entities_from_query
)
get_workspace_schema = _timed_call("get_workspace_schema", get_workspace_schema)
get_workspace_catalog = _timed_call("get_workspace_catalog", get_workspace_catalog)
select_tables = _timed_call("select_tables", select_tables)
select_tables_detailed = _timed_call(
    "select_tables", select_tables_detailed
)
generate_sql = _timed_call("generate_sql", generate_sql)
query_tables = _timed_call("query_tables", query_tables)
search_evidence = _timed_call("search_evidence", search_evidence)
fetch_document_chunks = _timed_call("fetch_document_chunks", fetch_document_chunks)
build_document_evidence = _timed_call("build_document_evidence", build_document_evidence)
run_cross_document_join = _timed_call("cross_document_join", run_cross_document_join)
finalize_structured = _timed_call("finalize_structured", finalize_structured)


# Final evidence and planner catalog are not silently count-limited here;
# authorization, finite tool-call count and query deadline remain enforced.

MAX_RECOVERY_ATTEMPTS = 2
MAX_TOOL_CALLS_PER_QUERY = 12
MAX_QUERY_EXECUTION_SECONDS = 60

_EXHAUSTIVE_CUE = re.compile(
    r"\b(?:all|every|each|entire|whole|complete|full|exhaustive(?:ly)?)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+"
    r"(?:units?|modules?|chapters?|sections?|topics?)\b",
    re.IGNORECASE,
)


MIN_RETRIEVAL_CONFIDENCE: float | None = None

DEFAULT_FAILURE_MESSAGE = (
    "I could not complete that request because the required data operation failed."
)

ERROR_MESSAGES = {
    "REQUEST_TIMEOUT": (
        "This request took longer than expected, so I stopped it. "
        "Please try again, or try a simpler question."
    ),
    "DATABASE_ERROR": (
        "I couldn't reach the workspace data just now. Please try again in a moment."
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
    graph.add_node("agent", _timed_node("agent", agent_node))
    graph.add_node("dispatch", _timed_node("dispatch", dispatch_node))
    graph.add_node("finish", _timed_node("finish", finish_node))
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


def _resources(state: AgentState) -> WorkspaceResources:
    """Request-scoped structured-resource metadata for this request's workspace.

    agent_node / dispatch_node put one WorkspaceResources on the state they
    hand to every helper, so catalog / schema / table-index work happens at
    most once per request and is reused by every later stage. An instance is
    only ever reused for the workspace it was built for; anything else gets a
    fresh, un-shared instance. Nothing here is module-level state.
    """
    workspace_id = int(state["workspace_id"])
    resources = state.get("resource_context")
    if isinstance(resources, WorkspaceResources) and (
        resources.workspace_id == workspace_id
    ):
        return resources
    return WorkspaceResources(workspace_id)


def _scope_document_ids(
    state: AgentState,
    resources: WorkspaceResources,
) -> set[int] | None:
    """Documents whose table details this request can need, or None for all.

    Union of everything the request already established: the planner's
    resolved documents, documents/tables matched by reference resolution, and
    the tables entity resolution located values in. Callers use it to avoid
    sampling/counting tables of unrelated documents; None (nothing resolved)
    means "no narrowing", exactly like the previous full-workspace read.
    """
    ids: set[int] = set(_candidate_document_ids(state, resources))
    for item in state.get("resolved_references", {}).get("references", []):
        if item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if isinstance(match, dict) and match.get("type") in {"document", "table"}:
            document_id = _coerce_int(match.get("document_id"))
            if document_id is not None:
                ids.add(document_id)

    table_names = set(_resolved_table_names(state))
    for entity in _resolved_entity_values(state):
        for location in entity.get("locations", []):
            if location.get("table_name"):
                table_names.add(str(location["table_name"]))
    if table_names:
        for table in resources.table_index(document_ids=_allowed_document_ids(state)):
            if str(table.get("table_name")) in table_names:
                document_id = _coerce_int(table.get("document_id"))
                if document_id is not None:
                    ids.add(document_id)
    allowed_ids = _allowed_document_ids(state)
    if allowed_ids is not None:
        before_authorization = set(ids)
        ids.intersection_update(allowed_ids)
        # Do not turn an explicitly resolved but unauthorized source into a
        # broad search over other documents. When nothing was resolved at all,
        # the allow-list remains the safe discovery boundary.
        if before_authorization and not ids:
            return set()
        return ids or set(allowed_ids)
    return ids or None


def _chitchat_reply(query: str) -> str | None:
    """Deterministic greeting/thanks handling: no LLM call, no retrieval."""
    normalized = re.sub(r"[^\w\s]", "", str(query or "").lower()).strip()
    return CHITCHAT_REPLIES.get(normalized)


def _resolved_entity_values_from_resolution(
    resolution: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract resolved entity matches from a resolution object."""
    if not isinstance(resolution, dict):
        return []

    entities: list[dict[str, Any]] = []
    for item in resolution.get("references", []) or []:
        if not isinstance(item, dict) or item.get("status") != "RESOLVED":
            continue
        match = item.get("match")
        if not isinstance(match, dict) or match.get("type") != "entity":
            continue
        value = match.get("value")
        locations = match.get("locations")
        if value and isinstance(locations, list) and locations:
            entities.append({"value": value, "locations": locations})
    return entities


def _merge_entity_fallback(
    primary: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """
    Add only deterministic entity matches from a structured fallback pass.

    Existing document/table resolutions and ambiguities stay authoritative;
    a fallback result only replaces a NOT_FOUND reference with a deterministic
    entity result.
    """
    merged = dict(primary)
    primary_refs = [dict(item) for item in primary.get("references", []) or []]
    fallback_by_reference = {
        str(item.get("reference") or ""): item
        for item in fallback.get("references", []) or []
        if isinstance(item, dict) and item.get("reference")
    }

    changed = False
    for index, current in enumerate(primary_refs):
        if current.get("status") != "NOT_FOUND":
            continue
        reference = str(current.get("reference") or "")
        candidate = fallback_by_reference.get(reference)
        if not isinstance(candidate, dict):
            continue
        candidate_status = candidate.get("status")
        if candidate_status not in {"RESOLVED", "AMBIGUOUS"}:
            continue
        if candidate_status == "RESOLVED":
            match = candidate.get("match")
            if not isinstance(match, dict) or match.get("type") != "entity":
                continue
        current.update(candidate)
        primary_refs[index] = current
        changed = True

    if not changed:
        return primary

    merged["references"] = primary_refs
    merged["has_ambiguity"] = any(
        item.get("status") == "AMBIGUOUS" for item in primary_refs
    )
    merged["has_resolved"] = any(
        item.get("status") == "RESOLVED" for item in primary_refs
    )
    return merged


def _merge_query_entity_fallback(
    primary: dict[str, Any],
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Append deterministic entities found directly in the current query."""
    if not entities:
        return primary

    merged = dict(primary)
    references = [dict(item) for item in primary.get("references", []) or []]
    existing_entities = {
        str(item.get("match", {}).get("value") or "").casefold()
        for item in references
        if item.get("status") == "RESOLVED"
        and isinstance(item.get("match"), dict)
        and item.get("match", {}).get("type") == "entity"
    }

    changed = False
    for entity in entities:
        value = str(entity.get("value") or "").strip()
        locations = entity.get("locations")
        if not value or not isinstance(locations, list) or not locations:
            continue
        if value.casefold() in existing_entities:
            continue
        match = {
            "type": "entity",
            "value": value,
            "locations": locations,
        }
        if entity.get("document_id") is not None:
            match["document_id"] = entity.get("document_id")
        if entity.get("document_title"):
            match["document_title"] = entity.get("document_title")
        references.append(
            {
                "reference": value,
                "status": "RESOLVED",
                "match": match,
            }
        )
        existing_entities.add(value.casefold())
        changed = True

    if not changed:
        return primary

    merged["references"] = references
    merged["has_resolved"] = True
    return merged


def _has_pending_value_reference(resolution: dict[str, Any] | None) -> bool:
    """
    True when the request carries a non-source reference (a person, id,
    value...) that did not resolve yet. Such a value still has to be looked
    up in structured data, so the request is not a pure document request.
    File names and generic catalog phrases ("the document") do not count.
    """
    if not isinstance(resolution, dict):
        return False
    for item in resolution.get("references", []) or []:
        if not isinstance(item, dict) or item.get("status") != "NOT_FOUND":
            continue
        reference = str(item.get("reference") or "").strip()
        if not reference or _is_generic_catalog_reference(reference):
            continue
        if _parse_contextual_reference(reference) is not None:
            continue
        if _FILE_EXTENSION_RE.search(reference):
            continue
        return True
    return False


def _build_plan(
    state: AgentState,
    understanding: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    return build_execution_plan(
        understanding,
        resolution,
        _resolved_source_shapes(state, understanding, resolution),
        pending_value_reference=_has_pending_value_reference(resolution),
    )


def _needs_entity_lookup(
    plan: dict[str, Any],
    understanding: dict[str, Any],
) -> bool:
    """
    Run the deterministic (local SQLite, no LLM) entity scan when a value may
    have to be looked up: a table query is planned, or the planner called the
    request structured/mixed. This keeps "Rahul's salary in salary.xlsx" working
    even if the planner forgot to list "Rahul" as a reference.
    """
    if "TABLE_QUERY" in plan.get("capabilities", []):
        return True
    return str(understanding.get("category_hint") or "").lower() in {
        "structured",
        "mixed",
    }


def _apply_scope_route(state: AgentState, plan: dict[str, Any]) -> dict[str, Any]:
    """Honour the Scope Guard's PRODUCT verdict.

    The Scope Guard already resolved the request's intent (including pronouns
    such as "who built it"). The planner sees only a bounded history and can
    mislabel such a question as chat history, so no product evidence is
    fetched. PRODUCT questions need exactly the product knowledge source.
    """
    if state.get("scope_route") != "PRODUCT":
        return plan
    return {
        **plan,
        "status": "OK",
        "capabilities": ["PRODUCT_KNOWLEDGE"],
        "fallback_capabilities": [],
        "cross_document": False,
        "clarification_options": [],
        "decision": {
            **(plan.get("decision") or {}),
            "category": "product",
            "resolved_document_ids": [],
            "reason": "QueryPilot product question (Scope Guard)",
            "cross_document_reason": "no document sources needed",
        },
    }


# --- SUPPORTING_SOURCE_DISCOVERY (patch) ----------------------------------
# Reference resolution only finds sources the user NAMED. A contract/policy
# can state a rule but never holds the records the rule is applied to, so a
# question like "was X met in period P?" resolved only the rule document and
# stayed document-only (cross_document=False). The planner already sees the
# authorized workspace_catalog (titles + column digest); it now declares which
# other catalog sources the answer needs (`data_sources_needed`). Code does not
# trust the title: it is resolved with the same exact-match resolver used for
# user-named sources, against the authorized catalog only.
MAX_SUPPORTING_SOURCES = 3


def _add_planner_supporting_sources(
    *,
    state: AgentState,
    understanding: dict[str, Any],
    resolution: dict[str, Any],
    schema: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    authorized_ids: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    """Add planner-declared supporting sources to the resolved sources.

    Returns (understanding, resolution, added_document_ids); unchanged inputs
    and [] when nothing applies."""
    raw = understanding.get("data_sources_needed")
    if not isinstance(raw, list) or state.get("scope_route") == "PRODUCT":
        return understanding, resolution, []
    hint = str(understanding.get("category_hint") or "").strip().lower()
    if hint in {"conversation", "product", "metadata"}:
        return understanding, resolution, []

    titles: list[str] = []
    for value in raw:
        title = str(value or "").strip()
        if title and title not in titles:
            titles.append(title)
    titles = titles[:MAX_SUPPORTING_SOURCES]
    if not titles:
        return understanding, resolution, []

    already: set[int] = set()
    for item in resolution.get("references", []) or []:
        match = item.get("match") if isinstance(item, dict) else None
        if (
            isinstance(item, dict)
            and item.get("status") == "RESOLVED"
            and isinstance(match, dict)
            and match.get("type") in {"document", "table"}
        ):
            document_id = _coerce_int(match.get("document_id"))
            if document_id is not None:
                already.add(document_id)

    found = resolve_references(
        workspace_id=state["workspace_id"],
        references=titles,
        category_hint="document",
        schema=schema,
        catalog=catalog,
    )
    allowed = None if authorized_ids is None else {int(v) for v in authorized_ids}

    references = [dict(item) for item in resolution.get("references", []) or []]
    added: list[int] = []
    for item in found.get("references", []) or []:
        match = item.get("match") if isinstance(item, dict) else None
        if item.get("status") != "RESOLVED" or not isinstance(match, dict):
            logger.info(
                "supporting source %r not used: status=%s",
                item.get("reference"),
                item.get("status"),
            )
            continue
        if match.get("type") not in {"document", "table"}:
            continue
        document_id = _coerce_int(match.get("document_id"))
        if document_id is None or document_id in already or document_id in added:
            continue
        if allowed is not None and document_id not in allowed:
            continue
        references.append(
            {
                "reference": item.get("reference"),
                "status": "RESOLVED",
                "match": {**match, "origin": "planner_data_source"},
            }
        )
        added.append(document_id)

    if not added:
        return understanding, resolution, []

    new_understanding = dict(understanding)
    if hint == "document":
        # Same invariant the planner prompt states: needing recorded data from
        # another source IS a mixed request.
        new_understanding["category_hint"] = "mixed"
    new_resolution = {**resolution, "references": references, "has_resolved": True}
    logger.info(
        "supporting sources added: document_ids=%s (category_hint %s -> %s)",
        added,
        hint,
        new_understanding.get("category_hint"),
    )
    return new_understanding, new_resolution, added


# --- END SUPPORTING_SOURCE_DISCOVERY ---------------------------------------


def _log_planner_decision(plan: dict[str, Any]) -> None:
    """One concise line: what was chosen and why. Never logs prompt text."""
    decision = plan.get("decision") or {}
    logger.info(
        "planner decision: category=%s operation=%s resolved_document_ids=%s "
        "capabilities=%s fallback_capabilities=%s cross_document=%s "
        "reason=%s cross_document_reason=%s",
        decision.get("category"),
        decision.get("operation"),
        decision.get("resolved_document_ids", []),
        plan.get("capabilities", []),
        plan.get("fallback_capabilities", []),
        bool(plan.get("cross_document")),
        decision.get("reason"),
        decision.get("cross_document_reason"),
    )


def agent_node(state: AgentState) -> dict[str, Any]:
    original_query = str(state.get("query", "")).strip()
    if not state.get("deadline_monotonic"):
        state = {**state, "deadline_monotonic": time.monotonic() + MAX_QUERY_EXECUTION_SECONDS}
    planning_query = original_query

    carry: dict[str, Any] = {}
    replay = resolve_replay(original_query, state.get("conversation_history", []))
    if replay:
        previous_task, effective_query = replay
        logger.info(
            "Correction replay activated: current=%r previous_user_task=%r",
            original_query,
            previous_task,
        )
        carry = {"query": effective_query, "conversation_history": []}
        planning_query = effective_query
        state = {**state, **carry}
    else:
        # A short follow-up often lacks the entity/source names needed to
        # rebuild a calculation plan (for example, "how is that explained?").
        # Enrich planner interpretation only; the graph's original `query`
        # remains unchanged for answer generation and API compatibility.
        contextual_query = contextual_planning_query(
            original_query,
            state.get("conversation_history", []),
        )
        if contextual_query != original_query:
            logger.info(
                "Contextual follow-up planning activated: current=%r",
                original_query,
            )
            planning_query = contextual_query
            state = {**state, "query": contextual_query}

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

    # One request-scoped metadata holder for the whole request. It rides on
    # the state returned below, so dispatch_node reuses what planning loaded.
    resources = _resources(state)
    state = {**state, "resource_context": resources}

    _emit_event(
        event="planning",
        status="started",
        message="Understanding the request.",
    )

    try:
        planner_started = time.monotonic()
        trace = current_trace()
        if trace is not None:
            trace.increment("llm_calls_planner")
        try:
            try:
                response = router.generate(
                    task="planner",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_context(state)},
                    ],
                    temperature=0.0,
                    response_format=PLANNER_RESPONSE_FORMAT,
                    include_reasoning=False,
                )
                understanding = _parse_understanding_with_repair(response)
            except Exception:
                # Planner/provider/schema failures must not take down the chat.
                # Continue with the raw user query as a retrieval safety net.
                logger.exception(
                    "Planner generation failed; continuing with raw-query retrieval"
                )
                understanding = _safe_planner_fallback(
                    "Planner generation failed; raw-query retrieval only."
                )
        finally:
            if trace is not None:
                trace.record_stage("llm.planner", time.monotonic() - planner_started)
        if understanding.pop("_safe_fallback", False):
            fallback_plan = {
                "status": "OK",
                "capabilities": ["DOCUMENT_SEARCH"],
                "actions": [],
                "fallback_capabilities": [],
                "clarification_options": [],
                "cross_document": False,
            }
            return {
                "resource_context": resources,
                "planning_query": planning_query,
                "query_understanding": understanding,
                "evidence_requirements": [],
                "resolved_references": {"references": []},
                "execution_plan": fallback_plan,
                "pending_clarification": False,
                "clarification_options": [],
                "tool_history": list(state.get("tool_history", [])) + [{"stage": "planner_safe_fallback"}],
            }
        try:
            evidence_requirements = normalize_requirements(understanding.get("requirements", []))
        except ValueError as exc:
            logger.warning("Planner requirements rejected; retaining raw-query retrieval: %s", exc)
            evidence_requirements = []
        before_fallback = len(evidence_requirements)
        evidence_requirements = ensure_query_fallback_requirement(
            evidence_requirements,
            query=planning_query or original_query,
            category=str(understanding.get("category_hint") or ""),
            operation=str(understanding.get("operation") or ""),
        )
        if not before_fallback and evidence_requirements:
            logger.warning(
                "Planner supplied no valid evidence requirements; using one "
                "raw-query requirement (category=%s operation=%s)",
                understanding.get("category_hint"),
                understanding.get("operation"),
            )
        understanding["requirements"] = evidence_requirements
        understanding["needs_instance_data"] = bool(understanding.get("needs_instance_data"))
        trace = current_trace()
        if trace is not None:
            trace.increment("planner_requirements", len(evidence_requirements))
        # Source/entity resolution must operate inside the same conversation
        # allow-list enforced by retrieval. Scope Guard's relevance list is
        # intentionally narrower, so use all authorized documents here and
        # let the planner rank/select the required ones.
        authorized_ids = _allowed_document_ids(state)
        resolver_schema = resources.table_index(document_ids=authorized_ids)
        resolver_catalog = resources.catalog()
        if authorized_ids is not None:
            authorized_set = set(authorized_ids)
            resolver_catalog = [
                document
                for document in resolver_catalog
                if _coerce_int(document.get("document_id")) in authorized_set
            ]

        resolution = resolve_references(
            workspace_id=state["workspace_id"],
            references=understanding.get("references", []),
            category_hint=understanding.get("category_hint"),
            schema=resolver_schema,
            catalog=resolver_catalog,
        )

        try:
            resolution = _resolve_contextual_references(state, resolution)
        except Exception:
            logger.exception("Contextual reference resolution failed; continuing")

        try:
            understanding, resolution, _supporting_ids = (
                _add_planner_supporting_sources(
                    state=state,
                    understanding=understanding,
                    resolution=resolution,
                    schema=resolver_schema,
                    catalog=resolver_catalog,
                    authorized_ids=authorized_ids,
                )
            )
        except Exception:
            logger.exception("Supporting-source resolution failed; continuing")

        plan = _build_plan(state, understanding, resolution)

        if (
            _needs_entity_lookup(plan, understanding)
            and not _resolved_entity_values_from_resolution(resolution)
            and _has_pending_value_reference(resolution)
            and understanding.get("references")
        ):
            try:
                fallback_resolution = resolve_references(
                    workspace_id=state["workspace_id"],
                    references=understanding.get("references", []),
                    category_hint="structured",
                    schema=resolver_schema,
                    catalog=resolver_catalog,
                )
                merged_resolution = _merge_entity_fallback(
                    resolution,
                    fallback_resolution,
                )
                if merged_resolution is not resolution:
                    resolution = merged_resolution
                    plan = _build_plan(state, understanding, resolution)
            except Exception:
                logger.exception("Structured entity-resolution fallback failed")

        if _needs_entity_lookup(
            plan, understanding
        ) and not _resolved_entity_values_from_resolution(resolution):
            try:
                schema = resources.table_index(document_ids=authorized_ids)
                query_entities = resolve_entities_from_query(
                    workspace_id=state["workspace_id"],
                    query=str(state.get("query", "")),
                    schema=schema,
                )
                updated_resolution = _merge_query_entity_fallback(
                    resolution,
                    query_entities,
                )
                if updated_resolution != resolution:
                    resolution = updated_resolution
                    plan = _build_plan(state, understanding, resolution)
                    logger.info(
                        "Current-query entity resolution activated: %s",
                        query_entities,
                    )
            except Exception:
                logger.exception("Current-query entity resolution failed")

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
    logger.info(
    "resolution: relevant_document_ids=%s refs=%s",
    state.get("relevant_document_ids"),
        [
            (r.get("reference"), r.get("status"), (r.get("match") or {}).get("document_id"))
            for r in resolution.get("references", [])
        ],
    )
    plan = _apply_scope_route(state, plan)
    _log_planner_decision(plan)

    clarification = plan.get("status") == "NEEDS_CLARIFICATION"
    message = "Clarification required." if clarification else "Request understood."

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
        **carry,
        "resource_context": resources,
        "planning_query": planning_query,
        "query_understanding": understanding,
        "evidence_requirements": evidence_requirements,
        "resolved_references": resolution,
        "execution_plan": plan,
        "pending_clarification": clarification,
        "clarification_options": plan.get("clarification_options", []),
        "tool_history": list(state.get("tool_history", []))
        + [
            {
                "stage": "query_understanding",
                "original_query": original_query,
                "understanding": understanding,
                "resolution": resolution,
                "plan": plan,
            }
        ],
    }


def _with_table_scope(
    state: AgentState,
    capabilities: list[str],
) -> AgentState:
    """
    Evidence selection for structured sources: a resolved document does not
    make every one of its tables relevant. Decide, from profiles and schema
    metadata, which tables this question needs BEFORE anything is read.
    The result travels in state["table_scope"]. An absent scope means no
    narrowing applies; an explicit empty scope means the selector ran and
    found no safe structured table for this request.
    """
    # Scope Guard's relevant IDs are candidates, not the final plan.  Keep
    # them available while the selector evaluates schemas, together with any
    # documents discovered through resolved entities/references.
    resolved_ids = set(_candidate_document_ids(state, _resources(state)))
    if "TABLE_QUERY" not in capabilities:
        return state

    try:
        resources = _resources(state)
        entity_tables = {
            str(location.get("table_name"))
            for entity in _resolved_entity_values(state)
            for location in entity.get("locations", [])
            if location.get("table_name")
        }
        allowed_ids = _allowed_document_ids(state)
        if allowed_ids is not None:
            allowed_set = set(allowed_ids)
            entity_tables = {
                str(table.get("table_name"))
                for table in resources.table_index(document_ids=allowed_ids)
                if str(table.get("table_name")) in entity_tables
                and _coerce_int(table.get("document_id")) in allowed_set
            }
        rule_dependent_mixed = _is_rule_dependent_mixed_request(state)

        # A mixed data-plus-rules request often names the data file but refers
        # to its governing formula/policy descriptively. Resolution may find
        # only the data source, so selecting tables only from resolved IDs
        # silently excludes structured policy tables from candidate discovery.
        # In this case, ask the normal table selector to rank every table in
        # the caller's authorized document scope; only selected tables flow to
        # retrieval. This widens discovery, never authorization.
        if rule_dependent_mixed:
            # Discover companion rule tables within the caller's authorized
            # document boundary, even when only the data source resolved.
            schema = resources.schema(
                document_ids=allowed_ids,
                include_sample_rows=True,
            )
            candidates = list(schema)
        else:
            # Ordinary requests remain scoped to their resolved sources and
            # tables found through entity resolution.
            needed_documents = set(resolved_ids)
            if entity_tables:
                needed_documents |= {
                    doc_id
                    for doc_id in (
                        _coerce_int(table.get("document_id"))
                        for table in resources.table_index(document_ids=allowed_ids)
                        if str(table.get("table_name")) in entity_tables
                    )
                    if doc_id is not None
                }
            if not resolved_ids and not entity_tables:
                schema_ids = set(allowed_ids) if allowed_ids is not None else None
                schema = resources.schema(
                    document_ids=sorted(schema_ids) if schema_ids is not None else None,
                    include_sample_rows=True,
                )
                candidates = list(schema)
            else:
                schema = resources.schema(
                    document_ids=sorted(needed_documents),
                    include_sample_rows=True,
                )
                candidates = [
                    item
                    for item in schema
                    if _coerce_int(item.get("document_id")) in resolved_ids
                    or str(item.get("table_name")) in entity_tables
                ]
        selection = select_tables_detailed(
            workspace_id=state["workspace_id"],
            query=effective_planning_query(state),
            understanding=state.get("query_understanding", {}) or {},
            candidates=candidates,
            # A table explicitly named by the user or located by entity
            # resolution is authoritative in every route, not only mixed
            # requests. These names are still intersected with candidates by
            # the selector, so they cannot widen authorization.
            forced_table_names=entity_tables | _resolved_table_names(state),
        )
    except Exception:
        logger.exception(
            "table scope selection failed; falling back to document evidence"
        )
        scoped = dict(state)
        # None means unrestricted schema. On failure, an explicit empty scope
        # prevents the table executor from interpreting the failure as a full
        # scan. Exact table/entity references remain in the state for text
        # fallback and diagnostics.
        scoped["table_scope"] = []
        scoped["table_roles"] = {}
        scoped["table_computed"] = []
        scoped["table_unmet"] = []
        scoped["table_selection_status"] = "unavailable"
        return scoped

    if selection is None:
        scoped = dict(state)
        scoped["table_scope"] = []
        scoped["table_roles"] = {}
        scoped["table_computed"] = []
        scoped["table_unmet"] = []
        scoped["table_selection_status"] = "no_match"
        return scoped

    selected = selection.tables
    if not selected:
        # A valid no-match is not the same as "no narrowing". Exact table and
        # entity references were passed as forced candidates above. If none
        # survived candidate validation, fail closed rather than treating an
        # absent scope as permission to query every table.
        scoped = dict(state)
        scoped["table_scope"] = []
        scoped["table_roles"] = {}
        scoped["table_computed"] = list(selection.computed)
        scoped["table_unmet"] = list(selection.unmet)
        scoped["table_selection_status"] = selection.status
        return scoped

    scoped = dict(state)
    scoped["table_scope"] = sorted(selected)
    scoped["table_roles"] = dict(selection.roles)
    scoped["table_computed"] = list(selection.computed)
    scoped["table_unmet"] = list(selection.unmet)
    scoped["table_selection_status"] = selection.status
    return scoped


def _annotate_table_identity(
    items: list[Any],
    resources: WorkspaceResources | None,
) -> list[Any]:
    """Tell the answer stage which table each result came from.

    Result items name the physical table only. The description stored with
    the table at ingestion (sheet, heading, the caption that introduces it)
    is what distinguishes, say, an original schedule from an amended one when
    both have the same columns. Items without a stored description are left
    unchanged.
    """
    if not items or resources is None:
        return items
    try:
        described = {
            str(entry.get("table_name")): entry
            for entry in resources.table_index()
            if entry.get("table_title") or entry.get("table_context")
        }
    except Exception:
        logger.exception("table identity: table index unavailable")
        return items
    if not described:
        return items

    annotated: list[Any] = []
    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            annotated.append(item)
            continue
        entries = [
            described[name]
            for name in sorted(_evidence_table_names(item))
            if name in described
        ]
        titles = list(dict.fromkeys(e["table_title"] for e in entries if e.get("table_title")))
        contexts = list(dict.fromkeys(e["table_context"] for e in entries if e.get("table_context")))
        if not titles and not contexts:
            annotated.append(item)
            continue
        content = dict(content)
        if titles:
            content.setdefault("table_title", "; ".join(titles))
        if contexts:
            content.setdefault("table_context", "; ".join(contexts)[:600])
        annotated.append({**item, "content": content})
    return annotated


def _evidence_check_notes(
    state: AgentState,
    resources: WorkspaceResources | None,
    evidence: list[Any],
) -> list[dict[str, Any]]:
    """Factual notes about what collection did and did not produce.

    These never gate the answer. They name needed sources that contributed
    nothing, requested items no evidence mentions, and (for a question that
    refers back to earlier answers) what was said earlier.
    """
    notes: list[dict[str, Any]] = []
    try:
        titles: dict[int, str] = {}
        if resources is not None:
            for entry in resources.catalog():
                doc_id = _coerce_int(entry.get("document_id"))
                if doc_id is not None:
                    titles[doc_id] = str(entry.get("title") or f"source {doc_id}")
        for note in (
            document_coverage_note(_resolved_document_ids(state), titles, evidence),
            requirement_coverage_note(state.get("evidence_requirements") or [], evidence),
            prior_conclusions_note(
                str(state.get("query") or ""),
                state.get("conversation_history") or [],
            ),
        ):
            if note is not None:
                notes.append(note)
    except Exception:
        logger.exception("evidence check notes failed; continuing without them")
    return notes


def _derivation_note(state: AgentState) -> dict[str, Any] | None:
    """Tell the answer stage which values must be derived, not read.

    Table selection already knows which requested values are not stored as
    columns ("computed") and which needed data it found in no table ("unmet").
    Without this note that knowledge stops at the selector, and the answer
    model can mistake the partial rows it was given for the complete input.
    """
    computed = [str(v).strip() for v in state.get("table_computed") or [] if str(v).strip()]
    unmet = [str(v).strip() for v in state.get("table_unmet") or [] if str(v).strip()]
    if not computed and not unmet:
        return None
    parts: list[str] = []
    if computed:
        parts.append(
            "Derived values the answer must compute (they are not stored as "
            "columns): "
            + "; ".join(computed[:6])
            + ". Compute each from its governing definition and the inputs in "
            "the evidence. Recorded rows may cover only part of the period or "
            "population the definition requires; check that before using them "
            "as the complete input."
        )
    if unmet:
        parts.append(
            "Needed data no table was found to hold: "
            + "; ".join(unmet[:6])
            + ". Look for it in the text evidence; if it is absent, say it "
            "cannot be established."
        )
    return {
        "source_type": "evidence_note",
        "note_kind": "derivation",
        "result_status": "SUCCESS",
        "content": " ".join(parts),
    }


def _table_scope(state: AgentState) -> set[str] | None:
    # Missing/None means selection was not applied. An explicit [] is a
    # meaningful deny-all result and must never widen back to the full schema.
    if "table_scope" not in state or state.get("table_scope") is None:
        return None
    return {str(name) for name in state.get("table_scope", [])}


def _join_table_scope(state: AgentState) -> set[str] | None:
    """Return the exact selected table scope for key-based companion lookups.

    A key join is a row filter over tables already selected for this request;
    it is not permission to broaden the physical table set. Widening to every
    sibling table after selection can reintroduce the same all-table scan that
    the selector is designed to prevent. Relevant companion tables must be
    selected by the normal query-specific selector before any table is read.
    """
    scope = _table_scope(state)
    # The key join is not an independent discovery route. If no table scope
    # was selected, it may still join text evidence but must not query the
    # entire structured corpus by default.
    return scope if scope is not None else set()


def _in_table_scope(
    state: AgentState,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scope = _table_scope(state)
    if scope is None:
        return items
    return [item for item in items if str(item.get("table_name")) in scope]


def _documents_without_text(coverage: Any) -> set[int]:
    """
    Resolved documents that own no text chunks (purely structured sources),
    from the coverage report. Text lookups against them can never match.
    """
    if not isinstance(coverage, list):
        return set()
    ids: set[int] = set()
    for entry in coverage:
        if isinstance(entry, dict) and entry.get("total") == 0:
            document_id = _coerce_int(entry.get("document_id"))
            if document_id is not None:
                ids.add(document_id)
    return ids


_TABULAR_SOURCE_TYPES = {"xlsx", "xls", "xlsm", "csv", "tsv", "ods", "excel", "spreadsheet"}
_TABULAR_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".ods")


def _is_tabular_document(document: dict[str, Any]) -> bool:
    """Spreadsheet-style source: its content lives only in tables, never in text."""
    source_type = str(document.get("source_type") or "").strip().lower().lstrip(".")
    if source_type in _TABULAR_SOURCE_TYPES or any(
        token in source_type for token in ("spreadsheet", "excel", "csv")
    ):
        return True
    title = str(document.get("title") or document.get("document_title") or "")
    return title.strip().lower().endswith(_TABULAR_SUFFIXES)


def _scope_is_table_only(state: AgentState, resources: WorkspaceResources) -> bool:
    """True when every document this request is about is a spreadsheet/CSV.

    Such a scope has no prose to search, so a text search can only come back
    empty and must not be reported as missing evidence, and no policy document
    can exist for the request to wait on. Unknown or unresolved scope returns
    False so the previous behavior is kept.
    """
    document_ids = {
        document_id
        for document_id in (
            _coerce_int(value) for value in _document_ids_for_search(state)
        )
        if document_id is not None
    }
    if not document_ids:
        return False
    by_id = {
        _coerce_int(entry.get("document_id")): entry
        for entry in resources.catalog()
        if isinstance(entry, dict)
    }
    for document_id in document_ids:
        entry = by_id.get(document_id)
        if entry is None or not _is_tabular_document(entry):
            return False
    return True


def _add_key_joined_view(items: list[Any]) -> list[Any]:
    """Relate complete tables of different sources on their shared key columns.

    Deterministic (no SQL, no LLM). Source tables whose every row is contained
    in the joined view are replaced by it so the answer stage gets one compact,
    already-aligned table instead of several raw ones; anything else is kept.
    """
    try:
        built = build_joined_view(items)
    except Exception:
        logger.exception("key-joined view failed; keeping the raw tables")
        return items
    if not built:
        return items
    view, absorbed = built
    kept = [item for position, item in enumerate(items) if position not in absorbed]
    content = view.get("content") or {}
    logger.info(
        "key_joined_view: tables=%d rows=%d columns=%d keys=%s replaced_raw=%d",
        len(view.get("sources") or []),
        int(content.get("row_count") or 0),
        len(content.get("columns") or []),
        content.get("join_keys"),
        len(absorbed),
    )
    return [view] + kept


def _evidence_table_names(item: Any) -> set[str]:
    if not isinstance(item, dict):
        return set()
    return {
        str(source.get("table_name"))
        for source in item.get("sources", []) or []
        if isinstance(source, dict) and source.get("table_name")
    }


def _evidence_rows(item: Any) -> list[dict[str, Any]]:
    content = item.get("content") if isinstance(item, dict) else None
    rows = content.get("rows") if isinstance(content, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _row_signature(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


_POPULATION_CONTEXT_PATTERNS = (
    r"\b(?:all|every|each|among|across|overall|entire|whole|population)\b",
    r"\b(?:percentage|percent|share|proportion|ratio)\s+of\s+(?:the\s+)?total\b",
    r"\b(?:sum|total|average|mean|count)\s+(?:of|across|among)\b",
    r"\b(?:highest|lowest|maximum|minimum|top|bottom|rank|ranking)\b",
    r"\b(?:compare|compared\s+with|difference\s+between|more\s+than|less\s+than|greater\s+than)\b",
)


def _query_requires_population_context(
    query: str,
    operation: str = "",
) -> bool:
    """
    Determine whether the requested answer needs source-wide context.

    This is intentionally domain-neutral. It does not know about employees,
    sales, salaries, or any other business field. It only recognizes query
    semantics that require more than one entity row, such as population
    aggregates, rankings, comparisons, or shares of a total.
    """
    text = str(query or "").strip()
    if not text:
        return False

    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation in {
        "list",
        "ranking",
        "rank",
        "compare",
        "comparison",
        "aggregate",
        "population",
    }:
        return True

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _POPULATION_CONTEXT_PATTERNS
    )


def _is_complete_table_evidence(item: Any) -> bool:
    content = item.get("content") if isinstance(item, dict) else None
    if not isinstance(content, dict) or content.get("truncated"):
        return False

    rows = content.get("rows")
    if not isinstance(rows, list):
        return False

    counts = content.get("source_table_row_counts")
    if isinstance(counts, dict) and len(counts) == 1:
        try:
            return len(rows) == int(next(iter(counts.values())))
        except (TypeError, ValueError):
            pass

    return bool(content.get("rows_returned", len(rows)) == len(rows))


def _table_names(item: Any) -> set[str]:
    return _evidence_table_names(item)


def _prioritize_focused_tables(
    structured: list[Any],
    *,
    query: str = "",
    operation: str = "",
) -> list[Any]:
    """Choose one canonical representation of table facts without losing semantics."""
    needs_population = _query_requires_population_context(query, operation)
    by_table: dict[str, dict[str, list[Any]]] = {}

    for item in structured:
        names = _table_names(item)
        content = item.get("content") if isinstance(item, dict) else None
        if len(names) != 1 or not isinstance(content, dict):
            continue
        role = str(content.get("evidence_role") or "")
        if role not in {"document_table", "query_result", "key_joined", "entity_row"}:
            continue
        name = next(iter(names))
        by_table.setdefault(name, {}).setdefault(role, []).append(item)

    drop_ids: set[int] = set()
    for table_name, groups in by_table.items():
        full_items = groups.get("document_table", [])
        query_items = groups.get("query_result", [])
        focused_items = groups.get("key_joined", []) + groups.get("entity_row", [])
        complete_full = any(_is_complete_table_evidence(item) for item in full_items)
        if not complete_full:
            continue

        full_rows: set[str] = set()
        target_full = next(
            (item for item in full_items if _is_complete_table_evidence(item)),
            full_items[0],
        )
        for row in _evidence_rows(target_full):
            full_rows.add(_row_signature(row))

        if needs_population:
            for item in query_items:
                semantics = (item.get("content") or {}).get("query_semantics") or {}
                kind = semantics.get("type")
                single_ranking = kind == "ranking" and semantics.get("limit") == 1
                if kind in {"aggregate", "grouped", "positional"} or single_ranking:
                    for full in full_items:
                        drop_ids.add(id(full))
                    continue
                if kind == "ranking":
                    continue
                rows = {_row_signature(r) for r in _evidence_rows(item)}
                if rows and rows <= full_rows:
                    drop_ids.add(id(item))
        else:
            for item in query_items + focused_items:
                rows = {_row_signature(r) for r in _evidence_rows(item)}
                if not rows or rows <= full_rows:
                    drop_ids.add(id(item))

        for item in focused_items:
            content = item.get("content") or {}
            if not isinstance(content, dict):
                continue
            keys = [
                str(key) for key in content.get("matched_keys", []) if str(key).strip()
            ]
            if not keys:
                continue
            target_content = target_full.setdefault("content", {})
            existing = [
                str(key)
                for key in target_content.get("matched_keys", [])
                if str(key).strip()
            ]
            target_content["matched_keys"] = list(dict.fromkeys(existing + keys))
            drop_ids.add(id(item))

    return [item for item in structured if id(item) not in drop_ids]


def _canonicalize_evidence(items: list[Any]) -> list[Any]:
    """Deterministically remove exact duplicate evidence while preserving provenance."""
    result: list[Any] = []
    seen_exact: set[tuple[Any, ...]] = set()
    table_seen_rows: dict[tuple[str, str], set[str]] = {}

    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue

        content = item.get("content")
        role = (
            str(content.get("evidence_role") or "") if isinstance(content, dict) else ""
        )
        names = tuple(sorted(_evidence_table_names(item)))

        if (
            names
            and isinstance(content, dict)
            and role in {"document_table", "key_joined", "entity_row", "query_result"}
        ):
            key = (names, role)
            rows = content.get("rows")
            if isinstance(rows, list):
                unique_rows = []
                seen_rows = table_seen_rows.setdefault(key, set())
                for row in rows:
                    signature = (
                        _row_signature(row) if isinstance(row, dict) else str(row)
                    )
                    if signature in seen_rows:
                        continue
                    seen_rows.add(signature)
                    unique_rows.append(row)
                content = dict(content)
                content["rows"] = unique_rows
                item = dict(item)
                item["content"] = content

            exact_key = (
                "table",
                names,
                role,
                _row_signature(
                    {
                        str(k): v
                        for k, v in content.items()
                        if k not in {"rows", "success", "interpretation"}
                    }
                ),
            )
            if exact_key in seen_exact and not content.get("rows"):
                continue
            seen_exact.add(exact_key)
            result.append(item)
            continue

        metadata = (
            item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        )
        exact_key = (
            str(item.get("source_type") or ""),
            _coerce_int(item.get("document_id")),
            metadata.get("chunk_index"),
            str(content or ""),
        )
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        result.append(item)

    return result


_NOTE_KINDS = {
    "evidence_note",
    "retrieval_assessment",
    "source_resolution",
    "document_coverage",
    "coverage_note",
}
_NON_DATA_STATUSES = {
    "",
    "NOT_FOUND",
    "RETRIEVAL_ERROR",
    "PARTIAL",
    "INSUFFICIENT_EVIDENCE",
}


def _restore_unaggregated_reductions(
    before: list[Any],
    after: list[Any],
    query: str,
) -> list[Any]:
    """
    Invariant: a complete table is only reduced to a subset when an aggregate
    covers its population. finalize_structured reduces every non-aggregated
    document table once ANY aggregate exists; if the question names a numeric
    column of a reduced table and no aggregate was derived for it, the
    population (total / highest / share) would be unrecoverable downstream and
    the answer model would rightly report it as missing. Restore such tables
    in full (they are small by construction: complete tables under the read cap).
    """
    from planner.services.evidence import to_number
    from retrieval.services.keys import is_id_column

    originals: dict[str, Any] = {}
    for item in before:
        names = _evidence_table_names(item)
        content = item.get("content") if isinstance(item, dict) else None
        if (
            len(names) == 1
            and isinstance(content, dict)
            and content.get("evidence_role") == "document_table"
            and _is_complete_table_evidence(item)
        ):
            originals[next(iter(names))] = item

    aggregated: set[str] = set()
    for item in after:
        content = item.get("content") if isinstance(item, dict) else None
        if (
            isinstance(content, dict)
            and content.get("evidence_role") == "aggregate_result"
        ):
            for source in (item.get("sources") or [])[:1]:
                if isinstance(source, dict) and source.get("table_name"):
                    aggregated.add(str(source["table_name"]))

    def tokens(text: Any) -> set[str]:
        found = re.findall(
            r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower()
        )
        return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in found}

    query_tokens = tokens(query)

    def merge_original(original: Any, content: dict[str, Any]) -> dict[str, Any]:
        merged = dict(original)
        merged_content = dict(original["content"])
        keys = list(
            dict.fromkeys(
                [str(k) for k in original["content"].get("matched_keys", []) or []]
                + [str(k) for k in content.get("matched_keys", []) or []]
            )
        )
        if keys:
            merged_content["matched_keys"] = keys
        merged_content["population_scope"] = "complete_population"
        merged["content"] = merged_content
        return merged

    def reducible(item: Any) -> tuple[str, Any] | None:
        content = item.get("content") if isinstance(item, dict) else None
        names = _evidence_table_names(item)
        if (
            isinstance(content, dict)
            and content.get("population_reduced")
            and len(names) == 1
            and next(iter(names)) in originals
            and next(iter(names)) not in aggregated
        ):
            table = next(iter(names))
            return table, originals[table]
        return None

    restored: list[Any] = []
    for item in after:
        found = reducible(item)
        if found is not None:
            table, original = found
            rows = _evidence_rows(original)
            columns = [
                str(c)
                for c in (
                    original["content"].get("columns")
                    or (list(rows[0]) if rows else [])
                )
            ]
            population_column = any(
                not is_id_column(column)
                and tokens(column) & query_tokens
                and sum(1 for r in rows if r.get(column) not in (None, "")) >= 2
                and all(
                    to_number(r.get(column)) is not None
                    for r in rows
                    if r.get(column) not in (None, "")
                )
                for column in columns
            )
            if population_column:
                logger.info("population guard: restored complete table %s", table)
                restored.append(merge_original(original, item["content"]))
                continue
        restored.append(item)

    def norm(value: Any) -> str:
        return str(value).strip().lower()

    def complete_rows(item: Any) -> list[dict[str, Any]]:
        content = item.get("content") if isinstance(item, dict) else None
        if (
            isinstance(content, dict)
            and content.get("evidence_role") == "document_table"
            and content.get("population_scope") == "complete_population"
            and not content.get("population_reduced")
            and _is_complete_table_evidence(item)
        ):
            return _evidence_rows(item)
        return []

    complete_tables = [
        rows for rows in (complete_rows(i) for i in restored) if len(rows) >= 2
    ]
    final: list[Any] = []
    for item in restored:
        found = reducible(item)
        if found is None:
            final.append(item)
            continue
        table, original = found
        original_rows = _evidence_rows(original)
        kept_rows = _evidence_rows(item)
        needs_lookup = False
        if len(original_rows) > len(kept_rows) >= 0 and original_rows:
            for column in original["content"].get("columns") or list(original_rows[0]):
                column = str(column)
                values = [
                    norm(r.get(column))
                    for r in original_rows
                    if r.get(column) not in (None, "")
                ]
                if len(values) < 2 or len(set(values)) != len(values):
                    continue
                if all(
                    to_number(r.get(column)) is not None
                    for r in original_rows
                    if r.get(column) not in (None, "")
                ):
                    continue
                key_values = set(values)
                kept_values = {
                    norm(r.get(column))
                    for r in kept_rows
                    if r.get(column) not in (None, "")
                }
                for rows in complete_tables:
                    for other in rows[0]:
                        referenced = {
                            norm(r.get(other))
                            for r in rows
                            if r.get(other) not in (None, "")
                        }
                        if (
                            len(referenced) >= 2
                            and referenced <= key_values
                            and not referenced <= kept_values
                        ):
                            needs_lookup = True
                            break
                    if needs_lookup:
                        break
                if needs_lookup:
                    break
        if needs_lookup:
            logger.info("population guard: restored referenced lookup table %s", table)
            final.append(merge_original(original, item["content"]))
        else:
            final.append(item)
    return final


def _reconcile_source_notes(
    notes: list[Any],
    structured: list[Any],
    workspace_id: Any,
    resources: WorkspaceResources | None = None,
) -> list[Any]:
    """
    Retrieval/coverage/identifier-lookup notes describe the TEXT search
    ("no passages", "mode=none included=0", "no row contains the identifier").
    For a source whose rows were retrieved they contradict the evidence and
    read as 'source unavailable'. Drop such notes for every source whose
    structured rows are present, matching by title, catalog names and file
    names (not only the table title).
    """
    if not notes:
        return notes
    present_ids: set[int] = set()
    names: set[str] = set()
    for item in structured:
        if (
            not isinstance(item, dict)
            or str(item.get("result_status", "")).upper() != "SUCCESS"
        ):
            continue
        content = item.get("content")
        if not isinstance(content, dict) or not (
            content.get("rows") or content.get("aggregates")
        ):
            continue
        for source in item.get("sources") or []:
            did = (
                _coerce_int(source.get("document_id"))
                if isinstance(source, dict)
                else None
            )
            if did is not None:
                present_ids.add(did)
        if content.get("document_title"):
            names.add(str(content["document_title"]))
    if present_ids:
        try:
            # Only titles/ids are needed here, never row counts.
            catalog = (
                resources.catalog()
                if resources is not None and resources.workspace_id == int(workspace_id)
                else get_workspace_catalog(int(workspace_id), include_row_counts=False)
            )
            for doc in catalog:
                if _coerce_int(doc.get("document_id")) in present_ids:
                    for key in (
                        "title",
                        "document_name",
                        "file_name",
                        "filename",
                        "name",
                    ):
                        if doc.get(key):
                            names.add(str(doc[key]))
        except Exception:
            logger.exception("note reconciliation: catalog unavailable")
    expanded: set[str] = set()
    for name in names:
        lowered = name.strip().lower()
        expanded.add(lowered)
        expanded.add(re.sub(r"\.[a-z0-9]{2,5}$", "", lowered))
    expanded = {n for n in expanded if len(n) >= 3}
    if not expanded:
        return notes

    kept: list[Any] = []
    for note in notes:
        if (
            isinstance(note, dict)
            and str(note.get("note_kind") or "").lower() == "identifier_lookup"
        ):
            # A source can have valid rows yet still lack the requested key.
            # Its unrelated rows do not contradict an exact identifier miss.
            kept.append(note)
            continue
        if (
            isinstance(note, dict)
            and str(note.get("source_type") or "").lower() in _NOTE_KINDS
            and str(note.get("result_status") or "").upper() in _NON_DATA_STATUSES
        ):
            blob = json.dumps(
                note.get("content"), ensure_ascii=False, default=str
            ).lower()
            if any(name in blob for name in expanded):
                logger.info("dropped note contradicted by retrieved rows: %.80s", blob)
                continue
        kept.append(note)
    return kept


def _evidence_manifest(evidence: list[Any]) -> list[dict[str, Any]]:
    """Compact, log-only view of exactly what the answer stage will receive."""
    manifest: list[dict[str, Any]] = []
    for position, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            manifest.append({"n": position, "type": type(item).__name__})
            continue
        content = item.get("content")
        entry: dict[str, Any] = {
            "n": position,
            "type": item.get("source_type"),
            "status": item.get("result_status"),
        }
        if isinstance(content, dict):
            entry["role"] = content.get("evidence_role")
            entry["scope"] = content.get("population_scope")
            entry["doc"] = content.get("document_title")
            rows = content.get("rows")
            if isinstance(rows, list):
                entry["rows"] = len(rows)
            if content.get("aggregates"):
                entry["aggregates"] = [
                    a.get("column")
                    for a in content["aggregates"]
                    if isinstance(a, dict)
                ]
        else:
            entry["text"] = str(content)[:60]
        manifest.append(entry)
    return manifest


def _collapse_subsumed_table_items(items: list[Any]) -> list[Any]:
    """
    Final semantic pass over structured evidence: a table item whose rows are
    all present, exactly, in another item of the same source(s) is a redundant
    raw representation of the same facts. Fold it into the surviving item
    (matched keys merged, so join provenance is kept) and drop it.

    Items with their own query semantics (aggregate / ranking / positional /
    grouped results) and aggregate_result items are never folded: they are
    execution results, not duplicates.
    """
    candidates: list[tuple[int, frozenset[str], set[str], dict[str, Any]]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        names = frozenset(_evidence_table_names(item))
        rows = content.get("rows") if isinstance(content, dict) else None
        if not names or not isinstance(rows, list) or not rows:
            continue
        if not all(isinstance(r, dict) for r in rows):
            continue
        candidates.append((position, names, {_row_signature(r) for r in rows}, item))

    drop: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for position, names, signatures, item in candidates:
        content = item["content"]
        if content.get("evidence_role") == "aggregate_result":
            continue
        semantics = content.get("query_semantics") or {}
        if isinstance(semantics, dict) and semantics.get("type") not in (
            None,
            "table_result",
        ):
            continue

        for other_pos, other_names, other_sigs, other in candidates:
            if other_pos == position or other_pos in drop or other_names != names:
                continue
            if (other.get("content") or {}).get("evidence_role") == "aggregate_result":
                continue
            larger = len(other_sigs) > len(signatures)
            equal_but_earlier = other_sigs == signatures and other_pos < position
            if not (signatures <= other_sigs and (larger or equal_but_earlier)):
                continue

            keys = [
                str(k) for k in content.get("matched_keys", []) or [] if str(k).strip()
            ]
            if keys:
                survivor = replacements.get(other_pos) or dict(other)
                survivor_content = dict(survivor.get("content") or {})
                existing = [
                    str(k)
                    for k in survivor_content.get("matched_keys", []) or []
                    if str(k).strip()
                ]
                survivor_content["matched_keys"] = list(dict.fromkeys(existing + keys))
                survivor["content"] = survivor_content
                replacements[other_pos] = survivor
            drop.add(position)
            break

    return [
        replacements.get(position, item)
        for position, item in enumerate(items)
        if position not in drop
    ]


def _ensure_citation_sync(evidence: list[Any]) -> list[Any]:
    """
    citation_index must equal the 1-based position in the stored final
    evidence. Re-assign (once, on the exact list that will be stored and
    sent) if anything upstream let them diverge.
    """
    in_sync = all(
        isinstance(item, dict) and item.get("citation_index") == position
        for position, item in enumerate(evidence, start=1)
    )
    if in_sync:
        return evidence
    logger.warning("citation indexes out of sync with evidence order; re-assigning")
    return assign_citation_indexes(list(evidence))


def _join_adds_nothing(
    state: AgentState,
    resources: WorkspaceResources,
    structured: list[Any],
    documents: list[Any],
    coverage_info: Any,
    resolved_doc_ids: list[int],
) -> str | None:
    """Reason the cross-document key join provably has nothing left to add.

    The join relates sources by a shared key. It can only contribute (a) rows
    of tables not yet read and (b) text passages of documents that have text.
    It is skipped ONLY when neither is possible:

    * no resolved entity is available as a join anchor, and
    * every resolved document is purely structured (coverage found no text
      chunks for it) and no text evidence exists, and
    * every in-scope table of every resolved document is already present as
      complete, successful, single-table evidence.

    Anything uncertain (no coverage report, a truncated or failed table, an
    entity anchor, a document with text) returns None and the join runs.
    """
    if _resolved_entity_values(state) or documents:
        return None

    if not set(resolved_doc_ids) <= _documents_without_text(coverage_info):
        return None

    scope = _table_scope(state)
    tables = [
        str(table["table_name"])
        for table in resources.table_index(resolved_doc_ids)
        if scope is None or str(table["table_name"]) in scope
    ]
    if not tables:
        return None

    complete: set[str] = set()
    for item in structured:
        if (
            not isinstance(item, dict)
            or str(item.get("source_type", "")).lower() != "table_query"
            or str(item.get("result_status", "")).upper() != "SUCCESS"
            or not _is_complete_table_evidence(item)
        ):
            continue
        names = _evidence_table_names(item)
        if len(names) == 1:
            complete |= names

    if all(name in complete for name in tables):
        return (
            "no join anchors; all in-scope tables of the resolved documents "
            "are already complete evidence and none has text to relate"
        )
    return None


def _narrow_structured_to_request(
    state: AgentState, structured: list[Any]
) -> list[Any]:  # QP-PATCH-V4
    """Keep only the spreadsheet rows the question names, before documents are budgeted.

    Whole business tables otherwise consume the evidence budget and leave the
    governing rule documents a handful of passages. Population questions
    (totals, rankings, comparisons, "all"...) are never narrowed.
    """
    query = effective_planning_query(state)
    operation = str((state.get("query_understanding") or {}).get("operation") or "")
    if _query_requires_population_context(query, operation):
        return structured
    try:
        narrowed, count = _narrow_spreadsheet_tables_v3(structured, query)
    except Exception:
        logger.exception("structured row narrowing failed; keeping all rows")
        return structured
    if count:
        logger.info(
            "structured narrowing: %d spreadsheet table(s) reduced to the rows "
            "the question names",
            count,
        )
    return narrowed


def dispatch_node(state: AgentState) -> dict[str, Any]:
    if not state.get("deadline_monotonic"):
        state = {**state, "deadline_monotonic": time.monotonic() + MAX_QUERY_EXECUTION_SECONDS}
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

    capabilities = list(plan.get("capabilities", []))

    # A user can explicitly ask for corpus-wide evidence even when the LLM
    # planner resolved only one named/relevant source. Preserve that intent;
    # the actual expansion below remains bounded by allowed_document_ids.
    if (
        requests_workspace_wide_cross_document_search(effective_planning_query(state))
        and not plan.get("cross_document")
    ):
        plan = {**plan, "cross_document": True}
        state = {**state, "execution_plan": plan}

    if (
        "SOURCE_METADATA" in capabilities
        and not _resolved_entity_values(state)
        and not _resolved_document_ids(state)
        and not _resolved_table_names(state)
        and not _named_missing_source_references(state)
    ):
        capabilities = [
            c for c in capabilities if c not in {"TABLE_QUERY", "DOCUMENT_SEARCH"}
        ]

    budget = _RecoveryBudget(MAX_RECOVERY_ATTEMPTS)
    # Reuse the metadata planning already loaded (it travels on the state);
    # if it did not arrive, build one holder here and share it from now on.
    resources = _resources(state)
    state = {**state, "resource_context": resources}
    state = _with_table_scope(state, capabilities)

    # Spreadsheet/CSV-only scope: there is no prose to search and no policy
    # document to wait for. Searching text there only produces a fake
    # "insufficient evidence" note, so answer from the tables alone.
    table_only_scope = False
    if "TABLE_QUERY" in capabilities and "DOCUMENT_SEARCH" in capabilities:
        try:
            table_only_scope = _scope_is_table_only(state, resources)
        except Exception:
            logger.exception("table-only scope check failed; keeping the plan")
    if table_only_scope:
        capabilities = [c for c in capabilities if c != "DOCUMENT_SEARCH"]
        plan = dict(state.get("execution_plan") or {})
        plan["capabilities"] = list(capabilities)
        plan["actions"] = [
            action for action in plan.get("actions", []) if action != "document_search"
        ]
        plan["fallback_capabilities"] = [
            c for c in plan.get("fallback_capabilities", []) if c != "DOCUMENT_SEARCH"
        ]
        state = {**state, "execution_plan": plan}
        logger.info(
            "table-only scope: every in-scope document is a spreadsheet/CSV; "
            "skipping text search and rule-document dependency"
        )

    # An explicit empty selection means the candidate tables were examined
    # and none could support this question. Do not let _run_table interpret an
    # absent table restriction as "all tables" and manufacture an aggregate
    # over a neighbouring layout table. Use the authorized document corpus
    # instead; _document_ids_for_search still includes Scope Guard candidates.
    selection_status = state.get("table_selection_status")
    if (
        selection_status in {"no_match", "invalid", "unavailable"}
        and "TABLE_QUERY" in capabilities
    ):
        capabilities = [item for item in capabilities if item != "TABLE_QUERY"]
        if "DOCUMENT_SEARCH" not in capabilities:
            capabilities.append("DOCUMENT_SEARCH")
        plan = dict(state.get("execution_plan") or {})
        plan["capabilities"] = list(capabilities)
        plan["actions"] = [
            action
            for action in plan.get("actions", [])
            if action != "table_query"
        ]
        if "document_search" not in plan["actions"]:
            plan["actions"].append("document_search")
        plan["fallback_capabilities"] = []
        state = {**state, "execution_plan": plan}
        logger.info(
            "table selection status=%s; skipping SQL and falling back to "
            "document search within authorized candidate documents",
            selection_status,
        )

    structured: list[Any] = []
    documents: list[Any] = []
    total_tool_calls = 0
    table_status: str | None = None
    table_result: dict[str, Any] | None = None
    retrieval_confidence = 0.0
    document_search_error = False
    dependency_evidence_found = False
    coverage_notes: list[Any] = []
    coverage_used = False
    coverage_info: Any = None
    document_fallback_used = False
    cross_document_plan = bool(plan.get("cross_document"))

    try:
        _check_deadline(state, "dispatch")

        if "PRODUCT_KNOWLEDGE" in capabilities:
            product = get_product_knowledge()
            product["result_status"] = "SUCCESS"
            structured.append(product)
            total_tool_calls = _count_tool_call(total_tool_calls)

        if "SOURCE_METADATA" in capabilities:
            structured.append(_run_source_metadata(state))
            total_tool_calls = _count_tool_call(total_tool_calls)

        if "TABLE_QUERY" in capabilities:
            table_result = _run_table(state, budget)
            structured.extend(table_result["evidence"])
            table_status = table_result["status"]
            total_tool_calls = _count_tool_call(total_tool_calls)

            # The plan started from a structured guess about a named document
            # (no entity, no named table). If that path found nothing, the
            # document's own content is the evidence: run DOCUMENT_SEARCH.
            if (
                table_status in {"NOT_FOUND", "RETRIEVAL_ERROR"}
                and "DOCUMENT_SEARCH" not in capabilities
                and "DOCUMENT_SEARCH" in plan.get("fallback_capabilities", [])
            ):
                capabilities.append("DOCUMENT_SEARCH")
                document_fallback_used = True
                logger.info(
                    "table_query returned %s; falling back to DOCUMENT_SEARCH "
                    "(plan fallback)",
                    table_status,
                )

        if "CONVERSATION_MEMORY" in capabilities:
            result = conversation_memory(
                query=state.get("query", ""),
                conversation_history=state.get("conversation_history", []),
            )
            memory_messages = result.get("messages", [])
            structured.append(
                {
                    "source_type": "conversation_memory",
                    "result_status": ("SUCCESS" if memory_messages else "NOT_FOUND"),
                    "content": memory_messages,
                    "metadata": {
                        "resolution": result.get("resolution", ""),
                    },
                }
            )
            total_tool_calls = _count_tool_call(total_tool_calls)

        table_not_found = table_status == "NOT_FOUND"

        resolved_doc_ids = _resolved_document_ids(state)
        source_document_ids = _document_ids_for_search(state)
        if cross_document_plan:
            source_document_ids = _cross_document_search_ids(state, source_document_ids)
        missing_named_sources = _named_missing_source_references(state)
        named_source_missing = bool(missing_named_sources) and not resolved_doc_ids

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

        # Capabilities are independent unless the request makes one depend on
        # another. A table lookup that matched nothing (e.g. a missing salary
        # row) must not suppress the document evidence (e.g. the governing
        # policy) the request also asked for. Only a named source that does
        # not exist makes document search pointless.
        skip_document_search = named_source_missing

        if "DOCUMENT_SEARCH" in capabilities and skip_document_search:
            logger.info(
                "Skipping document search: named source(s) not found in workspace"
            )
        elif "DOCUMENT_SEARCH" in capabilities and table_not_found:
            logger.info(
                "Table lookup matched no rows; document search still runs "
                "(independent capability)"
            )

        # Document text retrieval must not implicitly read every extracted
        # SQLite table belonging to a candidate document. Structured evidence
        # is fetched only through TABLE_QUERY, where table selection, scope
        # validation and the SQL allow-list are applied. This prevents
        # unrelated PDF-layout tables from entering the answer context or
        # generating row-fact searches. Spreadsheet/table-only requests are
        # routed through TABLE_QUERY by the planner.

        if "DOCUMENT_SEARCH" in capabilities and not skip_document_search:
            structured = _narrow_structured_to_request(state, structured)  # QP-PATCH-V4
            state = {**state, "tool_calls_used": total_tool_calls}
            docs = _run_documents(state, budget)
            if requires_dependency_discovery(state):
                docs = _supplement_dependency_evidence(state, docs, structured)
                dependency_evidence_found = bool(
                    docs.get("dependency_evidence_found")
                )
            else:  # QP-GEN-3b
                docs = _supplement_row_fact_evidence(state, docs, structured)
            retrieval_calls = max(1, int(docs.get("calls", 1)))
            for _ in range(retrieval_calls):
                total_tool_calls = _count_tool_call(total_tool_calls)
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
                coverage_notes = list(docs.get("notes") or [])
                coverage_used = bool(docs.get("coverage"))
                coverage_info = docs.get("coverage")

            if docs.get("degraded"):
                logger.warning(
                    "document_search ran DEGRADED (dense retrieval failed); "
                    "results are lexical-only."
                )

        join_notes: list[Any] = []
        join_keys: list[str] = []

        # The join is cross-document REASONING: it runs only when the plan
        # decided the request relates sources, not merely because >1 document
        # was retrieved (e.g. an independent multi-document summary).
        join_skip_reason = (
            _join_adds_nothing(
                state,
                resources,
                structured,
                documents,
                coverage_info,
                source_document_ids,
            )
            if cross_document_plan
            and len(source_document_ids) >= 2
            and "DOCUMENT_SEARCH" in capabilities
            and not skip_document_search
            else None
        )
        if join_skip_reason:
            logger.info("cross-document join skipped: %s", join_skip_reason)

        if (
            cross_document_plan
            and len(source_document_ids) >= 2
            and "DOCUMENT_SEARCH" in capabilities
            and not skip_document_search
            and not join_skip_reason
        ):
            try:
                understanding = state.get("query_understanding", {}) or {}
                join = run_cross_document_join(
                    workspace_id=state["workspace_id"],
                    query=effective_planning_query(state),
                    references=understanding.get("references", []),
                    resolved_references=state.get("resolved_references", {}).get(
                        "references", []
                    ),
                    document_ids=sorted(
                        set(source_document_ids)
                        | _document_ids_from_text_evidence(documents)
                    ),
                    entities=_resolved_entity_values(state),
                    structured=structured,
                    documents=documents,
                    skip_text_document_ids=_documents_without_text(coverage_info),
                    allowed_table_names=_join_table_scope(state),  # QP-FIX-3
                    table_roles=state.get("table_roles", {}),
                )
            except Exception:
                logger.exception("cross-document key join failed")
                join = None

            if join:
                join_keys = [str(k) for k in join.get("keys", []) or []]

                if join["tables"]:
                    structured.extend(join["tables"])
                    total_tool_calls = _count_tool_call(total_tool_calls)

                if join["documents"]:
                    documents = join["documents"] + list(documents)

                join_notes = join["notes"]

        operation = str(
            (state.get("query_understanding", {}) or {}).get("operation") or ""
        )

        structured = _prioritize_focused_tables(
            structured,
            query=effective_planning_query(state),
            operation=operation,
        )

        structured = _canonicalize_evidence(structured)
        documents = _canonicalize_evidence(list(documents))

        pre_finalize = list(structured)

        structured = finalize_structured(
            structured,
            workspace_id=state["workspace_id"],
            query=effective_planning_query(state),
            needs_population=_query_requires_population_context(
                effective_planning_query(state),
                operation,
            ),
            entity_values=[
                str(e.get("value"))
                for e in _resolved_entity_values(state)
                if e.get("value")
            ],
            join_keys=join_keys,
            run_query=query_tables,
        )

        structured = _restore_unaggregated_reductions(
            pre_finalize,
            structured,
            effective_planning_query(state),
        )

        structured = _collapse_subsumed_table_items(structured)
        structured = _add_key_joined_view(structured)
        structured = _annotate_table_identity(structured, resources)

        coverage_notes = reconcile_coverage_notes(
            coverage_notes,
            coverage_info,
            structured,
        )

        has_structured_document_evidence = _has_document_table_evidence(structured)
        trace = current_trace()
        if trace is not None:
            trace.increment("text_evidence_items", len(documents))
            trace.increment("structured_evidence_items", len(structured))
            trace.increment("recovery_attempts", budget.used)

        # Evidence collection is complete. Do not run a second semantic or
        # deterministic completeness gate here: the answer model receives the
        # collected evidence and planner checklist, then answers every supported
        # part while explicitly identifying unsupported facts.
        notes: list[Any] = []

        if (
            requires_dependency_discovery(state)
            and not dependency_evidence_found
            and not table_only_scope
        ):
            notes.append(
                {
                    "source_type": "evidence_note",
                    "result_status": "NOT_FOUND",
                    "content": (
                        "This calculation requires applicable policy/rule evidence, "
                        "but the rule-focused source search did not establish it. "
                        "Do not provide a numerical result until the governing "
                        "formula, eligibility conditions, deductions, caps and "
                        "thresholds can be verified from authorized evidence."
                    ),
                }
            )

        if (
            "TABLE_QUERY" in capabilities
            and "DOCUMENT_SEARCH" in capabilities
            and not document_fallback_used
            and not skip_document_search
            and not document_search_error
            and not documents
            and requires_document_evidence(
                state,
                skip_document_search=skip_document_search,
                has_structured_document_evidence=has_structured_document_evidence,
            )
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

        derivation_note = _derivation_note(state)
        if derivation_note is not None:
            notes.append(derivation_note)
        notes.extend(coverage_notes)
        notes.extend(join_notes)
        notes.extend(_evidence_check_notes(state, resources, structured + documents))
        notes = _canonicalize_evidence(notes)
        notes = _reconcile_source_notes(
            notes,
            structured,
            state["workspace_id"],
            resources,
        )

        evidence = _ensure_citation_sync(
            assign_citation_indexes(structured + documents + notes)
        )

        logger.info(
            "evidence_manifest: %s",
            json.dumps(
                _evidence_manifest(evidence),
                ensure_ascii=False,
                default=str,
            ),
        )

        return {
            "evidence": evidence,
            "final_evidence": evidence,
            # This compatibility flag reports whether any actual source data
            # was collected. It is not a completeness judgment or an answer gate.
            "evidence_sufficient": bool(structured or documents),
            "dependency_evidence_found": dependency_evidence_found,
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



def _stream_answer_with_fit(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None,
    planner_requirements: list[dict[str, Any]] | None = None,
):
    """Stream an answer from the complete evidence set without trimming."""
    yield from generate_answer_stream(
        query=query,
        evidence=evidence,
        conversation_history=conversation_history,
        planner_requirements=planner_requirements,
    )


def finish_node(state: AgentState) -> dict[str, Any]:
    trace = current_trace()
    if trace is not None:
        trace.increment("final_evidence_items", len(state.get("final_evidence") or state.get("evidence") or []))
        trace.increment("tool_calls_from_state", int(state.get("total_tool_calls") or 0))
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

    # Empty, partial, conflicting, and NOT_FOUND evidence all go through the
    # answer model. Retrieval status notes are context, not a reason to bypass
    # answer generation. The model must state limitations without inventing.
    evidence = list(state.get("final_evidence") or state.get("evidence") or [])
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

    evidence = _ensure_citation_sync(evidence)

    _emit_event(
        event="generating_answer",
        status="started",
        message="Generating the final answer.",
    )
    logger.info(
        "final_answer_evidence: items=%d chars=%d",
        len(evidence),
        len(json.dumps(evidence, ensure_ascii=False, default=str)),
    )

    try:
        parts: list[str] = []
        answer_started = time.perf_counter()
        first_token_seen = False
        answer_trace = current_trace()
        if answer_trace is not None:
            answer_trace.increment("llm_calls_answer")
        for delta in _stream_answer_with_fit(
            # Use the same contextualized task that planning and retrieval used.
            # A short follow-up like "please explain" is not self-contained;
            # passing only the raw utterance here loses the subject/calculation
            # and can cause the model to repeat the previous one-line result.
            query=effective_planning_query(state),
            evidence=evidence,
            conversation_history=state.get("conversation_history", []),
            planner_requirements=state.get("evidence_requirements") or [],
        ):
            if not first_token_seen:
                first_token_seen = True
                logger.info(
                    "[TIMING] stage=answer_first_token seconds=%.3f",
                    time.perf_counter() - answer_started,
                )
            parts.append(delta)
            _emit_event(
                event="answer_delta",
                status="streaming",
                message="",
                data={"delta": delta},
            )
        answer_elapsed = time.perf_counter() - answer_started
        logger.info(
            "[TIMING] stage=answer_stream_total seconds=%.3f chars=%d",
            answer_elapsed,
            sum(len(p) for p in parts),
        )
        if answer_trace is not None:
            answer_trace.record_stage("llm.answer_stream", answer_elapsed)
        answer = _parse_answer_response(
            "".join(parts),
            evidence_count=len(evidence),
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


def _build_context(state: AgentState) -> str:
    workspace_id = state.get("workspace_id")
    catalog: list[dict[str, Any]] = []

    if workspace_id:
        try:
            # The planner only needs ids/titles/types/table counts: use the
            # request-scoped table index (no sampling, no COUNT(*)).
            raw_catalog = _resources(state).catalog()
            relevant_ids = {
                _coerce_int(value)
                for value in state.get("relevant_document_ids", [])
                if _coerce_int(value) is not None
            }
            authorized_ids = _allowed_document_ids(state)
            if authorized_ids is not None:
                authorized_set = set(authorized_ids)
                raw_catalog = [
                    doc
                    for doc in raw_catalog
                    if _coerce_int(doc.get("document_id")) in authorized_set
                ]
                relevant_ids.intersection_update(authorized_set)
            # Scope Guard's picks are a ranking hint, not a boundary: keep every
            # authorized document, most relevant first, so the planner can still
            # see data that lives in a source Scope Guard did not pick.
            raw_catalog = sorted(
                raw_catalog,
                key=lambda doc: _coerce_int(doc.get("document_id"))
                not in relevant_ids,
            )
            column_digest = _planner_column_digest(
                state,
                [
                    document_id
                    for document_id in (
                        _coerce_int(doc.get("document_id")) for doc in raw_catalog
                    )
                    if document_id is not None
                ],
            )

            for document in raw_catalog:
                # Planner needs titles/ids/types to copy exact reference
                # strings. Table names, row counts and status are not needed
                # for reference extraction (execution resolves them later).
                entry: dict[str, Any] = {
                    "document_id": document.get("document_id"),
                    "title": document.get("title"),
                    "source_type": document.get("source_type"),
                }
                if relevant_ids:
                    entry["likely_relevant"] = (
                        _coerce_int(document.get("document_id")) in relevant_ids
                    )
                digest_columns = column_digest.get(
                    _coerce_int(document.get("document_id"))
                )
                if digest_columns:
                    entry["columns"] = digest_columns
                table_count = len(document.get("tables", []) or [])
                if table_count > 1:
                    entry["table_count"] = table_count
                catalog.append(entry)
        except Exception:
            logger.exception(
                "Could not load planner workspace catalog; continuing without it"
            )

    context: dict[str, Any] = {"user_query": state.get("query", "")}
    if catalog:
        # The catalog is already limited to the Scope Guard's relevant
        # documents, so the id list is not repeated.
        context["workspace_catalog"] = catalog
    elif state.get("relevant_document_ids"):
        relevant_ids = {
            _coerce_int(value)
            for value in state.get("relevant_document_ids", [])
            if _coerce_int(value) is not None
        }
        authorized_ids = _allowed_document_ids(state)
        if authorized_ids is not None:
            relevant_ids.intersection_update(authorized_ids)
        context["relevant_document_ids"] = sorted(relevant_ids)
    history = bound_history_for_planner(
        state.get("conversation_history", []),
        max_recent=2,
        max_user_chars=140,
        max_assistant_chars=120,
        max_summary_chars=160,
        max_total_chars=400,
    )
    if history:
        context["conversation_history"] = history
    return json.dumps(
        context,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def _safe_planner_fallback(reason: str) -> dict[str, Any]:
    return {
        "data_sources_needed": [],
        "category_hint": "document",
        "operation": "answer",
        "references": [],
        "requested_fields": [],
        "needs_instance_data": False,
        "requirements": [],
        "reason": str(reason or "Planner unavailable; raw-query retrieval only.")[:240],
        "_safe_fallback": True,
    }


def _planner_string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()[:max_chars]
        marker = text.casefold()
        if text and marker not in seen:
            seen.add(marker)
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _parse_understanding(response: Any) -> dict[str, Any]:
    try:
        choice = response.choices[0]
        content = choice.message.content
        data = json.loads(content)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AgentRuntimeError(
            "Query understanding returned invalid JSON.",
            code="PLANNING_ERROR",
        ) from exc

    if not isinstance(data, dict):
        raise AgentRuntimeError(
            "Query understanding returned an invalid object.",
            code="PLANNING_ERROR",
        )

    validation_error = next(iter(_PLANNER_VALIDATOR.iter_errors(data)), None)
    if validation_error is not None:
        path = "/".join(str(part) for part in validation_error.absolute_path)
        logger.warning(
            "Planner JSON failed local schema validation at %s: %s",
            path or "<root>",
            validation_error.message[:240],
        )
        raise AgentRuntimeError(
            "Query understanding did not match the planner schema.",
            code="PLANNING_SCHEMA_ERROR",
        )

    allowed_categories = {
        "metadata", "structured", "document", "mixed", "conversation", "product"
    }
    allowed_operations = {
        "describe", "lookup", "aggregate", "calculate", "filter", "compare",
        "list", "overview", "answer",
    }
    data_sources = _planner_string_list(
        data.get("data_sources_needed"), limit=20, max_chars=240
    )
    category = str(data.get("category_hint") or "document").strip().lower()
    operation = str(data.get("operation") or "answer").strip().lower()
    requirements = data.get("requirements")
    if not isinstance(requirements, list):
        requirements = []

    return {
        "data_sources_needed": data_sources,
        "category_hint": (
            category if category in allowed_categories
            else ("mixed" if data_sources else "document")
        ),
        "operation": operation if operation in allowed_operations else "answer",
        "references": _planner_string_list(
            data.get("references"), limit=30, max_chars=240
        ),
        "requested_fields": _planner_string_list(
            data.get("requested_fields"), limit=40, max_chars=240
        ),
        "needs_instance_data": data.get("needs_instance_data") is True,
        "requirements": requirements,
        "reason": str(data.get("reason") or "")[:240],
    }


def _parse_understanding_with_repair(response: Any) -> dict[str, Any]:
    """Parse one planner response, allow one constrained repair, then fail safe."""
    try:
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "Planner output reached its completion-token limit; using raw-query retrieval"
            )
            return _safe_planner_fallback(
                "Planner output was truncated; raw-query retrieval only."
            )
        return _parse_understanding(response)
    except AgentRuntimeError:
        try:
            malformed = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            malformed = ""
        trace = current_trace()
        if trace is not None:
            trace.increment("planner_repair_attempts")
            trace.increment("llm_calls_planner_repair")
        try:
            repaired = router.generate(
                task="planner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Repair the supplied planner output into valid QueryPilot planner JSON. "
                            "Use only these requirement type values: rule, fact, numeric, entity, "
                            "status, date, comparison, other. Use kind text, table, or either. "
                            "Never add facts, sources, entities, SQL or requirements. "
                            "Return a compact JSON object only."
                        ),
                    },
                    {"role": "user", "content": str(malformed or "")[:10000]},
                ],
                temperature=0.0,
                max_tokens=1400,
                response_format=PLANNER_RESPONSE_FORMAT,
                include_reasoning=False,
            )
            if getattr(repaired.choices[0], "finish_reason", None) == "length":
                return _safe_planner_fallback(
                    "Planner repair output was truncated; raw-query retrieval only."
                )
            return _parse_understanding(repaired)
        except Exception:
            logger.exception(
                "Planner response repair failed; using raw-query retrieval fallback"
            )
            return _safe_planner_fallback(
                "Planner output could not be validated; raw-query retrieval only."
            )


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


def _allowed_document_ids(state: AgentState) -> list[int] | None:
    """
    Conversation-level document allowlist supplied by the caller
    (state["allowed_document_ids"]).

    None = no conversation restriction (workspace scope still applies);
    [] = no documents allowed.
    """
    raw = state.get("allowed_document_ids")
    if raw is None:
        return None
    return [i for i in (_coerce_int(v) for v in raw) if i is not None]


def _resolved_source_matches(state: AgentState) -> list[dict[str, Any]]:
    """
    Every resolved document/table match, one per distinct source
    (a document and one of its tables count as the same source), in
    resolution order.
    """
    matches: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for item in state.get("resolved_references", {}).get("references", []):
        match = item.get("match", {})
        if item.get("status") != "RESOLVED" or not isinstance(match, dict):
            continue
        if match.get("type") not in {"document", "table"}:
            continue
        document_id = _coerce_int(match.get("document_id"))
        key = (
            document_id
            if document_id is not None
            else (
                match.get("title")
                or match.get("document_title")
                or match.get("table_name")
            )
        )
        if key in seen:
            continue
        seen.add(key)
        matches.append(match)
    return matches


def _run_source_metadata(state: AgentState) -> dict[str, Any]:
    """
    Preserve the tool's FOUND / NOT_FOUND state, look up EVERY resolved
    source (not just the first), never answer a question about a specific
    missing source with the whole catalog, and never read outside the
    conversation's allowed documents.
    """

    workspace_id = state["workspace_id"]
    allowed = _allowed_document_ids(state)
    references = state.get("resolved_references", {}).get("references", [])
    matches = _resolved_source_matches(state)

    def lookup(match: dict[str, Any]) -> dict[str, Any]:
        document_id = _coerce_int(match.get("document_id"))
        reference = None
        if document_id is None:
            reference = (
                match.get("title")
                or match.get("document_title")
                or match.get("table_name")
            )
        return source_metadata(
            workspace_id=workspace_id,
            reference=reference,
            document_id=document_id,
            allowed_document_ids=allowed,
        )

    if len(matches) == 1:
        metadata = lookup(matches[0])

    elif len(matches) > 1:
        results = [lookup(match) for match in matches]
        found = [r for r in results if r.get("status") != "NOT_FOUND"]
        documents = [
            doc for r in found for doc in r.get("content", {}).get("documents", [])
        ]
        metadata = {
            "source_type": "workspace_metadata",
            "workspace_id": workspace_id,
            "status": "FOUND" if found else "NOT_FOUND",
            "sources": [source for r in found for source in r.get("sources", [])],
            "content": {
                "evidence_role": "workspace_metadata",
                "document_count": len(documents),
                "document_names": [d.get("document_name") for d in documents],
                "documents": documents,
            },
        }

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
            catalog = source_metadata(
                workspace_id=workspace_id,
                allowed_document_ids=allowed,
            )
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

        metadata = source_metadata(
            workspace_id=workspace_id,
            allowed_document_ids=allowed,
        )

    metadata = dict(metadata)
    metadata["result_status"] = (
        "NOT_FOUND" if metadata.get("status") == "NOT_FOUND" else "SUCCESS"
    )
    return metadata


_TABLE_REFERENCE_WORDS = {"table", "sheet", "record"}
_DOCUMENT_REFERENCE_WORDS = {"document", "file", "doc", "pdf"}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
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


def _is_rule_dependent_mixed_request(state: AgentState) -> bool:
    """Whether structured rows must be preserved for a document-rule task."""
    plan = state.get("execution_plan") or {}
    decision = plan.get("decision") or {}
    understanding = state.get("query_understanding") or {}
    categories = {
        str(value or "").strip().lower()
        for value in (
            understanding.get("category_hint"),
            decision.get("category"),
        )
    }
    # Query-understanding and execution planning are independent stages. Do
    # not let a stale/more generic category hint mask the final planner's
    # explicit mixed decision when both structured and document evidence are
    # required.
    return "mixed" in categories and _plan_is_mixed(state)


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
        if deictic or ordinal is not None:
            return {"thing": None, "ordinal": ordinal, "deictic": deictic}
        return None

    if ordinal is None and not deictic:
        return None

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
        return None

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

    catalog = _resources(state).catalog()
    allowed_ids = _allowed_document_ids(state)
    if allowed_ids is not None:
        allowed_set = set(allowed_ids)
        catalog = [
            item
            for item in catalog
            if _coerce_int(item.get("document_id")) in allowed_set
        ]

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
        for doc in _resources(state).catalog()
    }
    return {
        document_id: bool(has_tables.get(document_id)) for document_id in document_ids
    }


def _document_query(state: AgentState) -> str:
    understanding = state.get("query_understanding", {})
    return " ".join(
        [
            str(understanding.get("operation", "answer")),
            " ".join(str(x) for x in understanding.get("requested_fields", [])),
            " ".join(str(x) for x in understanding.get("references", [])),
            effective_planning_query(state),
        ]
    ).strip()


def _resolved_document_ids(state: AgentState) -> list[int]:
    """
    Documents the user explicitly pointed at.

    Prefer the IDs already resolved by the planner. Fall back to the
    resolved-reference list for compatibility with older states.
    """
    decision_ids = (
        state.get("execution_plan", {})
        .get("decision", {})
        .get("resolved_document_ids", [])
    )

    if decision_ids:
        return sorted(
            {
                document_id
                for document_id in (_coerce_int(value) for value in decision_ids)
                if document_id is not None
            }
        )

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


def _candidate_document_ids(
    state: AgentState,
    resources: WorkspaceResources | None = None,
) -> list[int]:
    """Relevant, resolved and entity-located sources inside the allow-list."""
    table_index: list[dict[str, Any]] | None = None
    resolution = state.get("resolved_references", {}) or {}
    wants_table_index = any(
        isinstance(item, dict)
        and item.get("status") == "RESOLVED"
        and isinstance(item.get("match"), dict)
        and (
            item.get("match", {}).get("type") == "table"
            or any(
                isinstance(location, dict) and location.get("table_name")
                for location in item.get("match", {}).get("locations", []) or []
            )
        )
        for item in resolution.get("references", []) or []
    )
    if wants_table_index:
        try:
            table_index = (resources or _resources(state)).table_index(
                document_ids=_allowed_document_ids(state)
            )
        except Exception:
            logger.exception("Could not map resolved entity tables to document IDs")
    return candidate_document_ids(state, table_index)


def _document_ids_for_search(state: AgentState) -> list[int]:
    # Do not let a non-empty but incomplete plan hide Scope Guard's relevant
    # source candidates or entity-located source documents.
    return _candidate_document_ids(state)


def _workspace_document_ids_for_search(state: AgentState) -> list[int]:
    """List searchable documents inside the caller's authorization boundary."""
    allowed = _allowed_document_ids(state)
    if allowed is not None:
        allowed_set = set(allowed)
        if not allowed_set:
            return []
    else:
        allowed_set = None

    try:
        catalog = _resources(state).catalog()
    except Exception:
        logger.exception("Could not load authorized workspace document catalog")
        return []

    document_ids = {
        document_id
        for document_id in (
            _coerce_int(item.get("document_id"))
            for item in catalog
            if isinstance(item, dict)
        )
        if document_id is not None
        and (allowed_set is None or document_id in allowed_set)
    }
    return sorted(document_ids)


def _cross_document_search_ids(
    state: AgentState,
    candidate_ids: list[int],
) -> list[int]:
    """Expand an explicitly cross-document request only within authorization.

    If at least two source documents were resolved/relevant, preserve that
    narrower set. If the planner marked a cross-document task but only one
    candidate was found, search the authorized workspace corpus so an
    unresolved companion document is not silently omitted.
    """
    unique = sorted(
        {int(value) for value in candidate_ids if _coerce_int(value) is not None}
    )
    if len(unique) >= 2:
        return unique
    return _workspace_document_ids_for_search(state)


def _mark_success(chunks: list[Any]) -> list[Any]:
    for item in chunks:
        if isinstance(item, dict):
            item.setdefault("result_status", "SUCCESS")
    return chunks


def _has_document_table_evidence(items: list[Any]) -> bool:
    """
    Return True when structured evidence already contains successful rows
    read from a workspace table that belongs to an uploaded document.

    Structured table results are document-derived evidence even when they
    arrived through TABLE_QUERY rather than DOCUMENT_SEARCH. This keeps the
    evidence-state logic independent of file format, table name, or domain.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_type", "")).lower() != "table_query":
            continue
        if str(item.get("result_status", "")).upper() != "SUCCESS":
            continue

        sources = item.get("sources") or []
        if any(
            isinstance(source, dict)
            and _coerce_int(source.get("document_id")) is not None
            for source in sources
        ):
            return True

    return False


def _wants_exhaustive_read(state: AgentState) -> bool:
    """A list/enumerate request that asks for ALL items needs the whole source."""
    operation = normalized_operation(state.get("query_understanding"))
    if operation in OVERVIEW_OPERATIONS:
        return True
    if operation == "list":
        return bool(_EXHAUSTIVE_CUE.search(effective_planning_query(state)))
    return False



def _run_documents_with_coverage(
    state: AgentState,
    scope_ids: list[int],
    primary_query: str,
    *,
    answer_sized: bool = False,
    require_complete: bool = False,
) -> dict[str, Any] | None:
    """Load full chunks from the already-authorized, resolved documents.

    Segmenting groups complete chunks into manageable evidence items; it does
    not truncate chunk text. Returns None when loading fails so callers can
    fall back to ranked per-document retrieval.
    """
    workspace_id = state["workspace_id"]

    titles: dict[int, str] = {}
    try:
        for doc in _resources(state).catalog():
            doc_id = _coerce_int(doc.get("document_id"))
            if doc_id is not None:
                titles[doc_id] = str(doc.get("title") or f"document {doc_id}")
    except Exception:
        logger.exception("document coverage: catalog unavailable")

    def boost(document_id: int) -> tuple[set[int], bool]:
        result = search_evidence(
            query=primary_query,
            workspace_id=workspace_id,
            allowed_document_ids=[document_id],
        )
        if result.get("error_code"):
            return set(), False
        indexes = {
            (item.get("metadata") or {}).get("chunk_index")
            for item in result.get("evidence", [])
        }
        return (
            {i for i in indexes if isinstance(i, int)},
            bool(result.get("degraded")),
        )

    # Coverage retrieval is scoped by authorized document IDs. Include every
    # fetched chunk from those documents; segment size only groups chunks and
    # never truncates their content.
    sizing: dict[str, Any] = {
        "budget": None,
        "segment_char_limit": 8000,
    }

    _built = build_document_evidence(
        workspace_id=workspace_id,
        document_ids=scope_ids,
        question=effective_planning_query(state),
        understanding=state.get("query_understanding", {}) or {},
        titles=titles,
        fetch_chunks=fetch_document_chunks,
        boost_lookup=boost,
        require_complete=require_complete,
        **sizing,
    )
    return _built


def _search_evidence_variants(
    *,
    queries: list[str],
    workspace_id: int,
    document_ids: list[int] | None,
    max_calls: int = MAX_TOOL_CALLS_PER_QUERY,
) -> dict[str, Any]:
    """Search bounded planner queries and merge evidence without widening scope."""
    from hashlib import sha256

    targets = list(document_ids) if document_ids else [None]
    collected: dict[str, dict[str, Any]] = {}
    confidences: list[float] = []
    errors: list[tuple[str, str]] = []
    degraded = False
    calls = 0

    # QP-GEN-1c: share the call budget fairly between documents so a later
    # document is never starved by the queries spent on an earlier one.
    _call_cap = max(0, min(MAX_TOOL_CALLS_PER_QUERY, int(max_calls)))
    _per_target = max(1, min(_MAX_RETRIEVAL_QUERIES_CAP, _call_cap // max(1, len(targets))))
    for document_id in targets:
        for query in queries[:_per_target]:
            if calls >= max(0, min(MAX_TOOL_CALLS_PER_QUERY, int(max_calls))):
                logger.warning("document retrieval query budget reached; stopping variants")
                break
            kwargs = {
                "query": query,
                "workspace_id": workspace_id,
                "allowed_document_ids": [document_id] if document_id is not None else None,
            }
            try:
                result = search_evidence(**kwargs)
            except Exception as exc:
                logger.exception("Requirement retrieval failed for document_id=%s", document_id)
                errors.append(("RETRIEVAL_ERROR", str(exc)))
                calls += 1
                continue
            calls += 1
            if result.get("error_code"):
                errors.append((str(result.get("error_code")), str(result.get("error") or "Retrieval failed")))
                continue
            degraded = degraded or bool(result.get("degraded"))
            confidences.append(float(result.get("confidence") or 0.0))
            for item in result.get("evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                identity = (
                    f"{metadata.get('document_id', item.get('document_id'))}:"
                    f"{metadata.get('chunk_index', '')}"
                )
                if identity == ":":
                    identity = sha256(str(item.get("content") or "").encode("utf-8")).hexdigest()
                previous = collected.get(identity)
                if previous is None or float(item.get("score") or 0.0) > float(previous.get("score") or 0.0):
                    collected[identity] = item
        if calls >= max(0, min(MAX_TOOL_CALLS_PER_QUERY, int(max_calls))):
            break

    evidence = sorted(
        collected.values(),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    error = errors[0] if errors and not evidence else None
    return {
        "error_code": error[0] if error else None,
        "error": error[1] if error else None,
        "evidence": evidence,
        "confidence": max(confidences) if confidences else 0.0,
        "candidate_count": len(collected),
        "degraded": degraded,
        "calls": calls,
    }


# QP-GEN-1a: the planner's per-requirement queries (rules, exceptions, fees)
# used to be cut off after the raw question + effective question + one more.
_MAX_RETRIEVAL_QUERIES = 6
_MAX_RETRIEVAL_QUERIES_CAP = 10


def _retrieval_queries_for_state(state: AgentState) -> list[str]:
    original = str(state.get("query") or "").strip()
    effective = effective_planning_query(state)
    requirements = [
        requirement
        for requirement in (state.get("evidence_requirements") or [])
        if requirement.get("kind") in {"text", "either"}
    ]
    candidates = [original, effective]
    # Take requirement queries in rounds: every requirement's first query
    # before any requirement's second. Listing them requirement by
    # requirement let the first requirements use up the whole budget, so the
    # later ones were never searched.
    per_requirement = [list(requirement.get("queries", [])) for requirement in requirements]
    depth = max((len(queries) for queries in per_requirement), default=0)
    for level in range(depth):
        for queries in per_requirement:
            if level < len(queries):
                candidates.append(queries[level])
    limit = min(_MAX_RETRIEVAL_QUERIES_CAP, max(_MAX_RETRIEVAL_QUERIES, 2 + len(per_requirement)))
    queries: list[str] = []
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value and value.casefold() not in {item.casefold() for item in queries}:
            queries.append(value)
        if len(queries) >= limit:  # QP-GEN-1b
            break
    return queries or [effective] if effective else queries


def _run_documents(
    state: AgentState,
    budget: _RecoveryBudget,
) -> dict[str, Any]:
    primary_query = _document_query(state)
    primary_ids = _document_ids_for_search(state)
    if state.get("execution_plan", {}).get("cross_document"):
        primary_ids = _cross_document_search_ids(state, primary_ids)
    # Retrieve every source selected by the plan. For cross-document work,
    # _cross_document_search_ids has already expanded an incomplete candidate
    # set to the authorized workspace corpus. Retrieval scope and cross-source
    # reasoning remain separate decisions.
    multi_source = len(primary_ids) >= 2

    # A summary/overview must see the whole document, not only the top-k
    # chunks most similar to the word "summarize". Use document coverage for
    # it even when a single document is in scope.
    wants_whole_document = _wants_exhaustive_read(state)

    # Whole-document coverage is for: a summary/overview, several explicitly
    # named sources, or a mixed table+document request (existing behaviour).
    # A pointed question with no named source (e.g. "what is the standard
    # deduction?") is NOT read positionally: ranked retrieval over the allowed
    # corpus finds the passage that answers it.
    plan_cross = bool(state.get("execution_plan", {}).get("cross_document"))
    use_coverage = bool(primary_ids) and (
        wants_whole_document
        or (
            len(set(primary_ids)) >= 2
            and (multi_source or plan_cross or _plan_is_mixed(state))
        )
    )

    if use_coverage:
        try:
            covered = _run_documents_with_coverage(
                state,
                sorted(set(primary_ids)),
                primary_query,
                answer_sized=wants_whole_document,
                require_complete=wants_whole_document,
            )
        except Exception:
            logger.exception("document coverage failed; using ranked search")
            covered = None
        if covered is not None:
            return {
                "evidence": covered["evidence"],
                "confidence": covered["confidence"],
                "error": None,
                "degraded": covered["degraded"],
                "notes": covered["notes"],
                "coverage": covered["coverage"],
                "complete": covered.get("complete"),
                "calls": 1,
            }

    queries = _retrieval_queries_for_state(state)

    if multi_source:
        result = _search_evidence_variants(
            queries=queries,
            workspace_id=state["workspace_id"],
            document_ids=primary_ids,
            max_calls=max(0, MAX_TOOL_CALLS_PER_QUERY - int(state.get("tool_calls_used", 0))),
        )
        if result.get("error_code"):
            return {
                "evidence": [],
                "confidence": 0.0,
                "error": (str(result["error_code"]), result.get("error") or "Document retrieval failed."),
                "degraded": bool(result.get("degraded")),
                "calls": int(result.get("calls", 0)),
            }
        return {
            "evidence": _mark_success(list(result.get("evidence", []))),
            "confidence": float(result.get("confidence") or 0.0),
            "error": None,
            "degraded": bool(result.get("degraded")),
            "calls": int(result.get("calls", 0)),
        }

    result = _search_evidence_variants(
        queries=queries,
        workspace_id=state["workspace_id"],
        document_ids=primary_ids,
        max_calls=max(0, MAX_TOOL_CALLS_PER_QUERY - int(state.get("tool_calls_used", 0))),
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
            "calls": int(result.get("calls", 0)),
        }

    chunks = _mark_success(list(result.get("evidence", [])))
    confidence = float(result.get("confidence") or 0.0)
    degraded = bool(result.get("degraded"))

    if documents_insufficient(len(chunks), confidence, MIN_RETRIEVAL_CONFIDENCE):
        raw_query = effective_planning_query(state)

        scoped_to_resolved = bool(primary_ids)
        explicit_source_scope = bool(_resolved_document_ids(state)) or bool(
            _named_missing_source_references(state)
        )
        all_ids = _document_ids_for_search(state)
        retry_ids = primary_ids if scoped_to_resolved else (all_ids or primary_ids)

        # Scope Guard's relevant_document_ids are a preferred search set, not
        # an authorization boundary. If they produce insufficient evidence,
        # expand to every conversation-authorized document. An explicitly
        # resolved source remains hard-scoped and is never widened silently.
        if not explicit_source_scope:
            authorized = _allowed_document_ids(state)
            if authorized is not None:
                widened = list(dict.fromkeys(int(value) for value in authorized))
            else:
                try:
                    widened = [
                        value
                        for value in (
                            _coerce_int(doc.get("document_id"))
                            for doc in _resources(state).catalog()
                        )
                        if value is not None
                    ]
                except Exception:
                    logger.exception("Could not expand preferred document scope")
                    widened = []
            if widened:
                retry_ids = widened

        different = bool(raw_query) and (
            raw_query != primary_query or sorted(retry_ids) != sorted(primary_ids)
        )

        recovery_calls = 0
        if (
            different
            and not _past_deadline(state)
            and budget.try_consume()
            and int(state.get("tool_calls_used", 0)) + int(result.get("calls", 0)) < MAX_TOOL_CALLS_PER_QUERY
        ):
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
                recovery_calls = 1
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

    return {
        "evidence": chunks,
        "confidence": confidence,
        "error": None,
        "degraded": degraded,
        "calls": int(result.get("calls", 0)) + locals().get("recovery_calls", 0),
    }


def _document_ids_from_text_evidence(items: list[Any]) -> set[int]:
    """Extract source document IDs from retrieved text chunks."""
    result: set[int] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for value in (item.get("document_id"), metadata.get("document_id")):
            document_id = _coerce_int(value)
            if document_id is not None:
                result.add(document_id)
    return result


# QP-FIX-4: facts found in table ROWS drive extra text searches. A rule such
# as "members get extra days" is written in prose and the question never says
# it; only a row value ("<flag column> = Yes") reveals that it matters.
_FACT_MIN_ROWS = 4  # tables smaller than this are not "populations"
_FACT_MAX_DISTINCT = 6  # a column is a category/flag only if it has few values
_FACT_MAX_VALUE_CHARS = 30
_FACT_MAX_QUERIES = 3
_FACT_INCLUDE_QUESTION = False  # True: append the question to each fact query
_FACT_NUMERIC = re.compile(r"^[\s\d.,%+\-/:]+$")
# QP-FIX-6: a SMALL table (one entity's rows) has no variety to measure, but a
# yes/no style flag in it still points at a rule written in prose. Only these
# boolean-looking values are used from small tables; names, cities and other
# free text would only add noise to the search.
_FACT_BOOL_VALUES = {"yes", "no", "true", "false", "y", "n"}


def _table_is_rule_only(item: Any, roles: dict[str, str]) -> bool:  # QP-FIX-6
    names = {
        str(s.get("table_name"))
        for s in (item.get("sources") or [] if isinstance(item, dict) else [])
        if isinstance(s, dict) and s.get("table_name")
    }
    return bool(names) and all(roles.get(name) == "rule" for name in names)


def _row_fact_phrases(state: AgentState, structured: list[Any]) -> list[str]:
    """'<column name> <value> <value>' for low-variety text columns.

    Skips rule tables, id-like columns, numbers, dates and long text. Columns
    with the fewest distinct values (flags, tiers, statuses) come first.
    """
    from retrieval.services.keys import is_id_column

    roles = {
        str(name): str(role or "").strip().lower()
        for name, role in (state.get("table_roles") or {}).items()
    }
    scored: list[tuple[int, str]] = []
    for item in structured or []:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue
        rows = [r for r in (content.get("rows") or []) if isinstance(r, dict)]
        if len(rows) < _FACT_MIN_ROWS:
            # QP-FIX-6: small (entity) tables still contribute their yes/no flags
            if rows and not _table_is_rule_only(item, roles):
                for column in [str(c) for c in (content.get("columns") or list(rows[0]))]:
                    if is_id_column(column):
                        continue
                    flag_values = list(
                        dict.fromkeys(
                            str(r.get(column)).strip()
                            for r in rows
                            if r.get(column) not in (None, "")
                        )
                    )
                    if len(flag_values) == 1 and flag_values[0].lower() in _FACT_BOOL_VALUES:
                        label = re.sub(
                            r"[_\s]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", column)
                        ).strip()
                        scored.append((1, f"{label} {flag_values[0]}"))
            continue
        names = {
            str(s.get("table_name"))
            for s in item.get("sources") or []
            if isinstance(s, dict) and s.get("table_name")
        }
        if names and all(roles.get(name) == "rule" for name in names):
            continue
        columns = [str(c) for c in (content.get("columns") or list(rows[0]))]
        for column in columns:
            if is_id_column(column):
                continue
            values = [
                str(r.get(column)).strip()
                for r in rows
                if r.get(column) not in (None, "")
            ]
            distinct = list(dict.fromkeys(values))
            if len(distinct) < 2:
                continue
            if len(distinct) > min(_FACT_MAX_DISTINCT, max(2, len(rows) // 2)):
                continue
            if any(
                len(v) > _FACT_MAX_VALUE_CHARS or _FACT_NUMERIC.match(v)
                for v in distinct
            ):
                continue
            label = re.sub(
                r"[_\s]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", column)
            ).strip()
            scored.append((len(distinct), f"{label} {' '.join(distinct)}"))
    scored.sort(key=lambda pair: pair[0])
    phrases = list(dict.fromkeys(text for _, text in scored))
    return phrases[:_FACT_MAX_QUERIES]


def _doc_and_chunk_indexes(item: Any) -> tuple[int | None, set[int]]:  # QP-FIX-5
    """(document id, chunk indexes) of a text passage; indexes may be empty."""
    if not isinstance(item, dict):
        return None, set()
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    doc = _coerce_int(item.get("document_id", meta.get("document_id")))
    indexes = meta.get("chunk_indexes")
    if not isinstance(indexes, list):
        single = meta.get("chunk_index")
        indexes = [single] if isinstance(single, int) else []
    return doc, {i for i in indexes if isinstance(i, int)}


# QP-GEN-2a: a row value only matters if some prose says what it changes, so
# the search text asks for rules/conditions, not just for the value itself.
_FACT_RULE_SUFFIX = "rules conditions benefits exceptions"


def _structured_columns_for_log(structured: Any) -> list[list[str]]:  # QP-GEN-7b
    columns: list[list[str]] = []
    for item in structured or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict) and content.get("columns"):
            columns.append([str(c) for c in content.get("columns") or []][:25])
    return columns[:5]

_FACT_MAX_HITS_PER_PHRASE = 2
_FACT_MAX_HITS = 5


def _loaded_chunk_map(items: list[Any]) -> dict[int, set[int] | None]:
    """doc id -> chunk indexes already in the evidence (None = whole document)."""
    loaded: dict[int, set[int] | None] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        doc, indexes = _doc_and_chunk_indexes(item)
        if doc is None:
            continue
        if str(meta.get("retrieval_method") or "") == "full_document":
            loaded[doc] = None
            continue
        if doc in loaded and loaded[doc] is None:
            continue
        loaded.setdefault(doc, set()).update(indexes)
    return loaded


def _row_fact_hit_already_loaded(
    item: Any,
    already_loaded: set[int],
    loaded_chunks: dict[int, set[int] | None] | None,
) -> bool:
    """True when this exact passage (not merely its document) is already present."""
    doc, indexes = _doc_and_chunk_indexes(item)
    if doc is None:
        return False
    if doc in (already_loaded or set()):
        return True
    if loaded_chunks and doc in loaded_chunks:
        seen = loaded_chunks[doc]
        if seen is None or not indexes:
            return True
        return indexes <= seen
    return False


def _search_row_facts(
    state: AgentState,
    phrases: list[str],
    allowed_ids: Any,
    base_query: str,
    already_loaded: set[int],
    loaded_chunks: dict[int, set[int] | None] | None = None,
) -> list[Any]:
    """Passages found through row facts that are not already in the evidence."""
    hits: list[Any] = []
    seen: set[tuple[Any, Any]] = set()
    for phrase in phrases:
        if len(hits) >= _FACT_MAX_HITS:
            break
        _phrase_hits = 0
        text = f"{phrase} {base_query}" if _FACT_INCLUDE_QUESTION else phrase
        text = f"{text} {_FACT_RULE_SUFFIX}"
        try:
            found = search_evidence(
                query=text,
                workspace_id=int(state["workspace_id"]),
                allowed_document_ids=allowed_ids,
            )
        except Exception:
            logger.exception("row-fact text search failed: %r", phrase)
            continue
        if found.get("error_code"):
            continue
        for item in found.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            if _row_fact_hit_already_loaded(item, already_loaded, loaded_chunks):  # QP-GEN-2b
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            ident = (item.get("document_id"), meta.get("chunk_index"))
            if ident in seen:
                continue
            if _phrase_hits >= _FACT_MAX_HITS_PER_PHRASE or len(hits) >= _FACT_MAX_HITS:
                break
            seen.add(ident)
            _phrase_hits += 1
            hits.append(item)
    return hits


def _supplement_dependency_evidence(
    state: AgentState,
    document_result: dict[str, Any],
    structured: list[Any] | None = None,
) -> dict[str, Any]:
    """Search the authorized corpus for governing rules omitted by initial scope.

    Scope Guard and source resolution give high-value candidates, but they can
    miss a companion policy that is not explicitly named by the user. For
    mixed calculations, run one bounded ranked search over the conversation's
    authorized document set and merge the most relevant text evidence. This is
    retrieval-only: it does not widen authorization or select every workspace
    document for the answer.
    """
    if not requires_dependency_discovery(state):
        return document_result

    query = dependency_discovery_query(  # QP-PATCH-V4
        state,
        evidence=list(structured or []) + list(document_result.get("evidence") or []),
    )
    if not query.strip():
        return {**document_result, "dependency_evidence_found": False}

    allowed_ids = authorized_document_ids(state)
    try:
        discovered = search_evidence(
            query=query,
            workspace_id=int(state["workspace_id"]),
            allowed_document_ids=allowed_ids,
        )
    except Exception:
        logger.exception("Supplemental dependency-evidence search failed")
        return {**document_result, "dependency_evidence_found": False}

    if discovered.get("error_code"):
        logger.warning(
            "Supplemental dependency-evidence search failed: code=%s",
            discovered.get("error_code"),
        )
        return {**document_result, "dependency_evidence_found": False}

    supplemental = _mark_success(list(discovered.get("evidence", []) or []))
    primary = list(document_result.get("evidence", []) or [])
    coverage_query = query  # QP-FIX-4
    _fact_phrases = _row_fact_phrases(state, list(structured or []))
    if not _fact_phrases:  # QP-GEN-7a
        logger.info(
            "row-fact text search: no attribute phrases; structured columns=%s",
            _structured_columns_for_log(structured),
        )
    _fact_chunk_map: dict[int, set[int]] = {}  # QP-FIX-5: doc id -> chunks found via row facts
    if _fact_phrases:
        _fact_hits = _mark_success(
            _search_row_facts(
                state,
                _fact_phrases,
                allowed_ids,
                query,
                set(),
                loaded_chunks=_loaded_chunk_map(primary),  # QP-GEN-2c
            )
        )
        logger.info(
            "row-fact text search: phrases=%s new_chunks=%d new_document_ids=%s",
            _fact_phrases,
            len(_fact_hits),
            sorted(_document_ids_from_text_evidence(_fact_hits)),
        )
        if _fact_hits:
            for _hit in _fact_hits:  # QP-FIX-5: rules found via row facts must survive trimming
                if isinstance(_hit, dict):
                    _hit["protect_from_trim"] = True
                    _hd, _hc = _doc_and_chunk_indexes(_hit)
                    if _hd is not None:
                        _fact_chunk_map.setdefault(_hd, set()).update(_hc)
            supplemental = _fact_hits + supplemental
            coverage_query = f"{query} {' '.join(_fact_phrases)}"
    if not supplemental:
        return {**document_result, "dependency_evidence_found": False}

    # Ranked retrieval locates the companion source, but a few top passages
    # may omit a table, formula, exception, or neighbouring definition needed
    # for the calculation. Once a dependency document has been identified,
    # load its authorized chunks in reading order with the same answer-sized
    # coverage builder used for multi-document requests. This is still bounded
    # by the final evidence budget and never widens beyond discovered IDs.
    dependency_id_set = _document_ids_from_text_evidence(supplemental)
    if allowed_ids is not None:
        authorized_set = {int(value) for value in allowed_ids}
        dependency_id_set.intersection_update(authorized_set)
    # QP-PATCH: do not re-load documents the first pass already supplied, and
    # do not force whole-document completeness (that took 85% of the budget).
    already_loaded = _document_ids_from_text_evidence(primary)
    dependency_ids = sorted(dependency_id_set - already_loaded)
    dependency_coverage: list[Any] = []
    dependency_notes: list[Any] = []  # QP-PATCH-DEPENDENCY-FLOOR
    dependency_cov_info: list[Any] = []
    logger.info("Supplemental dependency load: authorized_document_ids=%s", dependency_ids)
    if dependency_ids:
        try:
            covered = _run_documents_with_coverage(
                state,
                dependency_ids,
                coverage_query,
                require_complete=False,
            )
        except Exception:
            logger.exception(
                "Supplemental dependency-document coverage failed; "
                "keeping ranked evidence"
            )
            covered = None
        if covered is not None:
            dependency_coverage = list(covered.get("evidence", []) or [])
            dependency_notes = list(covered.get("notes") or [])
            dependency_cov_info = list(covered.get("coverage") or [])
            # QP-FIX-5: a coverage passage that holds a chunk found through row
            # facts is protected from trimming. If the passage carries no chunk
            # info, every passage of that document is protected (it was loaded
            # because a row fact pointed there).
            _prot = 0
            for _cov in dependency_coverage:
                if not isinstance(_cov, dict):
                    continue
                _cd, _cc = _doc_and_chunk_indexes(_cov)
                if _cd is None or _cd not in _fact_chunk_map:
                    continue
                if not _cc or (_cc & _fact_chunk_map[_cd]):
                    _cov["protect_from_trim"] = True
                    _prot += 1
            logger.info(
                "row-fact protection: protected=%d of %d coverage passage(s) fact_chunks=%s",
                _prot,
                len(dependency_coverage),
                {k: sorted(v) for k, v in _fact_chunk_map.items()},
            )

    # Prefer coverage for documents it actually represented, avoiding duplicate
    # top-k chunks for the same document. Keep ranked results as a fallback if
    # full-document loading returned no usable passage for that source.
    covered_ids = _document_ids_from_text_evidence(dependency_coverage) | already_loaded
    # QP-GEN-2d: a passage found through a row fact is kept when THAT passage
    # is missing, even if other passages of its document are already loaded.
    _covered_map = _loaded_chunk_map(list(dependency_coverage) + list(primary))
    remaining_supplemental = [
        item
        for item in supplemental
        if (
            not _row_fact_hit_already_loaded(item, set(), _covered_map)
            if isinstance(item, dict) and item.get("protect_from_trim")
            else not (_document_ids_from_text_evidence([item]) & covered_ids)
        )
    ]
    # Preserve every non-duplicate dependency hit; the answer stage no longer
    # removes passages using a locally estimated token/character budget.
    combined = _canonicalize_evidence(
        dependency_coverage + remaining_supplemental + primary
    )
    old_confidence = float(document_result.get("confidence") or 0.0)
    new_confidence = float(discovered.get("confidence") or 0.0)
    found_ids = sorted(_document_ids_from_text_evidence(supplemental))
    logger.info(
        "Supplemental dependency evidence: ranked_chunks=%d coverage_items=%d "
        "document_ids=%s",
        len(supplemental),
        len(dependency_coverage),
        found_ids,
    )
    if dependency_ids and not dependency_coverage:  # QP-PATCH-DEPENDENCY-FLOOR
        dependency_notes.append(
            {
                "source_type": "evidence_note",
                "result_status": "SUCCESS",
                "content": (
                    "Document coverage. Source document(s) "
                    + ", ".join(str(i) for i in dependency_ids)
                    + " were identified as relevant, but only a few ranked "
                    "passages of them could be included. If the answer needs "
                    "content that is not in the provided passages, say it was "
                    "not found in the provided passages; do not guess."
                ),
            }
        )
    return {
        **document_result,
        "evidence": combined,
        "notes": list(document_result.get("notes") or []) + dependency_notes,
        "coverage": list(document_result.get("coverage") or []) + dependency_cov_info,
        "confidence": max(old_confidence, new_confidence),
        "error": None if combined else document_result.get("error"),
        "degraded": bool(document_result.get("degraded"))
        or bool(discovered.get("degraded")),
        "dependency_evidence_found": True,
    }


def _supplement_row_fact_evidence(
    state: AgentState,
    document_result: dict[str, Any],
    structured: list[Any] | None = None,
) -> dict[str, Any]:
    """QP-GEN-3a: let table rows pull in the prose rules that depend on them.

    Runs for requests that were routed to document search but are not
    "calculation" requests (those use _supplement_dependency_evidence). It
    takes yes/no or low-variety attributes of the matched rows and searches the
    authorized documents for what those attributes change. Bounded, protected
    from trimming, and a no-op when no such attribute exists.
    """
    try:
        phrases = _row_fact_phrases(state, list(structured or []))
        if not phrases:
            return document_result
        primary = list(document_result.get("evidence") or [])
        hits = _mark_success(
            _search_row_facts(
                state,
                phrases,
                authorized_document_ids(state),
                effective_planning_query(state),
                set(),
                loaded_chunks=_loaded_chunk_map(primary),
            )
        )
        logger.info(
            "row-fact text search (non-calculation): phrases=%s new_chunks=%d "
            "document_ids=%s",
            phrases,
            len(hits),
            sorted(_document_ids_from_text_evidence(hits)),
        )
        if not hits:
            return document_result
        for hit in hits:
            if isinstance(hit, dict):
                hit["protect_from_trim"] = True
        return {
            **document_result,
            "evidence": _canonicalize_evidence(hits + primary),
            "row_fact_evidence_found": True,
        }
    except Exception:
        logger.exception("row-fact supplement failed; keeping existing evidence")
        return document_result


def _apply_table_scope(
    state: AgentState,
    schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scope = _table_scope(state)
    if scope is None:
        return schema
    # An explicit table scope is authoritative. If none of the supplied
    # schema entries match it, return no tables rather than silently widening
    # back to the full schema (which defeats selection and can overflow the
    # answer context).
    return [item for item in schema if str(item.get("table_name")) in scope]


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


_PROBE_MAX_COLUMNS = 8
_PROBE_MAX_DISTINCT = 25
_PROBE_MAX_VALUE_CHARS = 60


def _is_empty_result(result: dict[str, Any] | None) -> bool:
    """True when a query that RAN produced nothing usable.

    Zero rows, or only rows whose every value is NULL (what an aggregate
    returns when its filter matched nothing).
    """
    if result is None:
        return False
    if int(result.get("row_count", 0) or 0) == 0:
        return True
    rows = [r for r in (result.get("rows") or []) if isinstance(r, dict)]
    return bool(rows) and all(all(v is None for v in r.values()) for r in rows)


def _probe_stored_values(
    state: AgentState,
    schema_for_sql: list[dict[str, Any]],
    allowed: set[str],
    tables_used: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Read-only look at what low-cardinality text columns really contain.

    Uses the same query_tables path (allow-list, validation, read-only), so no
    authorization or SQL-safety rule is bypassed.
    """
    by_name = {str(item.get("table_name")): item for item in schema_for_sql}
    names = [n for n in tables_used if n in by_name] or list(by_name)[:2]
    found: dict[str, dict[str, list[str]]] = {}
    budget_left = _PROBE_MAX_COLUMNS

    for table_name in names:
        for column in by_name[table_name].get("columns") or []:
            if budget_left <= 0 or _past_deadline(state):
                return found
            column = str(column)
            if '"' in column or table_name not in allowed:
                continue
            numeric_names = {
                str(c) for c in by_name[table_name].get("numeric_columns") or []
            }
            if column in numeric_names:
                continue  # numbers do not help fix a text/period filter
            budget_left -= 1
            try:
                result = query_tables(
                    workspace_id=state["workspace_id"],
                    sql=(
                        f'SELECT DISTINCT "{column}" FROM "{table_name}" '
                        f"LIMIT {_PROBE_MAX_DISTINCT + 1}"
                    ),
                    parameters=[],
                    allowed_table_names={table_name},
                )
            except ValueError as exc:
                if _is_infrastructure_error(str(exc)):
                    raise AgentRuntimeError(str(exc), code="DATABASE_ERROR") from exc
                continue
            if column in (result.get("date_columns") or {}):
                continue  # shown converted, not as stored; sample_rows has the raw form
            values = [
                r.get(column)
                for r in result.get("rows", [])
                if r.get(column) is not None
            ]
            if not values or len(values) > _PROBE_MAX_DISTINCT:
                continue  # empty, or too many values to be a category/period list
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in values
            ):
                continue  # plain numbers do not help fix a text/period filter
            found.setdefault(table_name, {})[column] = [
                str(v)[:_PROBE_MAX_VALUE_CHARS] for v in values
            ]
    return found


def _zero_rows_repair_guidance(
    schema_for_sql: list[dict[str, Any]],
    is_mixed: bool,
    stored_values: dict[str, dict[str, list[str]]] | None = None,
    parameters: Any = None,
) -> str:
    """
    Generic, evidence-driven repair guidance for an empty result that
    contradicts the actual source data. Keyed on table sizes from the
    authoritative schema, the previous parameters and the stored values -
    never on query wording. It never asks the model to drop a requested filter.
    """
    sizes = ", ".join(
        f"{item.get('table_name')}: {int(item.get('row_count') or 0)} rows"
        for item in schema_for_sql[:5]
    )
    guidance = (
        "The query returned zero rows although the source table(s) contain "
        f"rows ({sizes}), so a condition did not match what is stored - the "
        "data is not missing. KEEP every entity, date, period, status or "
        "category condition the question asks for; do NOT drop them. Instead "
        "rewrite each one so its literal or parameter matches the format and "
        "spelling actually stored in the column (shown below). Also check "
        "column names and CASTs, and remove only a JOIN or condition the "
        "question does not require."
    )
    if isinstance(parameters, (list, tuple)) and parameters:
        shown = ", ".join(repr(str(p))[:80] for p in parameters[:8])
        guidance += f" Parameter values used last time: [{shown}]."
    if stored_values:
        parts = [
            f"{table}.{column} = {values}"
            for table, columns in stored_values.items()
            for column, values in columns.items()
        ]
        guidance += " Values actually stored (distinct): " + "; ".join(parts) + "."
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
            query=effective_planning_query(state),
            schema=schema_for_sql,
            resolved_entities=entities,
            previous_attempt=previous_attempt,
            sql_required=True,
            planner_notes=planner_notes or None,
            query_semantics={
                "operation": str(
                    (state.get("query_understanding", {}) or {}).get("operation")
                    or "answer"
                ),
                "population_required": _query_requires_population_context(
                    effective_planning_query(state),
                    str(
                        (state.get("query_understanding", {}) or {}).get("operation")
                        or ""
                    ),
                ),
                "requested_fields": list(
                    (state.get("query_understanding", {}) or {}).get(
                        "requested_fields", []
                    )
                    or []
                )[:12],
            },
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
        "table_query attempt: repair=%s sql_preview=%r parameter_count=%d "
        "parameters=%.200r",
        previous_attempt is not None,
        str(sql)[:300],
        len(parameters) if isinstance(parameters, (list, tuple)) else 0,
        parameters,
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
            " Report this as no matching records, not as missing or inaccessible data."
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

    Top-N (LIMIT N, N > 1) is recognised as a ranking, not as a plain table
    result: the ordered rows ARE the answer and must not be treated as a
    duplicate of the full table downstream.
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

    order_limit = re.search(
        r"\bORDER BY\b(.+?)\bLIMIT\s+(\d+|\?)(?:\s+OFFSET\s+(\d+))?",
        upper_sql,
    )
    if order_limit:
        order_clause = order_limit.group(1).strip()
        limit_text = order_limit.group(2)
        limit = int(limit_text) if limit_text.isdigit() else None
        first_key = order_clause.split(",")[0].replace('"', "").strip()

        if first_key in {"ROWID", "ROWID ASC", "ROWID DESC"}:
            descending = first_key.endswith("DESC")
            return {
                "type": "positional",
                "selection": f"{'last' if descending else 'first'} {limit if limit else 'N'}",
                "basis": f"ORDER BY rowid LIMIT {limit_text}",
                "meaning": (
                    "The query deliberately selected rows by position from the "
                    "whole source table; it is not evidence about the table size."
                ),
            }

        direction = "DESC" if re.search(r"\bDESC\b", first_key) else "ASC"
        if limit == 1:
            meaning = (
                "The query deliberately selected the single highest/lowest "
                "ranked result according to its ORDER BY expression. The "
                "returned row is the ranking result, not an arbitrary sample."
            )
        else:
            meaning = (
                "The query deliberately selected the top-ranked rows by its "
                "ORDER BY expression. The returned rows, in the returned "
                "order, ARE the ranking result. Do not re-rank them or "
                "replace them with other rows of the source table."
            )
        return {
            "type": "ranking",
            "direction": direction,
            "operation": normalized_operation,
            "basis": f"ORDER BY ... LIMIT {limit_text}",
            "limit": limit,
            "meaning": meaning,
        }

    if re.search(r"\bGROUP BY\b", upper_sql):
        return {
            "type": "grouped",
            "operation": normalized_operation,
            "basis": "GROUP BY",
            "meaning": (
                "Each returned row is one group's aggregate, not a source row."
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
    # Do not let an incomplete plan source list hide request-scoped relevant
    # documents or a source found through entity resolution.
    resolved_document_ids: list[int] = list(_candidate_document_ids(state))
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

    if _is_rule_dependent_mixed_request(state):
        # The schema passed for this route is already restricted to the
        # caller's authorization scope and selected by the table selector.
        # Include its document IDs even when first-pass reference resolution
        # found only one of the data/rule sources.
        resolved_document_ids.extend(
            document_id
            for item in schema
            if (document_id := _coerce_int(item.get("document_id"))) is not None
        )

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
    *,
    allow_single_source: bool = False,
) -> dict[str, Any]:
    """Retrieve source tables independently, without LLM-generated SQL.

    Cross-document requests need source contents, not a model-generated join.
    For a mixed data-plus-policy request, the same row-preserving path can be
    used with one initially resolved structured source while the companion
    policy is discovered separately. Tables are read with allowlisted SELECT *
    queries, never joined across documents.
    """
    groups = _cross_document_source_groups(state, schema)
    if not groups or (len(groups) < 2 and not allow_single_source):
        return {"evidence": [], "status": "NOT_FOUND"}

    evidence: list[dict[str, Any]] = []
    any_success = False
    any_error = False
    big_tables: list[dict[str, Any]] = []
    operation = str(
        state.get("query_understanding", {}).get("operation") or "answer"
    ).lower()

    for group in groups:
        if _past_deadline(state):
            break

        for item in _in_table_scope(state, group):
            if _past_deadline(state):
                break

            table_name = str(item.get("table_name") or "")
            if not table_name:
                continue

            if int(item.get("row_count") or 0) > QUERY_ROW_CAP:
                # Larger than the read cap: SELECT * would return only the
                # first rows, which must never stand in for the population.
                big_tables.append(item)
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
                            "Rows extracted directly from an authorized source table "
                            "selected as relevant to this request. This is source "
                            "content for cross-document reasoning, including facts, "
                            "rules, rates, thresholds, or formulas represented in the "
                            "table. It was retrieved independently and was never joined "
                            "with another document."
                        ),
                    },
                    "result_status": "SUCCESS" if success else "NOT_FOUND",
                }
            )

    if big_tables:
        large = _run_large_tables_via_sql(state, budget, big_tables)
        evidence.extend(large["evidence"])
        any_success = any_success or large["success"]
        any_error = any_error or large["error"]

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
    resources = _resources(state)
    # Cheap existence/shape check first (no sampling, no COUNT(*)). Table
    # details are loaded below only for the documents this request needs, and
    # reuse anything an earlier stage already loaded.
    allowed_document_ids = _allowed_document_ids(state)
    table_index = resources.table_index(document_ids=allowed_document_ids)
    if not table_index:
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
    resolved_document_ids = _document_ids_for_search(state)
    entities = _resolved_entity_values(state)
    is_mixed = _plan_is_mixed(state)

    if _is_rule_dependent_mixed_request(state):
        # Do this before ordinary resolved-document narrowing. The policy may
        # be a companion source the resolver did not name-match, even though
        # cross-document intent is already known. Each selected table is read
        # independently with an allowlisted SELECT; no cross-document SQL join
        # is introduced.
        mixed_schema = resources.schema(
            document_ids=allowed_document_ids,
            include_sample_rows=True,
        )
        mixed_schema = _apply_table_scope(state, mixed_schema)
        if mixed_schema:
            logger.info(
                "table_query: mixed-request discovery selected %d table(s) "
                "across authorized documents",
                len(mixed_schema),
            )
            return _run_cross_document_tables(
                state,
                budget,
                mixed_schema,
                allow_single_source=True,
            )

    if len(set(resolved_document_ids)) >= 2 or (
        resolved_table_names
        and len(
            {
                _coerce_int(item.get("document_id"))
                for item in table_index
                if str(item.get("table_name")) in resolved_table_names
                and _coerce_int(item.get("document_id")) is not None
            }
        )
        >= 2
    ):
        # Cross-document groups only ever use the resolved documents' tables
        # (plus entity-located tables), all covered by _scope_document_ids.
        return _run_cross_document_tables(
            state,
            budget,
            resources.schema(document_ids=_scope_document_ids(state, resources)),
        )

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
                for item in table_index
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

    # Details for just the documents the request established; with nothing
    # resolved the whole workspace is the SQL scope, as before.
    # Sample rows are already fetched/cached for numeric detection; exposing them
    # lets SQL generation see the real stored format of every column.
    if resolved_document_ids or resolved_table_names:
        schema = resources.schema(
            document_ids=_scope_document_ids(state, resources),
            include_sample_rows=True,
        )
    else:
        schema = resources.schema(include_sample_rows=True)

    if resolved_table_names:
        filtered_schema = [
            item for item in schema if item.get("table_name") in resolved_table_names
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

    schema_for_sql = _apply_table_scope(state, schema_for_sql)
    allowed = {item["table_name"] for item in schema_for_sql}
    schema_by_table = {item["table_name"]: item for item in schema_for_sql}

    # In a mixed request, document rules may control eligibility, thresholds,
    # deductions, or the derived measure itself. Do not let generated SQL apply
    # an incomplete policy or aggregate away the employee/record-level inputs.
    # Preserve the authorized table rows; the companion rules are retrieved by
    # DOCUMENT_SEARCH and the final answer combines both evidence streams.
    if _is_rule_dependent_mixed_request(state):
        logger.info(
            "table_query: preserving source rows for rule-dependent mixed request; "
            "skipping generated SQL aggregation/filtering"
        )
        return _run_cross_document_tables(
            state,
            budget,
            schema_for_sql,
            allow_single_source=True,
        )

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

    operation = str(
        state.get("query_understanding", {}).get("operation") or "answer"
    ).lower()

    failed = first["result"] is None
    empty_result = _is_empty_result(first["result"])
    source_populated = any(
        int(item.get("row_count") or 0) > 0 for item in schema_for_sql
    )
    # A query that ran but found nothing, against a populated source or for an
    # entity we already located, is a repairable query problem. This does not
    # depend on the operation label or on entity resolution succeeding.
    empty_but_source_has_data = empty_result and (
        bool(entities) or source_populated
    )

    logger.info(
        "table_query recovery gate: failed=%s empty_result=%s entities=%d "
        "source_populated=%s operation=%s past_deadline=%s budget_used=%d/%d",
        failed,
        empty_result,
        len(entities),
        source_populated,
        operation,
        _past_deadline(state),
        budget.used,
        budget.limit,
    )

    if (failed or empty_but_source_has_data) and not _past_deadline(state):
        if budget.try_consume():
            previous: dict[str, Any] = {
                "sql": first["sql"],
                "parameters": first["parameters"],
            }
            stored_values: dict[str, dict[str, list[str]]] = {}
            if failed:
                previous["error"] = first["error"]
            else:
                previous["zero_rows"] = True
                try:
                    stored_values = _probe_stored_values(
                        state,
                        schema_for_sql,
                        allowed,
                        list((first["result"] or {}).get("tables_used", [])),
                    )
                except AgentRuntimeError:
                    raise
                except Exception:
                    logger.exception("stored-value probe failed; repairing without it")
                previous["error"] = _zero_rows_repair_guidance(
                    schema_for_sql,
                    is_mixed,
                    stored_values=stored_values,
                    parameters=first["parameters"],
                )

            logger.info(
                "table_query recovery: reason=%s probed_columns=%d",
                "error" if failed else "empty_result",
                sum(len(cols) for cols in stored_values.values()),
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

            if second is not None and _outcome_rank(second) >= _outcome_rank(first):
                chosen = second

    result = chosen["result"]

    if result is None:
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

    status = "SUCCESS" if int(result.get("row_count", 0) or 0) > 0 else "NOT_FOUND"
    return {
        "evidence": [
            {
                "source_type": "table_query",
                "workspace_id": state["workspace_id"],
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
                "content": {
                    **_table_content(
                        result,
                        schema_by_table,
                        sql=chosen["sql"],
                        operation=operation,
                    ),
                    "evidence_role": "query_result",
                },
                "result_status": status,
            }
        ],
        "status": status,
    }


def _planner_column_digest(
    state: AgentState,
    document_ids: list[int],
) -> dict[int, list[str]]:
    """Distinct column names per document, so the planner can see where data lives.

    Metadata only (no row sampling). Any failure returns {} and the planner
    simply runs without the digest.
    """
    if not document_ids:
        return {}
    try:
        schema = _resources(state).schema(document_ids=sorted(set(document_ids)))
    except Exception:
        logger.exception("planner column digest unavailable; continuing without it")
        return {}
    digest: dict[int, list[str]] = {}
    for table in schema:
        document_id = _coerce_int(table.get("document_id"))
        if document_id is None:
            continue
        bucket = digest.setdefault(document_id, [])
        for column in table.get("columns") or []:
            name = str(column)
            if name not in bucket:
                bucket.append(name)
    return digest


def _run_large_tables_via_sql(
    state: AgentState,
    budget: _RecoveryBudget,
    tables: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate tables that exceed the read cap in SQL instead of dumping rows.

    A table with more rows than query_tables can return is never read with
    SELECT *: the first rows would be mistaken for the whole population. Each
    document's large tables go through normal SQL generation (with one repair
    attempt from the shared recovery budget); values the table selector marked
    as computed are passed along as planner notes.
    """
    evidence: list[dict[str, Any]] = []
    success = False
    error = False
    operation = str(
        (state.get("query_understanding", {}) or {}).get("operation") or "answer"
    ).lower()

    notes = list(
        state.get("resolved_references", {}).get("contextual_notes", []) or []
    )
    computed = [str(item) for item in state.get("table_computed") or []]
    if computed:
        notes.append(
            "Derive these values in SQL over ALL rows: "
            + "; ".join(computed[:6])
            + "."
        )
    notes.append(
        "The table(s) hold more rows than can be returned. Compute counts, sums, "
        "averages, rates and rankings in SQL and return only the aggregated "
        "result, never raw rows."
    )

    by_document: dict[int | None, list[dict[str, Any]]] = {}
    for item in tables:
        by_document.setdefault(_coerce_int(item.get("document_id")), []).append(item)

    for group in by_document.values():
        if _past_deadline(state):
            break

        schema_by_table = {str(item["table_name"]): item for item in group}
        allowed = set(schema_by_table)

        chosen = _attempt_sql(
            state=state,
            schema_for_sql=group,
            entities=[],
            allowed=allowed,
            previous_attempt=None,
            planner_notes=notes,
        )

        if (
            chosen["result"] is None
            and not _past_deadline(state)
            and budget.try_consume()
        ):
            previous = {
                "sql": chosen["sql"],
                "parameters": chosen["parameters"],
                "error": chosen["error"],
            }
            try:
                second = _attempt_sql(
                    state=state,
                    schema_for_sql=group,
                    entities=[],
                    allowed=allowed,
                    previous_attempt=previous,
                    planner_notes=notes,
                )
            except AgentRuntimeError:
                raise
            except Exception:
                logger.exception("large-table SQL recovery failed")
                second = None
            if second is not None and _outcome_rank(second) >= _outcome_rank(chosen):
                chosen = second

        result = chosen["result"]
        if result is None:
            error = True
            logger.error("Large-table aggregation failed: %s", chosen["error"])
            evidence.append(
                {
                    "source_type": "table_query",
                    "workspace_id": state["workspace_id"],
                    "sources": [],
                    "content": {
                        "message": (
                            "A table too large to return in full could not be "
                            "aggregated, so no population-level figure is "
                            "available from it."
                        ),
                    },
                    "result_status": "RETRIEVAL_ERROR",
                }
            )
            continue

        status = "SUCCESS" if int(result.get("row_count", 0) or 0) > 0 else "NOT_FOUND"
        success = success or status == "SUCCESS"
        evidence.append(
            {
                "source_type": "table_query",
                "workspace_id": state["workspace_id"],
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
                "content": {
                    **_table_content(
                        result,
                        schema_by_table,
                        sql=chosen["sql"],
                        operation=operation,
                    ),
                    "evidence_role": "query_result",
                    "document_title": group[0].get("document_title"),
                },
                "result_status": status,
            }
        )

    return {"evidence": evidence, "success": success, "error": error}


agent_graph = build_agent_graph()