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
import re
import time
from typing import Any
from langgraph.graph import END, START, StateGraph
from planner.services.answer import _narrow_spreadsheet_tables_v3, _parse_answer_response, generate_answer_stream
from planner.services.cross_document_join import build_joined_view
from planner.services.coverage import document_coverage_note, prior_conclusions_note, requirement_coverage_note
from planner.services.evidence import assign_citation_indexes, reconcile_coverage_notes
from planner.services.source_scope import authorized_document_ids, effective_planning_query, requires_dependency_discovery, requires_document_evidence, requests_workspace_wide_cross_document_search
from planner.services.product_knowledge import get_product_knowledge
from planner.services.schema_registry import WorkspaceResources
from planner.services.state import AgentState
from planner.services.query_trace import QueryTrace, current_trace
from planner.tools.conversation_memory import conversation_memory
from planner.tools.source_metadata import source_metadata
from utils.timing import timed
from planner.services.agent_common import (
    AgentRuntimeError,
    MAX_QUERY_EXECUTION_SECONDS,
    MAX_TOOL_CALLS_PER_QUERY,
    _RecoveryBudget,
    _allowed_document_ids,
    _candidate_document_ids,
    _canonicalize_evidence,
    _coerce_int,
    _cross_document_search_ids,
    _document_ids_from_text_evidence,
    _emit_event,
    _evidence_table_names,
    _is_generic_catalog_reference,
    _is_rule_dependent_mixed_request,
    _loaded_chunk_map,
    _mark_success,
    _named_missing_source_references,
    _past_deadline,
    _query_requires_population_context,
    _resolved_document_ids,
    _resolved_entity_values,
    _resolved_table_names,
    _resources,
    _row_fact_phrases,
    _row_signature,
    _search_row_facts,
    _table_scope,
    finalize_structured,
    get_workspace_catalog,
    query_tables,
    run_cross_document_join,
    select_tables_detailed,
)
from planner.services.document_executor import (
    _run_documents,
    _supplement_dependency_evidence,
)
from planner.services.planning import (
    agent_node,
)
from planner.services.table_executor import (
    _run_table,
)

logger = logging.getLogger(__name__)

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


MAX_RECOVERY_ATTEMPTS = 2


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


def _check_deadline(state: AgentState, stage: str) -> None:
    if _past_deadline(state):
        logger.warning("Request budget exceeded before stage=%s", stage)
        raise AgentRuntimeError(
            "Request time budget exceeded.",
            code="REQUEST_TIMEOUT",
        )


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
            _coerce_int(value) for value in _candidate_document_ids(state)
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


def _evidence_rows(item: Any) -> list[dict[str, Any]]:
    content = item.get("content") if isinstance(item, dict) else None
    rows = content.get("rows") if isinstance(content, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


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
        names = _evidence_table_names(item)
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
        source_document_ids = _candidate_document_ids(state)
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


agent_graph = build_agent_graph()
