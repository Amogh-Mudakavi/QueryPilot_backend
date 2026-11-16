"""Planning stage of the agent: turns the user's question and the conversation into
an execution plan (planner call, reference and entity resolution, capability choice)."""

from __future__ import annotations

import logging
import json
import re
import time
from typing import Any
from llm.router import router
from planner.prompts.agent import PLANNER_RESPONSE_FORMAT, SYSTEM_PROMPT
from planner.services.history import bound_history_for_planner, resolve_replay
from planner.services.intent import build_execution_plan
from planner.services.source_scope import contextual_planning_query
from planner.services.state import AgentState
from planner.services.requirements import ensure_query_fallback_requirement, normalize_requirements
from planner.services.query_trace import current_trace
from planner.services.agent_common import (
    AgentRuntimeError,
    CHITCHAT_REPLIES,
    MAX_QUERY_EXECUTION_SECONDS,
    MAX_SUPPORTING_SOURCES,
    _FILE_EXTENSION_RE,
    _PLANNER_VALIDATOR,
    _allowed_document_ids,
    _coerce_int,
    _emit_event,
    _is_generic_catalog_reference,
    _parse_contextual_reference,
    _resources,
    resolve_entities_from_query,
    resolve_references,
)

logger = logging.getLogger("planner.services.agent_service")

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
