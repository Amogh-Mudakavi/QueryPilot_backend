"""Structured-data stage of the agent: table scope, SQL generation and execution,
recovery, cross-document table runs and the evidence built from the results."""

from __future__ import annotations

import logging
import json
import re
from typing import Any
from llm.exceptions import InvalidResponseError
from planner.services.source_scope import effective_planning_query
from planner.services.schema_registry import WorkspaceResources
from planner.services.state import AgentState
from planner.services.value_grounding import ground_question
from planner.tools.query_tables import MAX_RESULTS as QUERY_ROW_CAP
from planner.services.agent_common import (
    AgentRuntimeError,
    _INFRASTRUCTURE_ERRORS,
    _PROBE_MAX_COLUMNS,
    _PROBE_MAX_DISTINCT,
    _PROBE_MAX_VALUE_CHARS,
    _RecoveryBudget,
    _SQL_DECLINED,
    _SQL_OUTPUT_FAILURE,
    _allowed_document_ids,
    _candidate_document_ids,
    _coerce_int,
    _is_rule_dependent_mixed_request,
    _named_missing_source_references,
    _past_deadline,
    _plan_is_mixed,
    _query_requires_population_context,
    _resolved_entity_values,
    _resolved_table_names,
    _resources,
    _table_scope,
    generate_sql,
    query_tables,
)

logger = logging.getLogger("planner.services.agent_service")

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


def _in_table_scope(
    state: AgentState,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scope = _table_scope(state)
    if scope is None:
        return items
    return [item for item in items if str(item.get("table_name")) in scope]


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


def _is_infrastructure_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _INFRASTRUCTURE_ERRORS)


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


def _family_coverage(
    all_tables: list[dict[str, Any]],
    read_names: set[str],
) -> dict[str, Any]:
    """Say so when a result was read from only part of a split dataset.

    Tables of one document with the same columns are parts of one dataset
    (one table per page, sheet or period). A result read from some of them is
    not the whole population, even when it contains every row of the tables
    it did read.
    """
    def signature(item: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            re.sub(r"[^a-z0-9]+", "", str(c).casefold())
            for c in (item.get("columns") or [])
        )

    try:
        read = [t for t in all_tables if str(t.get("table_name")) in read_names]
        unread: dict[str, int] = {}
        for table in read:
            sig = signature(table)
            if not sig:
                continue
            document_id = _coerce_int(table.get("document_id"))
            for other in all_tables:
                name = str(other.get("table_name"))
                if name in read_names or name in unread:
                    continue
                if (
                    _coerce_int(other.get("document_id")) == document_id
                    and signature(other) == sig
                ):
                    unread[name] = int(other.get("row_count") or 0)
        if not unread:
            return {}
        read_rows = sum(int(t.get("row_count") or 0) for t in read)
        return {
            "coverage_partial": True,
            "coverage_note": (
                f"Read {len(read)} table(s) ({read_rows} rows) but {len(unread)} more "
                f"table(s) of the same document with the same columns "
                f"({sum(unread.values())} rows) were NOT read. They are parts of "
                "the same dataset, so this result is not the complete list or "
                "count for the source."
            ),
        }
    except Exception:
        logger.exception("family coverage check failed; continuing without it")
        return {}


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

    # Where do the question's own words really occur in the stored data? No
    # LLM call; a failure just means SQL generation runs without the hints.
    try:
        value_matches = ground_question(
            workspace_id=state["workspace_id"],
            query=effective_planning_query(state),
            schema=schema_for_sql,
        )
    except Exception:
        logger.exception("value grounding failed; generating SQL without it")
        value_matches = []
    if value_matches:
        logger.info(
            "value grounding: %s",
            [
                f"{m['word']}->{m['table']}.{m['column']}({m['rows_matching']})"
                for m in value_matches
            ],
        )

    try:
        decision = generate_sql(
            query=effective_planning_query(state),
            schema=schema_for_sql,
            resolved_entities=entities,
            previous_attempt=previous_attempt,
            sql_required=True,
            planner_notes=planner_notes or None,
            value_matches=value_matches or None,
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
    resolved_document_ids = _candidate_document_ids(state)
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
                    **_family_coverage(
                        schema, {str(n) for n in result.get("tables_used", [])}
                    ),
                },
                "result_status": status,
            }
        ],
        "status": status,
    }


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
