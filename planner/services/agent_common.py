"""Shared runtime pieces of the agent: errors, the timing wrappers around the tools,
deadline/scope helpers and the constants that planning, table execution and
document execution all use."""

from __future__ import annotations

import logging
import functools
import json
import re
import time
from jsonschema import Draft202012Validator
from typing import Any
from langgraph.config import get_stream_writer
from planner.prompts.agent import ACTION_SCHEMA
from planner.services.cross_document_join import run_cross_document_join
from planner.services.document_coverage import build_document_evidence
from planner.services.entity_resolution import resolve_entities_from_query, resolve_references
from planner.services.evidence import finalize_structured
from planner.services.source_scope import candidate_document_ids, effective_planning_query
from planner.services.schema_registry import WorkspaceResources, get_workspace_catalog, get_workspace_schema
from planner.services.sql_generation import generate_sql
from planner.services.state import AgentState
from planner.services.query_trace import current_trace
from planner.services.table_selection import select_tables, select_tables_detailed
from planner.tools.query_tables import query_tables
from planner.tools.search_evidence import fetch_document_chunks, search_evidence
from utils.timing import timed

logger = logging.getLogger("planner.services.agent_service")

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


MAX_TOOL_CALLS_PER_QUERY = 12


MAX_QUERY_EXECUTION_SECONDS = 60


_EXHAUSTIVE_CUE = re.compile(
    r"\b(?:all|every|each|entire|whole|complete|full|exhaustive(?:ly)?)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+"
    r"(?:units?|modules?|chapters?|sections?|topics?)\b",
    re.IGNORECASE,
)


MIN_RETRIEVAL_CONFIDENCE: float | None = None


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


def _table_scope(state: AgentState) -> set[str] | None:
    # Missing/None means selection was not applied. An explicit [] is a
    # meaningful deny-all result and must never widen back to the full schema.
    if "table_scope" not in state or state.get("table_scope") is None:
        return None
    return {str(name) for name in state.get("table_scope", [])}


def _evidence_table_names(item: Any) -> set[str]:
    if not isinstance(item, dict):
        return set()
    return {
        str(source.get("table_name"))
        for source in item.get("sources", []) or []
        if isinstance(source, dict) and source.get("table_name")
    }


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


# QP-GEN-1a: the planner's per-requirement queries (rules, exceptions, fees)
# used to be cut off after the raw question + effective question + one more.
_MAX_RETRIEVAL_QUERIES = 6


_MAX_RETRIEVAL_QUERIES_CAP = 10


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


_PROBE_MAX_COLUMNS = 8


_PROBE_MAX_DISTINCT = 25


_PROBE_MAX_VALUE_CHARS = 60
