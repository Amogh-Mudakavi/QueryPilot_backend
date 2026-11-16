"""Document-text stage of the agent: retrieval with coverage, row-fact search and
supplemental evidence."""

from __future__ import annotations

import logging
from typing import Any
from planner.services.intent import OVERVIEW_OPERATIONS, normalized_operation
from planner.services.source_scope import authorized_document_ids, dependency_discovery_query, effective_planning_query, requires_dependency_discovery
from planner.services.state import AgentState
from planner.services.agent_common import (
    MAX_TOOL_CALLS_PER_QUERY,
    MIN_RETRIEVAL_CONFIDENCE,
    _EXHAUSTIVE_CUE,
    _MAX_RETRIEVAL_QUERIES,
    _MAX_RETRIEVAL_QUERIES_CAP,
    _RecoveryBudget,
    _allowed_document_ids,
    _candidate_document_ids,
    _canonicalize_evidence,
    _coerce_int,
    _cross_document_search_ids,
    _doc_and_chunk_indexes,
    _document_ids_from_text_evidence,
    _loaded_chunk_map,
    _mark_success,
    _named_missing_source_references,
    _past_deadline,
    _plan_is_mixed,
    _resolved_document_ids,
    _resources,
    _row_fact_hit_already_loaded,
    _row_fact_phrases,
    _run_documents_with_coverage,
    _search_row_facts,
    search_evidence,
)

logger = logging.getLogger("planner.services.agent_service")

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


def _wants_exhaustive_read(state: AgentState) -> bool:
    """A list/enumerate request that asks for ALL items needs the whole source."""
    operation = normalized_operation(state.get("query_understanding"))
    if operation in OVERVIEW_OPERATIONS:
        return True
    if operation == "list":
        return bool(_EXHAUSTIVE_CUE.search(effective_planning_query(state)))
    return False


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


def documents_insufficient(
    chunk_count: int,
    confidence: float,
    min_retrieval_confidence: float | None,
) -> bool:
    """Whether a text retrieval result is too weak to guide another search."""
    if chunk_count <= 0:
        return True
    return (
        min_retrieval_confidence is not None
        and confidence < min_retrieval_confidence
    )


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
    primary_ids = _candidate_document_ids(state)
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
        all_ids = _candidate_document_ids(state)
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


def _structured_columns_for_log(structured: Any) -> list[list[str]]:  # QP-GEN-7b
    columns: list[list[str]] = []
    for item in structured or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict) and content.get("columns"):
            columns.append([str(c) for c in content.get("columns") or []][:25])
    return columns[:5]


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
