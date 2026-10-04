"""
Canonical streaming execution entrypoint for QueryPilot.
Flow:
    load conversation settings
        -> load conversation memory (if enabled)
        -> persist user message
        -> deterministic scope boundary (2.5)
        -> optional LLM scope boundary (2.6, fail-closed: a Scope Guard
           failure NEVER falls through to the agent)
        -> start title generation in the background (first message only)
        -> Evidence Agent (single LangGraph run)
        -> source extraction + citation extraction/validation
        -> grounding evaluation (deterministic, fail-closed)
        -> apply citation setting
        -> normalized metadata
        -> persist assistant message
        -> rolling conversation memory update (background, if enabled)
        -> apply title (already generated while the agent ran)
        -> final SSE event
Document scope:
    if the conversation has attached documents, retrieval is scoped to
    exactly those; otherwise it falls back to every workspace document.
Follow-up resource context (memory enabled only):
    the previous turn's deterministically resolved documents/tables
    are carried into the next request's planner state so contextual
    references ("that table", "the previous one", "the second table")
    resolve against what was actually established last turn. Catalog
    facts only (ids, titles, table names) - never evidence payloads,
    SQL, or raw internal state.
Correction replay:
    an explicit correction ("that's wrong", "I asked you top 3 ...") is
    resolved ONCE here against the history that was loaded BEFORE the current
    message was persisted. The scope guard judges the replayed task, not the
    bare correction. The agent performs its own replay and carries the
    effective query and an empty history through the whole graph.
"""
from __future__ import annotations
import contextvars
import logging
import re
import threading
import time
from collections.abc import Iterator
from typing import Any
from django.conf import settings
from django.db import connections
from chat.models import ConversationSettings, Message
from chat.services.citations import extract_citations, find_marker_numbers
from chat.services.conversation import (
    add_message,
    get_attached_document_ids,
    get_conversation,
    get_first_user_message,
)
from chat.services.events import ChatEvent
from chat.services.memory import (
    build_memory_context,
    update_rolling_summary,
)
from chat.services.title import (
    generate_conversation_title,
    is_meaningful_for_title,
)
from documents.models import Document
from llm.router import router
from planner.services.agent_service import agent_graph
from planner.services.history import resolve_replay
from planner.services.scope_guard_services import ScopeGuard
from planner.services.state import AgentState
logger = logging.getLogger(__name__)
REQUEST_BUDGET_SECONDS = 90
FAILURE_MESSAGE = "I couldn't complete this request. Please try again."
OUT_OF_SCOPE_MESSAGE = (
    "I can't help with this request. I can answer questions within "
    "the scope of the documents available in this workspace."
)
SCOPE_GUARD_FAILURE_MESSAGE = (
    "I couldn't verify whether this request is within scope, so I "
    "stopped it for safety. No document or table access was performed. "
    "Please try again in a moment."
)
SCOPE_GUARD_LLM_ENABLED = getattr(
    settings,
    "SCOPE_GUARD_LLM_ENABLED",
    True,
)
MAX_THOUGHTS = 30
MAX_THOUGHT_MESSAGE_CHARS = 300
PLANNER_CONTEXT_KEY = "planner_context"
MAX_PRIOR_CONTEXT_MATCHES = 5
TITLE_JOIN_TIMEOUT_SECONDS = 3.0
def _scoped_document_ids(conversation) -> list[int]:
    """
    Documents this conversation may draw on: its attached documents when
    there are any, otherwise every document in the workspace (previous
    behavior for conversations with nothing attached).
    """
    attached = get_attached_document_ids(conversation=conversation)
    if attached:
        return attached
    return list(
        Document.objects.filter(
            workspace_id=conversation.workspace_id,
        ).values_list("id", flat=True)
    )
class _TitleJob:
    """
    Runs the title LLM call in a background thread while the agent works.
    The thread performs NO database access: everything it needs is read on
    the request thread before it starts, and the result is applied on the
    request thread afterwards.
    """
    def __init__(
        self,
        *,
        user_query: str,
        document_profiles: list[str],
    ) -> None:
        self.title: str | None = None
        # ContextVars do not automatically cross threads. Copy the active
        # usage context so title usage is included in the same request log.
        self._context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=self._context.run,
            args=(self._run, user_query, document_profiles),
            daemon=True,
        )
        self._thread.start()
    def _run(
        self,
        user_query: str,
        document_profiles: list[str],
    ) -> None:
        try:
            self.title = generate_conversation_title(
                user_query=user_query,
                document_profiles=document_profiles,
            )
        except Exception:
            logger.exception("Title generation failed")
    def result(self, timeout: float) -> str | None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            return None
        return self.title
def _start_title_job(
    *,
    conversation,
    user_message: str,
    document_ids: list[int],
) -> _TitleJob | None:
    if conversation.title != "New Chat":
        return None
    first_user = get_first_user_message(conversation=conversation)
    user_query_for_title = next(
        (
            query
            for query in (first_user, user_message)
            if is_meaningful_for_title(query)
        ),
        None,
    )
    if user_query_for_title is None:
        return None
    documents = (
        Document.objects.filter(
            id__in=document_ids,
            workspace_id=conversation.workspace_id,
        )
        .select_related("metadata")
        .order_by("uploaded_at")[:3]
    )
    document_profiles: list[str] = []
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if metadata is not None and metadata.summary:
            document_profiles.append(metadata.summary)
    return _TitleJob(
        user_query=user_query_for_title,
        document_profiles=document_profiles,
    )
def _apply_title(
    *,
    conversation,
    job: _TitleJob | None,
) -> None:
    if job is None or conversation.title != "New Chat":
        return
    title = job.result(TITLE_JOIN_TIMEOUT_SECONDS)
    if not title:
        return
    conversation.title = title
    conversation.save(update_fields=["title", "updated_at"])
def _update_memory_in_background(conversation_id: int) -> None:
    """
    The rolling summary is only needed by the NEXT request, so it must not
    delay this one. Runs in its own thread with its own DB connection.
    """
    def run() -> None:
        try:
            conversation = get_conversation(conversation_id)
            update_rolling_summary(conversation=conversation)
        except Exception:
            logger.exception(
                "Rolling memory update failed: conversation_id=%s",
                conversation_id,
            )
        finally:
            connections.close_all()
    threading.Thread(target=run, daemon=True).start()
def _track(
    thoughts: list[dict[str, Any]],
    event: ChatEvent,
) -> ChatEvent:
    thoughts.append(
        {
            "event": event.event,
            "message": event.message,
        }
    )
    return event
def _bound_thoughts(
    thoughts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for item in thoughts[-MAX_THOUGHTS:]:
        bounded.append(
            {
                "event": str(item.get("event", "")),
                "message": str(item.get("message", ""))[:MAX_THOUGHT_MESSAGE_CHARS],
            }
        )
    return bounded
def _compact_planner_context(final_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Compact, deterministic record of the documents/tables this request
    resolved BY NAME, for the NEXT request's contextual reference
    resolution. Catalog facts only. Entities are deliberately
    excluded: they are values, not sources.
    """
    references = (final_result.get("resolved_references", {}) or {}).get(
        "references", []
    )
    matches: list[dict[str, Any]] = []
    for item in references:
        if not isinstance(item, dict) or item.get("status") != "RESOLVED":
            continue
        match = item.get("match", {})
        if not isinstance(match, dict) or match.get("type") not in {
            "document",
            "table",
        }:
            continue
        matches.append(
            {
                "type": match.get("type"),
                "document_id": match.get("document_id"),
                "title": match.get("title") or match.get("document_title"),
                "table_name": match.get("table_name"),
            }
        )
        if len(matches) >= MAX_PRIOR_CONTEXT_MATCHES:
            break
    return matches
def _prior_tool_history(conversation) -> list[dict[str, Any]]:
    """
    Seed the agent's tool_history with the PREVIOUS turn's resolved
    documents/tables so deterministic contextual resolution can anchor
    "that table" / "the previous one" / "the second table" to what was
    actually established last turn. Only the single most recent
    assistant message is used. Fails open to [] (previous behavior).
    """
    try:
        last_metadata = (
            Message.objects.filter(
                conversation=conversation,
                role="ASSISTANT",
            )
            .order_by("-id")
            .values_list("metadata", flat=True)
            .first()
        ) or {}
        prior_matches = last_metadata.get(PLANNER_CONTEXT_KEY) or []
        if not isinstance(prior_matches, list) or not prior_matches:
            return []
        return [
            {
                "stage": "query_understanding",
                "resolution": {
                    "references": [
                        {"status": "RESOLVED", "match": match}
                        for match in prior_matches
                        if isinstance(match, dict)
                    ],
                },
            }
        ]
    except Exception:
        logger.exception(
            "Could not load prior planner context (conversation_id=%s); "
            "continuing without it",
            conversation.id,
        )
        return []
def _has_meaningful_conversation_context(
    conversation_history: list[dict[str, str]],
) -> bool:
    """
    Return True when prior user/assistant content is available for context.
    A system summary alone is not enough to suppress the lightweight
    chitchat path. At least one real user/assistant turn must be present.
    """
    if not conversation_history:
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("role") or "").lower() in {"user", "assistant"}
        and bool(str(item.get("content") or "").strip())
        for item in conversation_history
    )
def _out_of_scope_response(
    *,
    conversation,
    thoughts: list[dict[str, Any]],
) -> ChatEvent:
    metadata: dict[str, Any] = {
        "sources": [],
        "citations": [],
        "thoughts": _bound_thoughts(thoughts),
        "grounding_passed": None,
        "route": "OUT",
    }
    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=OUT_OF_SCOPE_MESSAGE,
        metadata=metadata,
    )
    return ChatEvent(
        event="complete",
        message="Request completed.",
        data={
            "answer": OUT_OF_SCOPE_MESSAGE,
            "error_code": "",
            "conversation_title": conversation.title,
            **metadata,
        },
    )
def _scope_guard_failure_response(
    *,
    conversation,
    thoughts: list[dict[str, Any]],
) -> ChatEvent:
    """
    Safe deterministic stop used when Scope Guard classification itself
    fails. Scope Guard is a routing/security boundary: a broken guard
    must never fall through into planner / retrieval / SQL.
    """
    metadata: dict[str, Any] = {
        "sources": [],
        "citations": [],
        "thoughts": _bound_thoughts(thoughts),
        "grounding_passed": None,
        "route": "ERROR",
    }
    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=SCOPE_GUARD_FAILURE_MESSAGE,
        metadata=metadata,
    )
    return ChatEvent(
        event="error",
        message="Scope Guard classification failed; request stopped safely.",
        data={
            "answer": SCOPE_GUARD_FAILURE_MESSAGE,
            "error_code": "SCOPE_GUARD_ERROR",
            "conversation_title": conversation.title,
            **metadata,
        },
    )
CHITCHAT_FALLBACK = "I'm here 😄 What's up?"
def _chitchat_response(
    *,
    conversation,
    user_message: str,
    thoughts: list[dict[str, Any]],
) -> ChatEvent:
    system_prompt = (
        "You are QueryPilot's casual conversational assistant. "
        "Respond naturally to lightweight social conversation. "
        "Keep it concise (one to three sentences), relaxed, and "
        "lightly witty when it fits. Do not mention workspace documents "
        "or scope unless the user asks about them. Do not invent personal "
        "experiences or claim to have feelings. Answer only the user's "
        "current conversational message."
    )
    try:
        response = router.generate(
            task="chitchat",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=0.6,
            max_tokens=120,
            include_reasoning=False,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = CHITCHAT_FALLBACK
    except Exception:
        logger.exception(
            "Chitchat response generation failed; using fallback (conversation_id=%s)",
            conversation.id,
        )
        answer = CHITCHAT_FALLBACK
    metadata: dict[str, Any] = {
        "sources": [],
        "citations": [],
        "thoughts": _bound_thoughts(thoughts),
        "grounding_passed": None,
        "route": "CHITCHAT",
    }
    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=answer,
        metadata=metadata,
    )
    return ChatEvent(
        event="complete",
        message="Request completed.",
        data={
            "answer": answer,
            "error_code": "",
            "conversation_title": conversation.title,
            **metadata,
        },
    )
_CITATION_MARKER_PATTERN = re.compile(r" ?\[(\d{1,3})\]")
def _sanitize_citation_markers(
    answer: str,
    *,
    valid_indices: set[int],
    keep_valid: bool,
) -> str:
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if keep_valid and number in valid_indices:
            return match.group(0)
        return ""
    return _CITATION_MARKER_PATTERN.sub(replace, answer).strip()
def _evaluate_grounding(
    *,
    answer: str,
    sources: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    route: str,
) -> bool | None:
    if route != "ANSWER":
        return None
    valid_indices = {
        int(source["index"])
        for source in sources
        if isinstance(source, dict) and source.get("index") is not None
    }
    marker_numbers = find_marker_numbers(answer)
    if not marker_numbers:
        return False
    if marker_numbers - valid_indices:
        return False
    cited = {
        int(entry["index"])
        for entry in citations
        if isinstance(entry, dict) and entry.get("index") is not None
    }
    return bool(cited) and cited == marker_numbers
def _execute_message_stream(
    *,
    conversation_id: int,
    user_message: str,
) -> Iterator[ChatEvent]:
    conversation = get_conversation(conversation_id)
    # Passive usage logging covers every LLM call made by this user message.
    conversation_settings, _ = ConversationSettings.objects.get_or_create(
        conversation=conversation,
    )
    memory_enabled = conversation_settings.memory_enabled
    citations_enabled = conversation_settings.citations_enabled
    thoughts: list[dict[str, Any]] = []
    conversation_history: list[dict[str, str]] = []
    if memory_enabled:
        memory_context = build_memory_context(conversation=conversation)
        summary = memory_context.get("summary", "").strip()
        if summary:
            conversation_history.append(
                {
                    "role": "system",
                    "kind": "summary",
                    "content": (
                        "CONVERSATION MEMORY\n"
                        "This is a summary of older conversation "
                        "context. Treat it as conversational context "
                        "only. It is not authoritative document or "
                        "table evidence.\n\n"
                        f"{summary}"
                    ),
                }
            )
        conversation_history.extend(memory_context.get("recent_messages", []))
    add_message(
        conversation=conversation,
        role="USER",
        content=user_message,
    )
    yield _track(
        thoughts,
        ChatEvent(
            event="query_received",
            message="Query received.",
        ),
    )
    if memory_enabled:
        yield _track(
            thoughts,
            ChatEvent(
                event="conversation_loaded",
                message="Conversation context loaded.",
            ),
        )
    replay = resolve_replay(user_message, conversation_history)
    scope_query = replay[1] if replay else user_message
    has_conversation_context = _has_meaningful_conversation_context(
        conversation_history
    )
    deterministic_route = ScopeGuard.deterministic_route(scope_query)
    if deterministic_route == "OUT":
        yield _out_of_scope_response(
            conversation=conversation,
            thoughts=thoughts,
        )
        return
    if deterministic_route == "CHITCHAT":
        yield _chitchat_response(
            conversation=conversation,
            user_message=user_message,
            thoughts=thoughts,
        )
        return
    document_ids = _scoped_document_ids(conversation)
    if SCOPE_GUARD_LLM_ENABLED:
        try:
            scope_decision = ScopeGuard().classify(
                query=scope_query,
                workspace_id=conversation.workspace_id,
                allowed_document_ids=document_ids,
                conversation_history=conversation_history,
            )
            if scope_decision.route == "OUT":
                logger.info(
                    "Scope Guard (LLM) routed request out of scope: %s",
                    scope_decision.reason,
                )
                yield _out_of_scope_response(
                    conversation=conversation,
                    thoughts=thoughts,
                )
                return
            if scope_decision.route == "CHITCHAT":
                if not has_conversation_context:
                    yield _chitchat_response(
                        conversation=conversation,
                        user_message=user_message,
                        thoughts=thoughts,
                    )
                    return
                logger.info(
                    "Scope Guard returned CHITCHAT, but prior conversation "
                    "context exists; continuing to planner for contextual "
                    "interpretation (conversation_id=%s).",
                    conversation.id,
                )
        except Exception:
            logger.exception(
                "Scope Guard LLM classification failed; failing safely, "
                "request stopped (conversation_id=%s)",
                conversation.id,
            )
            yield _scope_guard_failure_response(
                conversation=conversation,
                thoughts=thoughts,
            )
            return
    title_job: _TitleJob | None = None
    try:
        title_job = _start_title_job(
            conversation=conversation,
            user_message=user_message,
            document_ids=document_ids,
        )
    except Exception:
        logger.exception(
            "Could not start title generation: conversation_id=%s",
            conversation.id,
        )
    state: AgentState = {
        "query": user_message,
        "workspace_id": conversation.workspace_id,
        "conversation_history": conversation_history,
        "relevant_document_ids": document_ids,
        "evidence": [],
        "retrieval_confidence": 0.0,
        "known_facts": [],
        "unresolved_questions": [],
        "last_tool": "",
        "tool_history": (_prior_tool_history(conversation) if memory_enabled else []),
        "iteration_count": 0,
        "total_tool_calls": 0,
        "evidence_sufficient": False,
        "final_evidence": [],
        "answer": "",
        "grounding_passed": False,
        "route": "",
        "error": "",
        "error_code": "",
        "tool_failed": False,
        "_pending_action": {},
        "insufficient_reasons": [],
        "recovery_attempts": 0,
        "deadline_monotonic": time.monotonic() + REQUEST_BUDGET_SECONDS,
    }
    yield _track(
        thoughts,
        ChatEvent(
            event="agent_started",
            message="Starting evidence agent.",
        ),
    )
    final_result: dict[str, Any] | None = None
    failure_code = ""
    try:
        for mode, chunk in agent_graph.stream(
            state,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                if not isinstance(chunk, dict):
                    continue
                event_name = chunk.get("event")
                message = chunk.get("message", "")
                event_data = chunk.get("data")
                event_status = chunk.get("status")
                if not event_name:
                    continue
                payload: dict[str, Any] = {}
                if event_status is not None:
                    payload["status"] = event_status
                if event_data is not None:
                    payload["data"] = event_data
                yield _track(
                    thoughts,
                    ChatEvent(
                        event=event_name,
                        message=message,
                        data=payload or None,
                    ),
                )
            elif mode == "values":
                if isinstance(chunk, dict):
                    final_result = chunk
    except Exception:
        logger.exception(
            "Agent run failed: conversation_id=%s",
            conversation.id,
        )
        failure_code = "EXECUTION_ERROR"
    if final_result is None and not failure_code:
        failure_code = "NO_FINAL_STATE"
    if failure_code:
        add_message(
            conversation=conversation,
            role="ASSISTANT",
            content=FAILURE_MESSAGE,
            metadata={
                "sources": [],
                "citations": [],
                "thoughts": _bound_thoughts(thoughts),
                "grounding_passed": None,
                "route": "ERROR",
            },
        )
        yield ChatEvent(
            event="error",
            message=FAILURE_MESSAGE,
            data={"error_code": failure_code},
        )
        return
    answer = final_result.get("answer", "")
    final_route = final_result.get("route", "")
    if not answer:
        answer = "I could not generate an answer from the available evidence."
    final_evidence = final_result.get("final_evidence", [])
    sources, citations = extract_citations(
        answer=answer,
        evidence=final_evidence,
        workspace_id=conversation.workspace_id,
    )
    grounding_passed = _evaluate_grounding(
        answer=answer,
        sources=sources,
        citations=citations,
        route=final_route,
    )
    answer = _sanitize_citation_markers(
        answer,
        valid_indices={source["index"] for source in sources},
        keep_valid=citations_enabled,
    )
    if not citations_enabled:
        citations = []
    metadata: dict[str, Any] = {
        "sources": sources,
        "citations": citations,
        "thoughts": _bound_thoughts(thoughts),
        "grounding_passed": grounding_passed,
        "route": final_route,
    }
    persisted_metadata = dict(metadata)
    if memory_enabled and final_route not in {
        "ERROR",
        "NEEDS_CLARIFICATION",
    }:
        persisted_metadata[PLANNER_CONTEXT_KEY] = _compact_planner_context(final_result)
    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=answer,
        metadata=persisted_metadata,
    )
    if final_route == "ERROR":
        yield ChatEvent(
            event="error",
            message=answer,
            data={
                "answer": answer,
                "error_code": final_result.get("error_code", ""),
                "conversation_title": conversation.title,
                **metadata,
            },
        )
        return
    if memory_enabled:
        _update_memory_in_background(conversation.id)
    try:
        _apply_title(conversation=conversation, job=title_job)
    except Exception:
        logger.exception(
            "Applying title failed: conversation_id=%s",
            conversation.id,
        )
    yield ChatEvent(
        event="complete",
        message="Request completed.",
        data={
            "answer": answer,
            "error_code": final_result.get("error_code", ""),
            "conversation_title": conversation.title,
            **metadata,
        },
    )


def execute_message_stream(
    *,
    conversation_id: int,
    user_message: str,
) -> Iterator[ChatEvent]:
    """Execute one chat request inside a passive LLM usage context."""
    from llm.usage import track_usage

    with track_usage(f"conversation={conversation_id}"):
        yield from _execute_message_stream(
            conversation_id=conversation_id,
            user_message=user_message,
        )
