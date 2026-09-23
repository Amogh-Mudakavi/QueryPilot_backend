"""
Canonical streaming execution entrypoint for QueryPilot.

Flow:

    load conversation settings
        ↓
    load conversation memory (if enabled)
        ↓
    persist user message
        ↓
    deterministic scope boundary (2.5)
        (clearly out-of-scope requests are refused here and never
         reach the planner, retrieval, Chroma, or SQL)
        ↓
    optional LLM scope boundary (2.6, fail-open, toggleable via
        settings.SCOPE_GUARD_LLM_ENABLED)
        ↓
    Evidence Agent
        (query understanding → workspace/entity resolution →
         execution planning → tool dispatch → evidence validation →
         answer generation, all inside one LangGraph run)
        ↓
    source extraction (chat.services.sources.build_sources)
    citation extraction/validation (chat.services.citations)
        ↓
    grounding evaluation (deterministic, fail-closed, from validated
    citations; independent of the citations_enabled display setting)
        ↓
    apply citation setting
        ↓
    normalized metadata (sources + citations + thoughts +
    grounding_passed + route)
        ↓
    persist assistant message (content + metadata)
        ↓
    update rolling conversation memory (if enabled)
        ↓
    conversation title
        ↓
    final SSE event (same metadata, not a second structure)

Sources, citations, and safe progress "thoughts" are all built
deterministically from data the agent already produced. None of this
calls an LLM a second time and none of it changes the answer generation
pipeline itself.

Scope boundaries:

    2.5 Deterministic (no LLM): ScopeGuard.deterministic_route() uses
        pure string logic. Clearly out-of-scope requests (general
        programming, external knowledge, prompt-injection attempts
        asking for code) receive a fixed refusal and never reach the
        planner, retrieval, Chroma, or SQL.

    2.6 LLM (optional, fail-open): requests the deterministic boundary
        could not decide are classified by the scope-guard LLM (task
        "scope_guard"). Only OUT blocks; DOCUMENT, CHITCHAT, and
        CLARIFY proceed to the agent. If the scope LLM fails for any
        reason, the request proceeds to the agent unchanged. This is
        not a second answer generator: it never produces user-visible
        prose.

Conversation memory:

    rolling summary of older messages
        +
    latest 10 completed messages
        +
    current user query

When memory is disabled for a conversation, the rolling summary and
recent conversation history are not supplied to the planner, and the
rolling summary is not updated after the response.

The rolling summary is conversational context only. It is never treated
as authoritative document or table evidence.

Citation setting:

    citations_enabled=True
        → citation markers and validated citation metadata are retained.

    citations_enabled=False
        → citation markers are removed from the answer and the
          `citations` metadata list is empty. This affects display only;
          grounding_passed is still computed from the validated citations.

Document scope:

    Workspace
        ↓
    all documents belonging to that workspace
        ↓
    every conversation in that workspace can use them

Documents are workspace-level sources, not conversation-level sources.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from django.conf import settings

from documents.models import Document

from chat.models import ConversationSettings
from chat.services.citations import extract_citations, find_marker_numbers
from chat.services.conversation import (
    add_message,
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

from llm.router import router
from planner.services.agent import agent_graph
from planner.services.scope_guard import ScopeGuard
from planner.services.state import AgentState

logger = logging.getLogger(__name__)

# Whole-request time budget. Checked by the agent between stages; a single
# LLM call cannot be interrupted from here, so ALSO set a per-call timeout
# in the provider client.
REQUEST_BUDGET_SECONDS = 90

FAILURE_MESSAGE = "I couldn't complete this request. Please try again."

# Fixed response for clearly out-of-scope requests. Emitted before the
# agent runs: no planner, no retrieval, no Chroma, or SQL.
OUT_OF_SCOPE_MESSAGE = (
    "I can't help with this request. I can answer questions within "
    "the scope of the documents available in this workspace."
)

# Optional second scope stage: the scope-guard LLM classifies requests
# the deterministic boundary could not decide. Fail-open, toggleable.
# Set SCOPE_GUARD_LLM_ENABLED = False in settings to disable (the
# deterministic boundary keeps working either way).
SCOPE_GUARD_LLM_ENABLED = getattr(
    settings,
    "SCOPE_GUARD_LLM_ENABLED",
    True,
)

# Bounds for the safe, user-facing progress "thoughts" persisted with
# an assistant message. These are the same events already emitted on
# the SSE stream today -- nothing new is captured, and nothing internal
# (raw LangGraph state, tool payloads, exception text) is included.
# This is only a defensive cap on stored size.
MAX_THOUGHTS = 30
MAX_THOUGHT_MESSAGE_CHARS = 300


def _generate_title_if_needed(
    *,
    conversation,
    user_message: str,
) -> None:
    """
    Generate a conversation title the first time a meaningful
    conversation message is processed.

    Titles are generated using documents belonging to the entire
    workspace because workspace documents are shared by all
    conversations in that workspace.
    """
    if conversation.title != "New Chat":
        return

    # Use the conversation's first meaningful user message as the
    # basis for title generation. Fall back to the supplied message
    # when no earlier user message exists.
    first_user = get_first_user_message(conversation=conversation)

    # First MEANINGFUL user message: a leading "Hello" must not title the
    # chat. If nothing meaningful exists yet, leave "New Chat" and try again
    # on a later turn. Once a title exists this function returns above.
    user_query_for_title = next(
        (
            query
            for query in (first_user, user_message)
            if is_meaningful_for_title(query)
        ),
        None,
    )

    if user_query_for_title is None:
        return

    # Workspace-level documents: documents belong directly to the
    # workspace. There is no conversation-specific document boundary.
    documents = (
        Document.objects.filter(
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

    title = generate_conversation_title(
        user_query=user_query_for_title,
        document_profiles=document_profiles,
    )

    conversation.title = title
    conversation.save(update_fields=["title", "updated_at"])


def _track(
    thoughts: list[dict[str, Any]],
    event: ChatEvent,
) -> ChatEvent:
    """
    Record a safe, user-facing progress event as a "thought" entry.

    Only ever called on ChatEvent objects that are already being sent
    out over SSE (see the yield sites below) -- this never captures
    anything that wasn't already intentionally user-facing, and never
    captures raw internal state.
    """
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
    """
    Return a bounded copy of the user-facing progress events.

    This prevents the persisted metadata from growing without limit
    while preserving the most recent progress events.
    """
    bounded: list[dict[str, Any]] = []

    for item in thoughts[-MAX_THOUGHTS:]:
        bounded.append(
            {
                "event": str(item.get("event", "")),
                "message": str(item.get("message", ""))[
                    :MAX_THOUGHT_MESSAGE_CHARS
                ],
            }
        )

    return bounded


def _out_of_scope_response(
    *,
    conversation,
    thoughts: list[dict[str, Any]],
) -> ChatEvent:
    """
    Persist the fixed out-of-scope refusal and build the final event.

    Used by both scope stages (deterministic and LLM). No planner,
    no retrieval, no Chroma, no SQL, no answer model ever runs.
    """
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


CHITCHAT_FALLBACK = "I'm here 😄 What's up?"


def _chitchat_response(
    *,
    conversation,
    user_message: str,
    thoughts: list[dict[str, Any]],
) -> ChatEvent:
    """
    Generate a short conversational response and emit it through the
    same final `complete` event contract used by the normal chat stream.

    Chitchat intentionally bypasses the evidence agent, retrieval,
    citations, grounding, and structured document-answer generation.
    """
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

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if not answer:
            answer = CHITCHAT_FALLBACK

    except Exception:
        logger.exception(
            "Chitchat response generation failed; using fallback "
            "(conversation_id=%s)",
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
    """
    Make the answer text agree with the citation metadata.

    - Markers that do not correspond to a real source are always removed
      (they would render as dead "[17]" text).
    - Valid markers are removed only when citations are disabled.

    Only the marker (and one preceding space) is touched. Newlines,
    indentation, tables and code blocks are left alone.
    """

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
    """
    Deterministic, fail-closed grounding check for a normal ANSWER.

    Passes only when ALL of the following hold:
      - the answer text carries at least one marker;
      - every marker in the text maps to a real source (no invalid marker);
      - the validated citation list is exactly the set of markers in the
        text (no citation without a marker, no marker without a citation).

    Non-ANSWER routes (NOT_FOUND, clarification, ERROR, ...) keep their
    existing semantics and are not evaluated (None). The result does not
    depend on the citations_enabled display setting.
    """
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


def execute_message_stream(
    *,
    conversation_id: int,
    user_message: str,
) -> Iterator[ChatEvent]:
    """
    Execute one QueryPilot message and stream its progress/events.

    Conversation settings are loaded before execution.

    `memory_enabled=True`:
        The planner receives rolling summary + latest 10 completed
        messages + current user query through AgentState["query"].

    `memory_enabled=False`:
        No conversation summary or recent message history is supplied
        to the planner.

    `citations_enabled=True`:
        Generated citation markers and validated citation metadata are
        preserved.

    `citations_enabled=False`:
        Numeric citation markers are removed from the answer and the
        citation metadata list is emptied. grounding_passed is still
        evaluated before markers are removed.

    After a successful assistant response is persisted, rolling memory
    is refreshed only when memory is enabled.
    """
    conversation = get_conversation(conversation_id)

    # ---------------------------------------------------------
    # 0. Load conversation settings
    #
    # Settings are created automatically for new conversations.
    # get_or_create() also protects older conversations created before
    # ConversationSettings existed.
    # ---------------------------------------------------------
    conversation_settings, _ = ConversationSettings.objects.get_or_create(
        conversation=conversation,
    )

    memory_enabled = conversation_settings.memory_enabled
    citations_enabled = conversation_settings.citations_enabled

    # Safe, user-facing progress events collected as this generator
    # runs, so they can be persisted alongside the assistant message
    # once it's saved.
    thoughts: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # 1. Load conversation memory when enabled
    #
    # Memory is loaded BEFORE persisting the current user message so
    # the current query is not duplicated in planner history.
    #
    # When memory is disabled, conversation_history remains empty.
    # ---------------------------------------------------------
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

        conversation_history.extend(
            memory_context.get("recent_messages", [])
        )

    # ---------------------------------------------------------
    # 2. Persist current user message
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2.5 Deterministic scope boundary (no LLM, no retrieval)
    #
    # Clearly out-of-scope requests (general programming, external
    # knowledge, prompt-injection attempts asking for code) are
    # refused here. They never reach the planner, retrieval, Chroma,
    # or SQL. Pure string logic: no LLM call, no measurable latency.
    # ---------------------------------------------------------
    deterministic_route = ScopeGuard.deterministic_route(user_message)

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

    # ---------------------------------------------------------
    # Workspace-level document boundary
    #
    # Every conversation belonging to this workspace has access
    # to the same document set. Loaded before the optional LLM
    # scope stage because that stage needs the allowed ID set.
    # ---------------------------------------------------------
    document_ids = list(
        Document.objects.filter(
            workspace_id=conversation.workspace_id,
        ).values_list("id", flat=True)
    )

    # ---------------------------------------------------------
    # 2.6 LLM scope boundary (optional, fail-open)
    #
    # Requests the deterministic boundary could not decide are
    # classified by the scope-guard LLM (task "scope_guard", already
    # configured in llm.config.TASK_MODELS). OUT stops normal agent
    # execution, CHITCHAT gets a short conversational response, and
    # DOCUMENT/CLARIFY continue to the agent. If the scope LLM fails
    # for any reason, the request proceeds to the agent unchanged
    # (fail-open).
    # ---------------------------------------------------------
    if SCOPE_GUARD_LLM_ENABLED:
        try:
            scope_decision = ScopeGuard().classify(
                query=user_message,
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
                yield _chitchat_response(
                    conversation=conversation,
                    user_message=user_message,
                    thoughts=thoughts,
                )
                return

        except Exception:
            logger.exception(
                "Scope Guard LLM classification failed; continuing to "
                "the agent (conversation_id=%s)",
                conversation.id,
            )

    # ---------------------------------------------------------
    # 3. Evidence Agent
    #
    # Query understanding, workspace/entity resolution, execution
    # planning, tool dispatch, and answer generation all happen
    # inside a single LangGraph run (planner.services.agent).
    # ---------------------------------------------------------
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
        "tool_history": [],
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

    # ---------------------------------------------------------
    # 8. Ensure agent completed
    #
    # On failure the user's message is already persisted, so persist a
    # matching assistant message too. History then stays paired and a
    # reload shows what happened instead of an unanswered question.
    # ---------------------------------------------------------
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
        answer = (
            "I could not generate an answer "
            "from the available evidence."
        )

    final_evidence = final_result.get("final_evidence", [])

    # ---------------------------------------------------------
    # 8.1 Extract sources/citations deterministically
    #
    # workspace_id is passed into citation extraction so source
    # resolution is independently restricted to this workspace.
    # ---------------------------------------------------------
    sources, citations = extract_citations(
        answer=answer,
        evidence=final_evidence,
        workspace_id=conversation.workspace_id,
    )

    # Evaluate BEFORE sanitizing/stripping so the display setting cannot
    # change the grounding verdict.
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

    # ---------------------------------------------------------
    # 10. Persist assistant answer (+ metadata)
    # ---------------------------------------------------------
    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=answer,
        metadata=metadata,
    )

    # ---------------------------------------------------------
    # 10.05 Internally handled agent failure
    #
    # The agent catches most failures and returns a normal final state
    # with route == "ERROR". The assistant message is persisted above so
    # history stays paired, but the client must be told the request
    # FAILED, not that it completed. Skip memory/title work on failures.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 10.1 Update rolling conversation memory
    #
    # Only update memory when the conversation setting allows memory.
    # Memory compression is separate from the evidence/answer pipeline.
    # A memory update failure must never invalidate a successful answer.
    # ---------------------------------------------------------
    if memory_enabled:
        try:
            update_rolling_summary(conversation=conversation)
        except Exception:
            logger.exception(
                "Rolling memory update failed: conversation_id=%s",
                conversation.id,
            )

    # ---------------------------------------------------------
    # 11. Conversation title
    # ---------------------------------------------------------
    try:
        _generate_title_if_needed(
            conversation=conversation,
            user_message=user_message,
        )
    except Exception:
        logger.exception(
            "Title generation failed: conversation_id=%s",
            conversation.id,
        )

    # ---------------------------------------------------------
    # 12. Final SSE event
    # ---------------------------------------------------------
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