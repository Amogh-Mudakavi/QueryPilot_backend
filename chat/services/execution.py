"""
chat/services/execution.py

Canonical streaming execution entrypoint for QueryPilot.

Flow:

    user message
        ↓
    persist user message
        ↓
    Scope Guard
        ↓
    Evidence Agent
        ↓
    evidence collection
        ↓
    answer generation
        ↓
    source extraction
        ↓
    persist assistant message
        ↓
    conversation title
        ↓
    final SSE event
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from chat.models import (
    ConversationDocument,
    Document,
)
from chat.services.conversation import (
    add_message,
    get_conversation,
    get_recent_history,
    get_first_user_message,
)
from chat.services.events import ChatEvent
from chat.services.title import (
    generate_conversation_title,
)

from planner.services.agent import agent_graph
from planner.services.state import AgentState


def _generate_title_if_needed(
    *,
    conversation,
    user_message: str,
) -> None:
    if conversation.title != "New Chat":
        return

    # Use the conversation's first meaningful user message as the
    # basis for title generation per design. Fall back to the
    # supplied `user_message` if no earlier user message exists.
    first_user = get_first_user_message(
        conversation=conversation
    )

    user_query_for_title = first_user or user_message

    document_links = list(
        ConversationDocument.objects.filter(
            conversation=conversation,
        ).select_related("document")
    )

    document_profiles: list[str] = []

    for link in document_links[:3]:
        metadata = getattr(
            link.document,
            "metadata",
            None,
        )

        if (
            metadata is not None
            and metadata.summary
        ):
            document_profiles.append(
                metadata.summary
            )

    title = generate_conversation_title(
        user_query=user_query_for_title,
        document_profiles=document_profiles,
    )

    conversation.title = title

    conversation.save(
        update_fields=[
            "title",
            "updated_at",
        ]
    )


def _build_sources(
    evidence: list[Any],
) -> list[dict[str, Any]]:
    if not evidence:
        return []

    document_ids: set[int] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        metadata = item.get(
            "metadata",
            {},
        )

        # Normal single-document evidence items may include a
        # top-level `document_id` or embed it under `metadata`.
        document_id = (
            item.get("document_id")
            or metadata.get("document_id")
        )

        # Table-query evidence embeds a `sources` list with
        # per-table `document_id` values. Collect those too so
        # we can later resolve document names.
        sources_list = item.get("sources") or metadata.get("sources")
        if isinstance(sources_list, list):
            for src in sources_list:
                try:
                    did = src.get("document_id")
                except Exception:
                    did = None

                if did is None:
                    continue

                try:
                    document_ids.add(int(did))
                except (TypeError, ValueError):
                    continue

        if document_id is None:
            continue

        try:
            document_ids.add(
                int(document_id)
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    document_names: dict[int, str] = {}

    if document_ids:
        documents = Document.objects.filter(
            id__in=document_ids,
        ).only(
            "id",
            "title",
        )

        document_names = {
            document.id: document.title
            for document in documents
        }

    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        metadata = item.get(
            "metadata",
            {},
        )

        document_id = (
            item.get("document_id")
            or metadata.get("document_id")
        )

        if document_id is None:
            continue

        try:
            document_id = int(document_id)
        except (
            TypeError,
            ValueError,
        ):
            continue

        chunk_index = (
            item.get("chunk_index")
            or metadata.get("chunk_index")
        )

        page_start = (
            item.get("page_start")
            or metadata.get("page_start")
        )

        page_end = (
            item.get("page_end")
            or metadata.get("page_end")
        )

        heading = (
            item.get("heading")
            or metadata.get("heading")
        )

        source_type = (
            item.get("source_type")
            or metadata.get("type")
            or "document"
        )

        table_name = (
            item.get("table_name")
            or metadata.get("table_name")
        )

        source_key = (
            document_id,
            chunk_index,
            page_start,
            page_end,
            heading,
            table_name,
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        source: dict[str, Any] = {
            "document_id": document_id,
            "document_name": document_names.get(
                document_id,
                item.get("document_name") or "",
            ),
            "source_type": source_type,
        }

        if chunk_index is not None:
            source["chunk_index"] = chunk_index

        if page_start is not None:
            source["page_start"] = page_start

        if page_end is not None:
            source["page_end"] = page_end

        if heading:
            source["heading"] = heading

        if table_name:
            source["table_name"] = table_name

        sources.append(source)

    # Handle table-query style evidence which embeds a `sources`
    # list with per-table document references. These items might
    # not appear with a top-level `document_id` so handle them
    # explicitly to ensure tables are reported as sources.
    for item in evidence:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata", {})
        sources_list = item.get("sources") or metadata.get("sources")

        if not isinstance(sources_list, list):
            continue

        for src in sources_list:
            try:
                doc_id = src.get("document_id")
            except Exception:
                doc_id = None

            if doc_id is None:
                continue

            try:
                doc_id = int(doc_id)
            except (TypeError, ValueError):
                continue

            table_name = src.get("table_name")

            source_key = (
                doc_id,
                None,
                None,
                None,
                None,
                table_name,
            )

            if source_key in seen:
                continue

            seen.add(source_key)

            source: dict[str, Any] = {
                "document_id": doc_id,
                "document_name": document_names.get(doc_id, ""),
                "source_type": "table",
            }

            if table_name:
                source["table_name"] = table_name

            sources.append(source)

    # Deduplicate table sources so the final `sources` list does
    # not contain repeated entries for the same document+table.
    final_sources: list[dict[str, Any]] = []
    seen_table_pairs: set[tuple[int, str]] = set()

    for s in sources:
        table = s.get("table_name")
        if table:
            key = (s.get("document_id"), table)
            if key in seen_table_pairs:
                continue
            seen_table_pairs.add(key)
            final_sources.append(s)
        else:
            final_sources.append(s)

    return final_sources


def execute_message_stream(
    *,
    conversation_id: int,
    user_message: str,
) -> Iterator[ChatEvent]:

    from llm.router import router
    from planner.services.scope_guard import ScopeGuard

    conversation = get_conversation(
        conversation_id
    )

    # ---------------------------------------------------------
    # 1. Persist user message
    # ---------------------------------------------------------

    add_message(
        conversation=conversation,
        role="USER",
        content=user_message,
    )

    yield ChatEvent(
        event="query_received",
        message="Query received.",
    )

    yield ChatEvent(
        event="conversation_loaded",
        message="Conversation context loaded.",
    )

    # ---------------------------------------------------------
    # 2. Load context
    # ---------------------------------------------------------

    conversation_history = get_recent_history(
        conversation=conversation,
        limit=20,
    )

    document_ids = list(
        ConversationDocument.objects.filter(
            conversation=conversation,
        ).values_list(
            "document_id",
            flat=True,
        )
    )

    # ---------------------------------------------------------
    # 3. Scope Guard
    # ---------------------------------------------------------

    yield ChatEvent(
        event="scope_check",
        message="Checking query scope.",
        data={
            "status": "started",
        },
    )

    guard = ScopeGuard()

    scope_result = guard.classify(
    query=user_message,
    workspace_id=conversation.workspace_id,
    allowed_document_ids=document_ids,
    conversation_history=conversation_history,
    )

    route = scope_result.route

    relevant_document_ids = (
        scope_result.relevant_document_ids
    )

    if (
        route == "DOCUMENT"
        and not relevant_document_ids
    ):
        relevant_document_ids = document_ids

    yield ChatEvent(
        event="scope_check",
        message="Query scope checked.",
        data={
            "status": "completed",
            "route": route,
            "relevant_document_ids": (
                relevant_document_ids
            ),
        },
    )

    # ---------------------------------------------------------
    # 4. OUT
    # ---------------------------------------------------------

    if route == "OUT":
        answer = (
            "I can only answer questions based on "
            "the documents available in this workspace."
        )

        add_message(
            conversation=conversation,
            role="ASSISTANT",
            content=answer,
        )

        yield ChatEvent(
            event="complete",
            message="Request completed.",
            data={
                "answer": answer,
                "route": route,
                "sources": [],
            },
        )

        return

    # ---------------------------------------------------------
    # 5. CLARIFY
    # ---------------------------------------------------------

    if route == "CLARIFY":
        answer = (
            "Could you clarify what you'd like to know "
            "from the uploaded documents?"
        )

        add_message(
            conversation=conversation,
            role="ASSISTANT",
            content=answer,
        )

        yield ChatEvent(
            event="complete",
            message="Clarification required.",
            data={
                "answer": answer,
                "route": route,
                "sources": [],
            },
        )

        return

    # ---------------------------------------------------------
    # 6. CHITCHAT
    # ---------------------------------------------------------

    if route == "CHITCHAT":

        yield ChatEvent(
            event="generating_answer",
            message="Generating response.",
            data={
                "status": "started",
            },
        )

        response = router.generate(
            task="chitchat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are QueryPilot's conversational "
                        "assistant. Keep the response concise."
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        yield ChatEvent(
            event="generating_answer",
            message="Response generated.",
            data={
                "status": "completed",
            },
        )

        add_message(
            conversation=conversation,
            role="ASSISTANT",
            content=answer,
        )

        _generate_title_if_needed(
            conversation=conversation,
            user_message=user_message,
        )

        yield ChatEvent(
            event="complete",
            message="Request completed.",
            data={
                "answer": answer,
                "route": route,
                "sources": [],
                "conversation_title": (
                    conversation.title
                ),
            },
        )

        return

    # ---------------------------------------------------------
    # 7. DOCUMENT → Evidence Agent
    # ---------------------------------------------------------

    state: AgentState = {
        "query": user_message,
        "workspace_id": conversation.workspace_id,
        "conversation_history": conversation_history,
        "relevant_document_ids": (
            relevant_document_ids
        ),
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
        "route": route,
        "error": "",
        "tool_failed": False,
        "_pending_action": {},
    }

    yield ChatEvent(
        event="agent_started",
        message="Starting evidence agent.",
    )

    final_result: dict[str, Any] | None = None

    for mode, chunk in agent_graph.stream(
        state,
        stream_mode=[
            "custom",
            "values",
        ],
    ):

        if mode == "custom":

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            event_name = chunk.get(
                "event"
            )

            message = chunk.get(
                "message",
                "",
            )

            event_data = chunk.get(
                "data"
            )

            event_status = chunk.get(
                "status"
            )

            if not event_name:
                continue

            payload: dict[str, Any] = {}

            if event_status is not None:
                payload["status"] = (
                    event_status
                )

            if event_data is not None:
                payload["data"] = (
                    event_data
                )

            yield ChatEvent(
                event=event_name,
                message=message,
                data=payload or None,
            )

        elif mode == "values":

            if isinstance(
                chunk,
                dict,
            ):
                final_result = chunk

    # ---------------------------------------------------------
    # 8. Ensure agent completed
    # ---------------------------------------------------------

    if final_result is None:
        raise RuntimeError(
            "QueryPilot agent produced no final state."
        )

    answer = final_result.get(
        "answer",
        "",
    )

    if not answer:
        answer = (
            "I could not generate an answer "
            "from the available evidence."
        )

    final_evidence = final_result.get(
        "final_evidence",
        [],
    )

    sources = _build_sources(
        final_evidence
    )

    # ---------------------------------------------------------
    # 9. Persist assistant answer
    # ---------------------------------------------------------

    add_message(
        conversation=conversation,
        role="ASSISTANT",
        content=answer,
    )

    # ---------------------------------------------------------
    # 10. Conversation title
    # ---------------------------------------------------------

    _generate_title_if_needed(
        conversation=conversation,
        user_message=user_message,
    )

    # ---------------------------------------------------------
    # 11. Final SSE event
    # ---------------------------------------------------------

    yield ChatEvent(
        event="complete",
        message="Request completed.",
        data={
            "answer": answer,
            "route": route,
            "grounding_passed": final_result.get(
                "grounding_passed"
            ),
            "sources": sources,
            "conversation_title": (
                conversation.title
            ),
        },
    )