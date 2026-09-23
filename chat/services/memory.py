"""
Conversation memory service for QueryPilot.

Memory design:

    - Message records remain the source of truth.
    - Only the latest 10 messages are kept as full conversational context.
    - Older messages are compressed into one rolling summary.
    - `ConversationMemory.summarized_until_message` records exactly how
      far the rolling summary has been updated.

Runtime context will eventually be:

    rolling summary
        +
    latest 10 messages
        +
    current user query

Important:
    The rolling summary is conversational context only. It must never be
    treated as authoritative document/table evidence.

This module owns memory creation, rolling-summary updates, and memory
context assembly. The HTTP layer and planner do not implement memory
logic directly.
"""

from __future__ import annotations

from django.db import transaction

from chat.models import (
    Conversation,
    ConversationMemory,
    Message,
)
from llm.router import router


# ---------------------------------------------------------
# Memory policy
# ---------------------------------------------------------

# The planner receives the latest 10 actual messages verbatim.
RECENT_MESSAGE_LIMIT = 10

# Keep the generated summary compact so it does not become a second
# large conversation history.
MAX_SUMMARY_CHARS = 6000

# Bound the amount of existing summary and message content sent to the
# summarization model in one call.
MAX_EXISTING_SUMMARY_CHARS = 5000
MAX_MESSAGE_CHARS = 3000


MEMORY_SUMMARY_SYSTEM_PROMPT = """
You are QueryPilot's conversation-memory summarizer.

Your job is to maintain a compact rolling summary of older conversation
context so the assistant can understand later follow-up questions.

Rules:

- Preserve important conversational context.
- Preserve the user's goals, decisions, constraints, named entities,
  references, and unresolved questions when they matter for future turns.
- Preserve relationships between entities when needed to understand
  later follow-up questions.
- Remove greetings, repetition, filler, and irrelevant wording.
- Do not invent facts.
- Do not turn guesses into facts.
- Do not treat the summary as document or table evidence.
- Do not fabricate information that was not present in the supplied
  conversation.
- Keep the summary concise.
- Return ONLY the updated summary text.
"""


def get_or_create_memory(
    *,
    conversation: Conversation,
) -> ConversationMemory:
    """
    Return the single memory record associated with a conversation.

    A conversation has at most one ConversationMemory record because the
    model relationship is OneToOneField.

    Memory creation is lazy: a memory row is created only when the
    conversation actually enters the memory system.
    """

    memory, _ = ConversationMemory.objects.get_or_create(
        conversation=conversation,
    )

    return memory


def get_recent_messages(
    *,
    conversation: Conversation,
    limit: int = RECENT_MESSAGE_LIMIT,
) -> list[Message]:
    """
    Return the newest `limit` messages in chronological order.

    Messages themselves remain the canonical conversation history.
    Nothing is duplicated into ConversationMemory.
    """

    messages = list(
        Message.objects.filter(
            conversation=conversation,
        )
        .order_by("-created_at", "-id")[:limit]
    )

    messages.reverse()

    return messages


def get_unsummarized_messages(
    *,
    conversation: Conversation,
    memory: ConversationMemory,
) -> list[Message]:
    """
    Return messages that are old enough to belong in the rolling summary
    but are not represented by the current summary checkpoint.

    The newest 10 messages are deliberately excluded. They remain available
    as full conversational context at runtime.

    The `summarized_until_message` checkpoint prevents the service from
    repeatedly summarizing the same historical messages.
    """

    recent_messages = get_recent_messages(
        conversation=conversation,
        limit=RECENT_MESSAGE_LIMIT,
    )

    recent_ids = {
        message.id
        for message in recent_messages
    }

    queryset = Message.objects.filter(
        conversation=conversation,
    ).order_by(
        "created_at",
        "id",
    )

    if memory.summarized_until_message_id is not None:
        queryset = queryset.filter(
            id__gt=memory.summarized_until_message_id,
        )

    messages = [
        message
        for message in queryset
        if message.id not in recent_ids
    ]

    return messages


def _format_messages_for_summary(
    messages: list[Message],
) -> str:
    """
    Convert messages into a bounded text representation for the
    summarization model.

    Only message role and content are supplied. Internal database fields,
    metadata, citations, tool payloads, and raw agent state are excluded.
    """

    parts: list[str] = []

    for message in messages:
        content = (message.content or "").strip()

        if not content:
            continue

        content = content[:MAX_MESSAGE_CHARS]

        parts.append(
            f"{message.role}: {content}"
        )

    return "\n\n".join(parts)


def _generate_updated_summary(
    *,
    existing_summary: str,
    messages: list[Message],
) -> str:
    """
    Produce one new rolling summary from the previous summary plus newly
    expired conversation messages.

    The summarizer is intentionally separate from the main reasoning agent.
    It exists only to compress conversational context.
    """

    message_context = _format_messages_for_summary(
        messages
    )

    if not message_context:
        return existing_summary.strip()

    existing_summary = (
        existing_summary or ""
    ).strip()[:MAX_EXISTING_SUMMARY_CHARS]

    prompt_parts = [
        "CURRENT ROLLING SUMMARY:",
        existing_summary or "(empty)",
        "",
        "NEW CONVERSATION CONTEXT TO INCORPORATE:",
        message_context,
        "",
        "Create the updated rolling summary.",
    ]

    prompt = "\n".join(prompt_parts)

    response = router.generate(
        task="chitchat",
        messages=[
            {
                "role": "system",
                "content": MEMORY_SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=1000,
        include_reasoning=False,
    )

    try:
        summary = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )
    except (
        AttributeError,
        IndexError,
    ) as exc:
        # Raise instead of returning the OLD summary: the caller advances
        # the checkpoint on success, which would mark these messages as
        # summarized even though nothing was incorporated (silent loss of
        # context). update_rolling_summary() catches this, leaves the
        # checkpoint alone and retries on the next turn. The chat request
        # itself is unaffected.
        raise ValueError("Summarizer returned an invalid response.") from exc

    if not summary:
        raise ValueError("Summarizer returned an empty summary.")

    return summary[:MAX_SUMMARY_CHARS].strip()


def update_rolling_summary(
    *,
    conversation: Conversation,
) -> bool:
    """
    Update the rolling summary with newly expired messages.

    Policy:

        latest 10 messages
            -> remain as full history

        everything older than those 10 messages
            -> belongs in the rolling summary

    The summary is updated incrementally. Already summarized messages are
    not summarized again because `summarized_until_message` acts as the
    checkpoint.

    Returns:
        True  -> summary was updated
        False -> no update was necessary

    Failure of the optional summary model is intentionally non-fatal.
    Conversation execution must continue even when memory compression
    cannot be refreshed.
    """

    memory = get_or_create_memory(
        conversation=conversation,
    )

    messages = get_unsummarized_messages(
        conversation=conversation,
        memory=memory,
    )

    if not messages:
        return False

    try:
        new_summary = _generate_updated_summary(
            existing_summary=memory.summary,
            messages=messages,
        )
    except Exception:
        # Memory is an optimization for context retention, not a reason
        # to fail the user's actual chat request.
        return False

    last_summarized_message = messages[-1]

    with transaction.atomic():
        memory = (
            ConversationMemory.objects
            .select_for_update()
            .get(
                pk=memory.pk,
            )
        )

        # Re-check the checkpoint after acquiring the lock. Another
        # request may have updated the same memory record concurrently.
        current_checkpoint = memory.summarized_until_message_id
        if (
            current_checkpoint is not None
            and current_checkpoint >= last_summarized_message.id
        ):
            # Already at or past this point (a concurrent request got
            # there first). Never move the checkpoint backwards.
            return False

        memory.summary = new_summary
        memory.summarized_until_message = (
            last_summarized_message
        )
        memory.save(
            update_fields=[
                "summary",
                "summarized_until_message",
                "updated_at",
            ]
        )

    return True


def build_memory_context(
    *,
    conversation: Conversation,
) -> dict:
    """
    Build the bounded conversational-memory payload used by the chat
    execution layer.

    Returns a dictionary containing:

        summary:
            Rolling summary of older conversation context.

        recent_messages:
            The newest 10 real Message records converted to simple
            role/content dictionaries.

    No document evidence is included here.
    """

    memory = get_or_create_memory(
        conversation=conversation,
    )

    recent_messages = get_recent_messages(
        conversation=conversation,
        limit=RECENT_MESSAGE_LIMIT,
    )

    return {
        "summary": (
            memory.summary.strip()
            if memory.summary
            else ""
        ),
        "recent_messages": [
            {
                "role": message.role.lower(),
                "content": message.content,
            }
            for message in recent_messages
        ],
    }