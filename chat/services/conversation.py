import logging

from django.db import transaction

from chat.models import (
    Conversation,
    Message,
    ConversationDocument,
    ConversationSettings,
)

from documents.models import Document


def create_conversation(
    *,
    workspace,
    user,
    title: str = "New Chat",
) -> Conversation:
    """
    Create a conversation together with its default settings.

    Every conversation gets exactly one ConversationSettings row.
    The operation is atomic so we never end up with a conversation
    without its corresponding settings.
    """
    with transaction.atomic():
        conversation = Conversation.objects.create(
            workspace=workspace,
            created_by=user,
            title=title,
        )

        ConversationSettings.objects.create(
            conversation=conversation,
        )

    return conversation


def get_conversation(
    conversation_id: int,
) -> Conversation:
    return Conversation.objects.get(
        id=conversation_id
    )


def list_conversations(
    workspace,
):
    return Conversation.objects.filter(
        workspace=workspace
    )


def add_message(
    *,
    conversation: Conversation,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> Message:
    """
    Persist a message and update the parent conversation timestamp.

    Message rows remain the source of truth for conversation history.
    """
    message = Message.objects.create(
        conversation=conversation,
        role=role,
        content=content,
        metadata=metadata or {},
    )

    # Message activity should update the parent conversation so list
    # ordering and Today/Yesterday grouping remain correct after the
    # conversation already has a title.
    conversation.save(update_fields=["updated_at"])

    return message


def attach_document(
    *,
    conversation: Conversation,
    document: Document,
) -> ConversationDocument:
    """
    Attach a document to a conversation.

    MVP rules:
        - Maximum 3 documents per conversation.
        - Document must belong to the same workspace.
        - Duplicate attachments are ignored through get_or_create().
    """
    attached_count = (
        ConversationDocument.objects.filter(
            conversation=conversation
        ).count()
    )

    if attached_count >= 3:
        raise ValueError(
            "Maximum of 3 documents allowed."
        )

    # Enforce that attached documents must belong to the same workspace.
    if getattr(document, "workspace_id", None) != conversation.workspace_id:
        raise ValueError(
            "Cannot attach a document from a different workspace to this conversation."
        )

    logger = logging.getLogger(__name__)
    logger.info(
        "attach_document: conversation_id=%s document_id=%s workspace_id=%s",
        conversation.id,
        getattr(document, "id", None),
        getattr(document, "workspace_id", None),
    )

    link, _ = ConversationDocument.objects.get_or_create(
        conversation=conversation,
        document=document,
    )

    return link


def get_recent_history(
    *,
    conversation: Conversation,
    limit: int = 10,
) -> list[dict]:
    """
    Return the most recent messages in chronological order.

    This helper is retained for places that still need direct recent
    history retrieval. Conversation memory uses the same 10-message
    policy at runtime.
    """
    messages = (
        Message.objects.filter(
            conversation=conversation
        )
        .order_by("-created_at")[:limit]
    )

    messages = reversed(list(messages))

    return [
        {
            "role": message.role.lower(),
            "content": message.content,
        }
        for message in messages
    ]


def get_first_user_message(
    *,
    conversation: Conversation,
) -> str:
    """
    Return the first (chronologically earliest) user message
    content for this conversation, or an empty string if none.
    """
    message = (
        Message.objects.filter(
            conversation=conversation,
            role="USER",
        )
        .order_by("created_at")
        .first()
    )

    if not message:
        return ""

    return (message.content or "").strip()