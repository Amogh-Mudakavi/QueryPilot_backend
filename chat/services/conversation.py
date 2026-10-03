"""Services Conversation module for QueryPilot."""

import logging

from django.db import transaction

from chat.models import (
    Conversation,
    ConversationDocument,
    ConversationSettings,
    Message,
)
from documents.models import Document

logger = logging.getLogger(__name__)

MAX_ATTACHED_DOCUMENTS = 3


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
    return Conversation.objects.get(id=conversation_id)


def list_conversations(
    workspace,
):
    return Conversation.objects.filter(workspace=workspace)


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
        - Attaching an already-attached document is a no-op (it is
          returned even when the conversation is already at the limit).

    The count check and the insert run in one transaction, with the
    conversation row locked, so concurrent attaches cannot exceed the limit.
    """

    if getattr(document, "workspace_id", None) != conversation.workspace_id:
        raise ValueError(
            "Cannot attach a document from a different workspace to this conversation."
        )

    with transaction.atomic():
        Conversation.objects.select_for_update().get(pk=conversation.pk)

        existing = ConversationDocument.objects.filter(
            conversation=conversation,
            document=document,
        ).first()

        if existing is not None:
            return existing

        attached_count = ConversationDocument.objects.filter(
            conversation=conversation
        ).count()

        if attached_count >= MAX_ATTACHED_DOCUMENTS:
            raise ValueError(f"Maximum of {MAX_ATTACHED_DOCUMENTS} documents allowed.")

        logger.info(
            "attach_document: conversation_id=%s document_id=%s workspace_id=%s",
            conversation.id,
            getattr(document, "id", None),
            getattr(document, "workspace_id", None),
        )

        link = ConversationDocument.objects.create(
            conversation=conversation,
            document=document,
        )

    return link


def get_attached_document_ids(
    *,
    conversation: Conversation,
) -> list[int]:
    """
    IDs of the documents explicitly attached to this conversation
    (same-workspace only, stable order). Empty when nothing is attached.
    """
    attached = ConversationDocument.objects.filter(
        conversation=conversation
    ).values_list("document_id", flat=True)

    return list(
        Document.objects.filter(
            id__in=list(attached),
            workspace_id=conversation.workspace_id,
        )
        .order_by("id")
        .values_list("id", flat=True)
    )


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
