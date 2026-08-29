from chat.models import (
    Conversation,
    Message,
    ConversationDocument,
)

from documents.models import Document


def create_conversation(
    *,
    workspace,
    user,
    title: str = "New Chat",
) -> Conversation:
    return Conversation.objects.create(
        workspace=workspace,
        created_by=user,
        title=title,
    )


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
) -> Message:
    return Message.objects.create(
        conversation=conversation,
        role=role,
        content=content,
    )


def attach_document(
    *,
    conversation: Conversation,
    document: Document,
) -> ConversationDocument:
    """
    MVP rule:
    maximum 3 attached documents.
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

    link, _ = (
        ConversationDocument.objects.get_or_create(
            conversation=conversation,
            document=document,
        )
    )

    return link


def get_recent_history(
    *,
    conversation: Conversation,
    limit: int = 10,
) -> list[dict]:
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