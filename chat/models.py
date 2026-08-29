from django.conf import settings
from django.db import models

from workspace.models import Workspace
from documents.models import Document


class Conversation(models.Model):
    """
    A chat session inside a workspace.

    Example:
        Food Delivery Policies
        HR Handbook
        New Chat
    """

    title = models.CharField(max_length=255, default="New Chat")

    workspace = models.ForeignKey(
        Workspace,
        on_delete=models.CASCADE,
        related_name="conversations",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conversations",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title


class Message(models.Model):
    """
    Individual chat message.
    """

    class Role(models.TextChoices):
        USER = "USER", "User"
        ASSISTANT = "ASSISTANT", "Assistant"
        SYSTEM = "SYSTEM", "System"

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(
        max_length=20,
        choices=Role.choices,
    )

    content = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role} | {self.conversation_id}"


class ConversationDocument(models.Model):
    """
    Documents attached to a conversation.

    A conversation may contain up to 3 documents.
    """

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="conversation_documents",
    )

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="conversation_documents",
    )

    attached_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("conversation", "document")

    def __str__(self) -> str:
        return (
            f"Conversation={self.conversation_id} "
            f"Document={self.document_id}"
        )