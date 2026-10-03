"""Models module for QueryPilot."""

from django.conf import settings
from django.db import models

from documents.models import Document
from workspace.models import Workspace


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


class ConversationSettings(models.Model):
    """
    Stores user-configurable settings for a single conversation.

    These settings control observable conversation behavior rather than
    internal AI infrastructure.
    """

    conversation = models.OneToOneField(
        Conversation,
        on_delete=models.CASCADE,
        related_name="settings",
    )

    memory_enabled = models.BooleanField(
        default=True,
        help_text="Whether previous conversation context is used.",
    )

    citations_enabled = models.BooleanField(
        default=True,
        help_text="Whether document citations are included in answers.",
    )

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Settings | Conversation={self.conversation_id}"


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

    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role} | {self.conversation_id}"


class ConversationMemory(models.Model):
    """
    Stores the rolling conversational memory for one conversation.

    Memory is intentionally kept separate from Message records.

    Design:
        - Message remains the source of truth for the full conversation.
        - `summary` stores compressed context from older messages.
        - `summarized_until_message` records the last Message already
          represented by the rolling summary.
        - The most recent 10 messages are loaded directly from Message
          at runtime and are NOT duplicated here.

    This means the agent can receive:

        rolling summary
            +
        latest 10 messages
            +
        current user query

    The summary is conversational context only. It is never treated as
    authoritative document or table evidence.
    """

    conversation = models.OneToOneField(
        Conversation,
        on_delete=models.CASCADE,
        related_name="memory",
    )

    summary = models.TextField(
        blank=True,
        default="",
        help_text=(
            "Rolling summary of older conversation context. "
            "This is context, not authoritative evidence."
        ),
    )

    summarized_until_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
        help_text=(
            "The latest message whose content is represented "
            "inside the rolling summary."
        ),
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    def __str__(self) -> str:
        return f"Memory | Conversation={self.conversation_id}"


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
        return f"Conversation={self.conversation_id} Document={self.document_id}"
