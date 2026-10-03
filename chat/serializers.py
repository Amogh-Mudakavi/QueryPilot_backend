"""Serializers module for QueryPilot."""

from rest_framework import serializers

from chat.models import (
    Conversation,
    ConversationSettings,
    Message,
)


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "metadata",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "role",
            "metadata",
            "created_at",
        ]


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "workspace",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "title",
            "created_by",
            "created_at",
            "updated_at",
        ]


class ConversationSettingsSerializer(serializers.ModelSerializer):
    """
    Serializer for user-configurable settings of a conversation.

    Only exposes settings that have an actual effect on conversation
    execution.
    """

    class Meta:
        model = ConversationSettings
        fields = [
            "conversation",
            "memory_enabled",
            "citations_enabled",
            "updated_at",
        ]
        read_only_fields = [
            "conversation",
            "updated_at",
        ]


class CreateConversationSerializer(serializers.Serializer):
    workspace_id = serializers.IntegerField()


class SendMessageSerializer(serializers.Serializer):
    conversation_id = serializers.IntegerField()

    message = serializers.CharField(
        allow_blank=False,
        trim_whitespace=True,
    )
