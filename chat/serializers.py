from rest_framework import serializers

from chat.models import (
    Conversation,
    Message,
)


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "role",
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


class CreateConversationSerializer(serializers.Serializer):
    workspace_id = serializers.IntegerField()


class SendMessageSerializer(serializers.Serializer):
    conversation_id = serializers.IntegerField()

    message = serializers.CharField(
        allow_blank=False,
        trim_whitespace=True,
    )