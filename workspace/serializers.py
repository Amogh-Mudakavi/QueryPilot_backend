"""Serializers module for QueryPilot."""

from rest_framework import serializers

from .models import Workspace


class WorkspaceSerializer(serializers.ModelSerializer):
    source_count = serializers.IntegerField(
        source="documents.count",
        read_only=True,
    )

    class Meta:
        model = Workspace
        fields = [
            "id",
            "title",
            "created_at",
            "updated_at",
            "source_count",
        ]


class WorkspaceCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workspace
        fields = [
            "title",
        ]


class WorkspaceUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workspace
        fields = [
            "title",
        ]

    def validate_title(self, value):
        value = value.strip()

        if not value:
            raise serializers.ValidationError("Workspace title cannot be empty.")

        return value


class WorkspaceSettingsSerializer(serializers.ModelSerializer):
    source_count = serializers.IntegerField(
        source="documents.count",
        read_only=True,
    )

    class Meta:
        model = Workspace
        fields = [
            "id",
            "title",
            "created_at",
            "source_count",
        ]
