"""Serializers module for QueryPilot."""

import re

from rest_framework import serializers

from .models import Workspace


def normalize_title(value: str) -> str:
    """Trim and collapse runs of whitespace so "My  docs " and "My docs" are the same name."""
    return re.sub(r"\s+", " ", value).strip()


class WorkspaceTitleMixin:
    def validate_title(self, value):
        value = normalize_title(value)

        if not value:
            raise serializers.ValidationError("Workspace title cannot be empty.")

        return value


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


class WorkspaceCreateSerializer(WorkspaceTitleMixin, serializers.ModelSerializer):
    class Meta:
        model = Workspace
        fields = [
            "title",
        ]


class WorkspaceUpdateSerializer(WorkspaceTitleMixin, serializers.ModelSerializer):
    class Meta:
        model = Workspace
        fields = [
            "title",
        ]


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