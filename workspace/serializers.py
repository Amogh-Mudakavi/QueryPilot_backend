from rest_framework import serializers

from .models import Workspace


class WorkspaceSerializer(serializers.ModelSerializer):

    class Meta:
        model = Workspace
        fields = [
            "id",
            "title",
            "created_at",
            "updated_at",
        ]


class WorkspaceCreateSerializer(serializers.ModelSerializer):

    class Meta:
        model = Workspace
        fields = [
            "title",
        ]