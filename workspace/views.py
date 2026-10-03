"""Views module for QueryPilot."""

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Workspace
from .serializers import (
    WorkspaceCreateSerializer,
    WorkspaceSerializer,
    WorkspaceSettingsSerializer,
    WorkspaceUpdateSerializer,
)


class WorkspaceCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = WorkspaceCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        workspace = Workspace.objects.create(
            owner=request.user,
            title=serializer.validated_data["title"],
        )

        return Response(
            WorkspaceSerializer(workspace).data,
            status=status.HTTP_201_CREATED,
        )


class WorkspaceListAPIView(ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = WorkspaceSerializer

    def get_queryset(self):
        return Workspace.objects.filter(owner=self.request.user).order_by("-updated_at")


class WorkspaceDetailAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get_workspace(self, request, workspace_id):
        return get_object_or_404(
            Workspace,
            id=workspace_id,
            owner=request.user,
        )

    def patch(self, request, workspace_id):
        workspace = self.get_workspace(
            request,
            workspace_id,
        )

        serializer = WorkspaceUpdateSerializer(
            workspace,
            data=request.data,
            partial=True,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(
            WorkspaceSerializer(workspace).data,
            status=status.HTTP_200_OK,
        )

    def delete(self, request, workspace_id):
        workspace = self.get_workspace(
            request,
            workspace_id,
        )

        workspace.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


class WorkspaceSettingsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, workspace_id):
        workspace = get_object_or_404(
            Workspace,
            id=workspace_id,
            owner=request.user,
        )

        return Response(
            WorkspaceSettingsSerializer(workspace).data,
            status=status.HTTP_200_OK,
        )
