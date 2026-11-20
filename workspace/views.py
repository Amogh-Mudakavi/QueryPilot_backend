"""Views module for QueryPilot."""

from django.db import IntegrityError, transaction
from django.db.models.functions import Lower
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.exceptions import APIException
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


class WorkspaceNameConflict(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_detail = "A workspace with that name already exists. Choose a different name."
    default_code = "workspace_name_conflict"


def title_taken(owner, title, exclude_id=None):
    qs = Workspace.objects.filter(owner=owner).annotate(t=Lower("title")).filter(t=title.lower())
    if exclude_id is not None:
        qs = qs.exclude(id=exclude_id)
    return qs.exists()


class WorkspaceCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = WorkspaceCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        title = serializer.validated_data["title"]

        # Friendly fast path; the database constraint below is what actually guarantees it.
        if title_taken(request.user, title):
            raise WorkspaceNameConflict()

        try:
            with transaction.atomic():
                workspace = Workspace.objects.create(owner=request.user, title=title)
        except IntegrityError:
            if title_taken(request.user, title):
                raise WorkspaceNameConflict()
            raise

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

        title = serializer.validated_data.get("title")
        if title is not None and title_taken(request.user, title, exclude_id=workspace.id):
            raise WorkspaceNameConflict()

        try:
            with transaction.atomic():
                serializer.save()
        except IntegrityError:
            if title is not None and title_taken(request.user, title, exclude_id=workspace.id):
                raise WorkspaceNameConflict()
            raise

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