from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Workspace
from .serializers import (
    WorkspaceSerializer,
    WorkspaceCreateSerializer,
)


class WorkspaceCreateAPIView(APIView):

    permission_classes = [IsAuthenticated]

    def post(self, request):

        serializer = WorkspaceCreateSerializer(
            data=request.data
        )

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

        return Workspace.objects.filter(
            owner=self.request.user
        ).order_by("-updated_at")


class WorkspaceDeleteAPIView(APIView):

    permission_classes = [IsAuthenticated]

    def delete(self, request, workspace_id):

        workspace = Workspace.objects.filter(
            id=workspace_id,
            owner=request.user,
        ).first()

        if not workspace:

            return Response(
                {"detail": "Workspace not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        workspace.delete()

        return Response(
            status=status.HTTP_204_NO_CONTENT
        )