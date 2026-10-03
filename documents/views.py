"""Views module for QueryPilot."""

import logging

from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from documents.ingestion.sqlite import SQLiteManager
from documents.ingestion.vectors import VectorStore
from utils.redis import publish_job
from workspace.models import Workspace

from .models import Document, IngestionJob
from .serializers import (
    DocumentListSerializer,
    DocumentUploadSerializer,
)

logger = logging.getLogger(__name__)


class DocumentUploadAPIView(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request):

        serializer = DocumentUploadSerializer(
            data=request.data,
            context={"request": request},
        )

        serializer.is_valid(raise_exception=True)

        workspace_id = serializer.validated_data["workspace_id"]

        workspace = get_object_or_404(
            Workspace.objects.select_for_update(),
            id=workspace_id,
            owner=request.user,
        )

        uploaded_file = serializer.validated_data["file"]

        file_hash = serializer.calculate_file_hash(uploaded_file)

        existing_document = Document.objects.filter(
            workspace=workspace,
            file_hash=file_hash,
        ).first()

        if existing_document:
            if existing_document.status == "failed":
                failed_document_id = existing_document.id
                cleanup_errors = []

                try:
                    VectorStore.delete_document(document_id=failed_document_id)

                except Exception:
                    logger.exception(
                        "Vector cleanup failed for stale failed document %s",
                        failed_document_id,
                    )

                    cleanup_errors.append("vectors")

                try:
                    SQLiteManager.drop_document_tables(
                        workspace_id=workspace.id,
                        document_id=failed_document_id,
                    )

                except Exception:
                    logger.exception(
                        "SQLite cleanup failed for stale failed document %s",
                        failed_document_id,
                    )

                    cleanup_errors.append("tables")

                try:
                    if existing_document.file:
                        existing_document.file.delete(save=False)

                except Exception:
                    logger.exception(
                        "File cleanup failed for stale failed document %s",
                        failed_document_id,
                    )

                    cleanup_errors.append("file")

                existing_document.delete()

                if cleanup_errors:
                    logger.warning(
                        "Stale failed document %s deleted with "
                        "possible cleanup issues: %s",
                        failed_document_id,
                        ", ".join(cleanup_errors),
                    )

            else:
                return Response(
                    {
                        "message": "Document already exists.",
                        "document_id": existing_document.id,
                        "status": existing_document.status,
                        "duplicate": True,
                    },
                    status=status.HTTP_200_OK,
                )

        document_count = (
            Document.objects.filter(
                workspace=workspace,
            )
            .exclude(
                status="failed",
            )
            .count()
        )

        if document_count >= 3:
            return Response(
                {"detail": ("Maximum 3 documents allowed per workspace.")},
                status=status.HTTP_400_BAD_REQUEST,
            )

        source_type = uploaded_file.name.rsplit(".", 1)[-1].lower()

        document = Document.objects.create(
            workspace=workspace,
            uploaded_by=request.user,
            title=uploaded_file.name,
            file=uploaded_file,
            source_type=source_type,
            file_hash=file_hash,
            status="pending",
        )

        job = IngestionJob.objects.create(
            document=document,
            status="pending",
        )

        transaction.on_commit(lambda: publish_job(job.id))

        return Response(
            {
                "message": "Document upload accepted.",
                "document_id": document.id,
                "ingestion_job_id": job.id,
                "status": document.status,
                "duplicate": False,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class DocumentListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):

        workspace_id = request.query_params.get("workspace_id")

        if not workspace_id:
            return Response(
                {"detail": ("Query parameter 'workspace_id' is required.")},
                status=status.HTTP_400_BAD_REQUEST,
            )

        workspace = get_object_or_404(
            Workspace,
            id=workspace_id,
            owner=request.user,
        )

        documents = (
            Document.objects.filter(
                workspace=workspace,
            )
            .exclude(
                status="failed",
            )
            .order_by("-uploaded_at")
        )

        serializer = DocumentListSerializer(
            documents,
            many=True,
        )

        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
        )


class DocumentStatusAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(
        self,
        request,
        document_id,
    ):

        document = get_object_or_404(
            Document.objects.select_related("ingestion_job"),
            id=document_id,
            workspace__owner=request.user,
        )

        job = getattr(
            document,
            "ingestion_job",
            None,
        )

        response = {
            "document_id": document.id,
            "status": document.status,
        }

        if job is not None:
            response.update(
                {
                    "ingestion_job_id": job.id,
                    "job_status": job.status,
                    "attempts": job.attempts,
                }
            )

        return Response(
            response,
            status=status.HTTP_200_OK,
        )


class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def delete(
        self,
        request,
        document_id,
    ):

        document = get_object_or_404(
            Document.objects.select_for_update(),
            id=document_id,
            workspace__owner=request.user,
        )

        if document.status in {
            "pending",
            "processing",
        }:
            return Response(
                {"detail": ("Document cannot be deleted while ingestion is active.")},
                status=status.HTTP_409_CONFLICT,
            )

        workspace_id = document.workspace_id

        cleanup_errors = []

        try:
            VectorStore.delete_document(document_id=document.id)

        except Exception:
            logger.exception(
                "Vector cleanup failed for document %s",
                document.id,
            )

            cleanup_errors.append("vectors")

        try:
            SQLiteManager.drop_document_tables(
                workspace_id=workspace_id,
                document_id=document.id,
            )

        except Exception:
            logger.exception(
                "SQLite cleanup failed for document %s",
                document.id,
            )

            cleanup_errors.append("tables")

        if document.file:
            document.file.delete(save=False)

        document.delete()

        if cleanup_errors:
            return Response(
                {
                    "message": (
                        "Document deleted, but cleanup of "
                        f"{', '.join(cleanup_errors)} may be "
                        "incomplete. This has been logged."
                    )
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {"message": "Document deleted successfully."},
            status=status.HTTP_200_OK,
        )
