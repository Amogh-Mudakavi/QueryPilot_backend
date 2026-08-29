from django.db import models

from accounts.models import User
from workspace.models import Workspace


class Document(models.Model):
    """
    Stores an uploaded document and the workspace it belongs to.

    A document always belongs to exactly one workspace.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("ready", "Ready"),
        ("failed", "Failed"),
    ]

    SOURCE_TYPE_CHOICES = [
        ("pdf", "PDF"),
        ("docx", "DOCX"),
        ("xlsx", "Excel"),
        ("csv", "CSV"),
    ]

    workspace = models.ForeignKey(
        Workspace,
        on_delete=models.CASCADE,
        related_name="documents",
    )

    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name="uploaded_documents",
    )

    title = models.CharField(
        max_length=255
    )

    file = models.FileField(
        upload_to="documents/"
    )

    source_type = models.CharField(
        max_length=10,
        choices=SOURCE_TYPE_CHOICES,
    )

    file_hash = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        db_index=True,
        help_text="SHA-256 hash of the uploaded file.",
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "workspace",
                    "file_hash",
                ],
                name="unique_workspace_document_hash",
            ),
        ]

    def __str__(self):
        return self.title


class IngestionJob(models.Model):
    """
    Tracks the asynchronous ingestion state of one document.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name="ingestion_job",
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
    )

    attempts = models.PositiveSmallIntegerField(
        default=0
    )

    error_message = models.TextField(
        blank=True,
        null=True,
    )

    started_at = models.DateTimeField(
        blank=True,
        null=True,
    )

    completed_at = models.DateTimeField(
        blank=True,
        null=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"IngestionJob {self.id} - {self.status}"


class WorkspaceTable(models.Model):
    """
    Catalogs a table extracted from a document.

    The actual table data lives in SQLite.
    """

    workspace = models.ForeignKey(
        Workspace,
        on_delete=models.CASCADE,
        related_name="tables",
    )

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="tables",
    )

    table_name = models.CharField(
        max_length=255
    )

    sqlite_table_name = models.CharField(
        max_length=255,
        unique=True,
    )

    columns = models.JSONField(
        default=list
    )

    row_count = models.PositiveIntegerField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["table_name"]

    def __str__(self):
        return self.table_name


class DocumentMetadata(models.Model):
    """
    Stores semantic metadata about a document.
    """

    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name="metadata",
    )

    summary = models.TextField(
        blank=True
    )

    topics = models.JSONField(
        default=list
    )

    keywords = models.JSONField(
        default=list
    )

    language = models.CharField(
        max_length=50,
        blank=True,
        null=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return f"Metadata - {self.document.title}"