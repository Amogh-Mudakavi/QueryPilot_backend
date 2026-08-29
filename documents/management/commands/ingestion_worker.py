"""
QueryPilot document ingestion worker.

Run with:

    python manage.py ingestion_worker

Startup:
    1. Recover pending jobs from the database.
    2. Process them.
    3. Start listening for new Redis notifications.
"""

from __future__ import annotations

from datetime import datetime, timezone

from django.core.management.base import BaseCommand

from documents.ingestion.pipeline import IngestionPipeline
from documents.models import IngestionJob
from utils.redis import get_pubsub


def process_ingestion_job(job_id: int) -> None:
    """
    Process one pending ingestion job.
    """

    job = (
        IngestionJob.objects
        .select_related("document")
        .filter(id=job_id)
        .first()
    )

    if job is None:
        print(f"[INGESTION] Job {job_id} not found.")
        return

    if job.status != "pending":
        print(
            f"[INGESTION] Ignoring job {job.id}; "
            f"status={job.status}"
        )
        return

    document = job.document

    try:
        job.attempts += 1
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        job.error_message = None

        job.save(
            update_fields=[
                "attempts",
                "status",
                "started_at",
                "error_message",
            ]
        )

        document.status = "processing"

        document.save(
            update_fields=[
                "status",
                "updated_at",
            ]
        )

        print(
            f"[INGESTION] Processing job {job.id}"
        )

        IngestionPipeline.run(document)

        completed_at = datetime.now(timezone.utc)

        job.status = "completed"
        job.completed_at = completed_at
        job.error_message = None

        job.save(
            update_fields=[
                "status",
                "completed_at",
                "error_message",
            ]
        )

        document.status = "ready"

        document.save(
            update_fields=[
                "status",
                "updated_at",
            ]
        )

        print(
            f"[INGESTION] Completed job {job.id}"
        )

    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)

        job.save(
            update_fields=[
                "status",
                "error_message",
            ]
        )

        document.status = "failed"

        document.save(
            update_fields=[
                "status",
                "updated_at",
            ]
        )

        print(
            f"[INGESTION] Failed job {job.id}: {exc}"
        )


def recover_pending_jobs() -> None:
    """
    Process jobs that were already pending when the worker started.
    """

    pending_job_ids = list(
        IngestionJob.objects
        .filter(status="pending")
        .order_by("id")
        .values_list("id", flat=True)
    )

    if not pending_job_ids:
        print(
            "[INGESTION] No pending jobs at startup."
        )
        return

    print(
        f"[INGESTION] Found "
        f"{len(pending_job_ids)} pending job(s) at startup."
    )

    for job_id in pending_job_ids:
        process_ingestion_job(job_id)


def start_worker() -> None:
    """
    Recover pending jobs first, then listen for Redis notifications.
    """

    recover_pending_jobs()

    pubsub = get_pubsub()

    print(
        "[INGESTION] Worker started. "
        "Listening on admin_jobs..."
    )

    for message in pubsub.listen():

        if message.get("type") != "message":
            continue

        try:
            job_id = int(message["data"])
        except (
            TypeError,
            ValueError,
        ):
            print(
                "[INGESTION] Ignoring invalid job ID: "
                f"{message.get('data')}"
            )
            continue

        print(
            f"[INGESTION] Received job {job_id}"
        )

        process_ingestion_job(job_id)


class Command(BaseCommand):
    """
    Django management command entrypoint.
    """

    help = (
        "Start the QueryPilot document ingestion worker."
    )

    def handle(
        self,
        *args,
        **options,
    ):
        self.stdout.write(
            self.style.SUCCESS(
                "Starting QueryPilot ingestion worker..."
            )
        )

        start_worker()