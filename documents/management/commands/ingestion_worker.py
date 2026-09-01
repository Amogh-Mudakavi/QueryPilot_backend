"""
QueryPilot document ingestion worker.

Run with:

    python manage.py ingestion_worker

Startup:
    1. Recover pending and stuck-processing jobs from the database.
    2. Process them.
    3. Start listening for new Redis notifications.

Safe to run as multiple concurrent processes (e.g. 3 worker replicas
in production) - job claiming uses select_for_update(skip_locked=True)
so only one process can ever pick up a given job.
"""

from __future__ import annotations

import logging

from datetime import datetime, timezone

from django.core.management.base import BaseCommand
from django.db import transaction

from documents.ingestion.pipeline import IngestionPipeline
from documents.models import IngestionJob
from utils.redis import get_pubsub

logger = logging.getLogger(__name__)


def _claim_job(job_id: int):
    """
    Atomically claims a job for this worker process, if it's still
    claimable.

    Uses select_for_update(skip_locked=True) so that when multiple
    worker processes receive the same Redis broadcast for this job_id
    (pub/sub delivers to every subscriber, not just one), only the
    first to reach this row wins the row lock and flips it to
    "processing". Every other worker's SELECT either sees the row is
    no longer "pending" (lock already released, status changed) or is
    skipped immediately via skip_locked instead of blocking - so it
    returns None right away rather than waiting and then double
    processing.

    Runs as its own short transaction, separate from the actual
    ingestion work below, so a slow pipeline run doesn't hold this row
    lock for the run's whole duration.

    Returns the claimed (job, document) tuple, or (None, None) if this
    process lost the race or the job isn't in a claimable state.
    """

    with transaction.atomic():
        job = (
            IngestionJob.objects
            .select_for_update(skip_locked=True)
            .select_related("document")
            .filter(
                id=job_id,
                status__in=["pending", "processing"],
            )
            .first()
        )

        if job is None:
            return None, None

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

        document = job.document

        document.status = "processing"

        document.save(
            update_fields=[
                "status",
                "updated_at",
            ]
        )

        return job, document


def process_ingestion_job(job_id: int) -> None:
    """
    Process one ingestion job, if this worker successfully claims it.
    """

    job, document = _claim_job(job_id)

    if job is None:
        logger.info(
            "[INGESTION] Job %s not claimable "
            "(already taken, missing, or in a terminal state).",
            job_id,
        )
        return

    try:
        logger.info(
            "[INGESTION] Processing job %s", job.id
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

        logger.info(
            "[INGESTION] Completed job %s", job.id
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

        logger.exception(
            "[INGESTION] Failed job %s", job.id
        )


def recover_pending_jobs() -> None:
    """
    Process jobs that were already pending or stuck in "processing"
    when the worker started.

    A job can be found "processing" at startup if a previous worker
    process died mid-ingestion (crash, OOM kill, deploy restart) after
    claiming the job but before finishing it. Recovering "processing"
    jobs here, not just "pending" ones, means such a job gets retried
    instead of sitting stuck forever with no worker aware of it.

    Safe to call from multiple worker processes starting at once -
    process_ingestion_job's claim is what actually serializes who
    gets each job, this function just enumerates candidates.
    """

    job_ids = list(
        IngestionJob.objects
        .filter(status__in=["pending", "processing"])
        .order_by("id")
        .values_list("id", flat=True)
    )

    if not job_ids:
        logger.info(
            "[INGESTION] No pending or stuck jobs at startup."
        )
        return

    logger.info(
        "[INGESTION] Found %d job(s) to recover at startup.",
        len(job_ids),
    )

    for job_id in job_ids:
        process_ingestion_job(job_id)


def start_worker() -> None:
    """
    Recover pending jobs first, then listen for Redis notifications.
    """

    recover_pending_jobs()

    pubsub = get_pubsub()

    logger.info(
        "[INGESTION] Worker started. Listening on admin_jobs..."
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
            logger.warning(
                "[INGESTION] Ignoring invalid job ID: %s",
                message.get("data"),
            )
            continue

        logger.info(
            "[INGESTION] Received job %s", job_id
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