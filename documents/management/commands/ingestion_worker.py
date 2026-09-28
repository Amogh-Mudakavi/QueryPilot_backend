"""
QueryPilot document ingestion worker.

Run with:

    python manage.py ingestion_worker

The worker is intended to stay alive continuously.

Flow:

    Redis queue
        ↓
    claim job in MySQL
        ↓
    ingestion pipeline
        ↓
    completed / retry / failed
"""

from __future__ import annotations

import logging
import os
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
)

from datetime import timedelta

import redis

from django.core.management.base import BaseCommand
from django.db import close_old_connections
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from documents.ingestion.pipeline import (
    IngestionPipeline,
)

from documents.ingestion.sqlite import (
    SQLiteManager,
)

from documents.ingestion.vectors import (
    VectorStore,
)

from documents.models import (
    IngestionJob,
)

from utils.redis import (
    enqueue_job,
    get_next_job,
)


logger = logging.getLogger(__name__)


WORKER_CONCURRENCY = int(
    os.getenv(
        "INGESTION_WORKER_CONCURRENCY",
        "3",
    )
)


JOB_STALE_SECONDS = int(
    os.getenv(
        "INGESTION_JOB_STALE_SECONDS",
        "1800",
    )
)


HEARTBEAT_SECONDS = int(
    os.getenv(
        "INGESTION_HEARTBEAT_SECONDS",
        "30",
    )
)


RECOVERY_INTERVAL_SECONDS = int(
    os.getenv(
        "INGESTION_RECOVERY_INTERVAL_SECONDS",
        "30",
    )
)


MAX_JOB_ATTEMPTS = int(
    os.getenv(
        "INGESTION_MAX_ATTEMPTS",
        "3",
    )
)


def _claim_job(job_id: int):

    now = timezone.now()

    stale_before = (
        now
        - timedelta(
            seconds=JOB_STALE_SECONDS
        )
    )

    with transaction.atomic():

        job = (
            IngestionJob.objects
            .select_for_update(
                skip_locked=True
            )
            .select_related(
                "document"
            )
            .filter(
                Q(
                    id=job_id,
                    status="pending",
                )
                |
                Q(
                    id=job_id,
                    status="processing",
                    updated_at__lt=stale_before,
                )
            )
            .first()
        )

        if job is None:
            return None, None

        job.attempts += 1
        job.status = "processing"
        job.started_at = now
        job.error_message = None

        job.save(
            update_fields=[
                "attempts",
                "status",
                "started_at",
                "error_message",
                "updated_at",
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


def _heartbeat(
    job_id: int,
    stop_event: threading.Event,
):

    close_old_connections()

    while not stop_event.wait(
        HEARTBEAT_SECONDS
    ):

        try:

            updated = (
                IngestionJob.objects
                .filter(
                    id=job_id,
                    status="processing",
                )
                .update(
                    updated_at=timezone.now()
                )
            )

            if updated == 0:
                break

        except Exception:

            logger.exception(
                "[INGESTION] Heartbeat failed "
                "for job %s",
                job_id,
            )

    close_old_connections()


def _mark_completed(
    job,
    document,
):

    now = timezone.now()

    job.status = "completed"
    job.completed_at = now
    job.error_message = None

    job.save(
        update_fields=[
            "status",
            "completed_at",
            "error_message",
            "updated_at",
        ]
    )

    document.status = "ready"

    document.save(
        update_fields=[
            "status",
            "updated_at",
        ]
    )


def _mark_failed(
    job,
    document,
    error,
):

    job.status = "failed"
    job.error_message = str(error)[:2000]

    job.save(
        update_fields=[
            "status",
            "error_message",
            "updated_at",
        ]
    )

    document.status = "failed"

    document.save(
        update_fields=[
            "status",
            "updated_at",
        ]
    )


def _delete_failed_document(document):

    document_id = document.id
    workspace_id = document.workspace_id
    cleanup_errors = []

    try:

        VectorStore.delete_document(
            document_id=document_id
        )

    except Exception:

        logger.exception(
            "[INGESTION] Vector cleanup failed "
            "for permanently failed document %s",
            document_id,
        )

        cleanup_errors.append(
            "vectors"
        )

    try:

        SQLiteManager.drop_document_tables(
            workspace_id=workspace_id,
            document_id=document_id,
        )

    except Exception:

        logger.exception(
            "[INGESTION] SQLite cleanup failed "
            "for permanently failed document %s",
            document_id,
        )

        cleanup_errors.append(
            "tables"
        )

    try:

        if document.file:

            document.file.delete(
                save=False
            )

    except Exception:

        logger.exception(
            "[INGESTION] File cleanup failed "
            "for permanently failed document %s",
            document_id,
        )

        cleanup_errors.append(
            "file"
        )

    try:

        document.delete()

    except Exception:

        logger.exception(
            "[INGESTION] Database cleanup failed "
            "for permanently failed document %s",
            document_id,
        )

        cleanup_errors.append(
            "database"
        )

    if cleanup_errors:

        logger.error(
            "[INGESTION] Permanently failed document %s "
            "was deleted, but cleanup may be incomplete: %s",
            document_id,
            ", ".join(cleanup_errors),
        )

    else:

        logger.info(
            "[INGESTION] Permanently failed document %s "
            "deleted successfully.",
            document_id,
        )


def _mark_for_retry(
    job,
    document,
    error,
):

    job.status = "pending"
    job.error_message = str(error)[:2000]

    job.save(
        update_fields=[
            "status",
            "error_message",
            "updated_at",
        ]
    )

    document.status = "pending"

    document.save(
        update_fields=[
            "status",
            "updated_at",
        ]
    )

    try:

        enqueue_job(job.id)

    except Exception:

        logger.exception(
            "[INGESTION] Failed to requeue "
            "job %s. Recovery sweep will retry it.",
            job.id,
        )


def process_ingestion_job(
    job_id: int,
) -> None:

    close_old_connections()

    job, document = _claim_job(
        job_id
    )

    if job is None:

        close_old_connections()
        return

    stop_event = threading.Event()

    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        args=(
            job.id,
            stop_event,
        ),
        daemon=True,
    )

    heartbeat_thread.start()

    try:

        logger.info(
            "[INGESTION] Processing job %s",
            job.id,
        )

        IngestionPipeline.run(
            document
        )

        _mark_completed(
            job,
            document,
        )

        logger.info(
            "[INGESTION] Completed job %s",
            job.id,
        )

    except Exception as exc:

        logger.exception(
            "[INGESTION] Failed job %s",
            job.id,
        )

        if job.attempts < MAX_JOB_ATTEMPTS:

            _mark_for_retry(
                job,
                document,
                exc,
            )

            logger.warning(
                "[INGESTION] Job %s requeued "
                "(attempt %s/%s).",
                job.id,
                job.attempts,
                MAX_JOB_ATTEMPTS,
            )

        else:

            _mark_failed(
                job,
                document,
                exc,
            )

            logger.error(
                "[INGESTION] Job %s permanently failed "
                "after %s attempts.",
                job.id,
                job.attempts,
            )

            _delete_failed_document(
                document
            )

    finally:

        stop_event.set()

        heartbeat_thread.join(
            timeout=2
        )

        close_old_connections()


def recover_pending_jobs():

    now = timezone.now()

    stale_before = (
        now
        - timedelta(
            seconds=JOB_STALE_SECONDS
        )
    )

    job_ids = list(
        IngestionJob.objects
        .filter(
            Q(status="pending")
            |
            Q(
                status="processing",
                updated_at__lt=stale_before,
            )
        )
        .values_list(
            "id",
            flat=True,
        )
    )

    if not job_ids:
        return

    logger.info(
        "[INGESTION] Recovery found %s job(s).",
        len(job_ids),
    )

    for job_id in job_ids:

        try:

            enqueue_job(
                job_id
            )

        except Exception:

            logger.exception(
                "[INGESTION] Could not requeue "
                "recovery job %s.",
                job_id,
            )


def _cleanup_finished_futures(
    futures,
):

    finished = []

    for future in futures:

        if future.done():

            try:
                future.result()

            except Exception:

                logger.exception(
                    "[INGESTION] Worker task crashed "
                    "outside normal job handling."
                )

            finished.append(
                future
            )

    for future in finished:

        futures.remove(
            future
        )


def start_worker():

    logger.info(
        "[INGESTION] Starting worker "
        "with concurrency=%s.",
        WORKER_CONCURRENCY,
    )

    logger.info(
        "[INGESTION] Loading embedding model..."
    )

    VectorStore.get_embedder()

    logger.info(
        "[INGESTION] Embedding model ready."
    )

    # Warm up the shared model with one short encode. This finishes
    # lazy initialization and pages the model weights in BEFORE the
    # first real job, and logs a timing baseline for this machine.
    warmup_started = time.monotonic()

    VectorStore.embed(
        [
            "QueryPilot embedding model warm-up."
        ]
    )

    logger.info(
        "[INGESTION] Embedding warm-up completed in %.2fs.",
        time.monotonic() - warmup_started,
    )

    recover_pending_jobs()

    futures = set()
    executor = ThreadPoolExecutor(
        max_workers=WORKER_CONCURRENCY
    )

    logger.info(
        "[INGESTION] Worker ready. "
        "Waiting for ingestion jobs."
    )

    last_recovery = time.monotonic()

    try:

        while True:

            _cleanup_finished_futures(
                futures
            )

            now = time.monotonic()

            if (
                now - last_recovery
                >= RECOVERY_INTERVAL_SECONDS
            ):

                recover_pending_jobs()
                last_recovery = now

            # Never submit more work than our
            # configured concurrency.
            if len(futures) >= WORKER_CONCURRENCY:

                time.sleep(0.25)
                continue

            try:

                job_id = get_next_job(
                    timeout=5
                )

            except redis.exceptions.RedisError:

                logger.exception(
                    "[INGESTION] Redis connection "
                    "error. Retrying."
                )

                time.sleep(2)
                continue

            if job_id is None:
                continue

            future = executor.submit(
                process_ingestion_job,
                job_id,
            )

            futures.add(
                future
            )

    except KeyboardInterrupt:

        logger.info(
            "[INGESTION] Shutdown requested."
        )

    finally:

        executor.shutdown(
            wait=True
        )


class Command(BaseCommand):

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
