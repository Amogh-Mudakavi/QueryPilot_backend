"""Apps module for QueryPilot."""

import logging
import os
import sys
import threading
import time

from django.apps import AppConfig

os.environ.setdefault(
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_VERBOSITY",
    "error",
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

logger = logging.getLogger(__name__)

# manage.py commands that embed text and therefore benefit from a warm model.
# Every other manage.py command (migrate, shell, collectstatic, ...) must not
# pay for loading it.
_WARMUP_MANAGE_COMMANDS = {"runserver", "ingestion_worker"}


def _should_warm_up() -> bool:
    """Warm the embedder only in processes that will actually serve/ingest."""
    if os.getenv("EMBEDDING_WARMUP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False

    entry = os.path.basename(sys.argv[0]) if sys.argv else ""
    if "pytest" in entry or "PYTEST_CURRENT_TEST" in os.environ:
        return False

    if entry == "manage.py":
        command = sys.argv[1] if len(sys.argv) > 1 else ""
        if command not in _WARMUP_MANAGE_COMMANDS:
            return False
        # runserver's autoreloader starts a parent and a child process; only
        # the child (RUN_MAIN=true) serves requests.
        if (
            command == "runserver"
            and "--noreload" not in sys.argv
            and os.environ.get("RUN_MAIN") != "true"
        ):
            return False

    # uvicorn / gunicorn / daphne / any other entry point serving requests.
    return True


def _warm_embedder() -> None:
    started = time.monotonic()
    try:
        # Imported here, not at module level, so Django start-up stays fast.
        from documents.ingestion.vectors import VectorStore

        # Loads the model AND runs one inference, so the ONNX session's
        # first-call setup cost is paid now rather than by the first user.
        VectorStore.embed(["embedding warm-up"])
        logger.info(
            "[EMBEDDING] warm-up complete in %.1fs.", time.monotonic() - started
        )
    except Exception:
        # Never block or crash the server: the model simply loads lazily on
        # the first real request, exactly as before.
        logger.exception("[EMBEDDING] warm-up failed; will load on first use.")


class DocumentsConfig(AppConfig):
    name = "documents"

    def ready(self):
        if not _should_warm_up():
            return
        # Background thread: server start-up is not delayed, and VectorStore's
        # own lock makes a request arriving mid-load wait for the one load
        # instead of starting a second one.
        threading.Thread(
            target=_warm_embedder,
            name="embedding-warmup",
            daemon=True,
        ).start()