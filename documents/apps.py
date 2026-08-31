# documents/apps.py

import os
import sys

from django.apps import AppConfig


class DocumentsConfig(AppConfig):

    name = "documents"

    def ready(self):

        if "runserver" not in sys.argv:
            return

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from documents.ingestion.vectors import (
            VectorStore,
        )

        print("Loading embedding model...")

        VectorStore.get_embedder()

        print("Embedding model ready.")