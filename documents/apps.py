# documents/apps.py

from django.apps import AppConfig


class DocumentsConfig(AppConfig):

    name = "documents"

    def ready(self):

        from documents.ingestion.vectors import (
            VectorStore,
        )

        VectorStore.get_embedder()