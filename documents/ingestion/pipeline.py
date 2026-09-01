"""
Main ingestion pipeline.

Coordinates all ingestion components.

Flow:

    Document
        ↓
    LlamaParse
        ↓
    Adapter
        ↓
    Text Nodes + Table Nodes
        ↓
    Chunker
        ↓
    Chroma

    Tables
        ↓
    TableProcessor
        ↓
    SQLite
        ↓
    Table Metadata
        ↓
    Chroma

    Chunks + Table Metadata
        ↓
    Document Profile
        ↓
    document_profiles
"""

import logging

from documents.ingestion.parser import (
    LlamaParseService,
    LlamaAdapter,
)

from documents.ingestion.chunker import (
    StructureAwareChunker,
)

from documents.ingestion.tables import (
    TableProcessor,
)

from documents.ingestion.sqlite import (
    SQLiteManager,
)

from documents.ingestion.vectors import (
    VectorStore,
)

from documents.ingestion.document_profile import (
    DocumentProfileBuilder,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Executes the complete ingestion flow
    for a single document.
    """

    @classmethod
    def run(
        cls,
        document,
    ):
        """
        Main orchestration entrypoint.

        If any step below fails, everything this run already wrote to
        Chroma/SQLite for this document is cleaned up before the
        exception is re-raised. Combined with SQLite being cleared up
        front (see below) and every Chroma write being an upsert, this
        makes it safe for the worker to retry a failed job from
        scratch - a retry never has to reason about what a previous,
        failed attempt already left behind.
        """

        workspace_id = document.workspace_id
        document_id = document.id

        logger.info(
            "Starting ingestion for document %s (workspace %s).",
            document_id,
            workspace_id,
        )

        try:
            cls._run(document, workspace_id, document_id)
        except Exception:
            logger.exception(
                "Ingestion failed for document %s; cleaning up "
                "partial writes.",
                document_id,
            )

            cls._cleanup_partial_ingestion(workspace_id, document_id)

            raise

        logger.info(
            "Completed ingestion for document %s.",
            document_id,
        )

    @classmethod
    def _run(cls, document, workspace_id, document_id):

        # ----------------------------------
        # Clear any SQLite tables left behind by a previous failed
        # attempt for this document, so create_table/insert_rows below
        # always start from a clean slate instead of appending onto
        # stale partial data on retry.
        # ----------------------------------

        SQLiteManager.drop_document_tables(
            workspace_id=workspace_id,
            document_id=document_id,
        )

        # ----------------------------------
        # Parse document
        # ----------------------------------

        result = (
            LlamaParseService.parse_document(
                document.file.path
            )
        )

        # ----------------------------------
        # Extract nodes
        # ----------------------------------

        text_nodes = (
            LlamaAdapter.extract_text_nodes(
                result
            )
        )

        table_nodes = (
            LlamaAdapter.extract_table_nodes(
                result
            )
        )

        logger.info(
            "Document %s: %d text node(s), %d table node(s).",
            document_id,
            len(text_nodes),
            len(table_nodes),
        )

        chunker = (
            StructureAwareChunker()
        )

        chunks = chunker.chunk(
            text_nodes
        )

        logger.info(
            "Document %s: generated %d chunk(s).",
            document_id,
            len(chunks),
        )

        VectorStore.store_chunks(
            workspace_id=workspace_id,
            document_id=document_id,
            chunks=chunks,
        )

        table_metadata = []

        for index, table_node in enumerate(
            table_nodes,
            start=1,
        ):

            processed = (
                TableProcessor.process(
                    table_node=table_node,
                    document_id=document_id,
                )
            )

            if not processed:
                continue

            table_metadata.append(
                processed["metadata"]
            )

            table_name = (
                f"ws_{workspace_id}"
                f"_doc_{document_id}"
                f"_table_{index}"
            )

            sqlite_payload = (
                processed[
                    "sqlite_payload"
                ]
            )

            headers = (
                sqlite_payload[
                    "headers"
                ]
            )

            rows = (
                sqlite_payload[
                    "rows"
                ]
            )

            SQLiteManager.create_table(
                table_name=table_name,
                columns=headers,
            )

            SQLiteManager.insert_rows(
                table_name=table_name,
                rows=rows,
            )

            VectorStore.store_table_metadata(
                workspace_id=workspace_id,
                document_id=document_id,
                table_name=table_name,
                summary=processed[
                    "semantic_text"
                ],
            )

        logger.info(
            "Document %s: processed %d table(s) into SQLite.",
            document_id,
            len(table_metadata),
        )

        DocumentProfileBuilder.create(
            workspace_id=workspace_id,
            document_id=document_id,
            document_title=document.title,
            chunks=chunks,
            table_metadata=table_metadata,
        )

    @staticmethod
    def _cleanup_partial_ingestion(workspace_id, document_id):
        """
        Best-effort removal of everything a failed run may have
        written, so a retry starts clean and a permanently-failed
        document doesn't leave orphaned vectors/tables behind.

        Deliberately swallows its own errors (logged, not raised) -
        this runs from inside an except block, and a secondary failure
        here should not mask or replace the original ingestion error.
        """

        try:
            VectorStore.delete_document(
                document_id=document_id
            )
        except Exception:
            logger.exception(
                "Cleanup: failed to delete vectors for document %s.",
                document_id,
            )

        try:
            SQLiteManager.drop_document_tables(
                workspace_id=workspace_id,
                document_id=document_id,
            )
        except Exception:
            logger.exception(
                "Cleanup: failed to drop SQLite tables for "
                "document %s.",
                document_id,
            )