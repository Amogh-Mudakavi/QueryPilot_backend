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
import time

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

        workspace_id = document.workspace_id
        document_id = document.id

        started = time.monotonic()

        logger.info(
            "Starting ingestion for document %s (workspace %s).",
            document_id,
            workspace_id,
        )

        try:
            cls._run(
                document,
                workspace_id,
                document_id,
            )

        except Exception:

            logger.exception(
                "Ingestion failed for document %s; cleaning up "
                "partial writes.",
                document_id,
            )

            cls._cleanup_partial_ingestion(
                workspace_id,
                document_id,
            )

            raise

        logger.info(
            "Completed ingestion for document %s in %.2fs.",
            document_id,
            time.monotonic() - started,
        )

    @classmethod
    def _run(
        cls,
        document,
        workspace_id,
        document_id,
    ):

        # Remove tables from a previous failed attempt.
        SQLiteManager.drop_document_tables(
            workspace_id=workspace_id,
            document_id=document_id,
        )

        result = LlamaParseService.parse_document(
            document.file.path,
            file_hash=document.file_hash,
        )

        text_nodes = LlamaAdapter.extract_text_nodes(
            result
        )

        table_nodes = LlamaAdapter.extract_table_nodes(
            result
        )

        # The raw LlamaParse result is no longer needed.
        # Release it before chunking to reduce peak memory.
        del result

        logger.info(
            "Document %s: %d text node(s), %d table node(s).",
            document_id,
            len(text_nodes),
            len(table_nodes),
        )

        chunker = StructureAwareChunker()

        chunks = chunker.chunk(
            text_nodes
        )

        # Chunking is complete; raw per-page text is no longer needed.
        del text_nodes

        logger.info(
            "Document %s: generated %d chunk(s).",
            document_id,
            len(chunks),
        )

        stage_started = time.monotonic()

        VectorStore.store_chunks(
            workspace_id=workspace_id,
            document_id=document_id,
            chunks=chunks,
        )

        logger.info(
            "Document %s: chunk embedding + Chroma upsert completed "
            "in %.2fs.",
            document_id,
            time.monotonic() - stage_started,
        )

        table_metadata = []

        for index, table_node in enumerate(
            table_nodes,
            start=1,
        ):

            # CHECKPOINT 3B: verify the exact table payload entering the
            # structured-data stage. If values are already empty here, the
            # problem is upstream of TableProcessor/SQLite.
            logger.info(
                "[CHECKPOINT 3B] TABLE PROCESSOR INPUT #%d: page=%s "
                "rows_count=%d headers=%r rows=%r.",
                index,
                table_node.get("page"),
                len(table_node.get("rows") or []),
                (table_node.get("rows") or [])[0]
                if (table_node.get("rows") or []) else [],
                (table_node.get("rows") or [])[1:]
                if len(table_node.get("rows") or []) > 1 else [],
            )

            processed = TableProcessor.process(
                table_node=table_node,
                document_id=document_id,
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

            sqlite_payload = processed[
                "sqlite_payload"
            ]

            headers = sqlite_payload[
                "headers"
            ]

            rows = sqlite_payload[
                "rows"
            ]

            # CHECKPOINT 4: exact payload immediately before SQLite insert.
            logger.info(
                "[CHECKPOINT 4] SQLITE INPUT #%d: table=%s headers=%r "
                "rows_count=%d rows=%r.",
                index,
                table_name,
                headers,
                len(rows),
                rows,
            )

            SQLiteManager.create_table(
                table_name=table_name,
                columns=headers,
            )

            SQLiteManager.insert_rows(
                table_name=table_name,
                rows=rows,
            )

            logger.info(
                "[CHECKPOINT 5] SQLITE INSERT COMPLETED #%d: table=%s rows_count=%d.",
                index,
                table_name,
                len(rows),
            )

            VectorStore.store_table_metadata(
                workspace_id=workspace_id,
                document_id=document_id,
                table_name=table_name,
                summary=processed["semantic_text"],
            )

        del table_nodes

        logger.info(
            "Document %s: processed %d table(s) into SQLite.",
            document_id,
            len(table_metadata),
        )

        stage_started = time.monotonic()

        DocumentProfileBuilder.create(
            workspace_id=workspace_id,
            document_id=document_id,
            document_title=document.title,
            chunks=chunks,
            table_metadata=table_metadata,
        )

        logger.info(
            "Document %s: document profile created in %.2fs.",
            document_id,
            time.monotonic() - stage_started,
        )

        del chunks
        del table_metadata

    @staticmethod
    def _cleanup_partial_ingestion(
        workspace_id,
        document_id,
    ):

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