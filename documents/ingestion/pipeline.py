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
        """

        workspace_id = document.workspace_id
        document_id = document.id

        print(
            f"Processing document {document_id}"
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

        print(
            f"Text Nodes: {len(text_nodes)}"
        )

        print(
            f"Table Nodes: {len(table_nodes)}"
        )

        

        chunker = (
            StructureAwareChunker()
        )

        chunks = chunker.chunk(
            text_nodes
        )

        print(
            f"Generated {len(chunks)} chunks"
        )


        VectorStore.store_chunks(
            workspace_id=workspace_id,
            document_id=document_id,
            chunks=chunks,
        )

        print(
            "Stored text chunks"
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

        print(
            "Processed tables"
        )


        print("BEFORE PROFILE")
        DocumentProfileBuilder.create(
            workspace_id=workspace_id,
            document_id=document_id,
            document_title=document.title,
            chunks=chunks,
            table_metadata=table_metadata,
        )

        print(
            "Stored document profile"
        )

        print(
            f"Completed document {document_id}"
        )