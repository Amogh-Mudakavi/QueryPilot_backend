import logging

from documents.ingestion.vectors import (
    VectorStore,
)

logger = logging.getLogger(__name__)


class DocumentProfileBuilder:
    """
    Builds a high-level document profile.

    Purpose:
        Planner retrieval.

    One profile = one document.

    Captures:

    - document title
    - section headings
    - first chunk preview
    - table metadata
    """
    @staticmethod
    def build_profile_text(
        document_title: str,
        chunks: list[dict],
        table_metadata: list[dict],
    ) -> str:

        lines = [
            f"Document: {document_title}",
            "",
            "Sections:",
            "",
        ]

        seen_headings = set()

        for chunk in chunks:

            heading = chunk.get(
                "heading",
                "",
            )

            if (
                not heading
                or heading in seen_headings
            ):
                continue

            seen_headings.add(
                heading
            )

            preview = chunk.get(
                "text",
                "",
            )[:300]

            lines.append(
                f"Heading: {heading}"
            )

            lines.append(
                f"Preview: {preview}"
            )

            lines.append("")

        lines.append(
            "Tables:"
        )

        lines.append("")

        for table in table_metadata:

            lines.append(
                f"Title: {table.get('title', '')}"
            )

            columns = table.get(
                "columns",
                [],
            )

            if columns:

                lines.append(
                    "Columns: "
                    + ", ".join(columns)
                )

            lines.append(
                f"Rows: {table.get('row_count', 0)}"
            )

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def create(
        cls,
        *,
        workspace_id: int,
        document_id: int,
        document_title: str,
        chunks: list[dict],
        table_metadata: list[dict],
    ) -> None:

        profile_text = (
            cls.build_profile_text(
                document_title=document_title,
                chunks=chunks,
                table_metadata=table_metadata,
            )
        )

        VectorStore.store_document_profile(
            workspace_id=workspace_id,
            document_id=document_id,
            document_name=document_title,
            profile_text=profile_text,
        )

        logger.info(
            "Stored document profile for document %s",
            document_id,
        )