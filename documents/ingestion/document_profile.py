import logging

from documents.ingestion.vectors import (
    VectorStore,
)

logger=logging.getLogger(__name__)

MAX_PROFILE_CHARS=12000
MAX_SECTION_PREVIEW_CHARS=300


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
        document_title:str,
        chunks:list[dict],
        table_metadata:list[dict],
    )->str:

        lines=[
            f"Document: {document_title}",
            "",
            "Sections:",
            "",
        ]

        seen_headings=set()
        current_size=sum(
            len(line)+1
            for line in lines
        )

        for chunk in chunks:

            heading=chunk.get(
                "heading",
                "",
            )

            if (
                not heading
                or heading in seen_headings
            ):
                continue

            preview=(
                chunk.get("text","")
                [:MAX_SECTION_PREVIEW_CHARS]
            )

            section_lines=[
                f"Heading: {heading}",
                f"Preview: {preview}",
                "",
            ]

            section_size=sum(
                len(line)+1
                for line in section_lines
            )

            if (
                current_size+section_size
                >MAX_PROFILE_CHARS
            ):
                break

            seen_headings.add(
                heading
            )

            lines.extend(
                section_lines
            )

            current_size+=section_size

        if current_size+len("Tables:\n\n")<=MAX_PROFILE_CHARS:

            lines.extend(
                [
                    "Tables:",
                    "",
                ]
            )

            current_size+=len(
                "Tables:\n\n"
            )

        for table in table_metadata:

            table_lines=[
                f"Title: {table.get('title','')}",
            ]

            columns=table.get(
                "columns",
                [],
            )

            if columns:

                table_lines.append(
                    "Columns: "
                    + ", ".join(
                        str(column)
                        for column in columns
                    )
                )

            table_lines.append(
                f"Rows: {table.get('row_count',0)}"
            )

            table_lines.append("")

            table_size=sum(
                len(line)+1
                for line in table_lines
            )

            if (
                current_size+table_size
                >MAX_PROFILE_CHARS
            ):
                break

            lines.extend(
                table_lines
            )

            current_size+=table_size

        return "\n".join(lines)


    @classmethod
    def create(
        cls,
        *,
        workspace_id:int,
        document_id:int,
        document_title:str,
        chunks:list[dict],
        table_metadata:list[dict],
    )->None:

        profile_text=cls.build_profile_text(
            document_title=document_title,
            chunks=chunks,
            table_metadata=table_metadata,
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