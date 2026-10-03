"""Ingestion Tables module for QueryPilot."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_SEMANTIC_CELL_CHARS = 500
MAX_SAMPLE_ROWS = 10


class TableProcessor:
    """
    Processes table nodes extracted from LlamaParse.

    Responsibilities:

    1. Extract table structure
    2. Normalize headers and rows
    3. Validate table quality
    4. Build retrieval metadata
    5. Build semantic representation
    """

    @staticmethod
    def extract_table(
        table_node: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(table_node, dict):
            raise ValueError(
                "Expected a dict table_node from LlamaParse, "
                f"got {type(table_node).__name__}."
            )

        raw_rows = table_node.get("rows") or []

        if not isinstance(raw_rows, (list, tuple)):
            raise ValueError("Table rows must be a list or tuple.")

        if raw_rows:
            headers = raw_rows[0]
            rows = raw_rows[1:]
        else:
            headers = []
            rows = []

        if not isinstance(headers, (list, tuple)):
            headers = [headers]

        return {
            "title": f"Table Page {table_node.get('page', 0)}",
            "page": table_node.get("page", 0),
            "headers": list(headers),
            "rows": rows,
        }

    @staticmethod
    def normalize_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:

        normalized_headers = []
        used_headers = set()

        for index, header in enumerate(headers):
            name = str(header).strip() or f"column_{index}"

            base_name = name
            suffix = 2

            while name.lower() in used_headers:
                name = f"{base_name}_{suffix}"
                suffix += 1

            used_headers.add(name.lower())
            normalized_headers.append(name)

        column_count = len(normalized_headers)
        normalized_rows = []

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue

            cleaned_row = []

            for index in range(column_count):
                if index < len(row):
                    value = row[index]

                    if value is None:
                        value = ""
                    else:
                        value = str(value).strip()
                else:
                    value = ""

                cleaned_row.append(value)

            if not any(cleaned_row):
                continue

            normalized_rows.append(cleaned_row)

        return normalized_headers, normalized_rows

    @staticmethod
    def validate_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> bool:

        if not headers:
            return False

        if len(headers) < 2:
            return False

        if not rows:
            return False

        return True

    @staticmethod
    def build_metadata(
        document_id: int,
        page: int,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> dict[str, Any]:

        return {
            "document_id": document_id,
            "page": page,
            "title": title,
            "columns": headers,
            "row_count": len(rows),
        }

    @staticmethod
    def serialize_for_embedding(
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:

        lines = [
            f"Table: {title}",
            "",
            "Columns:",
            ", ".join(headers),
            "",
            f"Contains {len(rows)} rows.",
            "",
            "Sample Rows:",
        ]

        for row in rows[:MAX_SAMPLE_ROWS]:
            values = []

            for value in row:
                text = str(value)

                if len(text) > MAX_SEMANTIC_CELL_CHARS:
                    text = text[:MAX_SEMANTIC_CELL_CHARS] + "..."

                values.append(text)

            lines.append(" | ".join(values))

        return "\n".join(lines)

    @classmethod
    def process(
        cls,
        table_node: dict[str, Any],
        document_id: int,
    ) -> dict[str, Any] | None:

        table = cls.extract_table(table_node)

        headers, rows = cls.normalize_table(
            table["headers"],
            table["rows"],
        )

        if not cls.validate_table(
            headers,
            rows,
        ):
            logger.info(
                "Table rejected for document %s (page %s): "
                "%d headers, %d rows after normalization.",
                document_id,
                table.get("page"),
                len(headers),
                len(rows),
            )
            return None

        metadata = cls.build_metadata(
            document_id=document_id,
            page=table["page"],
            title=table["title"],
            headers=headers,
            rows=rows,
        )

        semantic_text = cls.serialize_for_embedding(
            title=table["title"],
            headers=headers,
            rows=rows,
        )

        payload = {
            "sqlite_payload": {
                "headers": headers,
                "rows": rows,
            },
            "metadata": metadata,
            "semantic_text": semantic_text,
        }

        logger.debug(
            "Processed table for document %s (page %s): %d headers, %d rows.",
            document_id,
            table.get("page"),
            len(headers),
            len(rows),
        )

        return payload
