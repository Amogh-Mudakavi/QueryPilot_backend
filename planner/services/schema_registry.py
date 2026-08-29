from __future__ import annotations

import sqlite3
from typing import Any

from documents.ingestion.sqlite import SQLiteManager


def get_workspace_schema(
    workspace_id: int,
) -> list[dict[str, Any]]:
    """
    Return the authoritative schema for every table in one workspace.

    Also flags which columns are actually numeric, even though every
    column is physically stored as TEXT (see sqlite.py create_table).
    The SQL generator needs this to know when it must CAST - plain
    TEXT sorting/comparison on numeric-looking strings is wrong.
    """

    prefix = f"ws_{workspace_id}_"

    table_names = [
        table_name
        for table_name in SQLiteManager.list_tables()
        if table_name.startswith(prefix)
    ]

    if not table_names:
        return []

    connection = sqlite3.connect(
        f"file:{SQLiteManager.DB_PATH}?mode=ro",
        uri=True,
    )

    try:
        schema = []

        for table_name in table_names:
            columns = [
                row[1]
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]

            sample_rows = connection.execute(
                f'SELECT * FROM "{table_name}" LIMIT 5'
            ).fetchall()

            numeric_columns = _detect_numeric_columns(
                columns, sample_rows
            )

            document_id = _get_document_id(table_name)

            schema.append(
                {
                    "document_id": document_id,
                    "table_name": table_name,
                    "columns": columns,
                    "numeric_columns": numeric_columns,
                }
            )

        return schema

    finally:
        connection.close()


def _detect_numeric_columns(
    columns: list[str],
    sample_rows: list[tuple],
) -> list[str]:
    numeric = []

    for index, column in enumerate(columns):
        values = [
            row[index]
            for row in sample_rows
            if row[index] not in (None, "")
        ]

        if values and all(_is_number(value) for value in values):
            numeric.append(column)

    return numeric


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _get_document_id(table_name: str) -> int | None:
    parts = table_name.split("_")

    try:
        doc_index = parts.index("doc")
        return int(parts[doc_index + 1])
    except (ValueError, IndexError):
        return None