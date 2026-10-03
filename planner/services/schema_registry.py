"""Services Schema Registry module for QueryPilot."""

from __future__ import annotations

import sqlite3
from typing import Any

from documents.ingestion.sqlite import SQLiteManager
from documents.models import Document


def get_workspace_schema(
    workspace_id: int,
    allowed_document_ids: list[int] | None = None,
    include_sample_rows: bool = False,
) -> list[dict[str, Any]]:
    """Return authoritative SQLite schemas for tables in a workspace.

    When ``include_sample_rows`` is True, expose the sample rows already
    fetched for numeric-column detection so downstream table selection can
    reuse them without issuing another SELECT against the same table.
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    document_ids = (
        None
        if allowed_document_ids is None
        else {int(value) for value in allowed_document_ids}
    )
    if document_ids == set():
        return []

    prefix = f"ws_{workspace_id}_"
    table_names = [
        name for name in SQLiteManager.list_tables() if name.startswith(prefix)
    ]

    if document_ids is not None:
        table_names = [
            name for name in table_names if _get_document_id(name) in document_ids
        ]

    if not table_names:
        return []

    ids = {did for did in (_get_document_id(name) for name in table_names) if did}
    documents = {
        doc.id: doc
        for doc in Document.objects.filter(
            workspace_id=workspace_id,
            id__in=ids,
        ).only("id", "title", "source_type", "status")
    }
    valid_document_ids = set(documents)
    table_names = [
        name for name in table_names if _get_document_id(name) in valid_document_ids
    ]
    if not table_names:
        return []

    connection = sqlite3.connect(
        f"file:{SQLiteManager.DB_PATH}?mode=ro",
        uri=True,
    )
    try:
        schema: list[dict[str, Any]] = []
        for table_name in sorted(table_names):
            rows = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            columns = [row[1] for row in rows]
            sample_rows = connection.execute(
                f'SELECT * FROM "{table_name}" LIMIT 5'
            ).fetchall()
            document_id = _get_document_id(table_name)
            document = documents.get(document_id)

            item: dict[str, Any] = {
                "document_id": document_id,
                "document_title": getattr(document, "title", None),
                "source_type": getattr(document, "source_type", None),
                "document_status": getattr(document, "status", None),
                "table_name": table_name,
                "columns": columns,
                "numeric_columns": _detect_numeric_columns(columns, sample_rows),
                "row_count": _count_rows(connection, table_name),
            }

            if include_sample_rows:
                item["sample_rows"] = [
                    dict(zip(columns, row)) for row in sample_rows[:5]
                ]

            schema.append(item)
        return schema
    finally:
        connection.close()


def get_workspace_catalog(
    workspace_id: int,
    schema: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic workspace resource catalog.

    When schema is supplied, reuse it instead of rebuilding the SQLite
    schema for the same operation.
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    documents = list(
        Document.objects.filter(workspace_id=workspace_id)
        .only("id", "title", "source_type", "status", "uploaded_at")
        .order_by("uploaded_at", "id")
    )
    if schema is None:
        schema = get_workspace_schema(workspace_id)

    by_document: dict[int, list[dict[str, Any]]] = {}
    for item in schema:
        by_document.setdefault(int(item["document_id"]), []).append(item)

    catalog: list[dict[str, Any]] = []
    for document in documents:
        tables = by_document.get(document.id, [])
        catalog.append(
            {
                "document_id": document.id,
                "title": document.title,
                "source_type": document.source_type,
                "status": document.status,
                "tables": [
                    {
                        "table_name": table["table_name"],
                        "columns": table["columns"],
                        "row_count": table["row_count"],
                    }
                    for table in tables
                ],
            }
        )
    return catalog


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])


def _detect_numeric_columns(
    columns: list[str],
    sample_rows: list[tuple],
) -> list[str]:
    result: list[str] = []
    for index, column in enumerate(columns):
        values = [row[index] for row in sample_rows if row[index] not in (None, "")]
        if values and all(_is_number(value) for value in values):
            result.append(column)
    return result


def _is_number(value: Any) -> bool:

    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _get_document_id(table_name: str) -> int | None:
    parts = table_name.split("_")
    try:
        index = parts.index("doc")
        return int(parts[index + 1])
    except (ValueError, IndexError):
        return None
