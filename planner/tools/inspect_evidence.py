"""
planner/tools/inspect_evidence.py

Table-focused evidence inspection for QueryPilot.

Purpose:
    search_evidence() can discover that a relevant table exists.

    inspect_evidence() then exposes the table's REAL SQLite schema
    (the authoritative source) plus its semantic description, so the
    LangGraph agent has the exact column names it needs to write SQL.

This tool does NOT:
    - perform vector search
    - perform BM25
    - call an LLM
    - execute arbitrary SQL
    - decide what the answer should be
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from django.conf import settings

from documents.ingestion.vectors import VectorStore


def _database_path() -> str:
    path = Path(settings.BASE_DIR) / "storage" / "workspace_data.db"

    if not path.exists():
        raise ValueError(f"SQLite database not found: {path}")

    return str(path)


def _get_table_columns(workspace_id: int, table_name: str) -> list[str]:
    """
    Authoritative schema lookup. This is the real SQLite table, not
    whatever text got embedded into the vector store.
    """

    prefix = f"ws_{workspace_id}_"

    if not table_name.startswith(prefix):
        raise ValueError(
            "Table does not belong to this workspace."
        )

    connection = sqlite3.connect(
        f"file:{_database_path()}?mode=ro",
        uri=True,
    )

    try:
        exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (table_name,),
        ).fetchone()

        if not exists:
            raise ValueError(
                f"Table '{table_name}' does not exist."
            )

        rows = connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()

        return [row[1] for row in rows]
    finally:
        connection.close()


def _get_semantic_content(
    workspace_id: int,
    document_id: int,
    table_name: str,
) -> tuple[str | None, dict | None]:
    """
    Best-effort lookup of the human-readable description stored for
    semantic search. This is CONTEXT, never the source of truth for
    column names.
    """

    collection = VectorStore.knowledge_collection()

    records = collection.get(
        where={
            "workspace_id": workspace_id,
            "document_id": document_id,
            "type": "table",
            "table_name": table_name,
        },
        include=["documents", "metadatas"],
    )

    documents = records.get("documents", [])
    metadatas = records.get("metadatas", [])

    if documents:
        metadata = metadatas[0] if metadatas else {}
        return documents[0], metadata if isinstance(metadata, dict) else {}

    return None, None


def inspect_evidence(
    *,
    workspace_id: int,
    document_id: int,
    table_name: str,
) -> dict[str, Any]:
    """
    Inspect a discovered table.

    Returns the REAL SQLite column names (authoritative) plus the
    stored semantic description (context only).

    Raises:
        ValueError: if identifiers are invalid or the table
        doesn't exist / doesn't belong to this workspace.
    """

    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    if document_id <= 0:
        raise ValueError("document_id must be greater than zero.")

    table_name = table_name.strip()

    if not table_name:
        raise ValueError("table_name cannot be empty.")

    columns = _get_table_columns(workspace_id, table_name)

    content, metadata = _get_semantic_content(
        workspace_id,
        document_id,
        table_name,
    )

    return {
        "found": True,
        "source_type": "table_schema",
        "workspace_id": workspace_id,
        "document_id": document_id,
        "table_name": table_name,
        "columns": columns,
        "content": content,
        "metadata": metadata or {},
    }