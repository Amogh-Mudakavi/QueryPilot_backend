"""Services Schema Registry module for QueryPilot.

Two layers live here:

* ``get_workspace_schema`` / ``get_workspace_catalog``: the original stateless
  functions. Their signatures and return shapes are unchanged.
* ``WorkspaceResources``: a REQUEST-SCOPED holder that builds the same
  metadata lazily, at most once per request, and hands out filtered views of
  it. It is created per request, stored on the request's own state, and never
  shared between requests or workspaces (it is not a global cache).

Metadata is built in two tiers so a request only pays for what it uses:

1. table index   - which structured tables exist, their document and columns.
                   Three queries in total, whatever the number of tables.
2. table details - sample rows (numeric-column detection) and an exact row
                   count. Two queries per table, fetched only for the tables
                   a stage actually needs and never fetched twice.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable

from django.db import connections

from documents.ingestion.sqlite import SQLiteManager
from documents.models import Document

logger = logging.getLogger(__name__)

STRUCTURED_READ_DB = "structured_read"


def _normalize_document_ids(
    allowed_document_ids: Iterable[int] | None,
) -> set[int] | None:
    if allowed_document_ids is None:
        return None
    return {int(value) for value in allowed_document_ids}


def _load_table_index(
    connection,
    workspace_id: int,
    document_ids: set[int] | None,
    documents_by_id: dict[int, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the workspace's structured tables without sampling or counting.

    Authorization is identical to the original schema build: only tables with
    this workspace's ``ws_<id>_`` prefix are considered, optionally narrowed to
    ``document_ids``, and each table's document must still exist in this
    workspace. Table names are validated before they are used as identifiers.

    ``documents_by_id`` lets a caller that already loaded the workspace's
    documents skip the validation query; it must hold ONLY documents of
    ``workspace_id``.
    """
    prefix = f"ws_{workspace_id}_"

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """
        )
        all_table_names = [row[0] for row in cursor.fetchall()]

    table_names = [
        name
        for name in all_table_names
        if isinstance(name, str) and name.startswith(prefix)
    ]

    if document_ids is not None:
        table_names = [
            name for name in table_names if _get_document_id(name) in document_ids
        ]

    if not table_names:
        return []

    ids = {
        did for did in (_get_document_id(name) for name in table_names) if did
    }

    if documents_by_id is None:
        documents = {
            doc.id: doc
            for doc in Document.objects.filter(
                workspace_id=workspace_id,
                id__in=ids,
            ).only("id", "title", "source_type", "status")
        }
    else:
        documents = {
            doc_id: documents_by_id[doc_id] for doc_id in ids if doc_id in documents_by_id
        }

    table_names = [
        name for name in table_names if _get_document_id(name) in set(documents)
    ]

    if not table_names:
        return []

    validated = [
        SQLiteManager.validate_identifier(name, "table name")
        for name in sorted(table_names)
    ]

    # One COLUMNS query for every table (was one query per table).
    placeholders = ", ".join(["%s"] * len(validated))
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """,
            validated,
        )
        columns_by_table: dict[str, list[str]] = {}
        for table_name, column_name in cursor.fetchall():
            columns_by_table.setdefault(table_name, []).append(column_name)

    index: list[dict[str, Any]] = []
    for table_name in validated:
        columns = columns_by_table.get(table_name)
        if not columns:
            continue
        document_id = _get_document_id(table_name)
        document = documents.get(document_id)
        index.append(
            {
                "document_id": document_id,
                "document_title": getattr(document, "title", None),
                "source_type": getattr(document, "source_type", None),
                "document_status": getattr(document, "status", None),
                "table_name": table_name,
                "columns": columns,
            }
        )
    return index


def _load_table_details(
    connection,
    table_name: str,
    columns: list[str],
) -> dict[str, Any]:
    """Sample rows (for numeric-column detection) and the exact row count."""
    table_name = SQLiteManager.validate_identifier(table_name, "table name")

    # Table name is validated above because identifiers cannot be passed as
    # normal query parameters.
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 5")
        sample_rows = cursor.fetchall()

    return {
        "numeric_columns": _detect_numeric_columns(columns, sample_rows),
        "row_count": _count_rows(connection, table_name),
        "sample_rows": [dict(zip(columns, row)) for row in sample_rows[:5]],
    }


def _schema_item(
    index_item: dict[str, Any],
    details: dict[str, Any],
    include_sample_rows: bool,
) -> dict[str, Any]:
    item = dict(index_item)
    item["columns"] = list(index_item["columns"])  # never alias the cache
    item["numeric_columns"] = list(details["numeric_columns"])
    item["row_count"] = details["row_count"]
    if include_sample_rows:
        item["sample_rows"] = [dict(row) for row in details["sample_rows"]]
    return item


def get_workspace_schema(
    workspace_id: int,
    allowed_document_ids: list[int] | None = None,
    include_sample_rows: bool = False,
) -> list[dict[str, Any]]:
    """Return authoritative structured-data schemas for a workspace.

    The authoritative schema, sample rows, and row counts are read
    directly from RDS #2 through the dedicated read-only connection.

    When ``include_sample_rows`` is True, expose the sample rows already
    fetched for numeric-column detection so downstream table selection
    can reuse them without issuing another SELECT against the same table.

    Agent requests should prefer ``WorkspaceResources`` so this work happens
    once per request instead of once per stage.
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    document_ids = _normalize_document_ids(allowed_document_ids)
    if document_ids == set():
        return []

    connection = connections[STRUCTURED_READ_DB]
    try:
        index = _load_table_index(connection, workspace_id, document_ids)
        return [
            _schema_item(
                item,
                _load_table_details(connection, item["table_name"], item["columns"]),
                include_sample_rows,
            )
            for item in index
        ]
    finally:
        connection.close()


def _build_catalog(
    documents: Iterable[Any],
    schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
                        "columns": list(table["columns"]),
                        "row_count": table.get("row_count"),
                    }
                    for table in tables
                ],
            }
        )
    return catalog


def _load_workspace_documents(workspace_id: int) -> list[Any]:
    return list(
        Document.objects.filter(workspace_id=workspace_id)
        .only("id", "title", "source_type", "status", "uploaded_at")
        .order_by("uploaded_at", "id")
    )


def get_workspace_catalog(
    workspace_id: int,
    schema: list[dict[str, Any]] | None = None,
    include_row_counts: bool = True,
) -> list[dict[str, Any]]:
    """Return a deterministic workspace resource catalog.

    When schema is supplied, reuse it instead of rebuilding the
    structured-data schema for the same operation.

    ``include_row_counts=False`` lets a caller that only needs documents and
    table names skip the per-table sampling/COUNT work; table entries then
    carry ``row_count: None``. The default keeps the original behavior.
    """
    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    documents = _load_workspace_documents(workspace_id)

    if schema is None:
        if include_row_counts:
            schema = get_workspace_schema(workspace_id)
        else:
            connection = connections[STRUCTURED_READ_DB]
            try:
                schema = _load_table_index(connection, workspace_id, None)
            finally:
                connection.close()

    return _build_catalog(documents, schema)


class WorkspaceResources:
    """Structured-resource metadata for ONE request in ONE workspace.

    * Built lazily and at most once per tier (see module docstring).
    * Bound to a single ``workspace_id``; the agent refuses to use an instance
      for any other workspace.
    * Never module-level: the request's state carries the instance, so there
      is no cross-request or cross-tenant sharing and nothing to invalidate.
    * Every table that can appear in a view came through the same workspace
      prefix / document-ownership checks as ``get_workspace_schema``.
    """

    def __init__(self, workspace_id: int) -> None:
        if int(workspace_id) <= 0:
            raise ValueError("workspace_id must be greater than zero.")
        self.workspace_id = int(workspace_id)
        self._lock = threading.RLock()
        self._documents: list[Any] | None = None
        self._index: list[dict[str, Any]] | None = None
        self._details: dict[str, dict[str, Any]] = {}
        # Observability / tests: how much real work this request did.
        self.loads = {"documents": 0, "index": 0, "detail_tables": 0}

    def __repr__(self) -> str:  # keeps state dumps short
        return f"WorkspaceResources(workspace_id={self.workspace_id})"

    # -- loading ----------------------------------------------------------

    def _documents_loaded(self) -> list[Any]:
        if self._documents is None:
            started = time.perf_counter()
            self._documents = _load_workspace_documents(self.workspace_id)
            self.loads["documents"] += 1
            logger.info(
                "[RESOURCES] workspace=%s loaded=documents count=%d seconds=%.3f",
                self.workspace_id,
                len(self._documents),
                time.perf_counter() - started,
            )
        return self._documents

    def _index_loaded(self) -> list[dict[str, Any]]:
        if self._index is None:
            documents = {doc.id: doc for doc in self._documents_loaded()}
            started = time.perf_counter()
            connection = connections[STRUCTURED_READ_DB]
            try:
                self._index = _load_table_index(
                    connection,
                    self.workspace_id,
                    None,
                    documents_by_id=documents,
                )
            finally:
                connection.close()
            self.loads["index"] += 1
            logger.info(
                "[RESOURCES] workspace=%s loaded=table_index tables=%d seconds=%.3f",
                self.workspace_id,
                len(self._index),
                time.perf_counter() - started,
            )
        return self._index

    def _ensure_details(self, items: list[dict[str, Any]]) -> None:
        missing = [item for item in items if item["table_name"] not in self._details]
        if not missing:
            return
        started = time.perf_counter()
        connection = connections[STRUCTURED_READ_DB]
        try:
            for item in missing:
                self._details[item["table_name"]] = _load_table_details(
                    connection,
                    item["table_name"],
                    item["columns"],
                )
        finally:
            connection.close()
        self.loads["detail_tables"] += len(missing)
        logger.info(
            "[RESOURCES] workspace=%s loaded=table_details tables=%d seconds=%.3f",
            self.workspace_id,
            len(missing),
            time.perf_counter() - started,
        )

    # -- views ------------------------------------------------------------

    def table_index(
        self,
        document_ids: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Tables (document + columns only), optionally for given documents."""
        wanted = _normalize_document_ids(document_ids)
        if wanted == set():
            return []
        with self._lock:
            index = self._index_loaded()
            return [
                {**item, "columns": list(item["columns"])}
                for item in index
                if wanted is None or item["document_id"] in wanted
            ]

    def schema(
        self,
        document_ids: Iterable[int] | None = None,
        include_sample_rows: bool = False,
    ) -> list[dict[str, Any]]:
        """Same shape as ``get_workspace_schema`` for the given documents.

        Only tables of the requested documents are sampled/counted, and each
        table at most once per request.
        """
        wanted = _normalize_document_ids(document_ids)
        if wanted == set():
            return []
        with self._lock:
            index = self._index_loaded()
            items = [
                item
                for item in index
                if wanted is None or item["document_id"] in wanted
            ]
            self._ensure_details(items)
            return [
                _schema_item(item, self._details[item["table_name"]], include_sample_rows)
                for item in items
            ]

    def catalog(self, include_row_counts: bool = False) -> list[dict[str, Any]]:
        """Same shape as ``get_workspace_catalog``.

        Row counts are only populated when ``include_row_counts`` is True
        (which loads details for every table); otherwise they are None.
        """
        with self._lock:
            documents = self._documents_loaded()
            if include_row_counts:
                schema = self.schema()
            else:
                schema = self._index_loaded()
            return _build_catalog(documents, schema)


def _count_rows(
    connection,
    table_name: str,
) -> int:
    """Return the number of rows in one validated structured table."""

    table_name = SQLiteManager.validate_identifier(
        table_name,
        "table name",
    )

    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")

        return int(cursor.fetchone()[0])


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


def _is_number(
    value: Any,
) -> bool:
    try:
        float(str(value).replace(",", ""))
        return True

    except (
        TypeError,
        ValueError,
    ):
        return False


def _get_document_id(
    table_name: str,
) -> int | None:
    parts = table_name.split("_")

    try:
        index = parts.index("doc")

        return int(parts[index + 1])

    except (
        ValueError,
        IndexError,
    ):
        return None