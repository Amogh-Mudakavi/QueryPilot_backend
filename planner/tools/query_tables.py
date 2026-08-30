from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from django.conf import settings


logger = logging.getLogger(__name__)

MAX_RESULTS = 100
QUERY_TIMEOUT_SECONDS = 3.0


def query_tables(
    *,
    workspace_id: int,
    sql: str,
    parameters: list[Any] | tuple[Any, ...] | None = None,
    allowed_table_names: set[str] | None = None,
) -> dict[str, Any]:

    if workspace_id <= 0:
        raise ValueError(
            "workspace_id must be greater than zero."
        )

    sql = sql.strip()

    if not sql:
        raise ValueError(
            "SQL query cannot be empty."
        )

    if not sql.lower().startswith(
        ("select", "with")
    ):
        raise ValueError(
            "Only read-only SQL queries are allowed."
        )

    statement = (
        sql.rstrip()
        .rstrip(";")
        .rstrip()
    )

    if ";" in statement:
        raise ValueError(
            "Multiple SQL statements are not allowed."
        )

    allowed_tables = _get_workspace_tables(
        workspace_id
    )

    if not allowed_tables:
        raise ValueError(
            "No tables are available for this workspace."
        )

    if allowed_table_names is not None:
        allowed_tables.intersection_update(
            allowed_table_names
        )

        if not allowed_tables:
            raise ValueError(
                "No tables are allowed for this query."
            )

    logger.info(
        "query_tables: workspace_id=%s allowed_table_names=%s",
        workspace_id,
        sorted(allowed_tables),
    )

    connection = sqlite3.connect(
        f"file:{_database_path()}?mode=ro",
        uri=True,
    )

    connection.row_factory = sqlite3.Row

    tables_used: set[str] = set()

    connection.set_authorizer(
        _authorizer(
            allowed_tables,
            tables_used,
        )
    )

    start = time.monotonic()

    connection.set_progress_handler(
        lambda: int(
            time.monotonic() - start
            >= QUERY_TIMEOUT_SECONDS
        ),
        10_000,
    )

    try:
        cursor = connection.execute(
            sql,
            list(parameters or []),
        )

        rows = cursor.fetchmany(
            MAX_RESULTS
        )

        columns = [
            column[0]
            for column in (
                cursor.description or []
            )
        ]

        rows_list = [
            {
                column: row[column]
                for column in columns
            }
            for row in rows
        ]

        result = {
            "success": True,
            "columns": columns,
            "rows": rows_list,
            "row_count": len(rows_list),
            "truncated": len(rows_list) >= MAX_RESULTS,
            "tables_used": sorted(
                tables_used
            ),
        }

        logger.info(
            "query_tables: returned %d rows tables_used=%s",
            result["row_count"],
            result["tables_used"],
        )

        return result

    except sqlite3.Error as exc:
        raise ValueError(
            f"Table query failed: {exc}"
        ) from exc

    finally:
        connection.close()


def _database_path() -> str:
    path = (
        Path(settings.BASE_DIR)
        / "storage"
        / "workspace_data.db"
    )

    if not path.exists():
        raise ValueError(
            f"SQLite database not found: {path}"
        )

    return str(path)


def _get_workspace_tables(
    workspace_id: int,
) -> set[str]:

    connection = sqlite3.connect(
        f"file:{_database_path()}?mode=ro",
        uri=True,
    )

    try:
        prefix = f"ws_{workspace_id}_%"

        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name LIKE ?
            """,
            (prefix,),
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    finally:
        connection.close()


def _authorizer(
    allowed_tables: set[str],
    tables_used: set[str],
):

    forbidden = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_PRAGMA,
    }

    def check(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        source: str | None,
    ) -> int:

        if action in forbidden:
            return sqlite3.SQLITE_DENY

        if action == sqlite3.SQLITE_READ:

            table_name = arg1

            if (
                table_name is not None
                and table_name not in allowed_tables
                and table_name not in {
                    "sqlite_master",
                    "sqlite_temp_master",
                }
            ):
                return sqlite3.SQLITE_DENY

            if table_name in allowed_tables:
                tables_used.add(
                    table_name
                )

        return sqlite3.SQLITE_OK

    return check