from __future__ import annotations

import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from django.conf import settings


logger = logging.getLogger(__name__)

MAX_RESULTS = 100
QUERY_TIMEOUT_SECONDS = 3.0

MAX_PARAMETERS = 50
MAX_PARAMETER_STRING_LENGTH = 1000


def _validate_parameters(
    parameters: Any,
) -> list[Any]:
    """
    Enforce the scalar-only parameter contract at the tool boundary.

    Allowed: str, int, float (finite), bool, None.
    Rejected: dict, list, tuple, bytes, nested structures, anything else.

    Raised as ValueError so the caller handles it the same way as any
    other query_tables failure, before SQLite is ever touched.
    """

    if parameters is None:
        return []

    if not isinstance(parameters, (list, tuple)):
        raise ValueError(
            "SQL parameters must be a list of scalar values."
        )

    if len(parameters) > MAX_PARAMETERS:
        raise ValueError(
            f"Too many SQL parameters (max {MAX_PARAMETERS})."
        )

    validated: list[Any] = []

    for position, value in enumerate(parameters):

        if value is None:
            validated.append(value)
            continue

        # bool is a subclass of int, so it is covered here as well.
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                f"SQL parameter at position {position} must be a "
                f"string, number, boolean, or null; got "
                f"{type(value).__name__}."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"SQL parameter at position {position} must be a "
                f"finite number."
            )

        if (
            isinstance(value, str)
            and len(value) > MAX_PARAMETER_STRING_LENGTH
        ):
            raise ValueError(
                f"SQL parameter at position {position} is too long "
                f"(max {MAX_PARAMETER_STRING_LENGTH} characters)."
            )

        validated.append(value)

    return validated


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

    # Also covers needs_sql=true with sql=None from the SQL generator.
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(
            "SQL query cannot be empty."
        )

    sql = sql.strip()

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

    validated_parameters = _validate_parameters(
        parameters
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
            validated_parameters,
        )

        # Fetch one extra row so "truncated" is only True when there
        # really were more rows than the cap (not merely exactly the cap).
        fetched = cursor.fetchmany(
            MAX_RESULTS + 1
        )

        truncated = len(fetched) > MAX_RESULTS
        rows = fetched[:MAX_RESULTS]

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
            "truncated": truncated,
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
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

        prefix = f"ws_{workspace_id}_"

        return {
            row[0]
            for row in rows
            if isinstance(row[0], str)
            and row[0].startswith(prefix)
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

            # SQLite schema tables are global metadata, not workspace data.
            # Never expose them through planner-generated SQL.
            if table_name in {
                "sqlite_master",
                "sqlite_temp_master",
                "sqlite_schema",
                "sqlite_temp_schema",
            }:
                return sqlite3.SQLITE_DENY

            if table_name is not None and table_name not in allowed_tables:
                return sqlite3.SQLITE_DENY

            if table_name in allowed_tables:
                tables_used.add(
                    table_name
                )

        return sqlite3.SQLITE_OK

    return check