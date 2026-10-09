"""Tools Query Tables module for QueryPilot."""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

from django.db import connections

from documents.ingestion.sqlite import SQLiteManager


logger = logging.getLogger(__name__)


STRUCTURED_READ_DB = "structured_read"

MAX_RESULTS = 100
QUERY_TIMEOUT_SECONDS = 3.0

MAX_PARAMETERS = 50
MAX_PARAMETER_STRING_LENGTH = 1000


_DATE_NAME_TOKENS = frozenset({"date", "dob", "datetime", "timestamp"})

_EXCEL_EPOCH = date(1899, 12, 30)

_SERIAL_MIN = 20000.0
_SERIAL_MAX = 80000.0


_TABLE_REF_PATTERN = re.compile(
    r"\b(?:from|join)\s+"
    r"(?:"
    r"`([^`]+)`"
    r'|"([^"]+)"'
    r"|([A-Za-z_][A-Za-z0-9_]*)"
    r")",
    re.IGNORECASE,
)

_CTE_NAME_PATTERN = re.compile(
    r"(?:\bwith\s+(?:recursive\s+)?|,\s*)"
    r"(?:"
    r"`([^`]+)`"
    r'|"([^"]+)"'
    r"|([A-Za-z_][A-Za-z0-9_]*)"
    r")"
    r"\s*(?:\([^)]*\))?"
    r"\s+as\s*\(",
    re.IGNORECASE,
)

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"insert|update|delete|"
    r"create|drop|alter|truncate|rename|"
    r"grant|revoke|"
    r"call|execute|prepare|deallocate|"
    r"outfile|dumpfile|"
    r"procedure"
    r")\b",
    re.IGNORECASE,
)

_REPLACE_STATEMENT = re.compile(r"\breplace\s+into\b", re.IGNORECASE)

_FOR_UPDATE = re.compile(r"\bfor\s+update\b", re.IGNORECASE)
_FOR_SHARE = re.compile(r"\bfor\s+share\b", re.IGNORECASE)
_LOCK_IN_SHARE_MODE = re.compile(r"\block\s+in\s+share\s+mode\b", re.IGNORECASE)
_LOAD_FILE = re.compile(r"\bload_file\s*\(", re.IGNORECASE)

_QUOTED = re.compile(
    r"'(?:[^'\\]|\\.|'')*'"
    r'|"(?:[^"]|"")*"'
    r"|`[^`]*`",
    re.DOTALL,
)

_SERIAL_TEXT = re.compile(r"\d{5}(?:\.\d+)?")


def _strip_quoted(sql: str) -> str:
    """Replace every quoted literal/identifier with a space."""
    return _QUOTED.sub(" ", sql)


def _is_date_named(column: str) -> bool:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column))
    words = re.findall(r"[a-z0-9]+", spaced.lower())
    return any(word in _DATE_NAME_TOKENS for word in words)


def _serial_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not _SERIAL_TEXT.fullmatch(text):
            return None
        number = float(text)
    else:
        return None

    if not math.isfinite(number):
        return None

    return number if _SERIAL_MIN <= number <= _SERIAL_MAX else None


def _serial_to_iso(number: float) -> str:
    moment = datetime.combine(_EXCEL_EPOCH, datetime.min.time()) + timedelta(
        days=number
    )

    if number == int(number):
        return moment.date().isoformat()

    return moment.replace(microsecond=0).isoformat(sep=" ")


def _normalize_date_columns(
    columns: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Convert serial-valued date columns to ISO in place; report which."""

    converted: dict[str, str] = {}

    for column in columns:
        if not _is_date_named(column):
            continue

        present = [row.get(column) for row in rows if row.get(column) not in (None, "")]

        if not present or any(_serial_value(value) is None for value in present):
            continue

        for row in rows:
            value = row.get(column)
            if value not in (None, ""):
                serial = _serial_value(value)
                if serial is not None:
                    row[column] = _serial_to_iso(serial)

        converted[column] = "spreadsheet date serial number converted to calendar date"

    return converted


def _validate_parameters(parameters: Any) -> list[Any]:
    """
    Enforce the scalar-only parameter contract at the tool boundary.

    Allowed:  str, int, float, bool, None
    Rejected: dict, list, tuple, bytes, nested structures, anything else.
    """

    if parameters is None:
        return []

    if not isinstance(parameters, (list, tuple)):
        raise ValueError("SQL parameters must be a list of scalar values.")

    if len(parameters) > MAX_PARAMETERS:
        raise ValueError(f"Too many SQL parameters (max {MAX_PARAMETERS}).")

    validated: list[Any] = []

    for position, value in enumerate(parameters):
        if value is None:
            validated.append(value)
            continue

        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"SQL parameter at position {position} must be a string, "
                f"number, boolean, or null; got {type(value).__name__}."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"SQL parameter at position {position} must be a finite number."
            )

        if isinstance(value, str) and len(value) > MAX_PARAMETER_STRING_LENGTH:
            raise ValueError(
                f"SQL parameter at position {position} is too long "
                f"(max {MAX_PARAMETER_STRING_LENGTH} characters)."
            )

        validated.append(value)

    return validated


def _validate_sql(sql: str) -> None:
    """
    Validate the generated SQL before it reaches RDS #2.

    MySQL does not provide the SQLite authorizer API used by the old
    implementation, so the application explicitly rejects write,
    administrative, locking, and file-output operations.

    The database account used by this tool is also SELECT-only,
    providing a second independent security boundary.
    """

    sql = sql.strip()

    if not sql:
        raise ValueError("SQL query cannot be empty.")

    normalized = sql.lower()

    if not normalized.startswith(("select", "with")):
        raise ValueError("Only read-only SQL queries are allowed.")

    statement = sql.rstrip().rstrip(";").rstrip()

    if ";" in statement:
        raise ValueError("Multiple SQL statements are not allowed.")

    if "--" in sql or "#" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("SQL comments are not allowed.")

    code = _strip_quoted(sql)

    if "'" in code or '"' in code or "`" in code:
        raise ValueError("Malformed quoting in SQL.")

    if _FORBIDDEN_KEYWORDS.search(code) or _REPLACE_STATEMENT.search(code):
        raise ValueError("Write/schema/admin SQL is not allowed.")

    code_lower = code.lower()

    if _FOR_UPDATE.search(code_lower):
        raise ValueError("Row-locking queries are not allowed.")

    if _FOR_SHARE.search(code_lower) or _LOCK_IN_SHARE_MODE.search(code_lower):
        raise ValueError("Locking queries are not allowed.")

    if _LOAD_FILE.search(code_lower):
        raise ValueError("File-reading SQL is not allowed.")


def _extract_cte_names(sql: str) -> set[str]:
    """Return CTE aliases declared by a WITH clause (not physical tables)."""

    return {
        backtick or double_quoted or bare
        for (backtick, double_quoted, bare) in _CTE_NAME_PATTERN.findall(sql)
    }


def _extract_referenced_names(sql: str) -> set[str]:
    """Identifiers used after FROM/JOIN (physical tables and CTE names)."""

    return {
        quoted_backtick or quoted_double or bare
        for (quoted_backtick, quoted_double, bare) in _TABLE_REF_PATTERN.findall(sql)
    }


def _validate_tables(sql: str, allowed_tables: set[str]) -> set[str]:
    """Validate physical FROM/JOIN tables against the workspace allow-list."""

    cte_names = _extract_cte_names(sql)
    referenced = _extract_referenced_names(sql) - cte_names

    validated_references: set[str] = set()

    for table_name in referenced:
        validated_references.add(
            SQLiteManager.validate_identifier(table_name, "table name")
        )

    unknown = validated_references - allowed_tables

    if unknown:
        raise ValueError("Query references a table outside this workspace.")

    return validated_references


def _extract_tables_used(sql: str, allowed_tables: set[str]) -> set[str]:
    """Return the physical workspace tables actually referenced by the SQL."""

    cte_names = _extract_cte_names(sql)
    referenced = _extract_referenced_names(sql) - cte_names

    return {
        SQLiteManager.validate_identifier(table_name, "table name")
        for table_name in referenced
        if table_name in allowed_tables
    }


def _bind_placeholders(sql: str, parameters: list[Any]) -> tuple[str, list[Any] | None]:
    """
    Translate qmark (?) placeholders into the MySQL driver's %s style.

    The planner / join code writes SQLite-style `?` placeholders; the MySQL
    driver only understands `%s`. `?` inside quoted strings/identifiers is
    left alone. If the SQL contains no `?` it is assumed to already use the
    driver's native style and is passed through untouched. Literal `%` is
    escaped only when a translation actually happens.
    """

    out: list[str] = []
    quote: str | None = None
    count = 0
    i = 0

    while i < len(sql):
        ch = sql[i]

        if quote:
            out.append("%%" if ch == "%" else ch)
            if ch == "\\" and quote != "`" and i + 1 < len(sql):
                i += 1
                out.append("%%" if sql[i] == "%" else sql[i])
            elif ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
        elif ch == "?":
            count += 1
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)

        i += 1

    if count == 0:
        return sql, (parameters or None)

    if count != len(parameters):
        raise ValueError(
            f"SQL has {count} placeholder(s) but {len(parameters)} parameter(s) "
            "were supplied."
        )

    return "".join(out), parameters


def _like_prefix_pattern(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
    return escaped + "%"


def _get_workspace_tables(workspace_id: int, connection: Any = None) -> set[str]:
    """
    Discover the physical tables owned by a workspace.

    If a connection is supplied it is reused and left open for the caller.
    """

    owns_connection = connection is None

    if owns_connection:
        connection = connections[STRUCTURED_READ_DB]

    prefix = f"ws_{workspace_id}_"

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT TABLE_NAME
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_TYPE = 'BASE TABLE'
                  AND TABLE_NAME LIKE %s
                """,
                [_like_prefix_pattern(prefix)],
            )
            rows = cursor.fetchall()

        return {
            row[0]
            for row in rows
            if isinstance(row[0], str) and row[0].startswith(prefix)
        }

    finally:
        if owns_connection:
            connection.close()


def _resolve_allowed_tables(
    workspace_id: int,
    allowed_table_names: set[str] | None,
    connection: Any,
) -> set[str]:
    prefix = f"ws_{workspace_id}_"

    if allowed_table_names is None:
        allowed = _get_workspace_tables(workspace_id, connection)
        if not allowed:
            raise ValueError("No tables are available for this workspace.")
        return allowed

    allowed = {
        SQLiteManager.validate_identifier(name, "table name")
        for name in allowed_table_names
    }
    allowed = {name for name in allowed if name.startswith(prefix)}

    if not allowed:
        raise ValueError("No tables are allowed for this query.")

    return allowed


def query_tables(
    *,
    workspace_id: int,
    sql: str,
    parameters: list[Any] | tuple[Any, ...] | None = None,
    allowed_table_names: set[str] | None = None,
) -> dict[str, Any]:

    if workspace_id <= 0:
        raise ValueError("workspace_id must be greater than zero.")

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    sql = sql.strip()

    _validate_sql(sql)

    validated_parameters = _validate_parameters(parameters)

    connection = connections[STRUCTURED_READ_DB]

    start = time.monotonic()

    try:
        allowed_tables = _resolve_allowed_tables(
            workspace_id,
            allowed_table_names,
            connection,
        )

        logger.info(
            "query_tables: workspace_id=%s allowed_table_names=%s",
            workspace_id,
            sorted(allowed_tables),
        )

        _validate_tables(sql, allowed_tables)

        tables_used = _extract_tables_used(sql, allowed_tables)

        exec_sql, exec_params = _bind_placeholders(sql, validated_parameters)

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET SESSION max_execution_time = %s",
                    [int(QUERY_TIMEOUT_SECONDS * 1000)],
                )

                cursor.execute(
                    "SET SESSION sql_mode = "
                    "CONCAT_WS(',', NULLIF(@@SESSION.sql_mode, ''), 'ANSI_QUOTES')"
                )

                cursor.execute(exec_sql, exec_params)

                fetched = cursor.fetchmany(MAX_RESULTS + 1)

                truncated = len(fetched) > MAX_RESULTS

                rows = fetched[:MAX_RESULTS]

                columns = [column[0] for column in (cursor.description or [])]

            rows_list = [
                {column: row[index] for index, column in enumerate(columns)}
                for row in rows
            ]

            date_columns = _normalize_date_columns(columns, rows_list)

            result = {
                "success": True,
                "columns": columns,
                "rows": rows_list,
                "row_count": len(rows_list),
                "truncated": truncated,
                "tables_used": sorted(tables_used),
            }

            if date_columns:
                result["date_columns"] = date_columns

            logger.info(
                "query_tables: returned %d rows tables_used=%s elapsed=%.3fs",
                result["row_count"],
                result["tables_used"],
                time.monotonic() - start,
            )

            return result

        except Exception as exc:
            raise ValueError(f"Table query failed: {exc}") from exc

    finally:
        connection.close()