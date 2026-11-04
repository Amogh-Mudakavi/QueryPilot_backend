"""
Structured database module for QueryPilot.

Historical module name is retained for compatibility.

The physical structured-data store is now MySQL RDS #2.
"""

from __future__ import annotations

import hashlib
import re

from django.db import connections, transaction
from django.db.utils import DatabaseError, DataError


CONNECTION_TIMEOUT_SECONDS = 10

# MySQL limits every identifier (table, column) to 64 characters.
MAX_IDENTIFIER_LENGTH = 64

_HASH_SUFFIX_LENGTH = 8

# MySQL error codes that mean "this particular table's schema or data cannot
# be stored", as opposed to "the database is unavailable". They are
# deterministic: retrying the same input fails the same way every time.
_DETERMINISTIC_MYSQL_ERROR_CODES = frozenset(
    {
        1048,  # column cannot be null
        1054,  # unknown column
        1059,  # identifier name too long
        1060,  # duplicate column name
        1064,  # syntax error
        1067,  # invalid default value
        1071,  # specified key too long
        1074,  # column length too big
        1101,  # BLOB/TEXT cannot have a default
        1103,  # incorrect table name
        1117,  # too many columns
        1118,  # row size too large
        1136,  # column count does not match value count
        1166,  # incorrect column name
        1170,  # BLOB/TEXT used in key specification
        1264,  # value out of range
        1265,  # data truncated
        1366,  # incorrect string value
        1406,  # data too long
    }
)

_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


class SQLiteManager:
    """
    Compatibility manager for QueryPilot structured data.

    Historical class name is retained so the ingestion pipeline does
    not need to be rewritten.

    Physical storage:
        MySQL RDS #2

    Django database alias:
        structured

    Responsibilities:
        - Create tables
        - Insert rows
        - List tables
        - List document tables
        - Drop tables
        - Drop document tables
    """

    DB_ALIAS = "structured"

    @classmethod
    def get_connection(cls):
        """
        Return the Django database wrapper for RDS #2.
        """
        return connections[cls.DB_ALIAS]

    @staticmethod
    def validate_identifier(
        identifier,
        label="identifier",
    ):
        """
        Validate a MySQL identifier before interpolating it into SQL.

        Parameters cannot be used for identifiers, so table/column names
        must be strictly validated.
        """
        identifier = str(identifier)

        if not _IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(
                f"Invalid MySQL {label}: '{identifier}'."
            )

        if len(identifier) > MAX_IDENTIFIER_LENGTH:
            raise ValueError(
                f"MySQL {label} is longer than "
                f"{MAX_IDENTIFIER_LENGTH} characters: '{identifier}'."
            )

        return identifier

    @staticmethod
    def sanitize_column_name(
        column_name,
        index=None,
    ):
        """
        Turn extracted header text into a safe physical column name.

        Extracted headers are untrusted document content, so the result is
        guaranteed to be a valid MySQL identifier no matter what the
        document contained:

        - only ``[A-Za-z0-9_]`` characters (existing behavior);
        - never empty and never punctuation/non-ASCII only: such headers
          become ``column_<index>`` instead of collapsing to ``_``;
        - never starts with a digit (``2023`` -> ``c_2023``), which the
          planner's identifier validation would otherwise reject;
        - never longer than MAX_IDENTIFIER_LENGTH. Longer names are cut
          and given a short hash of the ORIGINAL header, so two different
          long headers sharing a prefix stay distinct and the same header
          always maps to the same column.

        Names that were already valid and short are returned unchanged.
        """
        raw = str(column_name).strip()

        cleaned = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            raw,
        )

        cleaned = re.sub(
            r"_+",
            "_",
            cleaned,
        )

        if not re.search(r"[A-Za-z0-9]", cleaned):
            return (
                f"column_{index}"
                if index is not None
                else "column"
            )

        if cleaned[0].isdigit():
            cleaned = f"c_{cleaned}"

        if len(cleaned) > MAX_IDENTIFIER_LENGTH:
            digest = hashlib.sha1(
                raw.encode("utf-8")
            ).hexdigest()[:_HASH_SUFFIX_LENGTH]

            keep = MAX_IDENTIFIER_LENGTH - _HASH_SUFFIX_LENGTH - 1

            cleaned = f"{cleaned[:keep].rstrip('_')}_{digest}"

        return cleaned

    @staticmethod
    def _unique_column_name(
        column,
        used_columns,
    ):
        """
        Return ``column`` or the first ``column_<n>`` (n >= 2) that is not
        already used, case-insensitively, without exceeding the MySQL
        identifier limit.
        """
        if column.lower() not in used_columns:
            return column

        suffix = 2

        while True:
            tail = f"_{suffix}"

            candidate = (
                column[: MAX_IDENTIFIER_LENGTH - len(tail)]
                + tail
            )

            if candidate.lower() not in used_columns:
                return candidate

            suffix += 1

    @staticmethod
    def is_deterministic_error(exc):
        """
        True when ``exc`` means this one table's schema/data cannot be
        stored, so retrying cannot help and the caller should degrade that
        table instead of failing the whole document.

        Connectivity problems, timeouts and unknown database errors return
        False and must keep propagating so the job retries.
        """
        if isinstance(exc, (ValueError, DataError)):
            return True

        if isinstance(exc, DatabaseError) and exc.args:
            code = exc.args[0]

            return (
                isinstance(code, int)
                and code in _DETERMINISTIC_MYSQL_ERROR_CODES
            )

        return False

    @classmethod
    def create_table(
        cls,
        table_name,
        columns,
        comment=None,
    ):
        if not columns:
            raise ValueError(
                f"Cannot create table '{table_name}' "
                "with no columns."
            )

        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        normalized_columns = []
        used_columns = set()

        for position, column in enumerate(columns):

            column = cls.sanitize_column_name(
                column,
                index=position,
            )

            column = cls._unique_column_name(
                column,
                used_columns,
            )

            column = cls.validate_identifier(
                column,
                "column name",
            )

            used_columns.add(
                column.lower()
            )

            normalized_columns.append(
                column
            )

        connection = cls.get_connection()

        column_sql = ", ".join(
            f"`{column}` TEXT"
            for column in normalized_columns
        )

        query = (
            f"CREATE TABLE IF NOT EXISTS "
            f"`{table_name}` ({column_sql})"
        )

        try:
            with transaction.atomic(
                using=cls.DB_ALIAS
            ):
                with connection.cursor() as cursor:
                    cursor.execute(query)

        except Exception:
            raise

        if comment:
            cls.set_table_comment(table_name, comment)

    @classmethod
    def set_table_comment(
        cls,
        table_name,
        comment,
    ):
        """Attach a short description to a physical table.

        The description (where the table came from: sheet, heading, nearby
        caption) is stored as the MySQL table comment, so it can be read back
        from information_schema together with the table names. It is only a
        label: failing to store it must never fail ingestion of the data.
        """
        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        text = str(comment or "").strip()[:1800]

        if not text:
            return

        try:
            with cls.get_connection().cursor() as cursor:
                cursor.execute(
                    f"ALTER TABLE `{table_name}` COMMENT = %s",
                    [text],
                )

        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Could not store the description of table %s; "
                "the table itself was created.",
                table_name,
            )

    @classmethod
    def insert_rows(
        cls,
        table_name,
        rows,
    ):
        if not rows:
            return

        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        expected_width = len(
            rows[0]
        )

        if expected_width == 0:
            return

        for row in rows:

            if len(row) != expected_width:
                raise ValueError(
                    f"Row width mismatch for table "
                    f"'{table_name}': expected "
                    f"{expected_width} columns, got "
                    f"{len(row)}."
                )

        placeholders = ",".join(
            "%s"
            for _ in range(expected_width)
        )

        query = (
            f"INSERT INTO `{table_name}` "
            f"VALUES ({placeholders})"
        )

        connection = cls.get_connection()

        try:
            with transaction.atomic(
                using=cls.DB_ALIAS
            ):
                with connection.cursor() as cursor:
                    cursor.executemany(
                        query,
                        rows,
                    )

        except Exception:
            raise

    @classmethod
    def list_tables(cls):
        """
        Return all structured tables in RDS #2.

        Uses MySQL information_schema instead of
        SQLite sqlite_master.
        """
        connection = cls.get_connection()

        query = """
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_TYPE = 'BASE TABLE'
        """

        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        return [
            row[0]
            for row in rows
        ]

    @classmethod
    def list_document_tables(
        cls,
        workspace_id,
        document_id,
    ):
        prefix = (
            f"ws_{workspace_id}"
            f"_doc_{document_id}_"
        )

        return [
            table
            for table in cls.list_tables()
            if table.startswith(prefix)
        ]

    @classmethod
    def drop_table(
        cls,
        table_name,
    ):
        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        query = (
            f"DROP TABLE IF EXISTS `{table_name}`"
        )

        connection = cls.get_connection()

        try:
            with transaction.atomic(
                using=cls.DB_ALIAS
            ):
                with connection.cursor() as cursor:
                    cursor.execute(query)

        except Exception:
            raise

    @classmethod
    def drop_document_tables(
        cls,
        workspace_id,
        document_id,
    ):
        prefix = (
            f"ws_{workspace_id}"
            f"_doc_{document_id}_"
        )

        # First discover the document's tables from RDS.
        tables = [
            table
            for table in cls.list_tables()
            if table.startswith(prefix)
        ]

        if not tables:
            return

        connection = cls.get_connection()

        try:
            with transaction.atomic(
                using=cls.DB_ALIAS
            ):
                with connection.cursor() as cursor:

                    for table_name in tables:

                        table_name = cls.validate_identifier(
                            table_name,
                            "table name",
                        )

                        cursor.execute(
                            f"DROP TABLE IF EXISTS `{table_name}`"
                        )

        except Exception:
            raise