"""
Structured database module for QueryPilot.

Historical module name is retained for compatibility.

The physical structured-data store is now MySQL RDS #2.
"""

from __future__ import annotations

import re

from django.db import connections, transaction


CONNECTION_TIMEOUT_SECONDS = 10

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

        return identifier

    @staticmethod
    def sanitize_column_name(
        column_name,
    ):
        """
        Preserve the existing QueryPilot column sanitisation behavior.
        """
        column_name = str(
            column_name
        ).strip()

        column_name = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            column_name,
        )

        column_name = re.sub(
            r"_+",
            "_",
            column_name,
        )

        return column_name or "column"

    @classmethod
    def create_table(
        cls,
        table_name,
        columns,
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

        # Preserve the current duplicate-column handling exactly.
        for column in columns:

            column = cls.sanitize_column_name(
                column
            )

            if column.lower() in used_columns:

                base = column
                suffix = 2

                while (
                    f"{base}_{suffix}".lower()
                    in used_columns
                ):
                    suffix += 1

                column = f"{base}_{suffix}"

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