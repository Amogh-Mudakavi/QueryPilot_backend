"""Ingestion Sqlite module for QueryPilot."""

import re
import sqlite3
import threading
from pathlib import Path

from django.conf import settings

MAX_QUERY_SECONDS = 3.0
CONNECTION_TIMEOUT_SECONDS = 10

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SQLiteManager:
    DB_PATH = Path(settings.BASE_DIR) / "storage" / "workspace_data.db"

    _wal_initialized = False
    _wal_lock = threading.Lock()

    @classmethod
    def get_connection(cls):

        cls.DB_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            cls.DB_PATH,
            timeout=CONNECTION_TIMEOUT_SECONDS,
        )

        connection.execute(f"PRAGMA busy_timeout={CONNECTION_TIMEOUT_SECONDS * 1000}")

        if not cls._wal_initialized:
            with cls._wal_lock:
                if not cls._wal_initialized:
                    connection.execute("PRAGMA journal_mode=WAL")

                    cls._wal_initialized = True

        connection.execute("PRAGMA synchronous=NORMAL")

        return connection

    @staticmethod
    def validate_identifier(
        identifier,
        label="identifier",
    ):

        identifier = str(identifier)

        if not _IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(f"Invalid SQLite {label}: '{identifier}'.")

        return identifier

    @staticmethod
    def sanitize_column_name(column_name):

        column_name = str(column_name).strip()

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
            raise ValueError(f"Cannot create table '{table_name}' with no columns.")

        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        normalized_columns = []
        used_columns = set()

        for index, column in enumerate(columns):
            column = cls.sanitize_column_name(column)

            if column.lower() in used_columns:
                base = column
                suffix = 2

                while f"{base}_{suffix}".lower() in used_columns:
                    suffix += 1

                column = f"{base}_{suffix}"

            used_columns.add(column.lower())
            normalized_columns.append(column)

        connection = cls.get_connection()

        try:
            column_sql = ", ".join(f'"{column}" TEXT' for column in normalized_columns)

            query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_sql})'

            connection.execute(query)
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

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

        expected_width = len(rows[0])

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

        connection = cls.get_connection()

        try:
            placeholders = ",".join("?" for _ in range(expected_width))

            query = f'INSERT INTO "{table_name}" VALUES ({placeholders})'

            connection.executemany(
                query,
                rows,
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    @classmethod
    def list_tables(cls):

        connection = cls.get_connection()

        try:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                """
            ).fetchall()

            return [row[0] for row in rows]

        finally:
            connection.close()

    @classmethod
    def list_document_tables(
        cls,
        workspace_id,
        document_id,
    ):

        prefix = f"ws_{workspace_id}_doc_{document_id}_"

        return [table for table in cls.list_tables() if table.startswith(prefix)]

    @classmethod
    def drop_table(
        cls,
        table_name,
    ):

        table_name = cls.validate_identifier(
            table_name,
            "table name",
        )

        connection = cls.get_connection()

        try:
            connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    @classmethod
    def drop_document_tables(
        cls,
        workspace_id,
        document_id,
    ):

        prefix = f"ws_{workspace_id}_doc_{document_id}_"

        connection = cls.get_connection()

        try:
            tables = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                """
            ).fetchall()

            document_tables = [row[0] for row in tables if row[0].startswith(prefix)]

            for table_name in document_tables:
                table_name = cls.validate_identifier(
                    table_name,
                    "table name",
                )

                connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()
