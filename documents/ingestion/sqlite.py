import re
import sqlite3
from pathlib import Path

from django.conf import settings


MAX_QUERY_SECONDS = 3.0

# How long a connection will wait for a lock held by another writer
# before giving up. Needed once more than one ingestion worker process
# can write to this file concurrently - without this, a second writer
# gets "database is locked" immediately instead of just waiting briefly.
CONNECTION_TIMEOUT_SECONDS = 10


class SQLiteManager:
    DB_PATH = (
        Path(settings.BASE_DIR)
        / "storage"
        / "workspace_data.db"
    )

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

        # WAL allows one writer + concurrent readers instead of locking
        # the whole file on every write. Safe to set on every connect -
        # it's a no-op if already enabled.
        connection.execute("PRAGMA journal_mode=WAL")

        return connection

    @staticmethod
    def sanitize_column_name(column_name):
        column_name = str(column_name).strip()
        column_name = re.sub(r"[^a-zA-Z0-9_]", "_", column_name)
        column_name = re.sub(r"_+", "_", column_name)
        return column_name or "column"

    @classmethod
    def create_table(cls, table_name, columns):
        if not columns:
            raise ValueError(
                f"Cannot create table '{table_name}' with no columns."
            )

        connection = cls.get_connection()
        try:
            columns = [
                cls.sanitize_column_name(column)
                for column in columns
            ]

            query = (
                f'CREATE TABLE IF NOT EXISTS "{table_name}" '
                f'({", ".join(f""" "{column}" TEXT""" for column in columns)})'
            )

            connection.execute(query)
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def insert_rows(cls, table_name, rows):
        if not rows:
            return

        expected_width = len(rows[0])

        # Every row must match the header width used to create the
        # table. TableProcessor.normalize_table already pads/truncates
        # rows to line up with headers, so a mismatch here means a bug
        # upstream - better to fail loudly now than let sqlite3 raise a
        # confusing "N values for M columns" error mid-batch, or worse,
        # silently misalign columns and rows.
        for row in rows:
            if len(row) != expected_width:
                raise ValueError(
                    f"Row width mismatch for table '{table_name}': "
                    f"expected {expected_width} columns, got {len(row)}."
                )

        connection = cls.get_connection()
        try:
            placeholders = ",".join(["?"] * expected_width)
            query = (
                f'INSERT INTO "{table_name}" '
                f"VALUES ({placeholders})"
            )
            connection.executemany(query, rows)
            connection.commit()
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
                WHERE type = 'table'
                """
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            connection.close()

    @classmethod
    def list_document_tables(cls, workspace_id, document_id):
        prefix = f"ws_{workspace_id}_doc_{document_id}_"
        return [
            table
            for table in cls.list_tables()
            if table.startswith(prefix)
        ]

    @classmethod
    def drop_table(cls, table_name):
        connection = cls.get_connection()
        try:
            connection.execute(
                f'DROP TABLE IF EXISTS "{table_name}"'
            )
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def drop_document_tables(cls, workspace_id, document_id):
        for table_name in cls.list_document_tables(
            workspace_id,
            document_id,
        ):
            cls.drop_table(table_name)