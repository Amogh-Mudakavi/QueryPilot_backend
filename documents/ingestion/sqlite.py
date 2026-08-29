import re
import sqlite3
from pathlib import Path

from django.conf import settings


MAX_QUERY_SECONDS = 3.0


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
        return sqlite3.connect(cls.DB_PATH)

    @staticmethod
    def sanitize_column_name(column_name):
        column_name = str(column_name).strip()
        column_name = re.sub(r"[^a-zA-Z0-9_]", "_", column_name)
        column_name = re.sub(r"_+", "_", column_name)
        return column_name or "column"

    @classmethod
    def create_table(cls, table_name, columns):
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

        connection = cls.get_connection()
        try:
            placeholders = ",".join(["?"] * len(rows[0]))
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
