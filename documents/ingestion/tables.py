"""Ingestion Tables module for QueryPilot."""

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

MAX_SEMANTIC_CELL_CHARS = 500
MAX_SAMPLE_ROWS = 10


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# --- SQL eligibility gate -------------------------------------------------
#
# A parser "table node" is only a candidate. Layout blocks (schedules, cards,
# two-column page designs) are detected as tables too, and their "header" is
# really a paragraph. Such tables are kept as readable text evidence instead
# of becoming a physical SQL table.

# A header longer than this, or with more words than HEADER_MAX_WORDS, reads
# as a sentence, not as a column label.
HEADER_MAX_CHARS = _env_int("INGESTION_TABLE_HEADER_MAX_CHARS", 64)
HEADER_MAX_WORDS = _env_int("INGESTION_TABLE_HEADER_MAX_WORDS", 12)

# Reject a table for SQL when at least this share of its real headers are
# sentence-like.
PROSE_HEADER_RATIO = _env_float("INGESTION_TABLE_PROSE_HEADER_RATIO", 0.5)

MAX_TABLE_COLUMNS = _env_int("INGESTION_TABLE_MAX_COLUMNS", 300)

# Safety cap for tables that are kept as text evidence.
FALLBACK_MAX_ROWS = _env_int("INGESTION_TABLE_FALLBACK_MAX_ROWS", 1000)

_GENERATED_HEADER = re.compile(r"^column_\d+(?:_\d+)?$")


class TableProcessor:
    """
    Processes table nodes extracted from LlamaParse.

    Responsibilities:

    1. Extract table structure
    2. Normalize headers and rows
    3. Validate table quality
    4. Build retrieval metadata
    5. Build semantic representation
    """

    @staticmethod
    def extract_table(
        table_node: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(table_node, dict):
            raise ValueError(
                "Expected a dict table_node from LlamaParse, "
                f"got {type(table_node).__name__}."
            )

        raw_rows = table_node.get("rows") or []

        if not isinstance(raw_rows, (list, tuple)):
            raise ValueError("Table rows must be a list or tuple.")

        if raw_rows:
            headers = raw_rows[0]
            rows = raw_rows[1:]
        else:
            headers = []
            rows = []

        if not isinstance(headers, (list, tuple)):
            headers = [headers]

        page = table_node.get("page", 0)
        sheet = " ".join(str(table_node.get("sheet") or "").split())
        heading = " ".join(str(table_node.get("heading") or "").split())
        before = [
            " ".join(str(value or "").split())
            for value in (table_node.get("before") or [])
        ]
        before = [value for value in before if value]
        after = " ".join(str(table_node.get("after") or "").split())

        # A table is identified by what introduces it. The nearest text just
        # before the table is the strongest signal (a caption or "the table
        # below" sentence); the heading and sheet name come after it.
        label = (
            (before[-1] if before else "")
            or heading
            or sheet
        )
        label = label[:120].strip()
        title = (
            f"{label} (page {page})"
            if label
            else f"Table Page {page}"
        )

        return {
            "title": title,
            "page": page,
            "sheet": sheet,
            "heading": heading,
            "before": before,
            "after": after,
            "headers": list(headers),
            "rows": rows,
        }

    @staticmethod
    def build_comment(table: dict[str, Any]) -> str:
        """Compact JSON describing where a table came from.

        Stored as the physical table's comment, so the planner can read it
        back from the same structured database without a schema migration.
        Kept well under MySQL's 2048-character table comment limit.
        """
        payload = {
            "title": str(table.get("title") or ""),
            "sheet": str(table.get("sheet") or ""),
            "heading": str(table.get("heading") or ""),
            "before": [str(v) for v in (table.get("before") or [])],
            "after": str(table.get("after") or ""),
            "page": table.get("page"),
        }
        limit = 1500
        for width in (320, 220, 140, 80, 40, 0):
            payload["before"] = [v[:width] for v in payload["before"]] if width else []
            payload["heading"] = payload["heading"][: max(width, 80)]
            payload["after"] = payload["after"][:width]
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(text) <= limit:
                return text
        return json.dumps(
            {"title": payload["title"][:200], "page": payload.get("page")},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def normalize_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:

        normalized_headers = []
        used_headers = set()

        for index, header in enumerate(headers):
            name = str(header).strip() or f"column_{index}"

            base_name = name
            suffix = 2

            while name.lower() in used_headers:
                name = f"{base_name}_{suffix}"
                suffix += 1

            used_headers.add(name.lower())
            normalized_headers.append(name)

        column_count = len(normalized_headers)
        normalized_rows = []

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue

            cleaned_row = []

            for index in range(column_count):
                if index < len(row):
                    value = row[index]

                    if value is None:
                        value = ""
                    else:
                        value = str(value).strip()
                else:
                    value = ""

                cleaned_row.append(value)

            if not any(cleaned_row):
                continue

            normalized_rows.append(cleaned_row)

        return normalized_headers, normalized_rows

    @staticmethod
    def is_prose_header(header: Any) -> bool:
        """True when a header cell reads like a sentence, not a column label."""
        text = " ".join(str(header or "").split())

        if not text:
            return False

        return (
            len(text) > HEADER_MAX_CHARS
            or len(text.split()) > HEADER_MAX_WORDS
        )

    @classmethod
    def sql_rejection_reason(
        cls,
        headers: list[str],
        rows: list[list[str]],
        strict: bool = True,
    ) -> str | None:
        """
        Why this table must NOT become a physical SQL table, or None.

        ``strict`` applies the header-prose check. It is meant for tables
        found INSIDE documents (PDF, DOCX), where layout blocks look like
        tables. Spreadsheets and CSV files are datasets by definition: their
        headers can legitimately be long (survey questions), and they rely on
        safe column naming instead.
        """
        if not headers:
            return "no headers"

        if len(headers) < 2:
            return "fewer than 2 columns"

        if not rows:
            return "no data rows"

        if len(headers) > MAX_TABLE_COLUMNS:
            return f"more than {MAX_TABLE_COLUMNS} columns"

        if strict:
            real_headers = [
                header
                for header in headers
                if header and not _GENERATED_HEADER.match(header)
            ]

            if real_headers:
                prose = sum(
                    1 for header in real_headers if cls.is_prose_header(header)
                )

                if prose / len(real_headers) >= PROSE_HEADER_RATIO:
                    return (
                        f"{prose}/{len(real_headers)} headers are "
                        "sentence-like (layout block, not a data table)"
                    )

        return None

    @classmethod
    def validate_table(
        cls,
        headers: list[str],
        rows: list[list[str]],
        strict: bool = True,
    ) -> bool:

        return cls.sql_rejection_reason(headers, rows, strict) is None

    @staticmethod
    def build_metadata(
        document_id: int,
        page: int,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> dict[str, Any]:

        return {
            "document_id": document_id,
            "page": page,
            "title": title,
            "columns": headers,
            "row_count": len(rows),
        }

    @staticmethod
    def serialize_for_embedding(
        title: str,
        headers: list[str],
        rows: list[list[str]],
        context: str = "",
    ) -> str:

        lines = [
            f"Table: {title}",
        ]

        if context:
            lines.extend(["", f"Context: {context}"])

        lines += [
            "",
            "Columns:",
            ", ".join(headers),
            "",
            f"Contains {len(rows)} rows.",
            "",
            "Sample Rows:",
        ]

        for row in rows[:MAX_SAMPLE_ROWS]:
            values = []

            for value in row:
                text = str(value)

                if len(text) > MAX_SEMANTIC_CELL_CHARS:
                    text = text[:MAX_SEMANTIC_CELL_CHARS] + "..."

                values.append(text)

            lines.append(" | ".join(values))

        return "\n".join(lines)

    @classmethod
    def process(
        cls,
        table_node: dict[str, Any],
        document_id: int,
        strict: bool = True,
    ) -> dict[str, Any] | None:

        table = cls.extract_table(table_node)

        headers, rows = cls.normalize_table(
            table["headers"],
            table["rows"],
        )

        reason = cls.sql_rejection_reason(
            headers,
            rows,
            strict,
        )

        if reason:
            logger.info(
                "Table not eligible for SQL for document %s (page %s): "
                "%s (%d headers, %d rows after normalization).",
                document_id,
                table.get("page"),
                reason,
                len(headers),
                len(rows),
            )
            return None

        metadata = cls.build_metadata(
            document_id=document_id,
            page=table["page"],
            title=table["title"],
            headers=headers,
            rows=rows,
        )

        context_parts = [
            part
            for part in (
                f"sheet {table['sheet']}" if table.get("sheet") else "",
                table.get("heading") or "",
                " ".join(table.get("before") or []),
                f"followed by: {table['after']}" if table.get("after") else "",
            )
            if part
        ]
        context_text = " | ".join(context_parts)[:900]

        semantic_text = cls.serialize_for_embedding(
            title=table["title"],
            headers=headers,
            rows=rows,
            context=context_text,
        )

        metadata["sheet"] = table.get("sheet") or ""
        metadata["context"] = context_text

        payload = {
            "sqlite_payload": {
                "headers": headers,
                "rows": rows,
            },
            "metadata": metadata,
            "semantic_text": semantic_text,
            "table_comment": cls.build_comment(table),
        }

        logger.debug(
            "Processed table for document %s (page %s): %d headers, %d rows.",
            document_id,
            table.get("page"),
            len(headers),
            len(rows),
        )

        return payload

    @staticmethod
    def _cell_text(value: Any) -> str:
        if value is None:
            return ""

        return " ".join(str(value).split())

    @classmethod
    def fallback_text_nodes(
        cls,
        table_node: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Render a table that is not stored in SQL as text nodes for the
        normal chunker, so its content stays searchable and citable.

        Every row becomes one self-contained line where each value keeps its
        column label ("Issue: Damaged item | Resolution: Submit an image"),
        so a retrieved row can never lose the header it belongs to. Headers
        that are really sentences are not used as labels. Never raises.
        """
        try:
            page = table_node.get("page") if isinstance(table_node, dict) else None
            page = page if isinstance(page, int) else 0

            raw_rows = table_node.get("rows") or []

            grid = [
                [cls._cell_text(cell) for cell in row]
                for row in raw_rows
                if isinstance(row, (list, tuple))
            ]

            grid = [row for row in grid if any(row)]

            lines: list[str] = []

            if not grid:
                fallback = table_node.get("markdown") or table_node.get("csv") or ""

                lines = [
                    line.strip()
                    for line in str(fallback).splitlines()
                    if line.strip()
                ]
            elif len(grid) == 1:
                lines = [" | ".join(cell for cell in grid[0] if cell)]
            else:
                header, data = grid[0], grid[1:]

                labels = [
                    "" if (not cell or cls.is_prose_header(cell)) else cell
                    for cell in header
                ]

                if any(labels):
                    lines.append(
                        "Columns: " + " | ".join(label for label in labels if label)
                    )

                for number, row in enumerate(data[:FALLBACK_MAX_ROWS], start=1):
                    parts = []

                    for position, cell in enumerate(row):
                        if not cell:
                            continue

                        label = labels[position] if position < len(labels) else ""

                        parts.append(f"{label}: {cell}" if label else cell)

                    if parts:
                        lines.append(f"Row {number}: " + " | ".join(parts))

                if len(data) > FALLBACK_MAX_ROWS:
                    lines.append(
                        f"({len(data) - FALLBACK_MAX_ROWS} more rows omitted)"
                    )

            if not lines:
                return []

            return [
                {
                    "page": page,
                    "text": "\n".join([f"## Table (page {page})", *lines]),
                }
            ]

        except Exception:
            logger.exception("Could not render table as text evidence.")
            return []

def parse_table_comment(comment: Any) -> dict[str, str]:
    """Read back the description stored with a table at ingestion time.

    The comment is a small JSON object (title, sheet, heading, the text that
    introduces the table and the text that follows it). Tables ingested
    before descriptions existed, or comments that are not JSON, simply have
    no description: both fields come back empty and nothing else changes.
    """
    empty = {"title": "", "context": ""}
    if isinstance(comment, bytes):
        comment = comment.decode("utf-8", "ignore")
    text = str(comment or "").strip()
    if not text.startswith("{"):
        return empty
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty

    title = " ".join(str(data.get("title") or "").split())
    parts: list[str] = []
    sheet = " ".join(str(data.get("sheet") or "").split())
    if sheet:
        parts.append(f"sheet: {sheet}")
    heading = " ".join(str(data.get("heading") or "").split())
    if heading:
        parts.append(f"heading: {heading}")
    before = [
        " ".join(str(value or "").split())
        for value in (data.get("before") or [])
        if isinstance(value, str)
    ]
    before = [value for value in before if value]
    if before:
        parts.append("introduced by: " + " ".join(before))
    after = " ".join(str(data.get("after") or "").split())
    if after:
        parts.append(f"followed by: {after}")
    return {"title": title, "context": " | ".join(parts)}
