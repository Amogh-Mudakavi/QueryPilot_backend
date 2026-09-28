"""
QueryPilot document parser.

Routing:
    PDF  → LiteParse  →  (failure / empty content)  → LlamaParse fallback
    DOCX → LlamaParse
    XLSX → LlamaParse
    CSV  → LlamaParse

Internal normalized shape produced by both paths:

    result
     └── items
          └── pages: list[SimpleNamespace]
               ├── page_number: int          ← always an int (1-based index if missing)
               └── items: list[SimpleNamespace]
                    ├── type: "text" | "table"
                    ├── md: str               ← markdown representation
                    ├── value: str            ← same as md
                    # table items additionally carry:
                    ├── rows: list[list[str]] ← 2-D; rows[0] = header row
                    ├── csv: str
                    └── html: str

LlamaAdapter reads this shape to produce text_nodes and table_nodes for
the rest of the pipeline.  Neither the adapter nor anything downstream
ever imports liteparse or llama_cloud directly.

[FIX] Salary-loss diagnosis changes (this file ONLY; no LlamaParse API
parameter was added, removed, or altered):
    1. PARSE_CACHE_SCHEMA_VERSION bumped 2 → 3. parse_document() caches the
       NORMALIZED result, so v2 cache entries hold normalized rows produced
       by the pre-fix normalization (numeric cells dropped to ""). Without a
       version bump, those poisoned rows are re-served on every re-ingest of
       the same file and every normalization fix appears to "not work".
    2. LLAMAPARSE_CACHE_BYPASS env switch: forces a fresh LlamaParse request
       and skips BOTH the cache read and the cache write. For diagnosis only;
       unset it after the fix is confirmed.
    3. [CHECKPOINT 0] runtime identity log: proves which parser module /
       bytecode the running worker is actually executing and that _cell_text()
       keeps numeric cells. Definitive stale-worker / duplicate-module detector.
    4. _cell_text() hardened: a cell dict/object whose "text" (or first
       attribute) is "" but whose "value" holds the data can no longer yield "".
    5. CHECKPOINT 1/2/3 value logs promoted from debug to INFO for the
       diagnostic run (revert to debug once the fix is confirmed).
"""

import csv as _csv
import hashlib
import io
import json
import os
from html.parser import HTMLParser
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from django.conf import settings
from liteparse import LiteParse

from utils.redis import get_redis_client


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIME type hints passed to LlamaParse on upload.
# LlamaParse also infers from the filename extension, but an explicit hint
# prevents misclassification of XLSX vs XLS vs CSV.
# ---------------------------------------------------------------------------
_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument"
             ".wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet",
    ".csv":  "text/csv",
    ".txt":  "text/plain",
}

# Suffixes where a table_count of 0 after LlamaParse is a red flag.
_STRUCTURED_SUFFIXES = frozenset({".xlsx", ".csv"})

# ---------------------------------------------------------------------------
# LlamaParse operational settings.
# These mirror the previously working parser so retries/polling/caching keep
# the same behavior while allowing the active tier/version to come from
# Django settings. The default tier is agentic for higher-fidelity structured
# document extraction.
# ---------------------------------------------------------------------------
PARSE_TIMEOUT_SECONDS = 600.0
PARSE_POLLING_INTERVAL = 2.0
PARSE_MAX_INTERVAL = 5.0
PARSE_CACHE_TTL_SECONDS = 6 * 60 * 60
PARSE_CACHE_MAX_BYTES = 15 * 1024 * 1024

PARSE_TIER = getattr(
    settings,
    "LLAMA_PARSE_TIER",
    "agentic",
)

PARSE_VERSION = getattr(
    settings,
    "LLAMA_PARSE_VERSION",
    "latest",
)

PARSE_CACHE_PREFIX = "querypilot:llamaparse:"

# Increment this whenever the shape/normalization of cached parse results
# changes. This prevents stale results produced by an older parser from
# bypassing the current normalization logic.
#
# [FIX] Bumped 2 → 3. The cache stores the NORMALIZED result (not the raw API
# response), so the v2 keys hold rows normalized by the pre-fix cell logic
# (numeric cells dropped to ""). Because the key did not change, the
# _cell_text numeric fix was masked: every re-ingest of the same file kept
# returning the poisoned v2 rows. v3 makes every existing entry unreachable;
# fresh, correct rows are cached under v3 on the next parse.
PARSE_CACHE_SCHEMA_VERSION = "3"

# [FIX] Runtime identity markers for CHECKPOINT 0.
# PARSER_MODULE_PATH proves which file the worker imported.
# PARSER_SOURCE_DIGEST is computed from the file AT IMPORT TIME: if it differs
# from `sha256sum documents/ingestion/parser.py` on disk, the running worker
# imported the file before the latest edit (stale bytecode).
PARSER_MODULE_PATH = str(Path(__file__).resolve())


def _parser_source_digest() -> str:
    try:
        return hashlib.sha256(
            Path(__file__).read_bytes(),
        ).hexdigest()[:12]
    except Exception:
        return "unavailable"


PARSER_SOURCE_DIGEST = _parser_source_digest()


class LlamaParseService:
    """
    Routes PDFs to LiteParse and every other supported format to LlamaParse.

    Both paths return the same internal SimpleNamespace shape so that
    LlamaAdapter and the rest of the pipeline are format-agnostic.
    """

    # ------------------------------------------------------------------
    # Singletons – lazy, double-checked, locked.
    # ------------------------------------------------------------------

    _liteparse: "LiteParse | None" = None
    _liteparse_lock = threading.Lock()

    _llama_client = None
    _llama_client_lock = threading.Lock()

    # ==================================================================
    # Parse cache helpers
    # ==================================================================

    @staticmethod
    def _cache_bypass_enabled() -> bool:
        """
        [FIX] Diagnosis switch. Set LLAMAPARSE_CACHE_BYPASS=1 (in the WORKER
        process environment) to force a fresh LlamaParse request on every
        ingestion: the Redis parse cache is neither read nor written while
        this is active. Read at call time so it can be toggled without
        changing code or restarting the worker.
        """
        return os.getenv(
            "LLAMAPARSE_CACHE_BYPASS",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _cache_key(file_hash: str) -> str:
        return (
            f"{PARSE_CACHE_PREFIX}"
            f"v{PARSE_CACHE_SCHEMA_VERSION}:"
            f"{PARSE_TIER}:"
            f"{PARSE_VERSION}:"
            f"{file_hash}:items"
        )

    @staticmethod
    def _to_cache_json(result) -> str:
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json()
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(mode="json"))
        if hasattr(result, "dict"):
            return json.dumps(result.dict())
        if isinstance(result, SimpleNamespace):
            return json.dumps(
                result,
                default=lambda obj: vars(obj),
            )
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        raise TypeError("Unsupported LlamaParse result type.")

    @staticmethod
    def _to_namespace(value):
        if isinstance(value, dict):
            return SimpleNamespace(
                **{
                    key: LlamaParseService._to_namespace(item)
                    for key, item in value.items()
                }
            )
        if isinstance(value, list):
            return [LlamaParseService._to_namespace(item) for item in value]
        return value

    @classmethod
    def _read_cache(cls, cache_key: str):
        client = get_redis_client()
        cached = client.get(cache_key)
        if not cached:
            return None
        try:
            payload = json.loads(cached)
            return cls._to_namespace(payload)
        except Exception:
            client.delete(cache_key)
            return None

    @classmethod
    def _write_cache(cls, cache_key: str, result) -> None:
        try:
            payload = cls._to_cache_json(result)
            if len(payload.encode("utf-8")) > PARSE_CACHE_MAX_BYTES:
                return
            get_redis_client().setex(
                cache_key,
                PARSE_CACHE_TTL_SECONDS,
                payload,
            )
        except Exception:
            # Cache failure must never fail ingestion.
            return

    @classmethod
    def _write_cache_if_allowed(cls, cache_key: str, result, name: str) -> None:
        """
        [FIX] Cache-write gate honoring LLAMAPARSE_CACHE_BYPASS. During
        diagnosis nothing is written, so a broken run can never poison the
        result a later diagnostic test would read.
        """
        if cls._cache_bypass_enabled():
            logger.info(
                "[LLAMAPARSE] CACHE BYPASS: result for %s NOT written to Redis.",
                name,
            )
            return
        cls._write_cache(cache_key, result)

    # ==================================================================
    # Generic helpers
    # ==================================================================

    @staticmethod
    def _value(obj, key, default=None):
        """Read key from a dict or an attribute from any object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _cell_text(cls, cell) -> str:
        """
        Convert one LlamaParse table cell to text without dropping scalars.

        LlamaParse may return cells as strings, numbers, dictionaries, or
        small model/object instances. Numeric cells are especially important
        for XLSX files, where salary/quantity/date-like values can arrive as
        Python scalars rather than objects with ``text``/``value`` fields.

        Zero is deliberately preserved; using ``x or ""`` here would turn
        legitimate 0 values into empty strings.

        [FIX] Hardening: when a dict/object exposes several of
        text/value/content, prefer the first NON-EMPTY value instead of the
        first non-None one, so {"text": "", "value": 25000.0} can no longer
        collapse to "". Plain floats/ints/strings/None behavior is unchanged:
            _cell_text(25000.0) == "25000.0"
            _cell_text(0)       == "0"
            _cell_text(None)    == ""
        """
        if cell is None:
            return ""

        if isinstance(cell, str):
            return cell.strip()

        if isinstance(cell, (int, float, complex)):
            return str(cell).strip()

        if isinstance(cell, dict):
            saw_value = False
            for key in ("text", "value", "content"):
                if key in cell and cell[key] is not None:
                    saw_value = True
                    text = cls._cell_text(cell[key])
                    if text:
                        return text
            # All present values were empty after conversion.
            return "" if saw_value else ""

        # LlamaParse SDK/model objects may expose the value under an
        # attribute. Prefer the first non-empty attribute value so an empty
        # "text" next to a populated "value" does not lose the data.
        saw_value = False
        for attr in ("text", "value", "content"):
            if hasattr(cell, attr):
                value = getattr(cell, attr)
                if value is not None:
                    saw_value = True
                    text = cls._cell_text(value)
                    if text:
                        return text

        if saw_value:
            # At least one recognized attribute existed but everything was
            # empty — the cell is genuinely empty. Do NOT str() the whole
            # object here (that would leak repr garbage).
            return ""

        # Last-resort scalar conversion. This keeps uncommon scalar-like
        # values (for example Decimal or date/datetime objects) instead of
        # silently turning them into empty strings.
        return str(cell).strip()

    # ==================================================================
    # Table representation helpers
    # ==================================================================

    @classmethod
    def _rows_to_representation(
        cls,
        all_rows: list[list[str]],
    ) -> dict:
        """
        Build markdown, CSV, and canonical rows from a 2-D list of strings.

        all_rows[0] is the header row.
        """
        if not all_rows:
            return {"rows": [], "csv": "", "html": "", "markdown": ""}

        header = all_rows[0]
        data   = all_rows[1:]
        width  = len(header)

        # Markdown
        md_lines = [
            "| " + " | ".join(str(c) for c in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in data:
            padded = list(row[:width]) + [""] * max(0, width - len(row))
            md_lines.append(
                "| " + " | ".join(str(c) for c in padded[:width]) + " |"
            )
        markdown = "\n".join(md_lines)

        # CSV
        buf = io.StringIO()
        _csv.writer(buf).writerows(all_rows)
        csv_text = buf.getvalue().strip()

        return {
            "rows":     all_rows,
            "csv":      csv_text,
            "html":     "",
            "markdown": markdown,
        }

    @classmethod
    def _normalize_raw_rows(cls, raw_rows: list) -> list[list[str]]:
        """
        Flatten a LlamaParse raw rows field to list[list[str]].

        LlamaParse may return rows as:
          list[list[str]]           → already flat strings
          list[list[dict]]          → cell dicts with a "text" key
          list[list[SimpleNamespace]] → cell objects with a .text attribute
        """
        result: list[list[str]] = []
        for row in raw_rows:
            if not isinstance(row, (list, tuple)):
                continue
            result.append([cls._cell_text(cell) for cell in row])
        return result

    @staticmethod
    def _csv_to_rows(csv_text: str) -> list[list[str]]:
        """Parse a CSV string into list[list[str]], skipping blank rows."""
        try:
            reader = _csv.reader(io.StringIO(csv_text))
            return [
                list(row)
                for row in reader
                if any(cell.strip() for cell in row)
            ]
        except Exception:
            return []

    # ==================================================================
    # Markdown → items parser
    # (used independently and as a fallback when LlamaParse rows=[])
    # ==================================================================

    @staticmethod
    def _split_md_row(line: str) -> list[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [cell.strip() for cell in line.split("|")]

    @classmethod
    def _is_md_separator(cls, line: str) -> bool:
        cells = cls._split_md_row(line)
        if not cells:
            return False
        return all(
            len(cell) >= 3 and cell.replace(":", "").replace("-", "") == ""
            for cell in cells
        )

    @classmethod
    def _markdown_to_items(cls, markdown: str) -> list[SimpleNamespace]:
        """
        Parse markdown text into a list of SimpleNamespace items.

        Recognises markdown pipe-tables and emits them as type="table"
        items with a populated rows field.  Everything else becomes a
        type="text" item.

        Used as a fallback when LlamaParse returns a table item whose
        rows field is empty but whose md field contains the table.
        """
        lines  = markdown.splitlines()
        items: list[SimpleNamespace] = []
        buf:   list[str] = []
        i = 0

        def flush_text() -> None:
            if not buf:
                return
            text = "\n".join(buf).strip()
            buf.clear()
            if text:
                items.append(SimpleNamespace(type="text", md=text, value=text))

        while i < len(lines):
            line = lines[i]
            # Pipe-table: current line has "|" and the NEXT line is a separator
            if (
                i + 1 < len(lines)
                and "|" in line
                and cls._is_md_separator(lines[i + 1])
            ):
                flush_text()
                header = cls._split_md_row(line)
                i += 2                          # skip header + separator
                data: list[list[str]] = []
                while i < len(lines):
                    rl = lines[i].strip()
                    if not rl or "|" not in rl:
                        break
                    data.append(cls._split_md_row(rl))
                    i += 1

                if header:
                    rep = cls._rows_to_representation([header] + data)
                    items.append(
                        SimpleNamespace(
                            type="table",
                            rows=rep["rows"],
                            csv=rep["csv"],
                            html=rep["html"],
                            md=rep["markdown"],
                            value=rep["markdown"],
                        )
                    )
                continue

            buf.append(line)
            if not line.strip():
                flush_text()
            i += 1

        flush_text()
        return items

    class _HTMLTableParser(HTMLParser):
        """Small stdlib HTML table parser used only as a fallback."""

        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._current_row: list[str] | None = None
            self._cell_buffer: list[str] | None = None

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag == "tr":
                self._current_row = []
            elif tag in {"td", "th"} and self._current_row is not None:
                self._cell_buffer = []

        def handle_data(self, data):
            if self._cell_buffer is not None:
                self._cell_buffer.append(data)

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in {"td", "th"} and self._cell_buffer is not None:
                text = "".join(self._cell_buffer).strip()
                if self._current_row is not None:
                    self._current_row.append(text)
                self._cell_buffer = None
            elif tag == "tr":
                if self._current_row:
                    self.rows.append(self._current_row)
                self._current_row = None
                self._cell_buffer = None

    @classmethod
    def _html_to_rows(cls, html_text: str) -> list[list[str]]:
        """Parse a simple HTML table into canonical 2-D rows."""
        if not html_text or "<table" not in html_text.lower():
            return []
        try:
            parser = cls._HTMLTableParser()
            parser.feed(html_text)
            parser.close()
            return parser.rows
        except Exception:
            return []

    # ==================================================================
    # Table extraction with multi-level fallback
    # ==================================================================

    @classmethod
    def _extract_table_rows(cls, item) -> list[list[str]]:
        """
        Extract a normalised 2-D row list from a LlamaParse table item.

        Priority order:
          1. item.rows  – structured list-of-lists (preferred path)
          2. item.md    – parse as markdown table
          3. item.csv   – parse as CSV

        This prevents the pipeline from silently discarding tables when
        LlamaParse returns the content only in structured rows, HTML/markdown,
        or CSV form.
        """
        # 1. Structured rows
        raw = cls._value(item, "rows", []) or []
        if raw:
            normalized = cls._normalize_raw_rows(raw)
            if normalized:
                return normalized

        # 2. Markdown/HTML fallback from the md field
        md = (
            cls._value(item, "md", None)
            or cls._value(item, "markdown", "")
            or ""
        )
        if md:
            html_rows = cls._html_to_rows(str(md))
            if html_rows:
                logger.debug(
                    "[PARSER] Recovered table rows from HTML-in-markdown "
                    "fallback (%d rows).",
                    len(html_rows),
                )
                return html_rows

            for md_item in cls._markdown_to_items(str(md)):
                if getattr(md_item, "type", None) == "table":
                    rows = getattr(md_item, "rows", [])
                    if rows:
                        logger.debug(
                            "[PARSER] Recovered table rows from markdown "
                            "fallback (%d rows).",
                            len(rows),
                        )
                        return rows

        # 3. Explicit HTML field fallback
        html_text = cls._value(item, "html", "") or ""
        html_rows = cls._html_to_rows(str(html_text))
        if html_rows:
            logger.debug(
                "[PARSER] Recovered table rows from HTML fallback (%d rows).",
                len(html_rows),
            )
            return html_rows

        # 4. CSV fallback
        csv_text = cls._value(item, "csv", "") or ""
        if csv_text:
            rows = cls._csv_to_rows(str(csv_text))
            if rows:
                logger.debug(
                    "[PARSER] Recovered table rows from CSV fallback "
                    "(%d rows).",
                    len(rows),
                )
                return rows

        return []

    @classmethod
    def _block_to_table(cls, block) -> "dict | None":
        """
        Build a table representation dict from a LiteParse table block.

        Expected block shape: {kind: "table", header: [...], rows: [[…]]}
        Returns None when the block contains no usable data.
        """
        header = cls._value(block, "header", []) or []
        rows   = cls._value(block, "rows",   []) or []

        header_strs = [cls._cell_text(c) for c in header]
        row_strs    = [
            [cls._cell_text(c) for c in row]
            for row in rows
            if isinstance(row, (list, tuple))
        ]

        if not header_strs and not row_strs:
            return None

        all_rows = ([header_strs] + row_strs) if header_strs else row_strs
        return cls._rows_to_representation(all_rows)

    # ==================================================================
    # LiteParse result normalisation
    # ==================================================================

    @classmethod
    def _normalize_liteparse_result(cls, result) -> SimpleNamespace:
        """
        Convert a LiteParse parse result into the internal shape.

        Page numbers are guaranteed to be integers after this function
        returns: any page whose page_num/page_number attribute is absent
        or None gets a 1-based sequential index.
        """
        pages = getattr(result, "pages", []) or []
        normalized_pages: list[SimpleNamespace] = []

        for page in pages:
            page_number = (
                cls._value(page, "page_num",    None)
                or cls._value(page, "page_number", None)
            )
            blocks = cls._value(page, "blocks", []) or []
            items:  list[SimpleNamespace] = []

            if blocks:
                for block in blocks:
                    kind = cls._value(block, "kind", "") or ""

                    # ── structured table ─────────────────────────────
                    if kind == "table":
                        table = cls._block_to_table(block)
                        if table is None:
                            continue
                        items.append(
                            SimpleNamespace(
                                type="table",
                                rows=table["rows"],
                                csv=table["csv"],
                                html=table["html"],
                                md=table["markdown"],
                                value=table["markdown"],
                            )
                        )
                        continue

                    # ── uncertain grid (keep as text, not a table) ───
                    if kind == "grid_fallback":
                        text = cls._value(block, "text", "")
                        if text:
                            items.append(
                                SimpleNamespace(
                                    type="text",
                                    md=str(text),
                                    value=str(text),
                                )
                            )
                        continue

                    # ── text / heading / list_item ───────────────────
                    text = cls._value(block, "text", "")
                    if not text:
                        continue
                    text = str(text).strip()

                    if kind == "heading":
                        level = cls._value(block, "level", 1)
                        try:
                            level = int(level)
                        except (TypeError, ValueError):
                            level = 1
                        level = max(1, min(level, 6))
                        text = "#" * level + " " + text

                    elif kind == "list_item":
                        marker  = cls._value(block, "marker", "-")
                        ordered = bool(cls._value(block, "ordered", False))
                        pfx     = marker or ("1." if ordered else "-")
                        text    = f"{pfx} {text}"

                    items.append(
                        SimpleNamespace(type="text", md=text, value=text)
                    )

            else:
                # Older LiteParse binding: no blocks, fall back to page text.
                # This is a plain-text safety path; no table extraction.
                text = cls._value(page, "text", "")
                if text:
                    items.append(
                        SimpleNamespace(
                            type="text",
                            md=str(text),
                            value=str(text),
                        )
                    )

            normalized_pages.append(
                SimpleNamespace(page_number=page_number, items=items)
            )

        # Assign sequential page numbers to pages that had no number.
        for idx, pg in enumerate(normalized_pages, start=1):
            if not isinstance(pg.page_number, int):
                pg.page_number = idx

        return SimpleNamespace(items=SimpleNamespace(pages=normalized_pages))

    # ==================================================================
    # LlamaParse result normalisation
    # ==================================================================

    @classmethod
    def _log_raw_llamaparse_table_checkpoint(cls, result, path: Path) -> None:
        """Diagnostic logging for the raw LlamaParse response.

        This is intentionally before normalization so we can distinguish:
            LlamaParse output problem
        from:
            QueryPilot normalization / adapter problem.

        For safety, only table-shaped fields are logged, with text previews
        capped to keep production logs manageable.
        """
        try:
            raw_items = cls._value(result, "items", None)
            raw_pages = cls._value(raw_items, "pages", []) or []

            logger.info(
                "[CHECKPOINT 1] RAW LlamaParse response for %s: type=%s pages=%d.",
                path.name,
                type(result).__name__,
                len(raw_pages),
            )

            raw_table_count = 0

            for page_idx, page in enumerate(raw_pages, start=1):
                page_items = cls._value(page, "items", []) or []
                logger.info(
                    "[CHECKPOINT 1] RAW page %d: items=%d.",
                    page_idx,
                    len(page_items),
                )

                for item_idx, item in enumerate(page_items, start=1):
                    item_type = cls._value(item, "type", None)
                    if item_type != "table":
                        continue

                    raw_table_count += 1
                    raw_rows = cls._value(item, "rows", []) or []
                    raw_md = (
                        cls._value(item, "md", None)
                        or cls._value(item, "markdown", "")
                        or ""
                    )
                    raw_csv = cls._value(item, "csv", "") or ""
                    raw_html = cls._value(item, "html", "") or ""

                    logger.info(
                        "[CHECKPOINT 1] RAW table page=%d item=%d "
                        "rows_type=%s rows_count=%d md_len=%d csv_len=%d html_len=%d.",
                        page_idx,
                        item_idx,
                        type(raw_rows).__name__,
                        len(raw_rows) if isinstance(raw_rows, (list, tuple)) else -1,
                        len(str(raw_md)),
                        len(str(raw_csv)),
                        len(str(raw_html)),
                    )
                    scalar_cells = sum(
                        1
                        for row in raw_rows
                        if isinstance(row, (list, tuple))
                        for cell in row
                        if isinstance(cell, (int, float, complex))
                    )
                    logger.info(
                        "[CHECKPOINT 1] RAW table summary: rows=%d numeric_cells=%d "
                        "md_len=%d csv_len=%d html_len=%d.",
                        len(raw_rows) if isinstance(raw_rows, (list, tuple)) else 0,
                        scalar_cells,
                        len(str(raw_md)),
                        len(str(raw_csv)),
                        len(str(raw_html)),
                    )

                    # [FIX] Promoted from debug to INFO for the salary-loss
                    # diagnosis: shows the actual raw cell values. Revert to
                    # logger.debug once the fix is confirmed.
                    logger.info("[CHECKPOINT 1] RAW table rows=%r", raw_rows)
                    logger.info(
                        "[CHECKPOINT 1] RAW table cell types (first 3 rows)=%r",
                        [
                            [type(cell).__name__ for cell in row]
                            for row in raw_rows[:3]
                            if isinstance(row, (list, tuple))
                        ],
                    )

            if raw_table_count == 0:
                logger.warning(
                    "[CHECKPOINT 1] RAW LlamaParse returned ZERO table items "
                    "for %s.",
                    path.name,
                )
            else:
                logger.info(
                    "[CHECKPOINT 1] RAW LlamaParse table count=%d for %s.",
                    raw_table_count,
                    path.name,
                )

        except Exception:
            logger.exception(
                "[CHECKPOINT 1] Failed while inspecting raw LlamaParse response for %s.",
                path.name,
            )

    @classmethod
    def _normalize_llamaparse_result(cls, result) -> SimpleNamespace:
        """
        Convert a raw LlamaParse API result into the internal shape.

        Expected shape when expand=["items"]:

            result.items.pages[i].items[j]
               where each item has: type, md, rows?, csv?, html?

        Table items with rows=[] are recovered via _extract_table_rows()
        which falls back to parsing the md and csv fields before giving up.
        """
        items_container = cls._value(result, "items", None)
        pages = cls._value(items_container, "pages", []) or []

        normalized_pages: list[SimpleNamespace] = []

        for idx, page in enumerate(pages, start=1):
            page_number = (
                cls._value(page, "page_number", None)
                or cls._value(page, "page_num",    None)
                or idx
            )
            normalized_items: list[SimpleNamespace] = []

            for item in (cls._value(page, "items", []) or []):
                item_type = cls._value(item, "type", "")

                if item_type == "table":
                    rows = cls._extract_table_rows(item)

                    md = (
                        cls._value(item, "md",       None)
                        or cls._value(item, "markdown", "")
                        or ""
                    )
                    csv_text = cls._value(item, "csv",  "") or ""
                    html     = cls._value(item, "html", "") or ""

                    # Build representations from rows if the API omitted them.
                    if rows and not md:
                        rep      = cls._rows_to_representation(rows)
                        md       = rep["markdown"]
                        csv_text = csv_text or rep["csv"]

                    logger.info(
                        "[CHECKPOINT 2] NORMALIZED table: rows_count=%d columns=%d.",
                        len(rows),
                        len(rows[0]) if rows else 0,
                    )
                    # [FIX] Promoted from debug to INFO for the diagnosis:
                    # shows the actual normalized values + Python types, so a
                    # float→"" loss at this exact boundary is visible. Revert
                    # to logger.debug once confirmed.
                    logger.info("[CHECKPOINT 2] NORMALIZED rows=%r", rows)
                    logger.info(
                        "[CHECKPOINT 2] NORMALIZED cell types (first 3 rows)=%r",
                        [
                            [type(cell).__name__ for cell in row]
                            for row in rows[:3]
                        ],
                    )

                    normalized_items.append(
                        SimpleNamespace(
                            type="table",
                            rows=rows,
                            csv=csv_text,
                            html=html,
                            md=md,
                            value=md,
                        )
                    )
                    continue

                # Non-table item
                text = (
                    cls._value(item, "md",    None)
                    or cls._value(item, "value", None)
                )
                if not text:
                    continue

                normalized_items.append(
                    SimpleNamespace(type="text", md=str(text), value=str(text))
                )

            normalized_pages.append(
                SimpleNamespace(page_number=page_number, items=normalized_items)
            )

        return SimpleNamespace(items=SimpleNamespace(pages=normalized_pages))

    # ==================================================================
    # Content check
    # ==================================================================

    @staticmethod
    def _has_content(result) -> bool:
        pages = getattr(getattr(result, "items", None), "pages", []) or []
        return any(getattr(p, "items", []) for p in pages)

    # ==================================================================
    # Client getters (lazy, thread-safe)
    # ==================================================================

    @classmethod
    def _get_liteparse(cls) -> LiteParse:
        if cls._liteparse is None:
            with cls._liteparse_lock:
                if cls._liteparse is None:
                    cls._liteparse = LiteParse(
                        ocr_enabled=False,
                        output_format="markdown",
                        image_mode="off",
                        extract_images=False,
                        extract_links=False,
                        extract_blocks=True,
                        quiet=True,
                    )
        return cls._liteparse

    @classmethod
    def _get_llama_client(cls):
        if cls._llama_client is None:
            with cls._llama_client_lock:
                if cls._llama_client is None:
                    try:
                        from llama_cloud import LlamaCloud
                    except ImportError as exc:
                        raise RuntimeError(
                            "llama-cloud is required for LlamaParse ingestion."
                        ) from exc

                    api_key = getattr(settings, "LLAMA_CLOUD_API_KEY", None)
                    if not api_key:
                        raise ValueError(
                            "LLAMA_CLOUD_API_KEY is not configured."
                        )

                    cls._llama_client = LlamaCloud(
                        api_key=api_key,
                        timeout=PARSE_TIMEOUT_SECONDS,
                    )
        return cls._llama_client

    # ==================================================================
    # LiteParse parser
    # ==================================================================

    @classmethod
    def _parse_with_liteparse(cls, path: Path) -> SimpleNamespace:
        """Parse a PDF with LiteParse.  Raises on failure or empty output."""
        parser     = cls._get_liteparse()
        result     = parser.parse(path)
        normalized = cls._normalize_liteparse_result(result)
        if not cls._has_content(normalized):
            raise ValueError("LiteParse returned no extractable content.")
        return normalized

    # ==================================================================
    # LlamaParse parser
    # ==================================================================

    @classmethod
    def _parse_with_llamaparse(cls, path: Path) -> SimpleNamespace:
        logger.info(
            "[LLAMAPARSE] Parsing: %s (%s)",
            path.name,
            path.suffix.lower() or "unknown",
        )

        client = cls._get_llama_client()
        mime   = _MIME.get(path.suffix.lower(), "application/octet-stream")

        # ── File upload ──────────────────────────────────────────────
        # Keep the previously working SDK upload form: pass the Path
        # directly so the SDK opens and uploads the real workbook/PDF.
        # Each parse uploads the file anew → a new file_id → no reuse of a
        # previous parse job or stale server-side result.
        logger.info("[LLAMAPARSE] Uploading: %s (%s)", path.name, mime)

        uploaded_file = client.files.create(
            file=path,
            purpose="parse",
        )

        logger.info(
            "[LLAMAPARSE] File uploaded: id=%s",
            getattr(uploaded_file, "id", "?"),
        )

        # ── Parse ────────────────────────────────────────────────────
        logger.info(
            "[LLAMAPARSE] Parse request: tier=%s version=%s expand=%s "
            "polling_interval=%ss max_interval=%ss timeout=%ss backoff=linear.",
            PARSE_TIER,
            PARSE_VERSION,
            ["items"],
            PARSE_POLLING_INTERVAL,
            PARSE_MAX_INTERVAL,
            PARSE_TIMEOUT_SECONDS,
        )

        started = time.perf_counter()
        result = client.parsing.parse(
            file_id=uploaded_file.id,
            tier=PARSE_TIER,
            version=PARSE_VERSION,
            expand=["items"],
            polling_interval=PARSE_POLLING_INTERVAL,
            max_interval=PARSE_MAX_INTERVAL,
            timeout=PARSE_TIMEOUT_SECONDS,
            backoff="linear",
        )

        logger.info(
            "[LLAMAPARSE] Parse returned in %.2fs.",
            time.perf_counter() - started,
        )

        # CHECKPOINT 1: inspect the raw response before QueryPilot touches it.
        cls._log_raw_llamaparse_table_checkpoint(result, path)

        normalized = cls._normalize_llamaparse_result(result)

        pages      = getattr(getattr(normalized, "items", None), "pages", []) or []
        item_count = sum(len(getattr(p, "items", [])) for p in pages)

        if item_count == 0:
            raise ValueError(
                "LlamaParse returned no extractable content."
            )

        table_count = sum(
            1
            for p    in pages
            for item in getattr(p, "items", [])
            if getattr(item, "type", None) == "table"
        )

        logger.info(
            "[LLAMAPARSE] Completed: %d page(s), %d item(s), %d table(s).",
            len(pages),
            item_count,
            table_count,
        )

        # Structured files with zero tables are a sign something is wrong.
        if table_count == 0 and path.suffix.lower() in _STRUCTURED_SUFFIXES:
            logger.warning(
                "[LLAMAPARSE] WARNING: %s parsed to 0 table items "
                "after all fallbacks (rows / md / csv). "
                "Verify the raw LlamaParse response if SQLite tables are expected.",
                path.name,
            )

        return normalized

    # ==================================================================
    # Public entry point
    # ==================================================================

    @classmethod
    def parse_document(
        cls,
        file_path: str,
        file_hash:  "str | None" = None,
    ) -> SimpleNamespace:

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document file not found at '{file_path}'."
            )

        suffix = path.suffix.lower()

        # [FIX] CHECKPOINT 0 — runtime identity + normalization sanity.
        # Proves which parser module/bytecode the running worker is actually
        # executing. If sanity_cell != '25000.0', the worker is running stale
        # code (pre-fix module still in memory, or a shadowing parser module
        # on sys.path — the module= path reveals which). If module= or
        # source_digest= differs from the file on disk
        # (`sha256sum documents/ingestion/parser.py`), the worker imported
        # parser.py before the latest edit — restart the worker.
        logger.info(
            "[CHECKPOINT 0] RUNTIME IDENTITY: module=%s source_digest=%s "
            "pid=%s tier=%s version=%s schema=v%s cache_bypass=%s | "
            "sanity _cell_text(25000.0)=%r _cell_text(0)=%r _cell_text(None)=%r",
            PARSER_MODULE_PATH,
            PARSER_SOURCE_DIGEST,
            os.getpid(),
            PARSE_TIER,
            PARSE_VERSION,
            PARSE_CACHE_SCHEMA_VERSION,
            cls._cache_bypass_enabled(),
            cls._cell_text(25000.0),
            cls._cell_text(0),
            cls._cell_text(None),
        )

        # LlamaParse results are cached; LiteParse remains local and fast.
        # This prevents retries from re-uploading and re-parsing the same
        # document. The cache key includes tier/version so changing the
        # parsing configuration automatically bypasses older results.
        #
        # [FIX] NOTE: this cache stores the NORMALIZED result, not the raw
        # API response. Any change to normalization logic MUST bump
        # PARSE_CACHE_SCHEMA_VERSION, otherwise stale (pre-fix) rows keep
        # being served and the fix appears not to work.
        if file_hash is None:
            file_hash = cls._file_hash(path)

        cache_key = cls._cache_key(file_hash)

        if cls._cache_bypass_enabled():
            logger.warning(
                "[LLAMAPARSE] CACHE BYPASS ACTIVE for %s — the Redis parse "
                "cache will NOT be read and NOT be written; a fresh "
                "LlamaParse request follows (key=%s).",
                path.name,
                cache_key,
            )
        else:
            cached_result = None
            if suffix != ".pdf":
                cached_result = cls._read_cache(cache_key)

            if cached_result is not None and suffix != ".pdf":
                logger.warning(
                    "[LLAMAPARSE] Cache HIT for %s (key=%s) — returning the "
                    "cached NORMALIZED result, NOT a fresh parse. Wrong "
                    "values here mean the entry predates the normalization "
                    "fix (the v3 schema bump should make this impossible; "
                    "if you still see a v3 hit, delete the key).",
                    path.name,
                    cache_key,
                )
                return cached_result

        # ── PDF: LiteParse first, LlamaParse on failure ───────────────
        if suffix == ".pdf":
            logger.info("[PARSER] Routing PDF to LiteParse: %s", path.name)
            started = time.perf_counter()

            try:
                normalized = cls._parse_with_liteparse(path)
                elapsed    = time.perf_counter() - started

                pages = getattr(getattr(normalized, "items", None), "pages", []) or []
                item_count  = sum(len(getattr(p, "items", [])) for p in pages)
                table_count = sum(
                    1
                    for p    in pages
                    for item in getattr(p, "items", [])
                    if getattr(item, "type", None) == "table"
                )

                logger.info(
                    "[LITEPARSE] Completed: %d page(s), %d item(s), "
                    "%d table(s), in %.3fs.",
                    len(pages),
                    item_count,
                    table_count,
                    elapsed,
                )

                return normalized

            except Exception as exc:
                logger.warning(
                    "[LITEPARSE] PDF parse failed for %s; "
                    "falling back to LlamaParse: %s",
                    path.name,
                    exc,
                    exc_info=True,
                )
                result = cls._parse_with_llamaparse(path)
                cls._write_cache_if_allowed(cache_key, result, path.name)
                return result

        # ── Non-PDF: LlamaParse directly ──────────────────────────────
        logger.info(
            "[PARSER] Routing non-PDF to LlamaParse: %s (%s)",
            path.name,
            suffix or "unknown",
        )
        result = cls._parse_with_llamaparse(path)
        cls._write_cache_if_allowed(cache_key, result, path.name)
        return result


# ============================================================================
# LlamaAdapter
# ============================================================================


class LlamaAdapter:
    """
    Converts the normalised parser output into QueryPilot primitives.

    Input:  result from LlamaParseService.parse_document()
    Output: text_nodes, table_nodes
    """

    @staticmethod
    def extract_text_nodes(result) -> list[dict]:
        text_nodes: list[dict] = []
        pages = getattr(getattr(result, "items", None), "pages", []) or []

        for page in pages:
            page_number = getattr(page, "page_number", None)

            for item in (getattr(page, "items", None) or []):
                if getattr(item, "type", None) == "table":
                    continue

                text = (
                    getattr(item, "md",    None)
                    or getattr(item, "value", None)
                )
                if not text:
                    continue

                text_nodes.append({"page": page_number, "text": str(text)})

        return text_nodes

    @staticmethod
    def extract_table_nodes(result) -> list[dict]:
        table_nodes: list[dict] = []
        pages = getattr(getattr(result, "items", None), "pages", []) or []

        for page in pages:
            page_number = getattr(page, "page_number", None)

            for item in (getattr(page, "items", None) or []):
                if getattr(item, "type", None) != "table":
                    continue

                rows = getattr(item, "rows", []) or []

                # CHECKPOINT 3: what actually leaves the parser/adapter.
                logger.info(
                    "[CHECKPOINT 3] ADAPTER table page=%s rows_count=%d columns=%d.",
                    page_number,
                    len(rows),
                    len(rows[0]) if rows else 0,
                )
                # [FIX] Promoted from debug to INFO for the diagnosis.
                logger.info("[CHECKPOINT 3] ADAPTER rows=%r", rows)

                table_nodes.append(
                    {
                        "page":     page_number,
                        "rows":     rows,
                        "csv":      getattr(item, "csv",   ""),
                        "html":     getattr(item, "html",  ""),
                        "markdown": getattr(item, "md",    ""),
                    }
                )

        return table_nodes  