"""Table structure recovery for tables found INSIDE documents (PDF, DOCX).

Why this exists
---------------
``TableProcessor.extract_table`` treats the first row of every table node as
the header. Document parsers do not guarantee that. A page layout often puts a
title line ("Q3 | Region North"), a note paragraph, or a blank band above the
real header, and one parser table frequently holds several stacked tables
(one per section) that each repeat the header row. When row 1 is not the
header, the physical table gets junk column names, the real header row and the
section titles end up as data rows, and tables that belong together look like
unrelated tables with different columns. A selector or SQL generator reading
that schema can then never find the right columns or all the rows.

What this module does (it knows no domain, column name or file format)
----------------------------------------------------------------------
1. Classifies every row of every table node as blank, spanning (a title or a
   note: one or two distinct texts, possibly repeated across merged cells),
   header-like, or data.
2. Finds the real header rows from evidence: a label-like row that repeats
   in the document is a header. A headerless piece of a table inherits the
   document's header for that column count, but only when its values have
   the same shape (numeric column stays numeric) as the rows that header
   already governs.
3. Cuts a node into segments at spanning rows and repeated headers. A short
   spanning text becomes the SECTION label of the rows below it; a long one is
   a NOTE and is returned as text evidence so nothing is lost.
4. Merges segments that share a header into ONE logical table with ``Section``
   and ``Source Page`` columns, when the same header repeats at least three
   times or carries section labels. One dataset becomes one table, so a list,
   count or total over it needs no knowledge of how the page was laid out.

Safety
------
* A node that already has a proper header row, no spanning rows and no
  repeated header is returned UNCHANGED (same object).
* If any piece of a node cannot be given a header with evidence, the whole node
  is returned unchanged and the old behaviour applies.
* Any exception returns the original nodes. The caller never loses a table.
* No rows are dropped except blank rows, repeated header rows and spanning
  rows, and the text of spanning rows is kept (section label or note).
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# A spanning text up to this long is a section label; longer is a note.
SECTION_LABEL_MAX_CHARS = _env_int("INGESTION_TABLE_SECTION_LABEL_MAX_CHARS", 90)
# A header cell is a short label, not a sentence.
HEADER_CELL_MAX_CHARS = _env_int("INGESTION_TABLE_HEADER_CELL_MAX_CHARS", 48)
HEADER_CELL_MAX_WORDS = _env_int("INGESTION_TABLE_HEADER_CELL_MAX_WORDS", 6)
# Same header this many times (or any section label) means one dataset.
MERGE_MIN_SEGMENTS = _env_int("INGESTION_TABLE_MERGE_MIN_SEGMENTS", 3)
# How many columns of a headerless piece must have the same value shape as the
# rows the inherited header already governs.
SHAPE_FIT_RATIO = 0.7

SECTION_COLUMN = "Section"
PAGE_COLUMN = "Source Page"

_NUMERIC = re.compile(r"^[\s$\u20b9\u20ac\u00a3+\-]*\d[\d,.\s]*%?$")
_EDGE_PUNCT = " |-:;,.\u2013\u2014"


# --------------------------------------------------------------------------
# Row helpers
# --------------------------------------------------------------------------

def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _is_numeric(text: str) -> bool:
    return bool(text) and bool(_NUMERIC.match(text))


def _distinct(cells: list[str]) -> list[str]:
    seen: list[str] = []
    for cell in cells:
        if cell and cell not in seen:
            seen.append(cell)
    return seen


def _key(cells: list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"[^a-z0-9]+", "", c.casefold()) for c in cells)


def _is_blank(cells: list[str]) -> bool:
    return not any(cells)


def _is_span(cells: list[str]) -> bool:
    """A title or note row: at most two distinct texts.

    Merged cells repeat their text across the columns they span, and a title
    band leaves the other columns empty. Rows with real data have several
    distinct values.
    """
    if len(cells) < 3:
        return False
    distinct = _distinct(cells)
    if not distinct or len(distinct) > 2:
        return False
    non_empty = sum(1 for c in cells if c)
    return non_empty <= 2 or len(distinct) < non_empty


def _is_label_like(cells: list[str]) -> bool:
    """Short, mostly non-numeric, non-repeated texts: how a header looks."""
    non_empty = [c for c in cells if c]
    if len(non_empty) < 2:
        return False
    if len(set(non_empty)) != len(non_empty):
        return False
    numeric = 0
    for cell in non_empty:
        if len(cell) > HEADER_CELL_MAX_CHARS or len(cell.split()) > HEADER_CELL_MAX_WORDS:
            return False
        if _is_numeric(cell):
            numeric += 1
    return numeric * 3 <= len(non_empty)


def _shape(rows: list[list[str]], ncols: int) -> list[str | None]:
    """Per column: 'num' or 'txt' by majority of its non-empty values."""
    shape: list[str | None] = []
    for index in range(ncols):
        values = [r[index] for r in rows if index < len(r) and r[index]]
        if not values:
            shape.append(None)
            continue
        numeric = sum(1 for v in values if _is_numeric(v))
        shape.append("num" if numeric * 10 >= len(values) * 8 else "txt")
    return shape


def _shapes_fit(left: list[str | None], right: list[str | None]) -> bool:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return False
    agree = sum(1 for a, b in pairs if a == b)
    return agree / len(pairs) >= SHAPE_FIT_RATIO


def _parse_span(cells: list[str]) -> tuple[str, str]:
    """Split a spanning row into (section label, note).

    Two texts where the longer contains the shorter are one title followed by
    a note that was merged into the same cell: the label is the longer text
    without the note. Otherwise a short text is a label and a long one a note.
    """
    distinct = _distinct(cells)
    if not distinct:
        return "", ""
    if len(distinct) == 2:
        first, second = distinct
        longer, shorter = (first, second) if len(first) >= len(second) else (second, first)
        if len(shorter) >= 20 and shorter in longer:
            label = longer.replace(shorter, "", 1).strip(_EDGE_PUNCT)
            if label and len(label) <= SECTION_LABEL_MAX_CHARS:
                return label, shorter
    labels: list[str] = []
    notes: list[str] = []
    for text in distinct:
        (labels if len(text) <= SECTION_LABEL_MAX_CHARS else notes).append(text)
    return " ".join(labels).strip(_EDGE_PUNCT), " ".join(notes)


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------

@dataclass
class _Segment:
    node_index: int
    page: Any
    section: str
    header: list[str] | None
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class _NodePlan:
    node_index: int
    node: dict[str, Any]
    ncols: int
    plain: bool
    segments: list[_Segment] = field(default_factory=list)
    notes: list[tuple[Any, str, str]] = field(default_factory=list)
    carry_out: str = ""
    failed: bool = False


def _grid(node: dict[str, Any]) -> list[list[str]]:
    raw = node.get("rows") or []
    rows = [
        [_clean(c) for c in row]
        for row in raw
        if isinstance(row, (list, tuple))
    ]
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _caption(node: dict[str, Any]) -> str:
    before = [str(v) for v in (node.get("before") or []) if str(v).strip()]
    text = (before[-1] if before else "") or str(node.get("heading") or "")
    return " ".join(text.split())[:120]


def _plan_node(
    index: int,
    node: dict[str, Any],
    repeated: set[tuple[str, ...]],
    carry_in: str,
) -> _NodePlan | None:
    grid = _grid(node)
    if len(grid) < 2:
        return None
    ncols = len(grid[0])
    if ncols < 2:
        return None

    first = next((r for r in grid if not _is_blank(r)), None)
    if first is None:
        return None
    internal_repeat = any(
        _key(r) in repeated for r in grid[1:] if not _is_blank(r)
    )
    structured = _is_span(first) or internal_repeat
    plain = not structured

    page = node.get("page")
    plan = _NodePlan(index, node, ncols, plain)

    if plain:
        # A normal table: first row is the header. It still takes part in
        # merging with look-alike tables elsewhere in the document.
        plan.segments.append(
            _Segment(index, page, carry_in, list(grid[0]), [list(r) for r in grid[1:] if not _is_blank(r)])
        )
        return plan

    section = carry_in
    current: _Segment | None = None
    last_data_section = None
    span_after_data = False

    for row in grid:
        if _is_blank(row):
            continue
        if _is_span(row):
            if current is not None and current.rows:
                plan.segments.append(current)
            current = None
            label, note = _parse_span(row)
            if label:
                section = label
                span_after_data = True
            if note:
                plan.notes.append((page, section, note))
            continue
        if _key(row) in repeated:
            if current is not None and current.rows:
                plan.segments.append(current)
            current = _Segment(index, page, section, list(row))
            span_after_data = False
            continue
        if current is None:
            current = _Segment(index, page, section, None)
        current.rows.append(list(row))
        last_data_section = section
        span_after_data = False

    if current is not None and current.rows:
        plan.segments.append(current)
    if span_after_data and section != last_data_section:
        plan.carry_out = section
    return plan


def _canonical_headers(
    plans: list[_NodePlan],
    repeated: set[tuple[str, ...]],
) -> dict[int, tuple[list[str], list[str | None]]]:
    """Per column count: the most common repeated header and the value shape of
    the rows it governs."""
    counts: dict[tuple[str, ...], int] = Counter()
    originals: dict[tuple[str, ...], list[str]] = {}
    rows_by_key: dict[tuple[str, ...], list[list[str]]] = {}
    for plan in plans:
        for segment in plan.segments:
            if segment.header is None or not segment.rows:
                continue
            key = _key(segment.header)
            if key not in repeated:
                continue
            counts[key] += 1
            originals.setdefault(key, segment.header)
            rows_by_key.setdefault(key, []).extend(segment.rows)

    best: dict[int, tuple[list[str], list[str | None]]] = {}
    best_count: dict[int, int] = {}
    for key, count in counts.items():
        header = originals[key]
        ncols = len(header)
        if count > best_count.get(ncols, 0):
            best_count[ncols] = count
            best[ncols] = (header, _shape(rows_by_key[key], ncols))
    return best


def _inherit_headers(
    plans: list[_NodePlan],
    canonical: dict[int, tuple[list[str], list[str | None]]],
) -> None:
    for plan in plans:
        if plan.plain:
            continue
        for segment in plan.segments:
            if segment.header is not None:
                continue
            found = canonical.get(plan.ncols)
            if found is None:
                plan.failed = True
                break
            header, shape = found
            if not _shapes_fit(_shape(segment.rows, plan.ncols), shape):
                plan.failed = True
                break
            segment.header = list(header)


def _plain_headerless(
    plans: list[_NodePlan],
    canonical: dict[int, tuple[list[str], list[str | None]]],
) -> None:
    """A plain node whose first row is data (a table continued from the
    previous page without a header) joins the document's header."""
    for plan in plans:
        if not plan.plain or not plan.segments:
            continue
        segment = plan.segments[0]
        header = segment.header or []
        if _is_label_like(header):
            continue
        found = canonical.get(plan.ncols)
        if found is None:
            continue
        canon_header, shape = found
        all_rows = [list(header)] + segment.rows
        if _shapes_fit(_shape(all_rows, plan.ncols), shape):
            segment.header = list(canon_header)
            segment.rows = all_rows
            plan.plain = False


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def restructure_table_nodes(
    table_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (table nodes with real structure, extra text nodes).

    The extra text nodes carry notes found inside table blocks. They should be
    stored as text evidence. Never raises: on any problem the original nodes
    come back unchanged with no extra text.
    """
    try:
        return _restructure(table_nodes)
    except Exception:
        logger.exception("Table structure recovery failed; keeping parser tables as they are.")
        return list(table_nodes), []


def _restructure(
    table_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not table_nodes:
        return list(table_nodes), []

    grids = [_grid(node) for node in table_nodes]

    # Rows that look like a header and occur more than once in the document.
    pool: Counter = Counter()
    for grid in grids:
        for row in grid:
            if _is_blank(row) or _is_span(row) or not _is_label_like(row):
                continue
            pool[_key(row)] += 1
    repeated = {key for key, count in pool.items() if count >= 2}

    plans: list[_NodePlan | None] = []
    carry = ""
    previous_page = None
    for index, node in enumerate(table_nodes):
        page = node.get("page")
        carry_in = carry if (previous_page is not None and page in (previous_page, _next_page(previous_page))) else ""
        plan = _plan_node(index, node, repeated, carry_in)
        plans.append(plan)
        carry = plan.carry_out if plan is not None else ""
        previous_page = page if plan is not None else previous_page

    live = [p for p in plans if p is not None]
    canonical = _canonical_headers(live, repeated)
    _inherit_headers(live, canonical)
    _plain_headerless(live, canonical)

    # A node that could not be repaired with evidence keeps the old behaviour.
    for plan in live:
        if plan.failed or any(s.header is None for s in plan.segments):
            logger.info(
                "Table structure: node %d (page %s) left as parsed (no evidence for its header).",
                plan.node_index,
                plan.node.get("page"),
            )
            plan.failed = True

    groups: dict[tuple[str, ...], list[_Segment]] = {}
    order: list[tuple[str, ...]] = []
    for plan in live:
        if plan.failed or not plan.segments:
            continue
        for segment in plan.segments:
            key = _key(segment.header or [])
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(segment)

    def merges(segments: list[_Segment]) -> bool:
        if len(segments) < 2:
            return False
        return len(segments) >= MERGE_MIN_SEGMENTS or any(s.section for s in segments)

    merged_keys = {key for key in order if merges(groups[key])}

    out: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    emitted: set[tuple[str, ...]] = set()

    for index, node in enumerate(table_nodes):
        plan = plans[index]
        if plan is None or plan.failed or not plan.segments:
            out.append(node)
            continue

        for page, section, text in plan.notes:
            label = f"{section}: " if section else ""
            notes.append({"page": page, "text": f"## Table note (page {page})\n{label}{text}"})

        keys = [_key(s.header or []) for s in plan.segments]
        in_merged = [k for k in keys if k in merged_keys]

        if not in_merged:
            if plan.plain and len(plan.segments) == 1:
                out.append(node)  # untouched: same object, same behaviour
                continue
            for segment in plan.segments:
                out.append(_build_node(table_nodes[segment.node_index], [segment], with_extra=False))
            continue

        for segment, key in zip(plan.segments, keys):
            if key in merged_keys:
                if key in emitted:
                    continue
                emitted.add(key)
                out.append(_build_node(node, groups[key], with_extra=True, nodes=table_nodes))
            else:
                out.append(_build_node(table_nodes[segment.node_index], [segment], with_extra=False))

    logger.info(
        "Table structure: %d parser table(s) -> %d logical table(s); %d note(s) kept as text.",
        len(table_nodes),
        len(out),
        len(notes),
    )
    return out, notes


def _next_page(page: Any) -> Any:
    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return None


def _unique_name(base: str, taken: list[str]) -> str:
    existing = {c.casefold() for c in taken}
    name, suffix = base, 2
    while name.casefold() in existing:
        name = f"{base} {suffix}"
        suffix += 1
    return name


def _build_node(
    template: dict[str, Any],
    segments: list[_Segment],
    *,
    with_extra: bool,
    nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    header = list(segments[0].header or [])
    pages = [s.page for s in segments if s.page is not None]
    node: dict[str, Any] = {
        "page": pages[0] if pages else template.get("page"),
        "csv": "",
        "html": "",
        "markdown": "",
        "sheet": template.get("sheet") or "",
        "heading": template.get("heading") or "",
        "before": list(template.get("before") or []),
        "after": template.get("after") or "",
    }

    if not with_extra:
        segment = segments[0]
        node["rows"] = [header] + [list(r) for r in segment.rows]
        if segment.section and not node["heading"]:
            node["heading"] = segment.section
        return node

    section_name = _unique_name(SECTION_COLUMN, header)
    page_name = _unique_name(PAGE_COLUMN, header + [section_name])
    rows: list[list[str]] = [header + [section_name, page_name]]
    for segment in segments:
        label = segment.section or (_caption(nodes[segment.node_index]) if nodes else "")
        for row in segment.rows:
            rows.append(list(row) + [label, "" if segment.page is None else str(segment.page)])
    node["rows"] = rows

    labels = [s.section for s in segments if s.section]
    lo, hi = (min(pages), max(pages)) if pages and all(isinstance(p, int) for p in pages) else (None, None)
    span = f", pages {lo}-{hi}" if lo is not None and lo != hi else ""
    if labels:
        node["heading"] = (
            f"Combined table of {len(segments)} parts ('{labels[0]}' to '{labels[-1]}'{span})"
        )
    else:
        node["heading"] = f"Combined table of {len(segments)} parts{span}"
    node["before"] = []
    node["after"] = ""
    return node
