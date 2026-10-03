"""
planner/services/evidence.py

Evidence finalization for structured sources.

Sits between structured execution / cross-document join and answer
generation. Everything here is deterministic, domain-neutral and derived
only from the evidence and the question's own tokens: no table names,
column names, join keys, filenames or entity names are known here.

Responsibilities
  1. Population semantics   every table item is tagged with the population
                            it represents (entity_row, focused_subset,
                            filtered_result, complete_population,
                            aggregate_result, grouped_result,
                            truncated_result).
  2. Aggregates upstream    when the question needs population figures
                            (total / share / highest / ...), numeric columns
                            that the question itself names are aggregated
                            over the COMPLETE population (in memory when the
                            table was fully read, otherwise by an allow-listed
                            read-only SQL aggregate). The extreme rows are
                            joined to the other sources through real shared
                            identifiers discovered from the evidence.
  3. Canonicalization       once aggregates exist AND the question is about
                            specific records found in the data (a matched
                            focus row exists), complete tables that are not
                            themselves aggregated are reduced to the rows that
                            matter (focus rows + rows joined to the extremes).
                            Their source size is preserved. A question with no
                            matched focus record is about the whole
                            population, so its tables are never reduced here:
                            row-level conditions, grouping and joins need
                            every row.
  4. Filtered populations   a question that names filters ("Engineering
                            employees rated 4.5 or higher", "Product employees
                            rated between 4.0 and 4.4") gets deterministic
                            results computed over the COMPLETE tables, joined
                            through an established identifier:
                              - selection_result  earliest / latest / oldest /
                                                  newest row of the qualifying
                                                  population
                              - aggregate_result  sum / average of the whole
                                                  qualifying population
                            Each comes from its own sentence of the question,
                            so one can never contaminate the other. Question
                            text is read by query_interpretation.py; this module
                            only computes. Whatever cannot be established
                            safely is left unresolved (nothing is invented).
  5. Citation indexing      indexes are assigned AFTER all of the above, on
                            the exact list that is sent to the answer model
                            and stored as final evidence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Union

from planner.services.query_interpretation import (
    AggregateIntent,
    NumericPredicate,
    SelectionIntent,
    extract_numeric_predicates,
    interpret_query,
)
from retrieval.services.keys import is_id_column

logger = logging.getLogger(__name__)

MAX_AGG_COLUMNS_PER_TABLE = 3
MAX_IDENTITY_COLUMNS = 3
MAX_RELATED_ROWS = 3
MAX_FOCUS_ROWS = 25
MIN_ROWS_TO_REDUCE = 4

SCOPE_ENTITY_ROW = "entity_row"
SCOPE_FOCUSED_SUBSET = "focused_subset"
SCOPE_FILTERED = "filtered_result"
SCOPE_COMPLETE = "complete_population"
SCOPE_AGGREGATE = "aggregate_result"
SCOPE_SELECTION = "selection_result"
SCOPE_GROUPED = "grouped_result"
SCOPE_TRUNCATED = "truncated_result"


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _content(item: Any) -> dict[str, Any]:
    content = item.get("content") if isinstance(item, dict) else None
    return content if isinstance(content, dict) else {}


def _rows(item: Any) -> list[dict[str, Any]]:
    rows = _content(item).get("rows")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _columns(item: Any) -> list[str]:
    columns = _content(item).get("columns")
    if isinstance(columns, list) and columns:
        return [str(c) for c in columns]
    rows = _rows(item)
    return [str(c) for c in rows[0]] if rows else []


def _table_name(item: Any) -> str | None:
    names = {
        str(s.get("table_name"))
        for s in (item.get("sources") or [] if isinstance(item, dict) else [])
        if isinstance(s, dict) and s.get("table_name")
    }
    return next(iter(names)) if len(names) == 1 else None


def _doc_id(item: Any) -> Any:
    for s in item.get("sources") or [] if isinstance(item, dict) else []:
        if isinstance(s, dict) and s.get("document_id") is not None:
            return s.get("document_id")
    return None


def _title(item: Any) -> str:
    content = _content(item)
    return str(content.get("document_title") or _table_name(item) or "source")


def _is_table_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and str(item.get("source_type", "")).lower() == "table_query"
        and _table_name(item) is not None
        and isinstance(_content(item).get("rows"), list)
        and str(item.get("result_status", "SUCCESS")).upper() == "SUCCESS"
    )


_FOCUSED_ROLES = frozenset({"entity_row", "key_joined"})
_DERIVED_SEMANTICS = frozenset({"aggregate", "grouped", "ranking", "positional"})


def is_complete_table(item: Any) -> bool:
    content = _content(item)
    rows = content.get("rows")
    if content.get("truncated") or not isinstance(rows, list):
        return False
    if content.get("population_reduced"):
        return False
    if content.get("evidence_role") in _FOCUSED_ROLES:
        return False
    semantics = (content.get("query_semantics") or {}).get("type")
    if semantics in _DERIVED_SEMANTICS:
        return False
    counts = content.get("source_table_row_counts")
    if isinstance(counts, dict) and len(counts) == 1:
        try:
            return len(rows) == int(next(iter(counts.values())))
        except (TypeError, ValueError):
            pass
    return bool(content.get("rows_returned", len(rows)) == len(rows))


def _norm_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[-_ ]", "", str(value).strip().upper())


_NUMBER = re.compile(r"^[\s$€£₹]*[-+]?\d[\d,]*(?:\.\d+)?\s*%?\s*$")


def to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value if value is not None else "").strip()
    if not text or not _NUMBER.match(text):
        return None
    try:
        return float(re.sub(r"[^\d.+-]", "", text))
    except ValueError:
        return None


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _tokens(text: Any) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", str(text or "").lower())}


def _num(value: float) -> float | int:
    rounded = round(value, 6)
    return int(rounded) if rounded == int(rounded) else rounded


def _id_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if is_id_column(c)]


def _identity(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    picked: list[str] = _id_columns(columns)[:2]
    for column in columns:
        if column not in picked and "name" in _tokens(column):
            picked.append(column)
            break
    return {c: row.get(c) for c in picked[:MAX_IDENTITY_COLUMNS]}


def discover_relationships(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identifier columns whose VALUES overlap across two different tables."""
    profiles: list[tuple[str, str, str, set[str]]] = []
    for item in tables:
        name = _table_name(item)
        if not name:
            continue
        rows = _rows(item)
        for column in _id_columns(_columns(item)):
            values = {_norm_cell(r.get(column)) for r in rows}
            values.discard("")
            if values:
                profiles.append((name, _title(item), column, values))

    found: list[dict[str, Any]] = []
    for i, (t1, title1, c1, v1) in enumerate(profiles):
        for t2, title2, c2, v2 in profiles[i + 1 :]:
            if t1 == t2:
                continue
            overlap = len(v1 & v2)
            if overlap:
                found.append(
                    {
                        "left": {"document": title1, "column": c1},
                        "right": {"document": title2, "column": c2},
                        "shared_values": overlap,
                    }
                )
    return found


def _numeric_columns_named_by_query(
    item: dict[str, Any],
    query_tokens: set[str],
) -> list[str]:
    rows = _rows(item)
    columns = _columns(item)
    chosen: list[tuple[int, str]] = []
    for column in columns:
        col_tokens = _tokens(column)
        if not col_tokens or not col_tokens <= query_tokens or is_id_column(column):
            continue
        values = [r.get(column) for r in rows if str(r.get(column) or "").strip()]
        if len(values) < 2 or any(to_number(v) is None for v in values):
            continue
        chosen.append((len(col_tokens), column))
    chosen.sort(key=lambda p: -p[0])
    return [c for _, c in chosen[:MAX_AGG_COLUMNS_PER_TABLE]]


def _related_rows(
    extreme: dict[str, Any],
    source: dict[str, Any],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rows of OTHER tables that share an identifier value with `extreme`."""
    id_values = {_norm_cell(extreme.get(c)) for c in _id_columns(_columns(source))}
    id_values.discard("")
    if not id_values:
        return []
    related: list[dict[str, Any]] = []
    for other in tables:
        if other is source:
            continue
        columns = _columns(other)
        id_cols = _id_columns(columns)
        if not id_cols:
            continue
        for row in _rows(other):
            if any(_norm_cell(row.get(c)) in id_values for c in id_cols):
                related.append({"document": _title(other), "row": row})
                break
        if len(related) >= MAX_RELATED_ROWS:
            break
    return related


def _aggregate_in_memory(
    item: dict[str, Any],
    column: str,
    tables: list[dict[str, Any]],
    focus_norms: set[str],
) -> dict[str, Any] | None:
    columns = _columns(item)
    scored = [(to_number(r.get(column)), r) for r in _rows(item)]
    scored = [(v, r) for v, r in scored if v is not None]
    if len(scored) < 2:
        return None
    values = [v for v, _ in scored]
    total = sum(values)
    hi, lo = max(values), min(values)
    max_rows = [r for v, r in scored if v == hi]
    min_rows = [r for v, r in scored if v == lo]

    entry: dict[str, Any] = {
        "column": column,
        "method": "computed from the complete table",
        "count": len(values),
        "sum": _num(total),
        "avg": _num(total / len(values)),
        "min": _num(lo),
        "max": _num(hi),
        "max_rows": [
            {
                "identity": _identity(r, columns),
                "value": _num(hi),
                "related": _related_rows(r, item, tables),
            }
            for r in max_rows[:2]
        ],
        "min_rows": [
            {"identity": _identity(r, columns), "value": _num(lo)} for r in min_rows[:2]
        ],
    }
    if len(max_rows) > 1:
        entry["max_tie"] = len(max_rows)

    focus = []
    for value, row in scored:
        if focus_norms and any(_norm_cell(v) in focus_norms for v in row.values()):
            focus.append(
                {
                    "identity": _identity(row, columns),
                    "value": _num(value),
                    "percent_of_total": _num(value / total * 100) if total else None,
                }
            )
    if focus:
        entry["focus"] = focus[:MAX_FOCUS_ROWS]
    return entry


def _aggregate_by_sql(
    run_query: Callable[..., dict[str, Any]],
    workspace_id: int,
    item: dict[str, Any],
    column: str,
    tables: list[dict[str, Any]],
    focus_norms: set[str],
) -> dict[str, Any] | None:
    """Allow-listed, read-only aggregate for tables that were truncated."""
    table = _table_name(item)
    if not table:
        return None
    q = _quote(column)
    expr = f"CAST(REPLACE(TRIM({q}), ',', '') AS REAL)"
    where = f"{q} IS NOT NULL AND TRIM({q}) <> ''"
    try:
        agg = run_query(
            workspace_id=workspace_id,
            sql=(
                f"SELECT COUNT({expr}) AS n, SUM({expr}) AS s, MIN({expr}) AS lo, "
                f"MAX({expr}) AS hi FROM {_quote(table)} WHERE {where}"
            ),
            parameters=[],
            allowed_table_names={table},
        )
        top = run_query(
            workspace_id=workspace_id,
            sql=(
                f"SELECT * FROM {_quote(table)} WHERE {where} "
                f"ORDER BY {expr} DESC LIMIT 2"
            ),
            parameters=[],
            allowed_table_names={table},
        )
    except Exception:
        logger.exception("population aggregate SQL failed: %s.%s", table, column)
        return None

    row = (agg.get("rows") or [{}])[0]
    if not isinstance(row, dict) or not row.get("n"):
        return None
    n, total = int(row["n"]), float(row["s"])
    hi, lo = float(row["hi"]), float(row["lo"])
    columns = _columns(item) or [str(c) for c in (top.get("columns") or [])]
    top_rows = [r for r in top.get("rows", []) if isinstance(r, dict)]
    entry: dict[str, Any] = {
        "column": column,
        "method": "computed by read-only SQL over the complete table",
        "count": n,
        "sum": _num(total),
        "avg": _num(total / n),
        "min": _num(lo),
        "max": _num(hi),
        "max_rows": [
            {
                "identity": _identity(r, columns),
                "value": _num(hi),
                "related": _related_rows(r, item, tables),
            }
            for r in top_rows[:1]
            if to_number(r.get(column)) == hi
        ],
    }
    focus = []
    for r in _rows(item):
        value = to_number(r.get(column))
        if (
            value is not None
            and focus_norms
            and any(_norm_cell(v) in focus_norms for v in r.values())
        ):
            focus.append(
                {
                    "identity": _identity(r, columns),
                    "value": _num(value),
                    "percent_of_total": _num(value / total * 100) if total else None,
                }
            )
    if focus:
        entry["focus"] = focus
    return entry


MAX_JOINED_ROWS = 5
MAX_QUALIFYING_ROWS_SHOWN = 25
MAX_QUALIFYING_KEYS_SHOWN = 50
MAX_TEXT_FILTER_CHARS = 80

_DATE_NAME_TOKENS = ("date", "dob", "datetime", "timestamp")
_EXCEL_EPOCH = datetime(1899, 12, 30)
_SERIAL_MIN = 20000.0
_SERIAL_MAX = 80000.0
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
)


class _Unresolved(Exception):
    """A deterministic result could not be established safely."""


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _root(token: str) -> str:
    """Crude stem so 'joined' and 'Join' (or 'hired' and 'hire') compare equal."""
    word = token.lower()
    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]
    for suffix in ("ing", "ed", "e"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _roots(text: Any) -> set[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    return {_root(token) for token in re.findall(r"[a-z0-9]+", spaced.lower())}


_DATE_ROOTS = frozenset(_root(token) for token in _DATE_NAME_TOKENS)
_GENERIC_DATE_ROOTS = _DATE_ROOTS | {_root("time")}


def _column_match_score(column: str, text: str) -> float:
    """Lexical overlap between a schema column and the words that name it."""
    col_tokens = _tokens(column)
    if not col_tokens:
        return 0.0
    score = len(col_tokens & _tokens(text)) / len(col_tokens)
    normalized_col = " ".join(re.findall(r"[a-z0-9]+", column.lower()))
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    if normalized_col and normalized_col in normalized_text:
        score += 0.5
    return score


@dataclass(frozen=True, eq=False)
class _Source:
    """One complete table: the population authority for its own columns."""

    item: dict[str, Any]
    name: str
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]

    def reference(self) -> dict[str, Any]:
        return {"table_name": self.name, "document_id": _doc_id(self.item)}


def _complete_sources(tables: list[dict[str, Any]]) -> list[_Source]:
    """Every table that was read completely, once per table (a full-table
    read is preferred over any other item for the same table)."""
    by_name: dict[str, _Source] = {}
    for item in tables:
        name = _table_name(item)
        if not name or not is_complete_table(item):
            continue
        is_full_read = _content(item).get("evidence_role") == "document_table"
        if name in by_name and not is_full_read:
            continue
        by_name[name] = _Source(item, name, _title(item), _columns(item), _rows(item))
    return list(by_name.values())


def _distinct_keys(source: _Source, column: str) -> set[str]:
    return {key for row in source.rows if (key := _norm_cell(row.get(column)))}


def _shared_key_columns(left: _Source, right: _Source) -> tuple[str, str]:
    """The identifier both tables carry: same column name AND overlapping
    values. Names of people and look-alike values are never a join key."""
    pairs = [
        (left_column, right_column)
        for left_column in _id_columns(left.columns)
        for right_column in _id_columns(right.columns)
        if _norm_cell(left_column) == _norm_cell(right_column)
        and _distinct_keys(left, left_column) & _distinct_keys(right, right_column)
    ]
    if len(pairs) != 1:
        raise _Unresolved(
            f"no single shared identifier between '{left.title}' and '{right.title}'"
        )
    return pairs[0]


@dataclass(frozen=True, eq=False)
class _CategoricalFilter:
    source: _Source
    column: str
    value: str

    def passes(self, row: dict[str, Any]) -> bool:
        return str(row.get(self.column) or "").strip().lower() == self.value.lower()


@dataclass(frozen=True, eq=False)
class _NumericFilter:
    source: _Source
    column: str
    predicate: NumericPredicate

    def passes(self, row: dict[str, Any]) -> bool:
        number = to_number(row.get(self.column))
        return number is not None and self.predicate.matches(number)


_Filter = Union[_CategoricalFilter, _NumericFilter]


def _distinct_text_values(source: _Source, column: str) -> list[str]:
    seen: dict[str, str] = {}
    for row in source.rows:
        value = row.get(column)
        if not _present(value) or to_number(value) is not None:
            continue
        text = str(value).strip()
        if 2 <= len(text) <= MAX_TEXT_FILTER_CHARS:
            seen.setdefault(text.lower(), text)
    return list(seen.values())


def _mentions(text_lower: str, value_lower: str) -> bool:
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(value_lower)}(?![a-z0-9])", text_lower)
    )


def _drop_contained_values(
    hits: dict[str, list[_CategoricalFilter]],
) -> dict[str, list[_CategoricalFilter]]:
    """'Product Design' mentioned: do not also filter on 'Product'."""
    return {
        value: filters
        for value, filters in hits.items()
        if not any(other != value and _mentions(other, value) for other in hits)
    }


def _validated_categorical(
    hits: dict[str, list[_CategoricalFilter]],
) -> list[_CategoricalFilter]:
    filters: list[_CategoricalFilter] = []
    value_by_column: dict[str, str] = {}
    for value, candidates in hits.items():
        column_names = {_norm_cell(candidate.column) for candidate in candidates}
        if len(column_names) > 1:
            raise _Unresolved(f"'{value}' appears in several different columns")
        column_name = next(iter(column_names))
        if value_by_column.setdefault(column_name, value) != value:
            raise _Unresolved("the question names several values of the same column")
        filters.extend(candidates)
    return filters


def _find_categorical_filters(
    sources: list[_Source], text: str
) -> list[_CategoricalFilter]:
    """Cell values that the sentence explicitly mentions (whole words only)."""
    lowered = text.lower()
    hits: dict[str, list[_CategoricalFilter]] = {}
    for source in sources:
        for column in source.columns:
            if is_id_column(column):
                continue
            for value in _distinct_text_values(source, column):
                if _mentions(lowered, value.lower()):
                    hits.setdefault(value.lower(), []).append(
                        _CategoricalFilter(source, column, value)
                    )
    return _validated_categorical(_drop_contained_values(hits))


def _numeric_columns(sources: list[_Source]) -> list[tuple[_Source, str]]:
    """Non-identifier columns whose every non-empty cell is a number."""
    found: list[tuple[_Source, str]] = []
    for source in sources:
        for column in source.columns:
            if is_id_column(column):
                continue
            cells = [
                row.get(column) for row in source.rows if _present(row.get(column))
            ]
            if cells and all(to_number(cell) is not None for cell in cells):
                found.append((source, column))
    return found


def _pick_numeric_column(sources: list[_Source], text: str) -> tuple[_Source, str]:
    """The one numeric column the words name. A tie is not guessed."""
    scored = [
        (score, source, column)
        for source, column in _numeric_columns(sources)
        if (score := _column_match_score(column, text)) > 0
    ]
    if not scored:
        raise _Unresolved(f"no numeric column matches '{text}'")
    best_score = max(score for score, _, _ in scored)
    best = [(source, column) for score, source, column in scored if score == best_score]
    if len(best) != 1:
        raise _Unresolved(f"several numeric columns match '{text}'")
    return best[0]


def _find_numeric_filters(sources: list[_Source], text: str) -> list[_NumericFilter]:
    filters: list[_NumericFilter] = []
    for predicate in extract_numeric_predicates(text):
        source, column = _pick_numeric_column(sources, predicate.context)
        filters.append(_NumericFilter(source, column, predicate))
    return filters


@dataclass(frozen=True, eq=False)
class _Population:
    anchor: _Source
    rows: list[dict[str, Any]]
    key_column: str | None
    keys: list[str]
    sources: list[_Source]
    categorical: list[_CategoricalFilter]
    numeric: list[_NumericFilter]
    relationships: list[dict[str, Any]]


def _passing_keys(flt: _Filter, key_column: str) -> set[str]:
    """Identifiers whose rows in the filter's table satisfy the filter.
    An identifier whose rows disagree is a conflict, not a coin flip."""
    outcomes: dict[str, set[bool]] = {}
    for row in flt.source.rows:
        key = _norm_cell(row.get(key_column))
        if key:
            outcomes.setdefault(key, set()).add(flt.passes(row))
    conflicting = sorted(key for key, seen in outcomes.items() if len(seen) > 1)
    if conflicting:
        raise _Unresolved(
            f"conflicting rows for {conflicting[:3]} in '{flt.source.title}'"
        )
    return {key for key, seen in outcomes.items() if seen == {True}}


def _one_row_per_entity(
    rows: list[dict[str, Any]],
    key_column: str | None,
) -> list[dict[str, Any]]:
    """Collapse exact duplicates; two different rows for one key is a conflict."""
    if key_column is None:
        unique: list[dict[str, Any]] = []
        for row in rows:
            if row not in unique:
                unique.append(row)
        return unique

    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _norm_cell(row.get(key_column))
        if not key:
            raise _Unresolved("a qualifying row has no identifier")
        if by_key.setdefault(key, row) != row:
            raise _Unresolved(f"conflicting rows for identifier {key}")
    return list(by_key.values())


def _unique_references(sources: Iterable[_Source]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    references: list[dict[str, Any]] = []
    for source in sources:
        if source.name not in seen:
            seen.add(source.name)
            references.append(source.reference())
    return references


def _build_population(
    anchor: _Source, sources: list[_Source], scope_text: str
) -> _Population:
    """Apply EVERY filter of the sentence to the complete anchor table."""
    categorical = _find_categorical_filters(sources, scope_text)
    numeric = _find_numeric_filters(sources, scope_text)
    if not categorical and not numeric:
        raise _Unresolved("the sentence defines no filter that can be identified")

    rows = list(anchor.rows)
    links: list[tuple[str, str, _Source]] = []
    for flt in [*categorical, *numeric]:
        if flt.source.name == anchor.name:
            rows = [row for row in rows if flt.passes(row)]
            continue
        anchor_column, other_column = _shared_key_columns(anchor, flt.source)
        passing = _passing_keys(flt, other_column)
        rows = [row for row in rows if _norm_cell(row.get(anchor_column)) in passing]
        links.append((anchor_column, other_column, flt.source))

    key_column = links[0][0] if links else None
    rows = _one_row_per_entity(rows, key_column)
    if not rows:
        raise _Unresolved("no row satisfies every filter")

    relationships: list[dict[str, Any]] = []
    for anchor_column, other_column, other in links:
        relationship = {
            "left": {"document": anchor.title, "column": anchor_column},
            "right": {"document": other.title, "column": other_column},
        }
        if relationship not in relationships:
            relationships.append(relationship)

    return _Population(
        anchor=anchor,
        rows=rows,
        key_column=key_column,
        keys=[str(row.get(key_column)) for row in rows] if key_column else [],
        sources=[
            anchor,
            *[
                flt.source
                for flt in [*categorical, *numeric]
                if flt.source.name != anchor.name
            ],
        ],
        categorical=categorical,
        numeric=numeric,
        relationships=relationships,
    )


def _describe_filters(population: _Population) -> dict[str, Any]:
    return {
        "categorical": [
            {"document": f.source.title, "column": f.column, "value": f.value}
            for f in population.categorical
        ],
        "numeric": [
            {
                "document": f.source.title,
                "column": f.column,
                "operator": f.predicate.operator,
                "lower": _num(f.predicate.lower),
                "upper": _num(f.predicate.upper)
                if f.predicate.upper is not None
                else None,
            }
            for f in population.numeric
        ],
    }


def _is_date_column(column: str) -> bool:
    return bool(_roots(column) & _DATE_ROOTS)


def _pick_date_column(sources: list[_Source], text: str) -> tuple[_Source, str]:
    """The date column the question is about. One candidate is used as is;
    several must be told apart by the question's own words, never guessed."""
    candidates = [
        (source, column)
        for source in sources
        for column in source.columns
        if _is_date_column(column)
        and any(_present(row.get(column)) for row in source.rows)
    ]
    if not candidates:
        raise _Unresolved("no date column in the complete tables")
    if len(candidates) == 1:
        return candidates[0]

    text_roots = _roots(text)
    scored = sorted(
        (
            (len((_roots(column) - _GENERIC_DATE_ROOTS) & text_roots), source, column)
            for source, column in candidates
        ),
        key=lambda entry: entry[0],
        reverse=True,
    )
    if scored[0][0] == 0 or scored[1][0] == scored[0][0]:
        raise _Unresolved("several date columns could match the selection")
    return scored[0][1], scored[0][2]


def _looks_like_serial(text: str) -> bool:
    return (
        bool(re.fullmatch(r"\d{5}(?:\.\d+)?", text))
        and _SERIAL_MIN <= float(text) <= _SERIAL_MAX
    )


def _parse_moment(text: str, fmt: str) -> datetime:
    if fmt == "serial":
        return _EXCEL_EPOCH + timedelta(days=float(text))
    return datetime.strptime(text, fmt)


def _detect_date_format(values: list[str]) -> str | None:
    """The one format every value fits. 05/06/2021 could be day-first or
    month-first: if the data cannot settle it, there is no format."""
    if not values:
        return None
    if all(_looks_like_serial(value) for value in values):
        return "serial"

    outcomes: dict[str, list[datetime]] = {}
    for fmt in _DATE_FORMATS:
        try:
            outcomes[fmt] = [_parse_moment(value, fmt) for value in values]
        except ValueError:
            continue
    if len({tuple(parsed) for parsed in outcomes.values()}) != 1:
        return None
    return next(iter(outcomes))


def _column_moments(
    all_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    column: str,
) -> list[datetime]:
    """Parse the date of every qualifying row. A missing or malformed date
    could be the earliest one, so it makes the selection unresolved."""
    raw = [
        str(row.get(column)).strip() if _present(row.get(column)) else ""
        for row in rows
    ]
    if not all(raw):
        raise _Unresolved(f"a qualifying row has no '{column}' value")

    whole_column = [
        str(row.get(column)).strip() for row in all_rows if _present(row.get(column))
    ]
    fmt = _detect_date_format(whole_column) or _detect_date_format(raw)
    if fmt is None:
        raise _Unresolved(f"'{column}' values are not in one unambiguous date format")
    try:
        return [_parse_moment(value, fmt) for value in raw]
    except ValueError as exc:
        raise _Unresolved(f"malformed '{column}' value: {exc}") from exc


@dataclass(frozen=True, eq=False)
class _Joined:
    """Rows of another table that belong to the selected entity."""

    source: _Source
    anchor_column: str
    other_column: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True, eq=False)
class _Selection:
    intent: SelectionIntent
    population: _Population
    date_column: str
    chosen_rows: list[dict[str, Any]]
    joined: list[_Joined]


def _joined_rows(
    anchor: _Source,
    chosen: list[dict[str, Any]],
    sources: list[_Source],
) -> list[_Joined]:
    """Fields of the SAME entity held by other tables, through the identifier."""
    joined: list[_Joined] = []
    for other in sources:
        if other.name == anchor.name:
            continue
        try:
            anchor_column, other_column = _shared_key_columns(anchor, other)
        except _Unresolved:
            continue
        keys = {_norm_cell(row.get(anchor_column)) for row in chosen}
        rows = [row for row in other.rows if _norm_cell(row.get(other_column)) in keys]
        if rows:
            joined.append(_Joined(other, anchor_column, other_column, rows))
    return joined


def _derive_selection(
    sources: list[_Source],
    intent: SelectionIntent | None,
) -> _Selection | None:
    if intent is None:
        return None
    try:
        anchor, date_column = _pick_date_column(sources, intent.scope_text)
        population = _build_population(anchor, sources, intent.scope_text)
        moments = _column_moments(anchor.rows, population.rows, date_column)
        target = min(moments) if intent.direction == "min" else max(moments)
        chosen = [
            row for row, moment in zip(population.rows, moments) if moment == target
        ]
    except _Unresolved as exc:
        logger.info("selection left unresolved: %s", exc)
        return None

    logger.info(
        "selection: %s %s -> %d of %d qualifying rows",
        intent.direction,
        date_column,
        len(chosen),
        len(population.rows),
    )
    return _Selection(
        intent, population, date_column, chosen, _joined_rows(anchor, chosen, sources)
    )


def _selection_item(
    selection: _Selection,
    *,
    workspace_id: int,
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    population = selection.population
    chosen = selection.chosen_rows
    key_column = population.key_column
    label = "earliest" if selection.intent.direction == "min" else "latest"

    body = {
        "direction": label,
        "selection_column": selection.date_column,
        "selected_value": chosen[0].get(selection.date_column),
        "tie": len(chosen) > 1,
        "selected_rows": [dict(row) for row in chosen],
        "key_column": key_column,
        "selected_keys": [str(row.get(key_column)) for row in chosen]
        if key_column
        else [],
        "qualifying_count": len(population.rows),
        "qualifying_keys": population.keys[:MAX_QUALIFYING_KEYS_SHOWN],
        "filters": _describe_filters(population),
        "joined_rows": [
            {
                "document": joined.source.title,
                "matched_on": joined.other_column,
                "row_count": len(joined.rows),
                "rows": joined.rows[:MAX_JOINED_ROWS],
            }
            for joined in selection.joined
        ],
        "relationships": population.relationships,
        "comparisons": comparisons,
    }
    return {
        "source_type": "table_query",
        "workspace_id": workspace_id,
        "sources": _unique_references(
            [*population.sources, *(joined.source for joined in selection.joined)]
        ),
        "content": {
            "evidence_role": "selection_result",
            "population_scope": SCOPE_SELECTION,
            "document_title": population.anchor.title,
            "population_rows": len(population.rows),
            "selection_result": body,
            "interpretation": (
                f"Chosen by the backend from the complete filtered population "
                f"({len(population.rows)} qualifying rows). Authoritative for this "
                "selection; a tie lists every tied row."
            ),
        },
        "result_status": "SUCCESS",
    }


@dataclass(frozen=True, eq=False)
class _FilteredAggregate:
    intent: AggregateIntent
    population: _Population
    metric_column: str
    pairs: list[tuple[dict[str, Any], float]]


def _derive_filtered_aggregate(
    sources: list[_Source],
    intent: AggregateIntent | None,
) -> _FilteredAggregate | None:
    if intent is None:
        return None
    try:
        anchor, metric_column = _pick_numeric_column(sources, intent.metric_phrase)
        population = _build_population(anchor, sources, intent.scope_text)
    except _Unresolved as exc:
        logger.info("filtered aggregate left unresolved: %s", exc)
        return None

    pairs = [
        (row, value)
        for row in population.rows
        if (value := to_number(row.get(metric_column))) is not None
    ]
    if not pairs:
        logger.info(
            "filtered aggregate left unresolved: no numeric %s values", metric_column
        )
        return None
    return _FilteredAggregate(intent, population, metric_column, pairs)


def _aggregate_entry(result: _FilteredAggregate) -> dict[str, Any]:
    population = result.population
    values = [value for _, value in result.pairs]
    total = sum(values)
    shown = result.pairs[:MAX_QUALIFYING_ROWS_SHOWN]
    entry: dict[str, Any] = {
        "column": result.metric_column,
        "function": result.intent.function,
        "method": "computed deterministically from the complete joined population",
        "count": len(values),
        "sum": _num(total),
        "avg": _num(total / len(values)),
        "filters": _describe_filters(population),
        "matched_keys": population.keys,
        "qualifying_rows": [
            {
                "identity": _identity(row, population.anchor.columns),
                "value": _num(value),
            }
            for row, value in shown
        ],
    }
    if len(result.pairs) > len(shown):
        entry["qualifying_rows_shown"] = len(shown)
    missing = len(population.rows) - len(values)
    if missing:
        entry["rows_without_value"] = missing
    return entry


def _aggregate_item(result: _FilteredAggregate, workspace_id: int) -> dict[str, Any]:
    population = result.population
    return {
        "source_type": "table_query",
        "workspace_id": workspace_id,
        "sources": _unique_references(population.sources),
        "content": {
            "evidence_role": "aggregate_result",
            "population_scope": SCOPE_AGGREGATE,
            "document_title": population.anchor.title,
            "population_rows": len(population.rows),
            "aggregates": [_aggregate_entry(result)],
            "relationships": population.relationships,
            "interpretation": (
                "Aggregate computed after applying every filter of its sentence to "
                "complete tables joined through a shared identifier."
            ),
        },
        "result_status": "SUCCESS",
    }


def _entity_number(
    selection: _Selection,
    row: dict[str, Any],
    source_name: str,
    column: str,
) -> float | None:
    """One entity's value of `column` in the given table, or None if unclear."""
    if selection.population.anchor.name == source_name:
        return to_number(row.get(column))
    for joined in selection.joined:
        if joined.source.name != source_name:
            continue
        key = _norm_cell(row.get(joined.anchor_column))
        numbers = [
            to_number(other.get(column))
            for other in joined.rows
            if _norm_cell(other.get(joined.other_column)) == key
        ]
        return numbers[0] if len(numbers) == 1 else None
    return None


def _comparisons(
    selection: _Selection | None,
    aggregate: _FilteredAggregate | None,
) -> list[dict[str, Any]]:
    """Selected entity's value minus the group average, on the aggregate's own
    field. Both results stay independent; only this subtraction combines them."""
    if selection is None or aggregate is None:
        return []
    average = sum(value for _, value in aggregate.pairs) / len(aggregate.pairs)
    source_name = aggregate.population.anchor.name

    comparisons: list[dict[str, Any]] = []
    for row in selection.chosen_rows:
        value = _entity_number(selection, row, source_name, aggregate.metric_column)
        if value is None:
            continue
        comparisons.append(
            {
                "entity": _identity(row, selection.population.anchor.columns),
                "column": aggregate.metric_column,
                "selected_value": _num(value),
                "group_average": _num(average),
                "group_size": len(aggregate.pairs),
                "difference": _num(value - average),
                "meaning": "difference = selected value minus group average",
            }
        )
    return comparisons


def _derive_aggregates(
    tables: list[dict[str, Any]],
    *,
    query: str,
    focus_norms: set[str],
    workspace_id: int,
    run_query: Callable[..., dict[str, Any]] | None,
    filtered: _FilteredAggregate | None = None,
    filter_columns: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, set[str]]]:
    """Return (aggregate evidence items, {id(table_item): needed row sigs})."""
    query_tokens = _tokens(query)
    items: list[dict[str, Any]] = []
    needed: dict[int, set[str]] = {}

    filtered_columns: set[tuple[str, str]] = set(filter_columns or ())
    if filtered is not None:
        items.append(_aggregate_item(filtered, workspace_id))
        filtered_columns.add((filtered.population.anchor.name, filtered.metric_column))

    for item in tables:
        if _content(item).get("evidence_role") not in {
            "document_table",
            "query_result",
        }:
            continue
        semantics = (_content(item).get("query_semantics") or {}).get("type")
        if semantics in {"aggregate", "grouped", "ranking", "positional"}:
            continue
        complete = is_complete_table(item)
        if not complete and not _content(item).get("truncated"):
            continue
        entries: list[dict[str, Any]] = []
        for column in _numeric_columns_named_by_query(item, query_tokens):
            if (_table_name(item), column) in filtered_columns:
                continue
            if complete:
                entry = _aggregate_in_memory(item, column, tables, focus_norms)
            elif run_query is not None:
                entry = _aggregate_by_sql(
                    run_query, workspace_id, item, column, tables, focus_norms
                )
            else:
                entry = None
            if entry:
                entries.append(entry)
        if not entries:
            continue

        sources = [{"table_name": _table_name(item), "document_id": _doc_id(item)}]
        for entry in entries:
            for extreme in entry.get("max_rows", []):
                for rel in extreme.get("related", []):
                    for other in tables:
                        if _title(other) == rel["document"] and _table_name(other):
                            src = {
                                "table_name": _table_name(other),
                                "document_id": _doc_id(other),
                            }
                            if src not in sources:
                                sources.append(src)
                            needed.setdefault(id(other), set()).add(
                                json.dumps(rel["row"], sort_keys=True, default=str)
                            )
        items.append(
            {
                "source_type": "table_query",
                "workspace_id": item.get("workspace_id"),
                "sources": sources,
                "content": {
                    "evidence_role": "aggregate_result",
                    "population_scope": SCOPE_AGGREGATE,
                    "document_title": _title(item),
                    "population_rows": _content(item).get(
                        "rows_returned", len(_rows(item))
                    ),
                    "aggregates": entries,
                    "formulas": {"percent_of_total": "value / sum * 100"},
                },
                "result_status": "SUCCESS",
            }
        )
    return items, needed


def _reduce_table(
    item: dict[str, Any],
    focus_norms: set[str],
    keep_sigs: set[str],
) -> dict[str, Any] | None:
    rows = _rows(item)
    if len(rows) < MIN_ROWS_TO_REDUCE or not is_complete_table(item):
        return None
    kept = [
        r
        for r in rows
        if json.dumps(r, sort_keys=True, default=str) in keep_sigs
        or (focus_norms and any(_norm_cell(v) in focus_norms for v in r.values()))
    ]
    if not kept or len(kept) >= len(rows):
        return None
    content = dict(_content(item))
    content["rows"] = kept
    content["row_count"] = len(kept)
    content["rows_returned"] = len(kept)
    content["population_reduced"] = True
    content["source_rows"] = len(rows)
    content["population_scope"] = SCOPE_FOCUSED_SUBSET
    reduced = dict(item)
    reduced["content"] = content
    return reduced


def _has_focus_rows(tables: list[dict[str, Any]], focus_norms: set[str]) -> bool:
    """True when at least one focus key / entity value is an actual cell of a
    retrieved table, i.e. the question is about specific records that exist in
    the data. A key that matches no row (a period label, a stray token) is not
    a focus record."""
    if not focus_norms:
        return False
    for item in tables:
        for row in _rows(item):
            if any(_norm_cell(v) in focus_norms for v in row.values()):
                return True
    return False


def _linked_identifiers(
    tables: list[dict[str, Any]], focus_norms: set[str]
) -> set[str]:
    """Identifier values carried by the focus rows themselves (one hop). A
    focus record that references another table through an identifier column
    (order -> product -> supplier) must keep the referenced rows too."""
    linked: set[str] = set()
    for item in tables:
        id_cols = _id_columns(_columns(item))
        if not id_cols:
            continue
        for row in _rows(item):
            if any(_norm_cell(v) in focus_norms for v in row.values()):
                linked.update(_norm_cell(row.get(c)) for c in id_cols)
    linked.discard("")
    return linked


def _scope_of(item: dict[str, Any]) -> str:
    content = _content(item)
    role = content.get("evidence_role")
    semantics = (content.get("query_semantics") or {}).get("type")
    if content.get("population_scope"):
        return str(content["population_scope"])
    if role == "aggregate_result" or semantics == "aggregate":
        return SCOPE_AGGREGATE
    if semantics == "grouped":
        return SCOPE_GROUPED
    if content.get("truncated"):
        return SCOPE_TRUNCATED
    if role == "entity_row":
        return SCOPE_ENTITY_ROW
    if role == "key_joined":
        return SCOPE_FOCUSED_SUBSET
    if semantics in {"ranking", "positional"}:
        return SCOPE_FILTERED
    if is_complete_table(item):
        return SCOPE_COMPLETE
    return SCOPE_FILTERED


def _focus_norms(
    tables: list[dict[str, Any]],
    join_keys: list[str] | None,
    entity_values: list[str] | None,
) -> set[str]:
    """Normalised values of the specific records the question is about."""
    norms = {_norm_cell(k) for k in (join_keys or []) if str(k).strip()}
    norms |= {_norm_cell(v) for v in (entity_values or []) if str(v).strip()}
    for item in tables:
        for key in _content(item).get("matched_keys", []) or []:
            norms.add(_norm_cell(key))
    norms.discard("")
    return norms


def _source_table_names(items: list[dict[str, Any]]) -> set[str]:
    return {
        str(source["table_name"])
        for item in items
        for source in item.get("sources", [])
        if isinstance(source, dict) and source.get("table_name")
    }


def _filter_columns(
    selection: _Selection | None,
    aggregate: _FilteredAggregate | None,
) -> set[tuple[str, str]]:
    """(table, column) pairs used as numeric filters by either result."""
    populations = [r.population for r in (selection, aggregate) if r is not None]
    return {
        (flt.source.name, flt.column)
        for population in populations
        for flt in population.numeric
    }


def _selection_items(
    selection: _Selection | None,
    aggregate: _FilteredAggregate | None,
    *,
    wants_comparison: bool,
    workspace_id: int,
) -> list[dict[str, Any]]:
    if selection is None:
        return []
    comparisons = _comparisons(selection, aggregate) if wants_comparison else []
    return [
        _selection_item(selection, workspace_id=workspace_id, comparisons=comparisons)
    ]


def finalize_structured(
    structured: list[Any],
    *,
    workspace_id: int,
    query: str,
    needs_population: bool,
    entity_values: list[str] | None = None,
    join_keys: list[str] | None = None,
    run_query: Callable[..., dict[str, Any]] | None = None,
) -> list[Any]:
    """Tag scopes, derive deterministic results, reduce redundant rows."""
    tables = [i for i in structured if _is_table_item(i)]
    focus_norms = _focus_norms(tables, join_keys, entity_values)
    relationships = discover_relationships(tables)

    intent = interpret_query(query)
    sources = _complete_sources(tables)
    selection = _derive_selection(sources, intent.selection)
    filtered = (
        _derive_filtered_aggregate(sources, intent.aggregate)
        if needs_population
        else None
    )
    selections = _selection_items(
        selection,
        filtered,
        wants_comparison=intent.wants_comparison,
        workspace_id=workspace_id,
    )

    aggregates: list[dict[str, Any]] = []
    needed: dict[int, set[str]] = {}
    if needs_population:
        try:
            aggregates, needed = _derive_aggregates(
                tables,
                query=query,
                focus_norms=focus_norms,
                workspace_id=workspace_id,
                run_query=run_query,
                filtered=filtered,
                filter_columns=_filter_columns(selection, filtered),
            )
        except Exception:
            logger.exception("population aggregate derivation failed")
            aggregates, needed = [], {}
            if filtered is not None:
                aggregates = [_aggregate_item(filtered, workspace_id)]

    protected_tables = _source_table_names([*selections, *aggregates])

    may_reduce = bool(aggregates) and _has_focus_rows(tables, focus_norms)
    if aggregates and not may_reduce:
        logger.info("population reduction skipped: no focus record matched a row")
    reduce_norms = (
        focus_norms | _linked_identifiers(tables, focus_norms)
        if may_reduce
        else focus_norms
    )

    result: list[Any] = []
    for item in structured:
        if not _is_table_item(item):
            result.append(item)
            continue
        if (
            may_reduce
            and _table_name(item) not in protected_tables
            and _content(item).get("evidence_role") == "document_table"
        ):
            reduced = _reduce_table(item, reduce_norms, needed.get(id(item), set()))
            if reduced is not None:
                item = reduced
        content = dict(_content(item))
        content["population_scope"] = content.get("population_scope") or _scope_of(item)
        item = dict(item)
        item["content"] = content
        result.append(item)

    if (
        aggregates
        and relationships
        and not aggregates[0]["content"].get("relationships")
    ):
        aggregates[0]["content"]["relationships"] = relationships
    return result + selections + aggregates


def reconcile_coverage_notes(
    notes: list[Any],
    coverage: Any,
    structured: list[Any],
) -> list[Any]:
    """
    Text-passage coverage says nothing about tabular content. Drop coverage
    notes that talk about a document whose structured rows were successfully
    retrieved and which simply owns no text passages, so 'coverage: none'
    can never be read as 'source unavailable'.
    """
    if not notes or not isinstance(coverage, list):
        return notes
    text_free = {
        str(e.get("document_id"))
        for e in coverage
        if isinstance(e, dict) and e.get("total") == 0
    }
    hide: set[str] = set()
    for item in structured:
        if _is_table_item(item) and str(_doc_id(item)) in text_free:
            title = _content(item).get("document_title")
            if title:
                hide.add(str(title))
    if not hide:
        return notes
    kept = []
    for note in notes:
        blob = json.dumps(note, ensure_ascii=False, default=str)
        if any(title in blob for title in hide):
            continue
        kept.append(note)
    return kept


def assign_citation_indexes(evidence: list[Any]) -> list[Any]:
    """1-based citation_index equal to list position. Call LAST."""
    indexed: list[Any] = []
    for position, item in enumerate(evidence, start=1):
        if isinstance(item, dict):
            item = dict(item)
            item["citation_index"] = position
        indexed.append(item)
    return indexed
