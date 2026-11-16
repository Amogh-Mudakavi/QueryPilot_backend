"""Ground the words of a question in the data that is actually stored.

Why this exists
---------------
SQL generation only sees column names, a few sample rows and the values of
columns that have very few distinct values. A concept that lives in a column
with many distinct values (a category with twenty members, a status with
thirty codes) is invisible, so the model guesses which column holds it, often
filtering a name column with LIKE and getting an empty or zero result that
looks like a real answer.

What it does
------------
It takes the content words of the question and looks each one up in the
stored text values of the tables in scope. It reports, per word, the table
and column where the word occurs, the exact stored values and how many rows
hold them. SQL generation receives that as ``stored_value_matches`` and can
filter the right column with the exact stored spelling.

Properties
----------
* No LLM call, no knowledge of any file, column name or domain.
* Read-only, through the same ``query_tables`` tool, so the table allow-list
  and SQL validation still apply.
* Bounded: few words, few queries, small tables only, small result.
* Best effort: any failure returns no matches and SQL generation proceeds as
  it did before.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

from planner.services.table_selection import _SELECTION_STOPWORDS, _tokens
from planner.tools.query_tables import query_tables

# Log under the agent's logger name so existing log searches keep working.
logger = logging.getLogger("planner.services.agent_service")

_MAX_TERMS = 5
_MIN_TERM_CHARS = 3
_MAX_TABLE_ROWS = 100_000
_PAIRS_PER_QUERY = 120
_COLUMNS_PER_TERM = 4
_VALUES_PER_COLUMN = 4
_MAX_MATCHES = 6
_VALUE_CHARS = 60
_MAX_QUERIES = 60

# Words that shape a question without naming anything stored.
_EXTRA_STOPWORDS = frozenset(
    "many much number count total list any also most least first last whether "
    "none every some other another per via here out off over under more less "
    "than just only own same such very".split()
)
_DATE_NAME_PARTS = ("date", "dob", "timestamp", "datetime")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(query: str) -> list[str]:
    """Content words of the question, normalised the way columns are tokenised.

    The longest words are the most specific, so when the question has more
    than ``_MAX_TERMS`` of them the longest are kept, in their original order.
    """
    found: list[str] = []
    for raw in _WORD.findall(str(query or "").lower()):
        if len(raw) < _MIN_TERM_CHARS:
            continue
        if raw in _SELECTION_STOPWORDS or raw in _EXTRA_STOPWORDS:
            continue
        normalized = next(iter(_tokens(raw)), raw)
        if len(normalized) < _MIN_TERM_CHARS:
            continue
        if normalized in _SELECTION_STOPWORDS or normalized in _EXTRA_STOPWORDS:
            continue
        if not re.fullmatch(r"[^\W_]+", normalized):
            continue  # only letters/digits are ever placed in a statement
        if normalized not in found:
            found.append(normalized)
    if len(found) <= _MAX_TERMS:
        return found
    keep = set(sorted(found, key=len, reverse=True)[:_MAX_TERMS])
    return [term for term in found if term in keep]


def _usable_name(name: str) -> bool:
    return bool(name) and "`" not in name and '"' not in name


def _text_columns(table: dict[str, Any]) -> list[str]:
    numeric = {str(c) for c in table.get("numeric_columns") or []}
    columns: list[str] = []
    for column in table.get("columns") or []:
        column = str(column)
        if column in numeric or not _usable_name(column):
            continue
        if any(part in column.casefold() for part in _DATE_NAME_PARTS):
            continue  # shown converted, never as stored; a word never names one
        columns.append(column)
    return columns


def _as_int(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return 0 if math.isnan(number) else int(number)


def ground_question(
    *,
    workspace_id: int,
    query: str,
    schema: list[dict[str, Any]],
    run_query: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return where the question's words occur in the stored data.

    Each match is ``{word, table, column, stored_values: [{value, rows}],
    rows_matching, whole_value}``. ``whole_value`` is true when a stored value
    is exactly the word, which is how a category or status looks, as opposed to
    a free-text column that merely mentions it.
    """
    run = run_query or query_tables
    terms = _terms(query)
    if not terms or not schema:
        return []

    budget = {"left": _MAX_QUERIES}

    def execute(sql: str, parameters: list[Any], table_name: str) -> list[dict[str, Any]]:
        if budget["left"] <= 0:
            return []
        budget["left"] -= 1
        try:
            result = run(
                workspace_id=workspace_id,
                sql=sql,
                parameters=parameters,
                allowed_table_names={table_name},
            )
        except Exception as exc:  # best effort: never block SQL generation
            logger.debug("value grounding query skipped: %s", str(exc)[:160])
            return []
        rows = result.get("rows") if isinstance(result, dict) else None
        return [row for row in rows or [] if isinstance(row, dict)]

    matches: list[dict[str, Any]] = []
    try:
        for table in schema:
            if not isinstance(table, dict):
                continue
            table_name = str(table.get("table_name") or "")
            row_count = table.get("row_count")
            if not _usable_name(table_name):
                continue
            if isinstance(row_count, int) and row_count > _MAX_TABLE_ROWS:
                continue
            columns = _text_columns(table)
            if not columns:
                continue

            # One query per table counts, for every (word, column) pair, the
            # rows whose value contains the word. Words are letters/digits
            # only, so they are safe to place in the statement as literals.
            pairs = [(term, column) for term in terms for column in columns]
            counts: dict[tuple[str, str], int] = {}
            for start in range(0, len(pairs), _PAIRS_PER_QUERY):
                chunk = pairs[start:start + _PAIRS_PER_QUERY]
                selects = ", ".join(
                    f"SUM(LOWER(`{column}`) LIKE '%{term}%') AS c{index}"
                    for index, (term, column) in enumerate(chunk)
                )
                rows = execute(f"SELECT {selects} FROM `{table_name}`", [], table_name)
                if not rows:
                    continue
                for index, (term, column) in enumerate(chunk):
                    hits = _as_int(rows[0].get(f"c{index}"))
                    if hits > 0:
                        counts[(term, column)] = hits

            for term in terms:
                hit_columns = sorted(
                    (c for (t, c) in counts if t == term),
                    key=lambda c: counts[(term, c)],
                    reverse=True,
                )
                for column in hit_columns[:_COLUMNS_PER_TERM]:
                    value_rows = execute(
                        f"SELECT `{column}` AS v, COUNT(*) AS n FROM `{table_name}` "
                        f"WHERE LOWER(`{column}`) LIKE ? GROUP BY `{column}` "
                        f"ORDER BY n DESC LIMIT {_VALUES_PER_COLUMN + 1}",
                        [f"%{term}%"],
                        table_name,
                    )
                    stored = [
                        {"value": str(row.get("v"))[:_VALUE_CHARS], "rows": _as_int(row.get("n"))}
                        for row in value_rows
                        if row.get("v") is not None
                    ]
                    if not stored:
                        continue
                    shown = stored[:_VALUES_PER_COLUMN]
                    whole = any(
                        _tokens(item["value"]) == frozenset({term}) for item in shown
                    )
                    matches.append(
                        {
                            "word": term,
                            "table": table_name,
                            "column": column,
                            "stored_values": shown,
                            "rows_matching": counts[(term, column)],
                            "whole_value": whole,
                            "_distinct": len(stored),
                        }
                    )
    except Exception:
        logger.exception("value grounding failed; continuing without it")
        return []

    # A column that holds the word as a whole value (a category) beats one that
    # only mentions it inside longer free text; fewer distinct matching values
    # means a more categorical column; then the column holding more rows.
    matches.sort(
        key=lambda m: (not m["whole_value"], m["_distinct"], -m["rows_matching"])
    )
    cleaned: list[dict[str, Any]] = []
    for match in matches[:_MAX_MATCHES]:
        match = dict(match)
        match.pop("_distinct", None)
        cleaned.append(match)
    return cleaned
