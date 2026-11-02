"""Services Sql Generation module for QueryPilot."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm.router import router

logger = logging.getLogger(__name__)


SQL_MAX_COMPLETION_TOKENS = 2600


SYSTEM_PROMPT = """
You generate one MySQL read-only SQL statement for QueryPilot.

INPUT
- user_question: the current request
- tables: authoritative table names, exact column names, numeric_columns, row counts, and sample_rows (a few real stored rows showing each column's stored format)
- resolved_entities: deterministic entity values plus the exact table/column locations
- query_semantics: planner operation, whether population-wide context is required, and ranking_limit (the exact N the user asked for, when it is a top/bottom N request)
- planner_notes: deterministic source/context notes

HARD RULES
1. Use ONLY table_name and column names present in the supplied schema. Never invent schema, joins, keys, aliases-as-identifiers, filters, or relationships.
2. Query only tables supplied in `tables`. Cross-document joins are prohibited here; each document is executed independently and combined later by the planner.
3. The structured read database is MySQL and extracted cell values may be stored as TEXT. For arithmetic, ordering, ranges, SUM/AVG/MIN/MAX, or numeric comparison, CAST only columns explicitly listed in numeric_columns to DECIMAL(30,10). Do not cast identifiers or free text. Use MySQL-compatible syntax and functions.
4. Use parameter placeholders `?` for user/entity values. Put the actual values in `parameters`.
5. Only one read-only SELECT/WITH query. No writes, DDL, administrative commands, file access, locking clauses, or multiple statements.
6. When resolved_entities are present, preserve their authoritative locations: filter an entity using one of the exact columns listed in that entity location, with its exact resolved value. Do not replace the entity with a name guessed from the question.
7. For population-level calculations (total, average, percentage/share, ranking, comparison, highest/lowest, count, all/every), prefer deterministic SQL aggregation/window functions. Return the requested derived value and the required aggregate/denominator rather than a large raw population when SQL can compute the result.
8. For list/all/every requests, return the actual requested rows. If the result can exceed the tool row cap, do not imply that the returned rows are the complete population.
9. Never encode business/policy rules in SQL when planner notes say those rules come from document evidence. Return the factual inputs needed for later rule application.
10. If a repair attempt is supplied, change only the part contradicted by the real error/result; do not broaden scope.
11. RANKING (top/bottom/highest/lowest N): ORDER BY the CAST numeric measure (never a text column), DESC for top/highest and ASC for bottom/lowest, and LIMIT exactly ranking_limit when it is given. If one entity can have several rows in the table (row_count is larger than the number of distinct entities), GROUP BY the entity key and rank by SUM(CAST(measure AS DECIMAL(30,10))) instead of ranking single rows. Add the entity key as a final ORDER BY tie-break so the order is stable. Select the entity key and the ranked measure so the ranking can be verified.
12. A LIMIT belongs only to the ranked rows. Company-wide totals or shares are computed from the whole population elsewhere; never try to get a total by summing the LIMITed rows.
13. IDENTIFIERS: Quote MySQL identifiers with backticks when needed, especially identifiers that are reserved words (for example `key`, `condition`, `rank`, or `value`). Do not use a reserved word as an unquoted alias. Quote identifiers, not string literals.
14. STORED VALUES: sample_rows and previous_attempt.error (when present) show how values are really stored. Express every filter from the question (entity, date, month, period, status, category) in the stored representation and spelling of the column you filter on, mapping the user's wording to it. Keep the meaning of the user's filter exactly. Never drop or widen a filter to get rows; if the stored values show the requested value does not exist, keep the filter.

OUTPUT
Return exactly one JSON object matching the provided schema. `needs_sql` must be true whenever the planner requested SQL. Keep SQL as short as correctness allows.
"""

SQL_GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sql_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "needs_sql": {"type": "boolean"},
                "sql": {"type": ["string", "null"]},
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": [
                            "string",
                            "number",
                            "boolean",
                            "null",
                        ],
                    },
                },
            },
            "required": [
                "needs_sql",
                "sql",
                "parameters",
            ],
            "additionalProperties": False,
        },
    },
}


_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NUMBER = r"(\d{1,3}|" + "|".join(_WORD_NUMBERS) + r")"
_RANK_WORDS = r"(?:top|bottom|highest|lowest|best|worst)"
_RANK_THEN_N = re.compile(r"\b" + _RANK_WORDS + r"\s+" + _NUMBER + r"\b", re.IGNORECASE)
_N_THEN_RANK = re.compile(r"\b" + _NUMBER + r"\s+" + _RANK_WORDS + r"\b", re.IGNORECASE)

_AGG_OR_CAST = re.compile(r"\b(CAST|SUM|AVG|COUNT|MIN|MAX|TOTAL)\s*\(", re.IGNORECASE)
_WINDOW_RANK = re.compile(r"\b(ROW_NUMBER|RANK|DENSE_RANK)\s*\(", re.IGNORECASE)


def requested_ranking_limit(query: str | None) -> int | None:
    """N from 'top 3', 'bottom five', '3 highest' ... else None."""
    text = str(query or "")
    for pattern in (_RANK_THEN_N, _N_THEN_RANK):
        match = pattern.search(text)
        if match:
            raw = match.group(1).lower()
            value = int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)
            if value and value > 0:
                return value
    return None


def _numeric_column_names(schema: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in schema or []:
        if not isinstance(item, dict):
            continue
        for column in item.get("numeric_columns") or []:
            if isinstance(column, str):
                names.add(column.strip().lower())
            elif isinstance(column, dict):
                value = column.get("name") or column.get("column")
                if value:
                    names.add(str(value).strip().lower())
    return names


def _first_order_key(sql: str) -> str | None:
    match = re.search(
        r"\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None
    return match.group(1).split(",")[0].strip()


def _ranking_problems(
    decision: Any,
    limit: int,
    schema: list[dict[str, Any]],
) -> list[str]:
    """Contract violations of a generated ranking query. Empty list = fine."""
    if not isinstance(decision, dict) or not decision.get("needs_sql"):
        return []
    sql = str(decision.get("sql") or "").strip()
    if not sql:
        return []

    normalized = re.sub(r"\s+", " ", sql)
    problems: list[str] = []

    has_window_rank = bool(_WINDOW_RANK.search(normalized))
    order_key = _first_order_key(normalized)
    limit_match = re.search(r"\bLIMIT\s+(\?|\d+)", normalized, re.IGNORECASE)

    if order_key is None and not has_window_rank:
        problems.append(
            "This is a top/bottom ranking request but the query has no ORDER BY "
            "on the ranked measure."
        )

    if limit_match is None:
        if not has_window_rank:
            problems.append(
                f"The user asked for exactly {limit} ranked rows but the query has no LIMIT."
            )
    elif limit_match.group(1).isdigit() and int(limit_match.group(1)) != limit:
        problems.append(
            f"The user asked for exactly {limit} ranked rows but the query uses "
            f"LIMIT {limit_match.group(1)}."
        )

    if order_key is not None and not _AGG_OR_CAST.search(order_key):
        bare = re.sub(r"\b(ASC|DESC)\b", "", order_key, flags=re.IGNORECASE)
        bare = bare.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
        bare = bare.split(".")[-1].strip().lower()
        if bare in _numeric_column_names(schema):
            problems.append(
                f"ORDER BY uses the numeric column '{bare}' without CAST(... AS DECIMAL(30,10)); "
                "cells are stored as TEXT, so this sorts alphabetically, not by value."
            )

    return problems


_SAMPLE_ROWS_IN_PROMPT = 3
_SAMPLE_VALUE_CHARS = 40


def _prompt_schema(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Schema as sent to the model: sample rows kept small (token budget)."""
    compact: list[Any] = []
    for item in schema or []:
        if not isinstance(item, dict):
            compact.append(item)
            continue
        item = dict(item)
        rows = item.get("sample_rows")
        if isinstance(rows, list):
            item["sample_rows"] = [
                {
                    str(key): (
                        value[:_SAMPLE_VALUE_CHARS] if isinstance(value, str) else value
                    )
                    for key, value in row.items()
                }
                for row in rows[:_SAMPLE_ROWS_IN_PROMPT]
                if isinstance(row, dict)
            ]
        compact.append(item)
    return compact


def _call_model(payload: dict[str, Any]) -> dict[str, Any]:
    response = router.generate(
        task="sql_generation",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=SQL_MAX_COMPLETION_TOKENS,
        response_format=SQL_GENERATION_SCHEMA,
        include_reasoning=False,
    )

    return json.loads(response.choices[0].message.content)



_MYSQL_RESERVED_IDENTIFIERS = frozenset({
    "accessible", "add", "all", "alter", "analyze", "and", "as", "asc",
    "before", "between", "by", "call", "case", "change", "check", "condition",
    "constraint", "continue", "create", "cross", "current_date", "current_time",
    "current_timestamp", "current_user", "database", "databases", "day_hour",
    "day_microsecond", "day_minute", "day_second", "dec", "decimal", "default",
    "delayed", "delete", "desc", "describe", "distinct", "distinctrow", "div",
    "drop", "dual", "each", "else", "elseif", "enclosed", "escaped", "exists",
    "explain", "false", "fetch", "for", "force", "foreign", "from", "fulltext",
    "grant", "group", "having", "high_priority", "hour_microsecond", "hour_minute",
    "hour_second", "if", "ignore", "in", "index", "infile", "inner", "inout",
    "insensitive", "insert", "int", "integer", "interval", "into", "is", "iterate",
    "join", "key", "keys", "kill", "leading", "leave", "left", "like", "limit",
    "lines", "load", "localtime", "localtimestamp", "lock", "long", "loop", "low_priority",
    "master_ssl_verify_server_cert", "match", "maxvalue", "mediumint", "minute_microsecond",
    "minute_second", "mod", "modifies", "natural", "not", "no_write_to_binlog", "null",
    "on", "optimize", "option", "optionally", "or", "order", "out", "outer", "outfile",
    "partition", "precision", "primary", "procedure", "purge", "range", "read", "reads",
    "read_write", "real", "recursive", "references", "regexp", "release", "rename",
    "repeat", "replace", "require", "restrict", "return", "revoke", "right", "rlike",
    "schema", "schemas", "second_microsecond", "select", "sensitive", "separator", "set",
    "show", "smallint", "spatial", "specific", "sql", "sql_big_result", "sql_calc_found_rows",
    "sql_small_result", "ssl", "starting", "straight_join", "table", "terminated", "then",
    "tinyint", "to", "trailing", "trigger", "true", "undo", "union", "unique", "unlock",
    "unsigned", "update", "usage", "use", "using", "utc_date", "utc_time", "utc_timestamp",
    "values", "varbinary", "varchar", "varying", "when", "where", "while", "with", "write",
    "xor", "year_month", "zerofill", "value", "status", "system", "rank", "rows", "row",
})


def _reserved_aliases(sql: str) -> set[str]:
    """Find aliases after AS outside literals and CAST type declarations."""
    aliases: set[str] = set()
    stack: list[bool] = []
    previous_word: str | None = None
    expect_alias = False
    quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote is not None:
            if ch == "\\" and quote != "`" and i + 1 < len(sql):
                i += 2
                continue
            if ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"', '`'):
            quote = ch
            i += 1
            continue
        if ch == "(":
            stack.append(bool(stack and stack[-1]) or (previous_word or "").casefold() == "cast")
            expect_alias = False
            previous_word = None
            i += 1
            continue
        if ch == ")":
            if stack:
                stack.pop()
            expect_alias = False
            previous_word = None
            i += 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", sql[i:])
        if match:
            token = match.group(0)
            folded = token.casefold()
            in_cast = any(stack)
            if expect_alias and not in_cast:
                aliases.add(folded)
            expect_alias = folded == "as" and not in_cast
            previous_word = token
            i += len(token)
            continue
        if not ch.isspace():
            if expect_alias:
                expect_alias = False
            if ch not in ".":
                previous_word = None
        i += 1
    return aliases


def _quote_reserved_schema_identifiers(sql: str, schema: list[dict[str, Any]]) -> str:
    """Backtick-quote reserved schema identifiers and aliases, not literals."""
    known: set[str] = set()
    for table in schema or []:
        if not isinstance(table, dict):
            continue
        table_name = table.get("table_name")
        if table_name:
            known.add(str(table_name).casefold())
        for column in table.get("columns") or []:
            if isinstance(column, str):
                known.add(column.casefold())
            elif isinstance(column, dict):
                name = column.get("name") or column.get("column")
                if name:
                    known.add(str(name).casefold())
    aliases = _reserved_aliases(sql)

    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(sql):
        ch = sql[i]
        if quote is not None:
            out.append(ch)
            if ch == "\\" and quote != "`" and i + 1 < len(sql):
                i += 1
                out.append(sql[i])
            elif ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 1
                    out.append(sql[i])
                else:
                    quote = None
            i += 1
            continue
        if ch in ("'", '"', '`'):
            quote = ch
            out.append(ch)
            i += 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", sql[i:])
        if match:
            token = match.group(0)
            folded = token.casefold()
            if (folded in known or folded in aliases) and folded in _MYSQL_RESERVED_IDENTIFIERS:
                out.append(f"`{token}`")
            else:
                out.append(token)
            i += len(token)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def generate_sql(
    *,
    query: str,
    schema: list[dict[str, Any]],
    resolved_entities: list[dict[str, Any]] | None = None,
    previous_attempt: dict[str, Any] | None = None,
    sql_required: bool = False,
    planner_notes: list[str] | None = None,
    query_semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_question": query,
        "tables": _prompt_schema(schema),
    }

    if resolved_entities:
        payload["resolved_entities"] = resolved_entities

    if sql_required:
        payload["sql_required"] = True

    if previous_attempt:
        payload["previous_attempt"] = previous_attempt

    if planner_notes:
        payload["planner_notes"] = [str(note) for note in planner_notes if note]

    semantics: dict[str, Any] = dict(query_semantics or {})
    limit = requested_ranking_limit(query)
    if limit is not None:
        semantics["ranking_limit"] = limit
    if semantics:
        payload["query_semantics"] = semantics

    decision = _call_model(payload)
    if isinstance(decision, dict) and isinstance(decision.get("sql"), str):
        decision["sql"] = _quote_reserved_schema_identifiers(decision["sql"], schema)

    if limit is None or previous_attempt:
        return decision

    problems = _ranking_problems(decision, limit, schema)
    if not problems:
        return decision

    logger.warning(
        "generated SQL breaks the ranking contract (limit=%s): %s | sql=%r",
        limit,
        problems,
        str(decision.get("sql"))[:300],
    )

    retry_payload = dict(payload)
    retry_payload["previous_attempt"] = {
        "sql": decision.get("sql"),
        "parameters": decision.get("parameters", []),
        "error": " ".join(problems),
    }
    try:
        retried = _call_model(retry_payload)
    except Exception:
        logger.exception("ranking-contract repair failed; keeping the first SQL")
        return decision

    if isinstance(retried, dict) and isinstance(retried.get("sql"), str):
        retried["sql"] = _quote_reserved_schema_identifiers(retried["sql"], schema)
    retry_problems = _ranking_problems(retried, limit, schema)
    if len(retry_problems) <= len(problems):
        if retry_problems:
            logger.warning("repaired SQL still has ranking issues: %s", retry_problems)
        return retried
    return decision
