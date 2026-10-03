"""Services Sql Generation module for QueryPilot."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm.router import router

logger = logging.getLogger(__name__)


SQL_MAX_COMPLETION_TOKENS = 1200


SYSTEM_PROMPT = """
You generate one SQLite read-only SQL statement for QueryPilot.

INPUT
- user_question: the current request
- tables: authoritative table names, exact column names, numeric_columns, and row counts
- resolved_entities: deterministic entity values plus the exact table/column locations
- query_semantics: planner operation, whether population-wide context is required, and ranking_limit (the exact N the user asked for, when it is a top/bottom N request)
- planner_notes: deterministic source/context notes

HARD RULES
1. Use ONLY table_name and column names present in the supplied schema. Never invent schema, joins, keys, aliases-as-identifiers, filters, or relationships.
2. Query only tables supplied in `tables`. Cross-document joins are prohibited here; each document is executed independently and combined later by the planner.
3. The SQLite database stores cells as TEXT. For arithmetic, ordering, ranges, SUM/AVG/MIN/MAX, or numeric comparison, CAST only columns explicitly listed in numeric_columns to REAL. Do not cast identifiers or free text.
4. Use parameter placeholders `?` for user/entity values. Put the actual values in `parameters`.
5. Only SELECT/WITH queries. No INSERT/UPDATE/DELETE/DDL, PRAGMA, ATTACH, or multiple statements.
6. When resolved_entities are present, preserve their authoritative locations: filter an entity using one of the exact columns listed in that entity location, with its exact resolved value. Do not replace the entity with a name guessed from the question.
7. For population-level calculations (total, average, percentage/share, ranking, comparison, highest/lowest, count, all/every), prefer deterministic SQL aggregation/window functions. Return the requested derived value and the required aggregate/denominator rather than a large raw population when SQL can compute the result.
8. For list/all/every requests, return the actual requested rows. If the result can exceed the tool row cap, do not imply that the returned rows are the complete population.
9. Never encode business/policy rules in SQL when planner notes say those rules come from document evidence. Return the factual inputs needed for later rule application.
10. If a repair attempt is supplied, change only the part contradicted by the real error/result; do not broaden scope.
11. RANKING (top/bottom/highest/lowest N): ORDER BY the CAST numeric measure (never a text column), DESC for top/highest and ASC for bottom/lowest, and LIMIT exactly ranking_limit when it is given. If one entity can have several rows in the table (row_count is larger than the number of distinct entities), GROUP BY the entity key and rank by SUM(CAST(measure AS REAL)) instead of ranking single rows. Add the entity key as a final ORDER BY tie-break so the order is stable. Select the entity key and the ranked measure so the ranking can be verified.
12. A LIMIT belongs only to the ranked rows. Company-wide totals or shares are computed from the whole population elsewhere; never try to get a total by summing the LIMITed rows.

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
                f"ORDER BY uses the numeric column '{bare}' without CAST(... AS REAL); "
                "cells are stored as TEXT, so this sorts alphabetically, not by value."
            )

    return problems


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
        "tables": schema,
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

    retry_problems = _ranking_problems(retried, limit, schema)
    if len(retry_problems) <= len(problems):
        if retry_problems:
            logger.warning("repaired SQL still has ranking issues: %s", retry_problems)
        return retried
    return decision
