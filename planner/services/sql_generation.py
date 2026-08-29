from __future__ import annotations

import json
from typing import Any

from llm.router import router


SYSTEM_PROMPT = """
You generate SQLite read-only SQL for QueryPilot.

You receive:
- the user's question
- the exact tables available in the workspace
- the exact SQLite column names
- for each table, a "numeric_columns" list

CRITICAL: every column is physically stored as TEXT in SQLite, even
numeric-looking ones like salary or price. Plain sorting or comparison
on these columns is WRONG (e.g. "99000" > "110000" as text, because
text comparison looks at characters, not value).

For ANY column listed in a table's "numeric_columns":
- Always wrap it in CAST(column AS REAL) when used in:
  ORDER BY, WHERE comparisons (>, <, >=, <=), or aggregate
  functions (SUM, AVG, MIN, MAX).
- Example:
    SELECT "Employee_Name", CAST("Monthly_Salary" AS REAL) AS salary
    FROM "ws_6_doc_17_table_1"
    ORDER BY CAST("Monthly_Salary" AS REAL) DESC
    LIMIT 1

CRITICAL — column selection for identified-entity lookups:
The user's question wording is NOT a reliable guide to which columns
are needed. A question may ask about "salary" and "leaves" but the
correct answer may depend on OTHER columns in that same row (e.g. an
entitlement, a grade, a join key) that the question never mentions,
because a separate policy document defines how those columns combine.

Therefore:
- If the WHERE clause identifies a SPECIFIC entity (matched by name,
  ID, or any other unique identifier) rather than filtering/aggregating
  across many rows, SELECT * for that row (or all columns via JOIN if
  multiple tables are involved) instead of only the columns literally
  mentioned in the question.
- Only use a narrow column list when the query is a genuine aggregate
  over many rows (SUM/AVG/COUNT/MIN/MAX/GROUP BY) where selecting every
  column would be meaningless, or when a single table has more than
  ~15 columns and most are obviously irrelevant (e.g. free-text notes
  fields unrelated to any numeric/categorical lookup).
- When in doubt for a single-row lookup, prefer SELECT * over a narrow
  SELECT. Extra columns are cheap. A missing column that a downstream
  policy needs is not recoverable later.

Rules:
- Use table and column names exactly as provided.
- Never invent or rename identifiers.
- Use JOIN when multiple tables are required.
- Only generate SELECT or WITH ... SELECT queries.
- Never generate INSERT, UPDATE, DELETE, REPLACE, CREATE, DROP,
  ALTER, ATTACH, DETACH, PRAGMA, or VACUUM.
- Use parameters for user values when practical.
- If table data is not required, return needs_sql=false.

Return only the JSON object matching the supplied schema.
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
                    "items": {},
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


def generate_sql(
    *,
    query: str,
    schema: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "user_question": query,
        "tables": schema,
    }

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
        max_tokens=500,
        response_format=SQL_GENERATION_SCHEMA,
        include_reasoning=False,
    )

    return json.loads(
        response.choices[0].message.content
    )