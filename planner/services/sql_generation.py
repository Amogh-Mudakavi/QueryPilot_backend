from __future__ import annotations

import json
from typing import Any

from llm.router import router


# Hard completion cap for SQL generation (still bounded, never unlimited).
#
# The JSON we need is tiny (<~300 tokens even for a multi-table JOIN), but on
# Groq this budget also covers hidden reasoning tokens for reasoning models,
# and `include_reasoning=False` only HIDES them, it does not stop them being
# generated. At 500 a "calculate ... using the policy" question could spend
# the whole budget reasoning and never emit the JSON (json_validate_failed /
# "max completion tokens reached"). 1500 gives ~5x headroom for the JSON
# itself. It is a ceiling, not a target: the model stops when the JSON is done.
SQL_MAX_COMPLETION_TOKENS = 1500


SYSTEM_PROMPT = """
You generate SQLite read-only SQL for QueryPilot.

You receive:
- the user's question
- the exact tables available in the workspace
- the exact SQLite column names
- for each table, a "numeric_columns" list

CRITICAL: every column is physically stored as TEXT in SQLite, even
numeric-looking ones like salary or price. Plain sorting, comparison, or
arithmetic on these columns is WRONG or unsafe because text coercion can
produce unintended results.

For ANY column listed in a table's "numeric_columns":
- Always wrap it in CAST(column AS REAL) when it participates in:
  - arithmetic expressions
  - ORDER BY
  - WHERE comparisons (>, <, >=, <=)
  - aggregate functions (SUM, AVG, MIN, MAX)
- Preserve the semantic meaning of the original column.
- Example:
    SELECT "Employee_Name",
           CAST("Monthly_Salary" AS REAL) AS salary
    FROM "ws_6_doc_17_table_1"
    ORDER BY CAST("Monthly_Salary" AS REAL) DESC
    LIMIT 1
- Arithmetic examples:
    CAST("Monthly_Salary" AS REAL) / 30
    CAST("Leave_Taken" AS REAL) - CAST("Entitled_Leave" AS REAL)
    CAST("Monthly_Salary" AS REAL) / 30 * CAST("Excess_Leave" AS REAL)

CRITICAL — COLUMN SELECTION FOR IDENTIFIED-ENTITY LOOKUPS:
The user's question wording is NOT a reliable guide to which columns
are needed. A question may ask about "salary" and "leaves" but the
correct answer may depend on OTHER columns in that same row (e.g. an
entitlement, a grade, a join key) because a separate document defines
how those columns combine.

CRITICAL — RESOLVED_ENTITIES IS AUTHORITATIVE:
You may receive a "resolved_entities" list. Each entry was found by
deterministic lookup against the ACTUAL stored data (not guessed from
the user's wording) and has the shape:
    {"value": "<exact stored value>", "locations": [{"table_name": ...,
     "column": ...}, ...]}
The user's question often contains only a PARTIAL form of this value
(e.g. the question says "Rahul" while the stored, resolved value is
"Rahul Sharma"). When resolved_entities is present, any WHERE clause
identifying that entity MUST filter on the exact resolved "value",
using the exact "column"/"table_name" given, NOT on a substring or
reconstruction of the user's own wording. Never use LIKE '%...%' with
a fragment of the user's question when an exact resolved value is
available for that entity - use exact equality against the resolved value
instead.

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

MIXED TABLE + DOCUMENT QUESTIONS:
When a request combines structured data with a rule, policy, definition,
threshold, or formula that should come from document evidence:

- Use SQL only to retrieve the factual structured inputs required from the
  relevant business/operational table(s).
- Do NOT use a document-derived policy table as a substitute for the real
  business/operational table merely because that policy table exists in the
  available schema.
- When the user explicitly names a source such as an employee sheet,
  sales sheet, inventory sheet, transaction table, or other operational
  dataset, anchor the SQL to the table that corresponds to that source.
- Do NOT JOIN to a policy/document-derived table merely because the question
  is mixed. Only JOIN another structured table when the requested factual
  result actually requires matching structured values from that table.
- The governing rule/formula itself is supplied by document retrieval and is
  not something SQL should invent or infer from similarly named columns in a
  policy table.
- Never replace a real operational dataset with a policy/document table.
- Prefer returning the complete relevant structured inputs needed by a
  downstream answer/calculation step rather than trying to encode a policy
  rule that belongs to document evidence.

POSITIONAL REFERENCES ("first", "second", "third", ..., "last" employee /
row / record):
- These mean POSITION IN THE SHEET over the WHOLE, UNFILTERED table:
  ORDER BY rowid LIMIT 1 OFFSET <n-1>   (n-th row)
  ORDER BY rowid DESC LIMIT 1           (last row)
- Do NOT sort by a data column and do NOT add any WHERE condition before
  taking the position. That includes conditions implied by the rest of
  the question ("has unpaid leave", "earns more than X", "is entitled to"):
  those are applied LATER, to the selected row, never before choosing it.
- Lookups and calculations select the SAME row: "salary deduction for the
  third employee" uses the same row as "salary of the third employee".
- Rankings ("highest", "lowest", "top", "second highest") are different:
  those ARE sorted by the ranked column.

SQL PARAMETERS:
- "parameters" must contain ONLY scalar values.
- Each parameter must be a string, number, boolean, or null.
- Never return objects, dictionaries, arrays, or nested JSON as a parameter.
- Example:
    "parameters": ["E005"]

Rules:
- Use table and column names exactly as provided.
- Never invent or rename identifiers.
- A document/file title (for example "salary.xlsx" or "policy.pdf") is NOT
  a SQLite table name unless that exact string appears in the supplied
  "tables" list as "table_name". Never use filenames or document titles
  directly in FROM/JOIN clauses.
- Use JOIN when multiple STRUCTURED BUSINESS/OPERATIONAL tables are actually
  required to answer the factual part of the question.
- Only generate SELECT or WITH ... SELECT queries.
- Never generate INSERT, UPDATE, DELETE, REPLACE, CREATE, DROP,
  ALTER, ATTACH, DETACH, PRAGMA, or VACUUM.
- NEVER query SQLite schema/catalog objects such as sqlite_master,
  sqlite_schema, sqlite_temp_master, sqlite_temp_schema, or
  pragma_table_list. The exact workspace tables are already supplied in
  the "tables" input; schema discovery is unnecessary and prohibited.
- When the user asks to inspect, enumerate, or show the contents of
  available tables, query the actual table name(s) supplied in "tables"
  directly. Never write a metadata query whose purpose is to discover
  table names.
- Use parameters for user values when practical.
- If table data is not required, return needs_sql=false, UNLESS the input has
  "sql_required": true. Then the caller has already decided this request
  needs a table query: you MUST return needs_sql=true with a query, and
  needs_sql=false is never acceptable.

REPAIR MODE:
You may receive a "previous_attempt" object: the SQL and parameters that
were already tried, plus EITHER an "error" from SQLite OR "zero_rows": true.
- Fix the real problem: a wrong table/column name, a wrong CAST, a filter
  that does not match the stored value, or an invalid schema-discovery
  query.
- Use only table and column names from "tables". Never invent identifiers.
- If the previous attempt queried sqlite_master, sqlite_schema,
  pragma_table_list, or any other schema/catalog object, discard that
  approach completely and query the actual allowed table name(s) supplied
  in "tables".
- If "zero_rows" is true and "resolved_entities" is present, that entity
  DOES exist in the data: filter on the exact resolved value, column and
  table, and do not add extra conditions taken from the user's wording.
- Do not return the previous SQL unchanged.
- If "previous_attempt"."sql" is null, your previous reply was invalid or
  cut off. Reply with ONLY the JSON object and the shortest valid query,
  with no explanation.
- If previous_attempt.error says you declined to write SQL, the question DOES
  need a table query: set needs_sql=true and return it.

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


def generate_sql(
    *,
    query: str,
    schema: list[dict[str, Any]],
    resolved_entities: list[dict[str, Any]] | None = None,
    previous_attempt: dict[str, Any] | None = None,
    sql_required: bool = False,
    planner_notes: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_question": query,
        "tables": schema,
    }

    if resolved_entities:
        # These are the authoritative values deterministic entity
        # resolution already found in the actual workspace data (not
        # a guess from the raw query text). When present, WHERE
        # clauses identifying a specific row should filter on these
        # exact values/columns rather than re-deriving a filter value
        # from the user's wording, which may be partial ("Rahul")
        # while the stored value is the full value ("Rahul Sharma").
        payload["resolved_entities"] = resolved_entities

    if sql_required:
        # The planner already chose TABLE_QUERY; declining is not allowed.
        payload["sql_required"] = True

    if previous_attempt:
        # Repair mode: one bounded retry, driven by the real failure.
        payload["previous_attempt"] = previous_attempt

    if planner_notes:
        # Deterministic notes produced by contextual source resolution.
        # These clarify WHICH resource was selected; they do not grant the
        # model permission to invent identifiers or broaden the SQL scope.
        payload["planner_notes"] = [str(note) for note in planner_notes if note]

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

    return json.loads(
        response.choices[0].message.content
    )
