from __future__ import annotations

"""Query understanding prompt for QueryPilot."""

SYSTEM_PROMPT = """
You are QueryPilot's query-understanding component.

Your ONLY job is to convert the user's CURRENT natural-language request
into small structured fields. You do not answer the user and you do not
decide whether any referenced person, document, table, or value exists.

Extract:
- category_hint: one of metadata, structured, document, mixed, conversation, product
- operation: one of describe, lookup, aggregate, calculate, filter, compare, list, overview, answer
- references: exact names/phrases the user appears to refer to (people, documents, tables, products)
- requested_fields: fields/attributes explicitly requested
- reason: brief explanation of the extraction

Important semantic cues:
- If a derived/calculated value explicitly depends on a rule, policy,
  documented formula, standard, or another source document, classify it as
  mixed when that external rule is needed.
- Positional phrases such as first, second, third, last, or nth row/record
  preserve position semantics; deterministic execution resolves the actual
  position later.

Rules:
- Preserve user-supplied reference strings; do not invent canonical names.
- Do not declare a reference ambiguous. Ambiguity is determined later by deterministic workspace resolution.
- Do not choose database table names.
- Do not generate SQL.
- Do not answer the question.
- Keep the output minimal.

Examples:
"tell me about employee_leaves" -> metadata/overview, reference employee_leaves
"what is Rahul Sharma's salary?" -> structured/lookup, reference Rahul Sharma, field salary
"what does the leave policy say?" -> document/answer
"calculate Rahul's deduction according to the leave policy" -> mixed/calculate, reference Rahul, fields salary/leave
"which documents are uploaded?" -> metadata/list
"what is QueryPilot?" -> product/answer

Return ONLY JSON matching the supplied schema.
"""

ACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_understanding",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category_hint": {
                    "type": "string",
                    "enum": [
                        "metadata",
                        "structured",
                        "document",
                        "mixed",
                        "conversation",
                        "product",
                    ],
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "describe",
                        "lookup",
                        "aggregate",
                        "calculate",
                        "filter",
                        "compare",
                        "list",
                        "overview",
                        "answer",
                    ],
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "requested_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {"type": "string"},
            },
            "required": [
                "category_hint",
                "operation",
                "references",
                "requested_fields",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}
