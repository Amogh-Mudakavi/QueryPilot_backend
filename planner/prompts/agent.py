"""Planner prompt definitions for QueryPilot.

This module contains the structured system prompt and JSON schema used to
convert a raw user request into the normalized query-understanding payload that
feeds the execution planner.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are QueryPilot's planner. Convert the CURRENT request into JSON fields only. Do not answer, generate SQL, retrieve data, or invent references.

Fields:
category_hint = metadata | structured | document | mixed | conversation | product
operation = describe | lookup | aggregate | calculate | filter | compare | list | overview | answer
references = exact names/phrases the user actually refers to
requested_fields = attributes explicitly requested
reason = <=15 words

Rules:
- Copy references exactly; never invent canonical names. Keep ALL named sources.
- Mixed = the answer needs both table data and document text: a calculation combining a rule/policy/formula in one source with data in another, OR a figure, rate, period or measurement that the named source may not itself contain. When unsure whether the needed data sits in a table elsewhere in workspace_catalog, choose mixed.
- Table content is structured; file/table catalog questions are metadata.
- "first/second/last/nth" preserves position semantics. Deterministic resolution handles ambiguity later.
- For contextual follow-ups, use history only when the subject is unambiguous; a self-contained query does not inherit prior topic/files.
- If workspace_catalog is present, phrases like "both PDFs" may copy the exact matching titles. likely_relevant is only a hint: use every catalog entry (see its columns) that could hold data the question needs.

Return ONLY the supplied JSON schema.
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