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
data_sources_needed = exact workspace_catalog titles of OTHER sources that must also be read (see rules); [] when none
reason = <=15 words

Rules:
- Copy references exactly; never invent canonical names. Keep ALL named sources.
- Mixed = the answer needs both table data and document text: a calculation combining a rule/policy/formula in one source with data in another, OR a figure, rate, period or measurement that the named source may not itself contain. When unsure whether the needed data sits in a table elsewhere in workspace_catalog, choose mixed.
- data_sources_needed: decide this first. Does the answer depend on recorded values (measurements, transactions, counts, dates, statuses for a particular period, entity or item) that the sources the user named cannot themselves contain? A rule, contract, policy or specification states what should hold; whether it held, or what follows from it, needs the records. If so, list the exact `title` of every workspace_catalog entry whose columns could hold those values, even when likely_relevant is false (that flag only reflects the wording of the question). Otherwise return []. Never list a title that is not in workspace_catalog, and never list a source merely because it exists.
- If data_sources_needed is not empty, category_hint must be mixed.
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
                "data_sources_needed": {
                    "type": "array",
                    "items": {"type": "string"},
                },
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
                "data_sources_needed",
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