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

The JSON object must contain exactly these root fields:
{ "data_sources_needed": [], "category_hint": "...", "operation": "...",
  "references": [], "requested_fields": [], "needs_instance_data": false,
  "requirements": [], "reason": "..." }

Rules:
- Copy references exactly; never invent canonical names. Keep ALL named sources.
- Mixed = the answer needs both table data and document text: a calculation combining a rule/policy/formula in one source with data in another, OR a figure, rate, period or measurement that the named source may not itself contain. When unsure whether the needed data sits in a table elsewhere in workspace_catalog, choose mixed.
- data_sources_needed: decide this first. Does the answer depend on recorded values (measurements, transactions, counts, dates, statuses for a particular period, entity or item) that the sources the user named cannot themselves contain? A rule, contract, policy or specification states what should hold; whether it held, or what follows from it, needs the records. If so, list the exact `title` of every workspace_catalog entry whose columns could hold those values, even when likely_relevant is false (that flag only reflects the wording of the question). Otherwise return []. Never list a title that is not in workspace_catalog, and never list a source merely because it exists.
- If data_sources_needed is not empty, category_hint must be mixed.
- Decompose substantive requests into the smallest useful evidence requirements (usually 1-8; never more than 12). Combine tightly related fields from the same record into one requirement. Each requirement describes a fact to establish, not a source or answer.
- Every requirement must use exactly this shape: {"id":"R1","kind":"text|table|either","type":"rule|fact|numeric|entity|status|date|comparison|other","description":"...","queries":["..."],"entities":[],"depends_on":[]}.
- `kind` must be text, table, or either. `type` must be rule, fact, numeric, entity, status, date, comparison, or other. Never use `calculation`, `lookup`, or any other value as a type; calculations use `numeric`.
- Use kind=text for passages/rules, table for recorded rows/aggregates, and either when either representation could establish the fact.
- queries must be short natural-language retrieval queries grounded in the requirement; never write SQL. SQL is generated only by the dedicated SQL-generation task. Never invent document titles, IDs, policies, people or values.
- dependencies use only IDs of other requirements and only when one fact is needed to identify or formulate another. Avoid artificial dependencies.
- Use [] for requirements on chitchat or pure conversation-memory requests. The raw user question is always searched independently as a safety net.
- Table content is structured; file/table catalog questions are metadata.
- "first/second/last/nth" preserves position semantics. Deterministic resolution handles ambiguity later.
- For contextual follow-ups, use history only when the subject is unambiguous; a self-contained query does not inherit prior topic/files.
- If workspace_catalog is present, phrases like "both PDFs" may copy the exact matching titles. likely_relevant is only a hint: use every catalog entry (see its columns) that could hold data the question needs.

Return one compact JSON object with exactly the root fields listed above. Do not use Markdown fences or add commentary.
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
                "needs_instance_data": {"type": "boolean"},
                "requirements": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string", "enum": ["text", "table", "either"]},
                            "type": {"type": "string", "enum": ["rule", "fact", "numeric", "entity", "status", "date", "comparison", "other"]},
                            "description": {"type": "string"},
                            "queries": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "kind", "type", "description", "queries", "entities", "depends_on"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
            },
            "required": [
                "data_sources_needed",
                "category_hint",
                "operation",
                "references",
                "requested_fields",
                "needs_instance_data",
                "requirements",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}
# Use JSON mode and validate the domain schema locally. Provider-side strict schema
# rejection turns a recoverable enum mistake into an HTTP 400 before repair can run.
PLANNER_RESPONSE_FORMAT = {"type": "json_object"}
