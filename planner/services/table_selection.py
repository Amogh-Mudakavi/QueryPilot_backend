"""
planner/services/table_selection.py

Evidence selection for structured sources.

A relevant DOCUMENT does not make every TABLE inside it relevant. Once the
documents for a request are resolved, this module decides -- from the
document profiles and the schema/catalog metadata (columns, row counts, a
couple of sample rows) -- which tables the question actually needs, BEFORE
any table is read. The executor then reads only those tables and the
cross-document key join only searches those tables.

Nothing here knows about any domain, column name, or file format. It never
executes model-written SQL: the model only chooses among table names that
came from the workspace schema, and the answer is validated against that
list. Any failure returns None, meaning "do not narrow" (previous behavior).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from documents.ingestion.vectors import VectorStore
from llm.router import router

logger = logging.getLogger(__name__)

SELECTION_MAX_COMPLETION_TOKENS = 800
MAX_PROFILE_CHARS = 500
MAX_SAMPLE_ROWS = 2
MAX_SAMPLE_CELL_CHARS = 120
MAX_COLUMNS_SHOWN = 40

SYSTEM_PROMPT = """
You choose which structured tables QueryPilot must read to answer a question.

You receive the user's question, the planner's understanding of it, and the
documents the question refers to. Each document lists its tables with
column names, row count and a few sample rows.

A relevant document does NOT make every table in it relevant. Select a
table only if the answer needs it for at least one of:
- values the question asks for,
- values used in a condition, filter, ranking or grouping in the question,
- input values of a requested calculation,
- a rule, policy, rate, threshold, tier or formula that the requested
  calculation or decision depends on (such tables may look like generic
  name/description or key/value tables: use the sample rows to tell),
- a linking key that is required to connect two tables you selected.

Do not select a table whose contents neither the question nor a needed
calculation uses.

Missing a required table makes the final answer wrong, while an extra table
only costs time. When you cannot tell whether a table is needed, include it.
Select at least one table from every listed document.

Use only table_name values from the input. Return only the JSON object.
"""


def _shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_SAMPLE_CELL_CHARS:
        return value[:MAX_SAMPLE_CELL_CHARS] + "..."
    return value


def _document_profiles(workspace_id: int, document_ids: list[int]) -> dict[int, str]:
    try:
        profiles = VectorStore.get_document_profiles(
            workspace_id=workspace_id,
            document_ids=document_ids,
        )
        return {
            int(document_id): str(profile)[:MAX_PROFILE_CHARS]
            for document_id, profile in profiles.items()
            if isinstance(profile, str) and profile.strip()
        }
    except Exception:
        logger.exception("table selection: document profiles unavailable")
        return {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_tables(
    *,
    workspace_id: int,
    query: str,
    understanding: dict[str, Any],
    candidates: list[dict[str, Any]],
    forced_table_names: set[str] | None = None,
) -> set[str] | None:
    """
    Return the table names to read, or None when no narrowing applies.

    `candidates` are schema-registry items of the resolved documents.
    `forced_table_names` (e.g. tables proven to contain a resolved entity)
    are always kept.
    """
    forced = set(forced_table_names or set())
    names = [str(c.get("table_name")) for c in candidates if c.get("table_name")]
    if len(names) <= 1:
        return None

    by_document: dict[int | None, list[dict[str, Any]]] = {}
    for item in candidates:
        by_document.setdefault(_as_int(item.get("document_id")), []).append(item)

    if all(len(tables) <= 1 for tables in by_document.values()):
        return None

    profiles = _document_profiles(
        workspace_id, [d for d in by_document if d is not None]
    )

    documents_payload = []
    for document_id, tables in by_document.items():
        documents_payload.append(
            {
                "document_id": document_id,
                "title": tables[0].get("document_title"),
                "profile": profiles.get(document_id, ""),
                "tables": [
                    {
                        "table_name": t.get("table_name"),
                        "columns": [str(c) for c in (t.get("columns") or [])][
                            :MAX_COLUMNS_SHOWN
                        ],
                        "row_count": t.get("row_count"),
                        "sample_rows": [
                            {k: _shorten(v) for k, v in row.items()}
                            for row in (t.get("sample_rows") or [])[:MAX_SAMPLE_ROWS]
                            if isinstance(row, dict)
                        ],
                    }
                    for t in tables
                ],
            }
        )

    payload = {
        "user_question": query,
        "understanding": {
            "operation": understanding.get("operation"),
            "requested_fields": understanding.get("requested_fields"),
            "references": understanding.get("references"),
        },
        "documents": documents_payload,
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "table_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {"type": "string", "enum": names},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["tables", "reason"],
                "additionalProperties": False,
            },
        },
    }

    try:
        response = router.generate(
            task="table_selection",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ],
            temperature=0.0,
            max_tokens=SELECTION_MAX_COMPLETION_TOKENS,
            response_format=response_format,
            include_reasoning=False,
        )
        data = json.loads(response.choices[0].message.content)
        chosen = {str(n) for n in data.get("tables", []) if str(n) in set(names)}
    except Exception:
        logger.exception("table selection failed; reading all candidate tables")
        return None

    if not chosen:
        return None

    for document_id, tables in by_document.items():
        doc_names = {str(t.get("table_name")) for t in tables}
        if not (doc_names & (chosen | forced)):
            chosen |= doc_names

    chosen |= forced & set(names)

    logger.info(
        "table selection: selected=%s of %s reason=%s",
        sorted(chosen),
        names,
        str(data.get("reason", ""))[:300],
    )
    return chosen
