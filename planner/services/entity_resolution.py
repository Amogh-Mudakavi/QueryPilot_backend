"""Services Entity Resolution module for QueryPilot."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from documents.ingestion.sqlite import SQLiteManager
from planner.services.schema_registry import get_workspace_catalog, get_workspace_schema


def normalize_text(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def resolve_references(
    *,
    workspace_id: int,
    references: list[str],
    category_hint: str | None = None,
) -> dict[str, Any]:
    """Resolve user references against authoritative workspace resources.

    Matching is exact/normalized only. Ambiguity is a fact discovered from
    the workspace catalog, not something predicted by the LLM.
    """
    schema = get_workspace_schema(workspace_id)
    catalog = get_workspace_catalog(workspace_id, schema=schema)
    results: list[dict[str, Any]] = []

    for raw in references:
        ref = str(raw or "").strip()
        if not ref:
            continue
        normalized = normalize_text(ref)
        candidates: list[dict[str, Any]] = []

        for document in catalog:
            title = str(document.get("title") or "")
            aliases = {normalize_text(title)}
            if "." in title:
                aliases.add(normalize_text(title.rsplit(".", 1)[0]))
            if normalized in aliases:
                candidates.append(
                    {
                        "type": "document",
                        "document_id": document["document_id"],
                        "title": title,
                        "source_type": document.get("source_type"),
                    }
                )

        for table in schema:
            table_name = str(table.get("table_name") or "")
            aliases = {
                normalize_text(table_name),
                normalize_text(_human_table_name(table_name)),
            }
            if normalized in aliases:
                candidates.append(
                    {
                        "type": "table",
                        "document_id": table.get("document_id"),
                        "document_title": table.get("document_title"),
                        "table_name": table_name,
                        "columns": table.get("columns", []),
                    }
                )

        entity_matches: list[dict[str, Any]] = []
        if category_hint in {"structured", "mixed"} and not candidates:
            entity_matches = _resolve_name_value(
                workspace_id=workspace_id,
                query=ref,
                schema=schema,
            )

        candidates.extend(entity_matches)
        unique = _dedupe_candidates(candidates)

        if len(unique) == 1:
            results.append(
                {
                    "reference": ref,
                    "status": "RESOLVED",
                    "match": unique[0],
                }
            )
        elif len(unique) > 1:
            results.append(
                {
                    "reference": ref,
                    "status": "AMBIGUOUS",
                    "matches": unique[:8],
                }
            )
        else:
            results.append(
                {
                    "reference": ref,
                    "status": "NOT_FOUND",
                    "matches": [],
                }
            )

    return {
        "references": results,
        "has_ambiguity": any(item["status"] == "AMBIGUOUS" for item in results),
        "has_resolved": any(item["status"] == "RESOLVED" for item in results),
    }


def _human_table_name(table_name: str) -> str:
    parts = table_name.split("_")
    if "table" in parts:
        try:
            return "table " + parts[parts.index("table") + 1]
        except IndexError:
            pass
    return table_name


def resolve_entities_from_query(
    *,
    workspace_id: int,
    query: str,
    schema: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Deterministically find entity values that are explicitly mentioned in
    the CURRENT user query. This is a fallback for cases where the planner
    fails to extract a person's/customer's name as a reference.

    A candidate is accepted only when the candidate's normalized tokens are
    all present in the query. The match is against real workspace values;
    this function never invents or fuzzily guesses an entity.
    """
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    if schema is None:
        schema = get_workspace_schema(workspace_id)
    if not schema:
        return []

    query_tokens = set(normalized_query.split())
    if not query_tokens:
        return []

    connection = sqlite3.connect(
        f"file:{SQLiteManager.DB_PATH}?mode=ro",
        uri=True,
    )
    grouped: dict[str, dict[str, Any]] = {}
    try:
        for table in schema:
            table_name = str(table.get("table_name") or "")
            if not table_name:
                continue

            for column in _entity_mention_columns(table.get("columns", [])):
                sql = (
                    f'SELECT DISTINCT "{column}" FROM "{table_name}" '
                    f'WHERE "{column}" IS NOT NULL'
                )
                try:
                    rows = connection.execute(sql).fetchall()
                except sqlite3.Error:
                    continue

                for row in rows:
                    value = str(row[0] or "").strip()
                    normalized_value = normalize_text(value)
                    if not normalized_value:
                        continue

                    value_tokens = set(normalized_value.split())
                    if not value_tokens.issubset(query_tokens):
                        continue

                    if len(value_tokens) == 1 and not _is_strong_entity_column(column):
                        continue

                    key = normalized_value
                    candidate = grouped.setdefault(
                        key,
                        {
                            "type": "entity",
                            "document_id": table.get("document_id"),
                            "document_title": table.get("document_title"),
                            "value": value,
                            "locations": [],
                        },
                    )
                    location = {
                        "table_name": table_name,
                        "column": column,
                    }
                    if location not in candidate["locations"]:
                        candidate["locations"].append(location)
    finally:
        connection.close()

    return list(grouped.values())


def _entity_mention_columns(columns: list[str]) -> list[str]:
    """Columns whose values are appropriate candidates for query mentions."""
    markers = (
        "name",
        "employee",
        "person",
        "customer",
        "client",
        "user",
        "product",
        "title",
    )
    return [
        str(column)
        for column in columns
        if any(marker in normalize_text(str(column)).split() for marker in markers)
    ]


def _is_strong_entity_column(column: str) -> bool:
    tokens = set(normalize_text(column).split())
    return bool(tokens & {"name", "employee", "person", "customer", "client", "user"})


def _resolve_name_value(
    *,
    workspace_id: int,
    query: str,
    schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    connection = sqlite3.connect(
        f"file:{SQLiteManager.DB_PATH}?mode=ro",
        uri=True,
    )
    try:
        grouped = _match_name_values(
            connection=connection,
            schema=schema,
            normalized_query=normalized_query,
            raw_query=query,
            mode="exact",
        )

        if not grouped:
            grouped = _match_name_values(
                connection=connection,
                schema=schema,
                normalized_query=normalized_query,
                raw_query=query,
                mode="partial",
            )
    finally:
        connection.close()

    return list(grouped.values())


def _match_name_values(
    *,
    connection: sqlite3.Connection,
    schema: list[dict[str, Any]],
    normalized_query: str,
    raw_query: str,
    mode: str,
) -> dict[str, dict[str, Any]]:
    query_tokens = set(normalized_query.split())
    grouped: dict[str, dict[str, Any]] = {}

    for table in schema:
        table_name = str(table.get("table_name") or "")
        for column in _name_like_columns(table.get("columns", [])):
            if mode == "exact":
                sql = (
                    f'SELECT DISTINCT "{column}" FROM "{table_name}" '
                    f'WHERE "{column}" IS NOT NULL '
                    f'AND LOWER(CAST("{column}" AS TEXT)) = LOWER(?) LIMIT 20'
                )
                params: tuple[Any, ...] = (raw_query,)
            else:
                sql = (
                    f'SELECT DISTINCT "{column}" FROM "{table_name}" '
                    f'WHERE "{column}" IS NOT NULL '
                    f'AND LOWER(CAST("{column}" AS TEXT)) LIKE ? LIMIT 50'
                )
                params = (f"%{raw_query.lower()}%",)

            try:
                rows = connection.execute(sql, params).fetchall()
            except sqlite3.Error:
                continue

            for row in rows:
                value = str(row[0] or "").strip()
                if not value:
                    continue
                value_key = normalize_text(value)
                if not value_key:
                    continue

                if mode == "exact":
                    if value_key != normalized_query:
                        continue
                else:
                    value_tokens = set(value_key.split())
                    if not query_tokens.issubset(value_tokens):
                        continue
                    if value_tokens == query_tokens:
                        continue

                candidate = grouped.setdefault(
                    value_key,
                    {
                        "type": "entity",
                        "document_id": table.get("document_id"),
                        "document_title": table.get("document_title"),
                        "value": value,
                        "locations": [],
                    },
                )
                location = {
                    "table_name": table_name,
                    "column": column,
                }
                if location not in candidate["locations"]:
                    candidate["locations"].append(location)

    return grouped


def _name_like_columns(columns: list[str]) -> list[str]:
    markers = (
        "name",
        "employee",
        "person",
        "customer",
        "client",
        "user",
        "product",
        "title",
        "id",
    )
    return [
        str(column)
        for column in columns
        if any(marker in normalize_text(str(column)).split() for marker in markers)
    ]


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate.get("type"),
            candidate.get("document_id"),
            candidate.get("table_name"),
            candidate.get("column"),
            candidate.get("value"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
