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
                candidates.append({
                    "type": "document",
                    "document_id": document["document_id"],
                    "title": title,
                    "source_type": document.get("source_type"),
                })

        for table in schema:
            table_name = str(table.get("table_name") or "")
            aliases = {
                normalize_text(table_name),
                normalize_text(_human_table_name(table_name)),
            }
            if normalized in aliases:
                candidates.append({
                    "type": "table",
                    "document_id": table.get("document_id"),
                    "document_title": table.get("document_title"),
                    "table_name": table_name,
                    "columns": table.get("columns", []),
                })

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
            results.append({
                "reference": ref,
                "status": "RESOLVED",
                "match": unique[0],
            })
        elif len(unique) > 1:
            results.append({
                "reference": ref,
                "status": "AMBIGUOUS",
                "matches": unique[:8],
            })
        else:
            results.append({
                "reference": ref,
                "status": "NOT_FOUND",
                "matches": [],
            })

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
        # Stage 1: exact normalized-value match. This is the safe,
        # unambiguous case and always takes priority when it finds
        # anything at all.
        grouped = _match_name_values(
            connection=connection,
            schema=schema,
            normalized_query=normalized_query,
            raw_query=query,
            mode="exact",
        )

        # Stage 2: controlled partial match, only attempted when the
        # exact pass found nothing. A candidate is accepted only if
        # every token in the query appears as a WHOLE token in the
        # candidate value (e.g. "rahul" matches "rahul sharma" via
        # the token "rahul", but not "farahul" or "rahulika", since
        # those don't contain "rahul" as a standalone token). This is
        # still deterministic set-membership on real workspace data,
        # not fuzzy/edit-distance matching, so it can't silently pick
        # a wrong person -- it can only ever find 0, 1, or many exact
        # token-level matches.
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
    """
    Search name-like columns for values matching the query.

    mode="exact": value must equal the query after normalization.
    mode="partial": every normalized query token must appear as a
        whole token in the value (order-independent), and the value
        must contain at least one token the query doesn't already
        have (otherwise this degenerates back into the exact case).
    """

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
                # A broad SQL LIKE narrows candidates cheaply; the
                # authoritative accept/reject decision still happens
                # in Python below via whole-token comparison, so a
                # loose SQL filter here cannot by itself cause a wrong
                # match.
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
                        # Identical token set already belongs to the
                        # exact stage; skip to avoid double-handling.
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
        "name", "employee", "person", "customer", "client",
        "user", "product", "title", "id",
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