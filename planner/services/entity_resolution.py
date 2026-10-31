"""Services Entity Resolution module for QueryPilot."""

from __future__ import annotations

import re
from typing import Any

from django.db import connections

from documents.ingestion.sqlite import SQLiteManager
from planner.services.schema_registry import get_workspace_catalog


STRUCTURED_READ_DB = "structured_read"


def normalize_text(
    value: str,
) -> str:
    value = str(value or "").strip().lower()

    value = re.sub(
        r"[\W\_]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    return " ".join(value.split())


def _light_schema_from_catalog(
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Table metadata (document, table name, columns) from a catalog.

    Reference resolution only needs names and columns, never sample rows or
    row counts, so it is built from the catalog's table-index tier.
    """

    schema: list[dict[str, Any]] = []

    for document in catalog:
        for table in document.get("tables") or []:
            table_name = table.get("table_name")

            if not table_name:
                continue

            schema.append(
                {
                    "document_id": document.get("document_id"),
                    "document_title": document.get("title"),
                    "source_type": document.get("source_type"),
                    "table_name": str(table_name),
                    "columns": list(table.get("columns") or []),
                }
            )

    schema.sort(key=lambda item: item["table_name"])

    return schema


def resolve_references(
    *,
    workspace_id: int,
    references: list[str],
    category_hint: str | None = None,
    schema: list[dict[str, Any]] | None = None,
    catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve user references against authoritative workspace resources.

    Matching is exact/normalized only. Ambiguity is a fact discovered from
    the workspace catalog, not something predicted by the LLM.

    ``schema`` / ``catalog`` are optional pre-loaded metadata (additive; the
    existing call signature is unchanged). When absent, the catalog is loaded
    WITHOUT row counts / sampling, and the table list is derived from it.
    """

    if catalog is None:
        catalog = get_workspace_catalog(
            workspace_id,
            include_row_counts=False,
        )

    if schema is None:
        schema = _light_schema_from_catalog(catalog)

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
                        "columns": table.get(
                            "columns",
                            [],
                        ),
                    }
                )

        entity_matches: list[dict[str, Any]] = []

        if (
            category_hint
            in {
                "structured",
                "mixed",
            }
            and not candidates
        ):
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

    results = _narrow_ambiguous_by_resolved_context(results)  # QP-PATCH

    return {
        "references": results,
        "has_ambiguity": any(item["status"] == "AMBIGUOUS" for item in results),
        "has_resolved": any(item["status"] == "RESOLVED" for item in results),
    }


_ID_LIKE_CELL = re.compile(r"^(?=.*\d)[a-z0-9][a-z0-9._\-]{2,39}$")  # QP-PATCH


def _entity_row_cells(
    connection,
    table_name: str,
    column: str,
    value: str,
    limit: int = 5,
) -> set[str]:
    """Lower-cased cells of the row(s) where `column` equals `value`."""
    cells: set[str] = set()
    try:
        table_name = SQLiteManager.validate_identifier(table_name, "table name")
        column = SQLiteManager.validate_identifier(column, "column name")
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM `{table_name}` "
                f"WHERE LOWER(CAST(`{column}` AS CHAR)) = LOWER(%s) "
                f"LIMIT {int(limit)}",
                (value,),
            )
            for row in cursor.fetchall():
                for cell in row:
                    text = str(cell).strip().lower() if cell is not None else ""
                    if text:
                        cells.add(text)
    except Exception:
        return set()
    return cells


def _narrow_ambiguous_by_resolved_context(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve an AMBIGUOUS entity reference using already-RESOLVED entities.

    Example: 'laptop' matches two products, but order NM-10002 (RESOLVED) links
    to exactly one of them through an identifier cell (e.g. P201). Narrowing
    only happens when EXACTLY ONE candidate is linked; otherwise the reference
    stays AMBIGUOUS and the user is asked, exactly as before.
    """
    resolved = [
        item["match"]
        for item in results
        if item.get("status") == "RESOLVED"
        and isinstance(item.get("match"), dict)
        and item["match"].get("type") == "entity"
        and item["match"].get("locations")
    ]
    ambiguous = [
        item
        for item in results
        if item.get("status") == "AMBIGUOUS"
        and any(m.get("type") == "entity" for m in item.get("matches", []))
    ]
    if not resolved or not ambiguous:
        return results

    connection = connections[STRUCTURED_READ_DB]
    try:
        context: set[str] = set()
        for entity in resolved:
            for loc in entity["locations"]:
                context |= _entity_row_cells(
                    connection, loc["table_name"], loc["column"], entity["value"]
                )
        if not context:
            return results
        context_norm = {normalize_text(c) for c in context}

        for item in ambiguous:
            linked: list[dict[str, Any]] = []
            for match in item["matches"]:
                if match.get("type") != "entity":
                    continue
                cells: set[str] = set()
                for loc in match.get("locations", []):
                    cells |= _entity_row_cells(
                        connection, loc["table_name"], loc["column"], match["value"]
                    )
                id_overlap = {
                    c for c in cells if _ID_LIKE_CELL.match(c) and c in context
                }
                name_hit = normalize_text(match.get("value", "")) in context_norm
                if id_overlap or name_hit:
                    linked.append(match)
            if len(linked) == 1:
                item["status"] = "RESOLVED"
                item["match"] = linked[0]
                item.pop("matches", None)
    finally:
        connection.close()

    return results


def _human_table_name(
    table_name: str,
) -> str:
    parts = table_name.split("_")

    if "table" in parts:
        try:
            return "table " + parts[parts.index("table") + 1]

        except IndexError:
            pass

    return table_name


def _fetch_column_values(
    connection,
    table_name: str,
    specs: list[tuple[str, str, tuple[Any, ...]]],
) -> dict[str, list[tuple[Any, ...]]]:
    """Fetch values for several columns of ONE table in one round trip.

    ``specs`` is a list of ``(column, where_clause, params)``; the column and
    table identifiers must already be validated by the caller. Each column is
    selected as its own UNION branch, so every column still sees exactly the
    rows its own WHERE selects (UNION de-duplicates, matching the previous
    per-column ``SELECT DISTINCT``). If the combined statement fails for any
    reason, each column is executed on its own as before, so one bad column
    only drops that column.

    Returns ``{column: [row, ...]}`` where each row is a 1-tuple (the value).
    """

    out: dict[str, list[tuple[Any, ...]]] = {column: [] for column, _, _ in specs}

    def run_single(column: str, where: str, params: tuple[Any, ...]) -> None:
        sql = f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE {where}"
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params) if params else cursor.execute(sql)
                out[column] = [(row[0],) for row in cursor.fetchall()]
        except Exception:
            out[column] = []

    if len(specs) <= 1:
        for column, where, params in specs:
            run_single(column, where, params)
        return out

    branches: list[str] = []
    all_params: list[Any] = []

    for index, (column, where, params) in enumerate(specs):
        branches.append(
            f"(SELECT {index} AS qp_idx, `{column}` AS qp_val "
            f"FROM `{table_name}` WHERE {where})"
        )
        all_params.extend(params)

    sql = " UNION ".join(branches)

    try:
        with connection.cursor() as cursor:
            if all_params:
                cursor.execute(sql, tuple(all_params))
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()

        for row in rows:
            out[specs[int(row[0])][0]].append((row[1],))

    except Exception:
        for column, where, params in specs:
            run_single(column, where, params)

    return out


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
        schema = _light_schema_from_catalog(
            get_workspace_catalog(
                workspace_id,
                include_row_counts=False,
            )
        )

    if not schema:
        return []

    query_tokens = set(normalized_query.split())

    if not query_tokens:
        return []

    connection = connections[STRUCTURED_READ_DB]

    grouped: dict[
        str,
        dict[str, Any],
    ] = {}

    try:
        for table in schema:
            table_name = str(table.get("table_name") or "")

            if not table_name:
                continue

            table_name = SQLiteManager.validate_identifier(
                table_name,
                "table name",
            )

            columns: list[str] = []

            for column in _entity_mention_columns(
                table.get(
                    "columns",
                    [],
                )
            ):
                column = SQLiteManager.validate_identifier(
                    column,
                    "column name",
                )

                if column not in columns:
                    columns.append(column)

            if not columns:
                continue

            fetched = _fetch_column_values(
                connection,
                table_name,
                [(column, f"`{column}` IS NOT NULL", ()) for column in columns],
            )

            for column in columns:
                for row in fetched.get(column, []):
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


def _entity_mention_columns(
    columns: list[str],
) -> list[str]:

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


def _is_strong_entity_column(
    column: str,
) -> bool:

    tokens = set(normalize_text(column).split())

    return bool(
        tokens
        & {
            "name",
            "employee",
            "person",
            "customer",
            "client",
            "user",
        }
    )


def _resolve_name_value(
    *,
    workspace_id: int,
    query: str,
    schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    normalized_query = normalize_text(query)

    if not normalized_query:
        return []

    connection = connections[STRUCTURED_READ_DB]

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
    connection,
    schema: list[dict[str, Any]],
    normalized_query: str,
    raw_query: str,
    mode: str,
) -> dict[str, dict[str, Any]]:

    query_tokens = set(normalized_query.split())

    grouped: dict[
        str,
        dict[str, Any],
    ] = {}

    for table in schema:
        table_name = str(table.get("table_name") or "")

        if not table_name:
            continue

        table_name = SQLiteManager.validate_identifier(
            table_name,
            "table name",
        )

        columns: list[str] = []

        for column in _name_like_columns(
            table.get(
                "columns",
                [],
            )
        ):
            column = SQLiteManager.validate_identifier(
                column,
                "column name",
            )

            if column not in columns:
                columns.append(column)

        if not columns:
            continue

        specs: list[tuple[str, str, tuple[Any, ...]]] = []

        for column in columns:
            if mode == "exact":
                where = (
                    f"`{column}` IS NOT NULL "
                    f"AND LOWER(CAST(`{column}` AS CHAR)) = LOWER(%s) "
                    f"LIMIT 20"
                )
                params: tuple[Any, ...] = (raw_query,)

            else:
                where = (
                    f"`{column}` IS NOT NULL "
                    f"AND LOWER(CAST(`{column}` AS CHAR)) LIKE %s "
                    f"LIMIT 50"
                )
                _escaped = (  # QP-PATCH
                    raw_query.lower()
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                params = (f"%{_escaped}%",)

            specs.append((column, where, params))

        fetched = _fetch_column_values(
            connection,
            table_name,
            specs,
        )

        for column in columns:
            for row in fetched.get(column, []):
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


def _name_like_columns(
    columns: list[str],
) -> list[str]:

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


def _dedupe_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:

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