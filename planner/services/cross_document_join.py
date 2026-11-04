"""
planner/services/cross_document_join.py

Format-independent, key-driven companion-evidence retrieval for
cross-document requests.

Flow (deterministic, no LLM, no hardcoded ids/names/files):

  1. Anchors    the entity the user asked about (planner references or
                deterministic entity resolution only).
  2. Entity     rows of that entity fetched by exact resolved value, so the
     rows       source row is present even when its sheet has > 100 rows.
  3. Keys       stable identifiers found next to the anchor, in ANY source:
                PDF chunk text or SQLite row (plus ids typed in the query).
  4. Lookup     every resolved document is searched with those keys:
                  - text chunks : exact key match (embedding-independent)
                  - SQLite tables: parameterised exact match, read-only,
                                   allow-listed per table
  5. Notes      a resolved document that contains none of the keys yields
                an explicit NOT_FOUND note, so the answer model reports
                missing evidence instead of substituting another record.

PDF<->PDF, XLSX<->XLSX, PDF<->XLSX and XLSX<->PDF differ only in which of
the two lookup channels (text / table) produces or consumes the key.
Workspace isolation is inherited from the underlying tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Iterable

from retrieval.services.keys import (
    dedupe_keys,
    drop_source_name_keys,
    extract_id_tokens,
    is_anchor_reference,
    is_id_column,
    keys_from_rows,
    keys_from_text,
    normalize_key,
    text_contains_key,
)

logger = logging.getLogger(__name__)

MAX_JOIN_KEYS = 25
MAX_ROWS_PER_KEY = 10
MAX_ANCHORS = 3
MAX_COLUMNS_PER_LOOKUP = 45

# CAST ... AS CHAR is valid on MySQL (CAST AS TEXT is not) and is also
# accepted by SQLite (CHAR has TEXT affinity), so one expression serves both.
_NORMALIZED_CELL = (
    "REPLACE(REPLACE(REPLACE(UPPER(TRIM(CAST({col} AS CHAR))),'-',''),'_',''),' ','')"
)


def _norm(text: Any) -> str:
    return " ".join(re.findall(r"\w+", str(text or "").lower()))


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def derive_anchors(
    *,
    query: str,
    references: Iterable[Any],
    resolved_references: Iterable[dict[str, Any]],
    entities: Iterable[dict[str, Any]],
) -> list[str]:
    source_refs: set[str] = set()
    for item in resolved_references or []:
        match = item.get("match") or {}
        if item.get("status") == "RESOLVED" and match.get("type") in {
            "document",
            "table",
        }:
            source_refs.add(_norm(item.get("reference")))

    strong = [str(e.get("value")) for e in entities or [] if e.get("value")]
    weak: list[str] = []
    for ref in references or []:
        text = str(ref or "").strip()
        if not is_anchor_reference(text) or _norm(text) in source_refs:
            continue
        if extract_id_tokens(text):
            continue
        (strong if re.search(r"[A-Z0-9]", text) else weak).append(text)

    anchors = list(dict.fromkeys(strong))
    return anchors[:MAX_ANCHORS]


def _rows_of(item: dict[str, Any]) -> list[dict[str, Any]]:
    content = item.get("content")
    rows = content.get("rows") if isinstance(content, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def collect_keys(
    *,
    query: str,
    references: Iterable[Any],
    anchors: list[str],
    table_items: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    limit: int = MAX_JOIN_KEYS,
    source_names: Iterable[str] = (),
) -> list[str]:
    """Identifier keys used to look for companion evidence in other sources.

    ``source_names`` are the titles/file names of the sources themselves. An
    identifier-shaped fragment of a source name (a version tag, a year in a
    file name) identifies the source, not a record, so it is never a key.
    """
    keys: list[str] = []
    for text in [query, *[str(r) for r in references or []]]:
        keys.extend(token for _, _, token in extract_id_tokens(str(text)))
    for item in table_items:
        keys.extend(keys_from_rows(_rows_of(item), anchors))
    for item in documents:
        keys.extend(keys_from_text(str(item.get("content", "")), anchors))

    return drop_source_name_keys(dedupe_keys(keys), source_names)[:limit]


_SQL_CELL_SEPARATORS = re.compile(r"[-_ ]")
MAX_QUERY_PARAMETERS = 50


def _cell_norm(value: Any) -> str:
    """Python twin of _NORMALIZED_CELL so in-memory and SQL matching agree."""
    if value is None:
        return ""
    return _SQL_CELL_SEPARATORS.sub("", str(value).strip().upper())


def _attribute_rows(
    rows: list[dict[str, Any]],
    cols_for_key: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Assign each row to every key whose normalised form equals a cell."""
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in cols_for_key}
    targets = {k: normalize_key(k) for k in cols_for_key}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, cols in cols_for_key.items():
            if any(_cell_norm(row.get(c)) == targets[key] for c in cols):
                out[key].append(row)
    return out


def _loaded_full_tables(
    items: list[dict[str, Any]],
) -> dict[str, tuple[list[str], list[dict[str, Any]], dict[str, Any]]]:
    """Tables already read completely: name -> (columns, rows, evidence item)."""
    loaded: dict[str, tuple[list[str], list[dict[str, Any]], dict[str, Any]]] = {}
    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue
        if content.get("evidence_role") != "document_table":
            continue
        if content.get("truncated") or not isinstance(content.get("rows"), list):
            continue
        names = {
            str(s.get("table_name"))
            for s in item.get("sources", []) or []
            if isinstance(s, dict) and s.get("table_name")
        }
        if len(names) != 1:
            continue
        loaded[next(iter(names))] = (
            [str(c) for c in content.get("columns", []) or []],
            _rows_of(item),
            item,
        )
    return loaded


def _lookup_table_rows(
    run_query: Callable[..., dict[str, Any]],
    workspace_id: int,
    table: dict[str, Any],
    cols_for_key: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """One batched, parameterised statement per (table, key chunk)."""
    by_key: dict[str, list[dict[str, Any]]] = {k: [] for k in cols_for_key}
    out_columns: list[str] = []
    groups: dict[tuple[str, ...], list[str]] = {}
    for key, cols in cols_for_key.items():
        groups.setdefault(tuple(cols), []).append(key)

    for cols, group_keys in groups.items():
        step = max(1, MAX_QUERY_PARAMETERS // len(cols))
        for i in range(0, len(group_keys), step):
            chunk = group_keys[i : i + step]
            where = " OR ".join(
                f"{_NORMALIZED_CELL.format(col=_quote(c))} = ?"
                for _ in chunk
                for c in cols
            )
            res = run_query(
                workspace_id=workspace_id,
                sql=(
                    f"SELECT * FROM {_quote(table['table_name'])} "
                    f"WHERE {where} "
                    f"LIMIT {(MAX_ROWS_PER_KEY + 1) * len(chunk)}"
                ),
                parameters=[normalize_key(k) for k in chunk for _ in cols],
                allowed_table_names={table["table_name"]},
            )
            out_columns = res.get("columns", out_columns)
            attributed = _attribute_rows(
                res.get("rows", []), {k: list(cols) for k in chunk}
            )
            for key, rows in attributed.items():
                by_key[key].extend(rows)
    return by_key, out_columns


def _row_sig(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _index_seen_rows(items: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Index rows that should suppress exact duplicate evidence.

    Full tables loaded as ``document_table`` are intentionally NOT indexed
    here. Cross-document key lookups must be allowed to emit a focused
    ``key_joined`` evidence item even when the same full table was already
    loaded. Otherwise the exact companion rows are deduplicated away before
    answer generation, leaving the model without an explicit join result for
    population queries. Other evidence roles keep the normal duplicate guard.
    """
    seen: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if (
            isinstance(content, dict)
            and content.get("evidence_role") == "document_table"
        ):
            continue
        rows = _rows_of(item)
        if not rows:
            continue
        names = {
            str(s.get("table_name"))
            for s in item.get("sources", []) or []
            if isinstance(s, dict) and s.get("table_name")
        }
        for name in names:
            seen.setdefault(name, set()).update(_row_sig(r) for r in rows)
    return seen


def _table_evidence(
    *,
    workspace_id: int,
    table: dict[str, Any],
    columns: list[str],
    rows: list[dict[str, Any]],
    role: str,
    matched: list[str],
    interpretation: str,
) -> dict[str, Any]:
    return {
        "source_type": "table_query",
        "workspace_id": workspace_id,
        "sources": [
            {
                "table_name": table["table_name"],
                "document_id": _as_int(table.get("document_id")),
            }
        ],
        "content": {
            "success": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "rows_returned": len(rows),
            "truncated": False,
            "tables_used": [table["table_name"]],
            "evidence_role": role,
            "matched_keys": matched,
            "document_title": table.get("document_title"),
            "interpretation": interpretation,
        },
        "result_status": "SUCCESS",
    }


def _light_schema_and_titles(
    workspace_id: int,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Table metadata for the join WITHOUT sampling or counting any table.

    The join only needs, per table: document id/title, table name, columns.
    ``get_workspace_catalog(include_row_counts=False)`` yields exactly that
    from the table-index tier (documents query + column metadata only) and
    applies the same workspace-prefix / document-ownership checks as the
    full schema build. Previously the join built the FULL schema (a sample
    SELECT and a COUNT per table across the whole workspace) first.
    """
    from planner.services.schema_registry import get_workspace_catalog

    catalog = get_workspace_catalog(workspace_id, include_row_counts=False)

    schema: list[dict[str, Any]] = []
    titles: dict[int, str] = {}
    for doc in catalog:
        did = _as_int(doc.get("document_id"))
        if did is None:
            continue
        title = str(doc.get("title") or f"document {did}")
        titles[did] = title
        for table in doc.get("tables") or []:
            name = table.get("table_name")
            if not name:
                continue
            schema.append(
                {
                    "document_id": did,
                    "document_title": title,
                    "source_type": doc.get("source_type"),
                    "table_name": str(name),
                    "columns": [str(c) for c in table.get("columns") or []],
                }
            )
    schema.sort(key=lambda t: t["table_name"])
    return schema, titles


def run_cross_document_join(
    *,
    workspace_id: int,
    query: str,
    references: Iterable[Any],
    resolved_references: Iterable[dict[str, Any]],
    document_ids: Iterable[int],
    entities: Iterable[dict[str, Any]],
    structured: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    get_schema: Callable[..., list[dict[str, Any]]] | None = None,
    get_catalog: Callable[..., list[dict[str, Any]]] | None = None,
    run_query: Callable[..., dict[str, Any]] | None = None,
    text_lookup: Callable[..., dict[str, Any]] | None = None,
    skip_text_document_ids: Iterable[int] | None = None,
    allowed_table_names: Iterable[str] | None = None,
    table_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Returns {"keys", "tables", "documents", "notes", "anchors"}.

    `tables` / `documents` contain ONLY evidence not already present in the
    `structured` / `documents` passed in. Never raises for retrieval
    problems: they are logged and reported through `notes`.
    """
    # When the caller injects neither loader (the production call site) use
    # the light metadata path. Injected loaders keep their original contract.
    use_light_metadata = get_schema is None and get_catalog is None

    if not use_light_metadata:
        if get_schema is None:
            from planner.services.schema_registry import (
                get_workspace_schema as get_schema,
            )
        if get_catalog is None:
            from planner.services.schema_registry import (
                get_workspace_catalog as get_catalog,
            )
    if run_query is None:
        from planner.tools.query_tables import query_tables as run_query
    if text_lookup is None:
        from planner.tools.search_evidence import search_evidence_by_keys as text_lookup

    doc_ids = sorted({d for d in (_as_int(x) for x in document_ids) if d is not None})
    entities = [e for e in entities or [] if isinstance(e, dict)]
    references = list(references or [])
    outcome: dict[str, Any] = {
        "keys": [],
        "tables": [],
        "documents": [],
        "notes": [],
        "anchors": [],
    }
    if len(doc_ids) < 2:
        return outcome

    anchors = derive_anchors(
        query=query,
        references=references,
        resolved_references=list(resolved_references or []),
        entities=entities,
    )
    outcome["anchors"] = anchors

    titles: dict[int, str] = {}
    schema: list[dict[str, Any]] = []
    metadata_loaded = False

    if use_light_metadata:
        try:
            schema, titles = _light_schema_and_titles(workspace_id)
            metadata_loaded = True
        except Exception:
            logger.exception("cross-document join: light metadata unavailable")
            schema, titles = [], {}

    if not metadata_loaded:
        if get_schema is None or get_catalog is None:
            # Light path failed: fall back to the original full loaders.
            from planner.services.schema_registry import (
                get_workspace_catalog as _default_catalog,
                get_workspace_schema as _default_schema,
            )

            get_schema = get_schema or _default_schema
            get_catalog = get_catalog or _default_catalog
        try:
            schema = list(get_schema(workspace_id, allowed_document_ids=None))
        except Exception:
            logger.exception("cross-document join: schema unavailable")
            schema = []

    entity_tables = {
        str(loc.get("table_name"))
        for e in entities
        for loc in e.get("locations", []) or []
        if loc.get("table_name")
    }
    relevant = [
        t
        for t in schema
        if _as_int(t.get("document_id")) in doc_ids
        or str(t.get("table_name")) in entity_tables
    ]
    if allowed_table_names is not None:
        allowed = {str(n) for n in allowed_table_names}
        relevant = [t for t in relevant if str(t.get("table_name")) in allowed]

    # A table selector can identify sources that contribute governing rules
    # rather than entity records. Do not search those rule tables for keys
    # extracted from data rows, and do not run a document-level key lookup
    # against a source whose in-scope tables are all explicitly rule tables.
    # Unlabelled tables retain the previous behaviour for backward compatibility.
    raw_roles = table_roles if isinstance(table_roles, dict) else {}
    normalized_roles = {
        str(name): str(role or "").strip().lower()
        for name, role in raw_roles.items()
    }
    tables_by_document: dict[int, list[str]] = {}
    for table in relevant:
        did = _as_int(table.get("document_id"))
        table_name = str(table.get("table_name") or "")
        if did in doc_ids and table_name:
            tables_by_document.setdefault(did, []).append(table_name)

    rule_only_document_ids = {
        did
        for did, table_names in tables_by_document.items()
        if table_names
        and all(normalized_roles.get(name) == "rule" for name in table_names)
    }
    rule_table_names = {
        name for name, role in normalized_roles.items() if role == "rule"
    }
    doc_tables = [
        t
        for t in relevant
        if _as_int(t.get("document_id")) in doc_ids
        and str(t.get("table_name") or "") not in rule_table_names
    ]
    lookup_document_ids = [
        did for did in doc_ids if did not in rule_only_document_ids
    ]
    if rule_only_document_ids:
        logger.info(
            "cross-document join: skipping identifier lookup for rule-only "
            "source document_ids=%s",
            sorted(rule_only_document_ids),
        )

    if not metadata_loaded:
        try:
            for d in get_catalog(workspace_id, schema=schema):
                did = _as_int(d.get("document_id"))
                if did is not None:
                    titles[did] = str(d.get("title") or f"document {did}")
        except Exception:
            logger.exception("cross-document join: catalog unavailable")

    seen_rows = _index_seen_rows(structured)
    new_tables: list[dict[str, Any]] = []

    def add_rows(table, columns, rows, role, matched, interpretation):
        seen = seen_rows.setdefault(str(table["table_name"]), set())
        fresh = [r for r in rows if _row_sig(r) not in seen]
        if not fresh:
            return
        seen.update(_row_sig(r) for r in fresh)
        new_tables.append(
            _table_evidence(
                workspace_id=workspace_id,
                table=table,
                columns=columns,
                rows=fresh,
                role=role,
                matched=matched,
                interpretation=interpretation,
            )
        )

    by_name = {str(t.get("table_name")): t for t in relevant}
    already_loaded = _loaded_full_tables(structured)
    for entity in entities:
        value = entity.get("value")
        for loc in entity.get("locations", []) or []:
            table = by_name.get(str(loc.get("table_name")))
            column = loc.get("column")
            if table is None or not column or value is None:
                continue
            if str(table["table_name"]) in already_loaded:
                continue
            try:
                res = run_query(
                    workspace_id=workspace_id,
                    sql=(
                        f"SELECT * FROM {_quote(table['table_name'])} "
                        f"WHERE {_quote(column)} = ? LIMIT {MAX_ROWS_PER_KEY}"
                    ),
                    parameters=[value],
                    allowed_table_names={table["table_name"]},
                )
            except Exception:
                logger.exception("cross-document join: entity row lookup failed")
                continue
            add_rows(
                table,
                res.get("columns", []),
                res.get("rows", []),
                "entity_row",
                [],
                "Row(s) of the resolved entity, fetched by exact resolved "
                "value so they are present even if the sheet is large.",
            )

    # Only data-bearing evidence should contribute companion lookup keys.
    # Otherwise an identifier appearing in a policy example (or in a rule
    # table) can be mistaken for an entity key from the requested records.
    key_source_tables = [
        item
        for item in [*structured, *new_tables]
        if not (
            {
                str(source.get("table_name"))
                for source in item.get("sources", []) or []
                if isinstance(source, dict) and source.get("table_name")
            }
            & rule_table_names
        )
    ]
    key_source_documents = [
        item
        for item in documents
        if _as_int(item.get("document_id")) not in rule_only_document_ids
    ]
    source_names = [
        *titles.values(),
        *(
            str(item.get("reference") or "")
            for item in resolved_references or []
            if isinstance(item, dict)
            and (item.get("match") or {}).get("type") in {"document", "table"}
        ),
    ]
    keys = collect_keys(
        query=query,
        references=references,
        anchors=anchors,
        table_items=key_source_tables,
        documents=key_source_documents,
        limit=MAX_JOIN_KEYS + 1,
        source_names=source_names,
    )
    if len(keys) > MAX_JOIN_KEYS:
        keys = keys[:MAX_JOIN_KEYS]
        outcome["notes"].append(
            {
                "source_type": "evidence_note",
                "result_status": "PARTIAL",
                "content": (
                    f"More than {MAX_JOIN_KEYS} candidate identifiers were "
                    f"found; only the first {MAX_JOIN_KEYS} were used for "
                    "companion lookups, so the matched population may be "
                    "incomplete. Say so if the answer depends on the full "
                    "population."
                ),
            }
        )
    outcome["keys"] = keys
    logger.info(
        "cross-document join: anchors=%s keys=%s document_ids=%s",
        anchors,
        keys,
        doc_ids,
    )
    if not keys:
        outcome["tables"] = new_tables
        return outcome

    matched: dict[str, set[int]] = {k: set() for k in keys}
    lookup_failed: set[int] = set()
    non_identifying: set[str] = set()

    seen_chunks = set()
    for d in documents:
        meta = d.get("metadata") or {}
        did = _as_int(d.get("document_id"))
        seen_chunks.add((did, meta.get("chunk_index")))
        for index in meta.get("chunk_indexes") or []:
            seen_chunks.add((did, index))
    for d in documents:
        did = _as_int(d.get("document_id"))
        for k in keys:
            if did is not None and text_contains_key(str(d.get("content", "")), k):
                matched[k].add(did)

    no_text = {
        d for d in (_as_int(x) for x in skip_text_document_ids or []) if d is not None
    }
    new_docs: list[dict[str, Any]] = []
    for did in lookup_document_ids:
        if did in no_text:
            continue
        try:
            res = text_lookup(
                keys=keys,
                workspace_id=workspace_id,
                allowed_document_ids=[did],
            )
        except Exception:
            logger.exception("cross-document join: text lookup failed doc=%s", did)
            lookup_failed.add(did)
            continue
        if res.get("error_code"):
            logger.warning(
                "cross-document join: text lookup error doc=%s code=%s",
                did,
                res.get("error_code"),
            )
            lookup_failed.add(did)
            continue
        for k, dids in (res.get("matched_keys") or {}).items():
            matched.setdefault(k, set()).update(int(x) for x in dids)
        non_identifying.update(res.get("non_identifying") or [])
        for ev in res.get("evidence", []):
            ident = (
                _as_int(ev.get("document_id")),
                (ev.get("metadata") or {}).get("chunk_index"),
            )
            if ident in seen_chunks:
                continue
            seen_chunks.add(ident)
            ev = dict(ev)
            ev["result_status"] = "SUCCESS"
            ev["evidence_role"] = "key_joined"
            new_docs.append(ev)

    # `structured` is not mutated by the entity pass above (only new_tables
    # grows), so the tables computed earlier are still current.
    loaded_tables = already_loaded
    for table in doc_tables:
        tname = str(table["table_name"])
        columns = [str(c) for c in table.get("columns", [])][:MAX_COLUMNS_PER_LOOKUP]
        did = _as_int(table.get("document_id"))

        cols_for_key: dict[str, list[str]] = {}
        for key in keys:
            cols = (
                [c for c in columns if is_id_column(c)]
                if normalize_key(key).isdigit()
                else columns
            )
            if cols:
                cols_for_key[key] = cols
        if not cols_for_key:
            continue

        used_cols: list[str] = columns
        loaded = loaded_tables.get(tname)
        if loaded is not None:
            used_cols, loaded_rows, loaded_item = loaded
            by_key = _attribute_rows(loaded_rows, cols_for_key)
            tagged = [k for k in keys if 0 < len(by_key.get(k, [])) <= MAX_ROWS_PER_KEY]
            non_identifying.update(
                k for k in keys if len(by_key.get(k, [])) > MAX_ROWS_PER_KEY
            )
            if tagged:
                content = loaded_item.setdefault("content", {})
                content["matched_keys"] = list(
                    dict.fromkeys(list(content.get("matched_keys") or []) + tagged)
                )
                if did is not None:
                    for k in tagged:
                        matched[k].add(did)
            continue
        else:
            try:
                by_key, used_cols = _lookup_table_rows(
                    run_query, workspace_id, table, cols_for_key
                )
            except Exception:
                logger.exception(
                    "cross-document join: keyed table lookup failed table=%s",
                    tname,
                )
                if did is not None:
                    lookup_failed.add(did)
                continue

        gathered: list[dict[str, Any]] = []
        gathered_sigs: set[str] = set()
        hit_keys: list[str] = []
        for key in keys:
            rows = by_key.get(key, [])
            if len(rows) > MAX_ROWS_PER_KEY:
                non_identifying.add(key)
                continue
            if rows:
                hit_keys.append(key)
                if did is not None:
                    matched[key].add(did)
                for row in rows:
                    sig = _row_sig(row)
                    if sig not in gathered_sigs:
                        gathered_sigs.add(sig)
                        gathered.append(row)
        if gathered:
            add_rows(
                table,
                used_cols,
                gathered,
                "key_joined",
                hit_keys,
                "Row(s) whose cells equal stable identifier(s) resolved "
                "from the request or another source. For population queries "
                "this may contain multiple matched records. Join key(s): "
                + ", ".join(hit_keys),
            )

    usable = [k for k in keys if k not in non_identifying]
    if usable:
        for did in lookup_document_ids:
            title = titles.get(did, f"document {did}")
            if did in lookup_failed:
                outcome["notes"].append(
                    {
                        "source_type": "evidence_note",
                        "note_kind": "identifier_lookup",
                        "result_status": "RETRIEVAL_ERROR",
                        "content": (
                            f"Searching '{title}' for identifier(s) "
                            f"{', '.join(usable)} failed, so companion "
                            "evidence from it may be missing. Do not "
                            "guess values that would come from it."
                        ),
                    }
                )
            elif not any(did in matched.get(k, set()) for k in usable):
                outcome["notes"].append(
                    {
                        "source_type": "evidence_note",
                        "note_kind": "identifier_lookup",
                        "result_status": "NOT_FOUND",
                        "content": (
                            f"Identifier(s) {', '.join(usable)} were "
                            f"resolved for the requested entity, but no "
                            f"passage or row in '{title}' contains them. "
                            "If the question needs an entity-specific "
                            "record from that source, state that it was "
                            "not found; never substitute another "
                            "entity's record."
                        ),
                    }
                )

    outcome["tables"] = new_tables
    outcome["documents"] = new_docs
    return outcome


# ---------------------------------------------------------------------------
# Deterministic key join across complete tables (any mix of documents).
#
# Questions such as "which orders were late, and what was refunded for each?"
# need rows from several tables related through a shared identifier. Aligning
# them by eye in the answer model is slow and error prone, and an LLM-written
# JOIN across documents is unsafe. This joins rows that were already read
# through the authorized table tools, in plain Python: no SQL, no LLM. Keys are
# discovered from the data (same column name + identifier-like name + real
# value overlap), only complete tables take part, the join is a LEFT join from a
# hub table so no hub row is lost, and a source table is replaced only when
# every one of its rows is present in the joined view.
# ---------------------------------------------------------------------------

_KJ_MAX_ROWS = 400
_KJ_MAX_COLUMNS = 40
_KJ_MIN_OVERLAP = 0.5

# Last word of a column name that marks it as an identifier-like column.
_KJ_KEY_TOKENS = {
    "id", "ids", "code", "key", "ref", "reference", "no", "number", "num",
    "sku", "uid", "uuid",
}


def _kj_col_tokens(name: str) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or ""))
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def _kj_norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _kj_looks_like_key(name: str) -> bool:
    tokens = _kj_col_tokens(name)
    return bool(tokens) and tokens[-1] in _KJ_KEY_TOKENS


def _kj_norm_val(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().upper()
    if not text or text in {"NULL", "NONE", "NAN"}:
        return None
    return re.sub(r"[\s_\-]", "", text)


def _kj_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    content = item.get("content")
    rows = content.get("rows") if isinstance(content, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _kj_columns(item: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    content = item.get("content") or {}
    cols = content.get("columns")
    if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
        return list(cols)
    return list(rows[0].keys()) if rows else []


def _kj_table_name(item: dict[str, Any]) -> str:
    for source in item.get("sources") or []:
        if isinstance(source, dict) and source.get("table_name"):
            return str(source["table_name"])
    return ""


def _kj_eligible(item: Any) -> bool:
    """A complete, successful, raw table read (never an aggregate/filtered result)."""
    if not isinstance(item, dict):
        return False
    if str(item.get("source_type") or "").lower() != "table_query":
        return False
    if str(item.get("result_status") or "").upper() != "SUCCESS":
        return False
    content = item.get("content")
    if not isinstance(content, dict) or content.get("truncated"):
        return False
    if content.get("evidence_role") != "document_table":
        return False
    return bool(_kj_rows(item)) and bool(_kj_table_name(item))


def _kj_distinct(rows: list[dict[str, Any]], column: str) -> tuple[set[str], int, int]:
    values: set[str] = set()
    non_null = 0
    for row in rows:
        value = _kj_norm_val(row.get(column))
        if value is None:
            continue
        non_null += 1
        values.add(value)
    return values, non_null, len(values)


def _kj_find_edges(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join candidates between every pair of tables."""
    edges: list[dict[str, Any]] = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            a, b = tables[i], tables[j]
            for col_a in a["columns"]:
                if not _kj_looks_like_key(col_a):
                    continue
                for col_b in b["columns"]:
                    if _kj_norm_col(col_a) != _kj_norm_col(col_b):
                        continue
                    vals_a, nn_a, d_a = _kj_distinct(a["rows"], col_a)
                    vals_b, nn_b, d_b = _kj_distinct(b["rows"], col_b)
                    if not vals_a or not vals_b:
                        continue
                    overlap = len(vals_a & vals_b)
                    if overlap == 0:
                        continue
                    if overlap / min(len(vals_a), len(vals_b)) < _KJ_MIN_OVERLAP:
                        continue
                    # At least one side must be a real key (one row per value),
                    # otherwise this is two fact tables sharing a label column.
                    unique_a = nn_a > 0 and d_a / nn_a >= 0.9
                    unique_b = nn_b > 0 and d_b / nn_b >= 0.9
                    if not (unique_a or unique_b):
                        continue
                    edges.append(
                        {
                            "a": i, "col_a": col_a,
                            "b": j, "col_b": col_b,
                            "overlap": overlap,
                            "distinct_a": d_a, "distinct_b": d_b,
                        }
                    )
    return edges


def _kj_components(count: int, edges: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        parent[find(edge["a"])] = find(edge["b"])
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def _kj_label(table: dict[str, Any]) -> str:
    title = str(table.get("title") or "").strip()
    title = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", title)
    name = table["name"]
    suffix = re.search(r"(table_\d+)$", name)
    base = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or name
    return f"{base}.{suffix.group(1)}" if suffix else base


def _kj_join_component(
    tables: list[dict[str, Any]],
    members: list[int],
    edges: list[dict[str, Any]],
) -> dict[str, Any] | None:
    member_set = set(members)
    local_edges = [e for e in edges if e["a"] in member_set and e["b"] in member_set]

    # Hub: the table that owns the most distinct key values (ties: more rows).
    def hub_score(index: int) -> tuple[int, int]:
        best = 0
        for e in local_edges:
            if e["a"] == index:
                best = max(best, e["distinct_a"])
            elif e["b"] == index:
                best = max(best, e["distinct_b"])
        return best, len(tables[index]["rows"])

    hub = max(members, key=hub_score)

    used_names: set[str] = set()
    out_columns: list[str] = []
    origin: dict[str, str] = {}

    def claim(table_index: int, column: str) -> str:
        if column not in used_names:
            used_names.add(column)
            return column
        candidate = f"{_kj_label(tables[table_index])}.{column}"
        n = 2
        unique = candidate
        while unique in used_names:
            unique = f"{candidate}_{n}"
            n += 1
        used_names.add(unique)
        return unique

    # Output name of every (table index, source column) already in the view.
    name_of: dict[tuple[int, str], str] = {}
    joined: list[dict[str, Any]] = []
    hub_rows = tables[hub]["rows"]
    for column in tables[hub]["columns"]:
        out = claim(hub, column)
        name_of[(hub, column)] = out
        out_columns.append(out)
        origin[out] = tables[hub]["name"]
    for row in hub_rows:
        joined.append({name_of[(hub, c)]: row.get(c) for c in tables[hub]["columns"]})

    in_view = {hub}
    absorbed: dict[int, bool] = {hub: True}  # table index -> every row represented
    keys_used: list[str] = []
    pending = set(members) - in_view

    while pending:
        # Next table connected to something already in the view (best overlap first).
        candidates = []
        for e in local_edges:
            if e["a"] in in_view and e["b"] in pending:
                candidates.append((e["overlap"], e["b"], e["a"], e["col_a"], e["col_b"]))
            elif e["b"] in in_view and e["a"] in pending:
                candidates.append((e["overlap"], e["a"], e["b"], e["col_b"], e["col_a"]))
        if not candidates:
            break
        candidates.sort(key=lambda c: (-c[0], c[1]))
        _, new_index, have_index, have_col, new_col = candidates[0]
        new_table = tables[new_index]
        view_key = name_of[(have_index, have_col)]

        index_by_key: dict[str, list[dict[str, Any]]] = {}
        for row in new_table["rows"]:
            key = _kj_norm_val(row.get(new_col))
            if key is not None:
                index_by_key.setdefault(key, []).append(row)

        # A column that already exists in the view under the same name and holds
        # the same value on every matched row (for example a payment method or
        # transaction id repeated in a second table) adds nothing: keep one copy.
        duplicate_columns: set[str] = set()
        for column in new_table["columns"]:
            if column == new_col or column not in used_names:
                continue
            compared = 0
            same = True
            for row in joined:
                key = _kj_norm_val(row.get(view_key))
                for match in index_by_key.get(key, []) if key is not None else []:
                    left, right = _kj_norm_val(row.get(column)), _kj_norm_val(match.get(column))
                    if left is None or right is None:
                        continue
                    compared += 1
                    if left != right:
                        same = False
                        break
                if not same:
                    break
            if same and compared > 0:
                duplicate_columns.add(column)

        new_columns = [
            c for c in new_table["columns"]
            if c != new_col and c not in duplicate_columns
        ]
        added: dict[str, str] = {}
        for column in new_columns:
            out = claim(new_index, column)
            added[column] = out
            name_of[(new_index, column)] = out
            out_columns.append(out)
            origin[out] = new_table["name"]
        for column in duplicate_columns:
            name_of[(new_index, column)] = column
        # The shared key already lives in the view under the existing name.
        name_of[(new_index, new_col)] = view_key

        matched_keys: set[str] = set()
        next_rows: list[dict[str, Any]] = []
        for row in joined:
            key = _kj_norm_val(row.get(view_key))
            matches = index_by_key.get(key, []) if key is not None else []
            if not matches:
                merged = dict(row)
                for out in added.values():
                    merged[out] = None
                next_rows.append(merged)
                continue
            matched_keys.add(key)
            for match in matches:
                merged = dict(row)
                for column, out in added.items():
                    merged[out] = match.get(column)
                next_rows.append(merged)
            if len(next_rows) > _KJ_MAX_ROWS:
                return None
        if len(next_rows) > _KJ_MAX_ROWS or len(out_columns) > _KJ_MAX_COLUMNS:
            return None

        orphan = any(
            (_kj_norm_val(r.get(new_col)) is None)
            or (_kj_norm_val(r.get(new_col)) not in matched_keys)
            for r in new_table["rows"]
        )
        absorbed[new_index] = not orphan
        joined = next_rows
        in_view.add(new_index)
        pending.discard(new_index)
        if view_key not in keys_used:
            keys_used.append(view_key)

    if len(in_view) < 2:
        return None
    return {
        "hub": hub,
        "members": sorted(in_view),
        "rows": joined,
        "columns": out_columns,
        "origin": origin,
        "keys": keys_used,
        "absorbed": {i for i, ok in absorbed.items() if ok},
    }


def build_joined_view(
    items: list[Any],
) -> tuple[dict[str, Any], set[int]] | None:
    """Join the complete raw tables found in ``items`` on discovered keys.

    Returns ``(evidence_item, absorbed_positions)`` or ``None`` when fewer than
    two tables relate to each other. ``absorbed_positions`` are the positions in
    ``items`` whose rows are fully contained in the evidence item.
    """
    positions = [i for i, item in enumerate(items) if _kj_eligible(item)]
    if len(positions) < 2:
        return None

    tables: list[dict[str, Any]] = []
    for position in positions:
        item = items[position]
        rows = _kj_rows(item)
        content = item.get("content") or {}
        source = next(
            (s for s in item.get("sources") or [] if isinstance(s, dict)), {}
        )
        tables.append(
            {
                "position": position,
                "name": _kj_table_name(item),
                "document_id": source.get("document_id"),
                "title": content.get("document_title") or "",
                "rows": rows,
                "columns": _kj_columns(item, rows),
            }
        )

    edges = _kj_find_edges(tables)
    if not edges:
        return None
    groups = _kj_components(len(tables), edges)
    if not groups:
        return None

    # One joined view per request: the largest connected group.
    groups.sort(key=lambda g: (-len(g), g))
    result = _kj_join_component(tables, groups[0], edges)
    if result is None:
        return None

    members = result["members"]
    sources = [
        {
            "table_name": tables[i]["name"],
            "document_id": tables[i]["document_id"],
        }
        for i in members
    ]
    titles = list(
        dict.fromkeys(str(tables[i]["title"]) for i in members if tables[i]["title"])
    )
    described = "; ".join(
        f"{tables[i]['title'] or tables[i]['name']} ({tables[i]['name']})"
        for i in members
    )
    hub_title = tables[result["hub"]]["title"] or tables[result["hub"]]["name"]
    item = {
        "source_type": "table_query",
        "workspace_id": next(
            (
                items[tables[i]["position"]].get("workspace_id")
                for i in members
                if items[tables[i]["position"]].get("workspace_id") is not None
            ),
            None,
        ),
        "sources": sources,
        "content": {
            "success": True,
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": len(result["rows"]),
            "rows_returned": len(result["rows"]),
            "truncated": False,
            "tables_used": [s["table_name"] for s in sources],
            "evidence_role": "key_joined",
            "join_keys": result["keys"],
            "document_title": " + ".join(titles) if titles else None,
            "column_sources": result["origin"],
            "interpretation": (
                "Complete rows of several source tables joined deterministically "
                "on the shared identifier column(s) "
                + ", ".join(result["keys"])
                + f", starting from {hub_title}. One output row per matching "
                "combination; an empty cell means the other table has no row for "
                "that key. Tables joined: "
                + described
                + ". Use this view for cross-table questions (filter, compare, "
                "list); values are exact source values and nothing is estimated."
            ),
        },
        "result_status": "SUCCESS",
    }
    absorbed = {tables[i]["position"] for i in members if i in result["absorbed"]}
    return item, absorbed
