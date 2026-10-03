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

_NORMALIZED_CELL = (
    "REPLACE(REPLACE(REPLACE(UPPER(TRIM(CAST({col} AS TEXT))),'-',''),'_',''),' ','')"
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


def _anchors_from_query(query: str) -> list[str]:
    anchors: list[str] = []
    for m in re.finditer(
        r"[A-Z][a-z]+(?:['’]s)?(?:\s+[A-Z][a-z]+(?:['’]s)?)*", query or ""
    ):
        text = m.group(0)
        if m.start() == 0:
            text = " ".join(text.split()[1:])
        text = re.sub(r"['’]s\b", "", text).strip()
        if text:
            anchors.append(text)
    return anchors


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
) -> list[str]:
    keys: list[str] = []
    for text in [query, *[str(r) for r in references or []]]:
        keys.extend(token for _, _, token in extract_id_tokens(str(text)))
    for item in table_items:
        keys.extend(keys_from_rows(_rows_of(item), anchors))
    for item in documents:
        keys.extend(keys_from_text(str(item.get("content", "")), anchors))

    return dedupe_keys(keys)[:limit]


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
) -> dict[str, Any]:
    """
    Returns {"keys", "tables", "documents", "notes", "anchors"}.

    `tables` / `documents` contain ONLY evidence not already present in the
    `structured` / `documents` passed in. Never raises for retrieval
    problems: they are logged and reported through `notes`.
    """
    if get_schema is None:
        from planner.services.schema_registry import get_workspace_schema as get_schema
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
    doc_tables = [t for t in relevant if _as_int(t.get("document_id")) in doc_ids]

    titles: dict[int, str] = {}
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

    keys = collect_keys(
        query=query,
        references=references,
        anchors=anchors,
        table_items=[*structured, *new_tables],
        documents=documents,
        limit=MAX_JOIN_KEYS + 1,
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
    for did in doc_ids:
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

    loaded_tables = _loaded_full_tables(structured)
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
        for did in doc_ids:
            title = titles.get(did, f"document {did}")
            if did in lookup_failed:
                outcome["notes"].append(
                    {
                        "source_type": "evidence_note",
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
