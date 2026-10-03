"""Services Answer module for QueryPilot."""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from llm.router import router
from planner.prompts.answer_prompts import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT
from planner.services.history import bound_history_for_answer
from retrieval.services.keys import is_id_column

logger = logging.getLogger(__name__)


PROVIDER_TPM_LIMIT = 8000
TOTAL_REQUEST_TOKEN_CEILING = 5200
ANSWER_MAX_OUTPUT_TOKENS = 1500
ANSWER_STREAM_MAX_OUTPUT_TOKENS = 1500
MIN_OUTPUT_TOKENS = 1000


RETRY_MAX_OUTPUT_TOKENS = 2400
RETRY_TOTAL_CEILING = 5800
REQUEST_OVERHEAD_TOKENS = 200
CHARS_PER_TOKEN = 3.0
MAX_TEXT_CHARS_PER_ITEM = 1800
MAX_NOTE_CHARS = 400
MAX_CELL_CHARS = 120
MIN_ROWS_WHEN_TRIMMED = 3
MIN_ROWS_FOR_COVERAGE_REDUCTION = 6


HISTORY_MAX_MESSAGES = 2
HISTORY_MAX_CHARS_PER_MESSAGE = 240
HISTORY_MAX_QUERY_WORDS = 15


MAX_TOTAL_EVIDENCE_CHARS = 8000
MAX_EVIDENCE_ITEMS = 24
MAX_CHARS_PER_EVIDENCE = MAX_TEXT_CHARS_PER_ITEM
MAX_TABLE_ROWS_IN_PAYLOAD = 30
DEFAULT_CURRENCY_SYMBOL = None


ANSWER_CONTRACT_RULES = ""
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "querypilot_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["answer", "citations"],
            "additionalProperties": False,
        },
    },
}


def _estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def _system_tokens() -> int:
    return _estimate_tokens(SYSTEM_PROMPT)


def _payload_char_budget() -> int:
    """Characters the whole user payload may take so that system prompt +
    payload + full completion stay under the request ceiling."""
    allowed = (
        TOTAL_REQUEST_TOKEN_CEILING
        - ANSWER_MAX_OUTPUT_TOKENS
        - REQUEST_OVERHEAD_TOKENS
        - _system_tokens()
    )
    return max(int(allowed * CHARS_PER_TOKEN), 4000)


def _completion_budget(payload: str) -> tuple[int, int]:
    """(max_tokens, estimated_input_tokens) from the ACTUAL serialized input."""
    input_tokens = (
        _system_tokens() + _estimate_tokens(payload) + REQUEST_OVERHEAD_TOKENS
    )
    room = TOTAL_REQUEST_TOKEN_CEILING - input_tokens
    max_tokens = min(ANSWER_MAX_OUTPUT_TOKENS, max(room, MIN_OUTPUT_TOKENS))
    total = input_tokens + max_tokens
    if total > TOTAL_REQUEST_TOKEN_CEILING:
        logger.warning(
            "answer request above target ceiling: input~%d completion=%d total~%d "
            "(provider limit %d). Sending as-is; the provider layer decides.",
            input_tokens,
            max_tokens,
            total,
            PROVIDER_TPM_LIMIT,
        )
    else:
        logger.info(
            "answer_budget: input~%d completion=%d total~%d (limit %d)",
            input_tokens,
            max_tokens,
            total,
            PROVIDER_TPM_LIMIT,
        )
    return max_tokens, input_tokens


_INLINE_CITATION = re.compile(r"\s?\[\d{1,3}\]")


def _bound_history(
    history: list[dict[str, Any]], query: str = ""
) -> list[dict[str, str]]:
    return bound_history_for_answer(history, current_query=query)


def _answer_history(
    history: list[dict[str, Any]] | None,
    query: str,
) -> list[dict[str, str]]:
    """
    Keep only what is needed to interpret the current query.

    The planner has already resolved references and the answer facts come
    from validated current evidence, so earlier assistant answers are not
    resent. A self-contained (longer) query gets no history at all; a short,
    possibly elliptical follow-up gets the last couple of turns, shortened.
    """
    try:
        from planner.services.history import is_correction_query

        if is_correction_query(query):
            return []
    except Exception:
        logger.exception("correction-query detection failed; continuing safely")

    if not history or len(str(query or "").split()) > HISTORY_MAX_QUERY_WORDS:
        return []

    try:
        bounded = _bound_history(history, query)
    except Exception:
        logger.exception("history bounding failed; sending no history")
        return []

    trimmed: list[dict[str, str]] = []
    for message in bounded[-HISTORY_MAX_MESSAGES:]:
        if not isinstance(message, dict):
            continue
        text = _INLINE_CITATION.sub("", str(message.get("content") or "")).strip()
        if not text:
            continue
        trimmed.append(
            {
                "role": str(message.get("role") or "user"),
                "content": text[:HISTORY_MAX_CHARS_PER_MESSAGE],
            }
        )
    return trimmed


_NUMERIC_ONLY = re.compile(r"^[\s\d.,%₹$€£+\-]+$")
_NOTE_TYPES = {"evidence_note", "source_resolution", "retrieval_assessment"}


def _size(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        )
    except (TypeError, ValueError):
        return len(str(value))


def _cell(value: Any) -> Any:
    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and abs(value) < 1e15
    ):
        return int(value)
    return value[:MAX_CELL_CHARS] if isinstance(value, str) else value


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _slim(value: Any) -> Any:
    """Drop empty values and clip long strings, recursively."""
    if isinstance(value, dict):
        out = {k: _slim(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if not _empty(v)}
    if isinstance(value, list):
        return [_slim(v) for v in value]
    if isinstance(value, str):
        return value[: MAX_CELL_CHARS * 2]
    return value


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _collect_strings(value: Any, out: set[str], depth: int = 0) -> None:
    """Non-numeric identifying strings inside a value (cells, aggregate parts)."""
    if depth > 6:
        return
    if isinstance(value, str):
        text = _norm(value)
        if 2 <= len(text) <= 80 and not _NUMERIC_ONLY.match(text):
            out.add(text)
    elif isinstance(value, dict):
        for inner in value.values():
            _collect_strings(inner, out, depth + 1)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _collect_strings(inner, out, depth + 1)


def _short_table_ref(name: Any) -> str | None:
    parts = str(name or "").split("_")
    if "table" in parts:
        try:
            return "table " + parts[parts.index("table") + 1]
        except IndexError:
            return None
    return None


def _compact_table(item: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
    columns = [str(c) for c in content.get("columns") or []]
    rows = content.get("rows") if isinstance(content.get("rows"), list) else []
    if not columns and rows and isinstance(rows[0], dict):
        columns = [str(c) for c in rows[0]]
    packed: list[Any] = []
    for row in rows:
        if isinstance(row, dict):
            packed.append([_cell(row.get(c)) for c in columns])
        elif isinstance(row, (list, tuple)):
            packed.append([_cell(v) for v in row])
        else:
            packed.append(row)

    if (
        columns
        and packed
        and all(isinstance(r, list) and len(r) == len(columns) for r in packed)
    ):
        keep = [j for j in range(len(columns)) if any(not _empty(r[j]) for r in packed)]
        if len(keep) != len(columns):
            columns = [columns[j] for j in keep]
            packed = [[r[j] for j in keep] for r in packed]
    document = content.get("document_title")
    if not document:
        for source in item.get("sources") or []:
            if isinstance(source, dict) and source.get("table_name"):
                document = (
                    _short_table_ref(source["table_name"]) or source["table_name"]
                )
                break
    out: dict[str, Any] = {
        "document": document,
        "scope": content.get("population_scope"),
        "columns": columns,
        "rows": packed,
    }
    source_rows = content.get("source_rows")
    if source_rows is None:
        counts = content.get("source_table_row_counts")
        if isinstance(counts, dict) and len(counts) == 1:
            source_rows = next(iter(counts.values()))
    if source_rows is not None and source_rows != len(packed):
        out["source_rows"] = source_rows
    if content.get("truncated"):
        out["truncated"] = True
    semantics = content.get("query_semantics")
    if isinstance(semantics, dict) and semantics.get("type") not in (
        None,
        "table_result",
    ):
        out["query_type"] = semantics.get("type")
        if semantics.get("selection") is not None:
            out["selection"] = semantics.get("selection")
        if semantics.get("direction"):
            out["direction"] = semantics.get("direction")
        if semantics.get("limit") is not None:
            out["limit"] = semantics.get("limit")
    if content.get("message"):
        out["message"] = str(content["message"])[:300]
    return {k: v for k, v in out.items() if not _empty(v) or k in ("rows", "columns")}


def _compact_metadata(content: dict[str, Any]) -> dict[str, Any]:
    documents = []
    for doc in content.get("documents", []) or []:
        if not isinstance(doc, dict):
            continue
        entry = {
            "document_name": doc.get("document_name") or doc.get("title"),
            "file_type": doc.get("file_type") or doc.get("source_type"),
            "document_id": doc.get("document_id"),
        }
        if doc.get("tables"):
            entry["tables"] = [
                {
                    "table": _short_table_ref(t.get("table_name"))
                    or t.get("table_name"),
                    "columns": t.get("columns"),
                    "rows": t.get("row_count"),
                }
                for t in doc["tables"]
                if isinstance(t, dict)
            ]
        elif doc.get("table_count"):
            entry["table_count"] = doc.get("table_count")
        documents.append(entry)
    out: dict[str, Any] = {
        "document_count": content.get("document_count", len(documents)),
        "documents": documents,
    }
    if content.get("available_documents"):
        out["available_documents"] = content["available_documents"]
    return out


def _compact_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"content": str(item)[:MAX_TEXT_CHARS_PER_ITEM]}
    compact: dict[str, Any] = {}
    source_type = item.get("source_type")
    if source_type:
        compact["source_type"] = source_type
    status = str(item.get("result_status") or "").upper()
    if status and status != "SUCCESS":
        compact["result_status"] = status
    content = item.get("content", "")
    if not isinstance(content, dict):
        if source_type == "product_knowledge":
            compact["content"] = str(content)[:10000]
            compact["scope"] = "querypilot_product"
            compact["authoritative"] = True
            return compact

        is_note = source_type in _NOTE_TYPES
        compact["content"] = str(content)[
            : MAX_NOTE_CHARS if is_note else MAX_TEXT_CHARS_PER_ITEM
        ]
        title = item.get("document_title") or item.get("title")
        if isinstance(title, str) and title and not is_note:
            compact["document"] = title[:120]
        return compact
    role = content.get("evidence_role")
    if source_type == "workspace_metadata" or role == "workspace_metadata":
        compact["content"] = _compact_metadata(content)
    elif role == "aggregate_result":
        compact.pop("source_type", None)
        compact["content"] = _slim(
            {
                "scope": "aggregate_result",
                "document": content.get("document_title"),
                "population_rows": content.get("population_rows"),
                "aggregates": content.get("aggregates"),
                "formulas": content.get("formulas"),
                "relationships": content.get("relationships"),
            }
        )
    elif role == "selection_result":
        compact.pop("source_type", None)
        compact["content"] = _slim(
            {
                "scope": "selection_result",
                "document": content.get("document_title"),
                "population_rows": content.get("population_rows"),
                "selection_result": content.get("selection_result"),
                "interpretation": content.get("interpretation"),
            }
        )
    elif "rows" in content or "columns" in content:
        compact.pop("source_type", None)
        compact["content"] = _compact_table(item, content)
    else:
        compact["content"] = {
            k: (str(v)[:300] if isinstance(v, str) else v)
            for k, v in content.items()
            if k in {"message", "error"}
        } or str(content)[:300]
    return compact


_compact_item_for_prompt = _compact_item


_PRODUCT_SECTION_HEADER = re.compile(r"^[A-Z][A-Z0-9 /&'().\-]{2,60}$")

_PRODUCT_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "be",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "give",
        "how",
        "in",
        "is",
        "it",
        "its",
        "list",
        "me",
        "of",
        "on",
        "or",
        "please",
        "show",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "would",
        "you",
        "your",
    }
)


def _split_product_sections(content: str) -> list[tuple[str, str]]:
    """[(header, body)] in document order. A header is a short ALL-CAPS
    line; everything up to the next header is that section's body."""
    sections: list[tuple[str, str]] = []
    header = ""
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip("\n")
        if header or text:
            sections.append((header, text))

    for raw_line in str(content).splitlines():
        line = raw_line.strip()
        if line and _PRODUCT_SECTION_HEADER.match(line):
            flush()
            header = line
            body = []
        else:
            body.append(raw_line)
    flush()
    return sections


def _section_score(header: str, body: str, tokens: list[str]) -> int:
    """Header hits count double: the header names the section's topic."""
    header_lower = header.lower()
    body_lower = body.lower()
    score = 0
    for token in tokens:
        if token in header_lower:
            score += 2
        score += body_lower.count(token)
    return score


def _relevant_product_sections(content: str, query: str, max_chars: int) -> str:
    """
    Reduce an over-budget product knowledge document to whole sections,
    most relevant to the current query first (document order on ties;
    the result keeps document order). At least one section - the most
    relevant - is always kept even if it alone exceeds max_chars: the
    fact the query needs is never deleted to satisfy a character budget.
    """
    sections = _split_product_sections(content)
    if not sections:
        return content

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
        if len(token) > 2 and token not in _PRODUCT_QUERY_STOPWORDS
    ]

    if tokens:
        ranked = sorted(
            range(len(sections)),
            key=lambda i: (
                -_section_score(sections[i][0], sections[i][1], tokens),
                i,
            ),
        )
    else:
        ranked = list(range(len(sections)))

    def section_text(index: int) -> str:
        header, body = sections[index]
        return f"{header}\n\n{body}".strip() if header else body

    selected: list[int] = []
    used = 0
    for index in ranked:
        length = len(section_text(index)) + 2
        if selected and used + length > max_chars:
            continue
        selected.append(index)
        used += length

    if not selected:
        return content

    return "\n\n".join(section_text(index) for index in sorted(selected))


def _is_table(item: dict[str, Any]) -> bool:
    content = item.get("content")
    return (
        isinstance(content, dict)
        and content.get("scope") != "aggregate_result"
        and isinstance(content.get("rows"), list)
        and isinstance(content.get("columns"), list)
    )


def _anchor_strings(evidence: list[Any]) -> set[str]:
    """
    Identifying values of the records the question is actually about:
    matched join keys, the rows of focused results (entity / key-joined /
    query results) and the named rows inside validated aggregates (extremes,
    related records). Domain-neutral: values only, never column names.
    """
    anchors: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            _collect_strings(metadata.get("matched_keys"), anchors)
        if not isinstance(content, dict):
            continue
        _collect_strings(content.get("matched_keys"), anchors)
        _collect_strings(content.get("join_keys"), anchors)
        role = content.get("evidence_role")
        if role == "aggregate_result":
            _collect_strings(content.get("aggregates"), anchors)
            _collect_strings(content.get("relationships"), anchors)
        elif role == "selection_result":
            _collect_strings(content.get("selection_result"), anchors)
        elif role in {"entity_row", "key_joined", "query_result"}:
            _collect_strings(content.get("rows"), anchors)
    return anchors


def _anchored_rows(rows: list[Any], anchors: set[str]) -> list[bool]:
    """
    A row is anchored when it holds an anchor value that is selective in this
    table (so a value shared by most rows, such as a category, does not pull
    the whole table in).
    """
    per_row: list[set[str]] = []
    for row in rows:
        found: set[str] = set()
        _collect_strings(row, found)
        per_row.append(found)
    counts = Counter(s for found in per_row for s in found)
    limit = max(1, len(rows) // 3)
    return [
        any(s in anchors and counts[s] <= limit for s in found) for found in per_row
    ]


def _dedupe_notes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            key = (item.get("source_type"), item.get("result_status"), content)
            if key in seen:
                continue
            seen.add(key)
        out.append(item)
    return out


def _reduce_covered_tables(
    items: list[dict[str, Any]], anchors: set[str]
) -> list[dict[str, Any]]:
    """
    When a validated aggregate already covers a table's whole population, the
    other rows of that table are redundant raw representations of the same
    facts. Keep the rows the question is about (anchored) and say, explicitly,
    that population figures live in the aggregate item. Tables without such
    coverage are never reduced here.
    """
    coverage: dict[str, tuple[int, str]] = {}
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or content.get("scope") != "aggregate_result":
            continue
        document = content.get("document")
        try:
            population = int(content.get("population_rows"))
        except (TypeError, ValueError):
            continue
        if document:
            text = json.dumps(
                [content.get("aggregates"), content.get("formulas")],
                ensure_ascii=False,
                default=str,
            ).lower()
            previous = coverage.get(document)
            coverage[document] = (
                max(population, previous[0]) if previous else population,
                (previous[1] if previous else "") + text,
            )
    if not coverage or not anchors:
        return items
    for item in items:
        if not _is_table(item):
            continue
        content = item["content"]
        rows = content["rows"]
        entry = coverage.get(content.get("document"))
        if entry is None or len(rows) <= MIN_ROWS_FOR_COVERAGE_REDUCTION:
            continue
        population, aggregate_text = entry
        if population < len(rows):
            continue

        if not any(str(c).lower() in aggregate_text for c in content["columns"]):
            continue
        flags = _anchored_rows(rows, anchors)
        kept = [row for row, flag in zip(rows, flags) if flag]
        if not kept or len(kept) == len(rows):
            continue
        content["source_rows"] = content.get("source_rows", len(rows))
        content["rows_covered_by_aggregate"] = len(rows) - len(kept)
        content["scope"] = "aggregate_covered_subset"
        content["rows"] = kept
    return items


def _dedupe_rows_across_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    One authoritative representation per row: a row that already appears in a
    larger item of the same source and column set is not sent twice. An item
    left with nothing but duplicates is dropped unless it carries its own
    semantics (ranking/aggregate type, message); then it keeps that and points
    to the item that holds the rows.

    Ranking / positional execution results are never touched: their rows ARE
    the answer (an ordered, limited selection), not a redundant copy.
    """
    tables = [(pos, item) for pos, item in enumerate(items) if _is_table(item)]
    tables.sort(key=lambda pair: -len(pair[1]["content"]["rows"]))
    seen: dict[tuple[Any, ...], dict[str, int]] = {}
    drop: set[int] = set()
    for pos, item in tables:
        content = item["content"]
        if content.get("query_type") in {"ranking", "positional"}:
            continue
        identity = (content.get("document"), tuple(content["columns"]))
        registry = seen.setdefault(identity, {})
        fresh, owners = [], []
        for row in content["rows"]:
            key = json.dumps(row, ensure_ascii=False, default=str)
            if key in registry:
                owners.append(registry[key])
            else:
                registry[key] = item["citation_index"]
                fresh.append(row)
        if not owners:
            continue
        if fresh:
            content["rows"] = fresh
            content["other_rows_in"] = sorted(set(owners))
        elif any(k in content for k in ("query_type", "message", "truncated")):
            content["rows"] = []
            content["rows_in"] = sorted(set(owners))
        else:
            drop.add(pos)
    return [item for pos, item in enumerate(items) if pos not in drop]


_QUESTION_FILLER = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "and",
        "or",
        "in",
        "on",
        "with",
        "by",
        "me",
        "my",
        "give",
        "show",
        "list",
        "tell",
        "all",
        "those",
        "these",
        "that",
        "their",
        "them",
        "also",
        "what",
        "who",
        "which",
        "is",
        "are",
        "was",
        "were",
        "please",
        "then",
        "than",
        "from",
        "who",
        "whose",
        "each",
        "every",
    }
)


def _word_roots(text: Any) -> set[str]:
    """Lower-case words without plurals, filler words and bare numbers."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    words = re.findall(r"[a-z0-9]+", spaced.lower())
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in words
        if word not in _QUESTION_FILLER and not word.isdigit()
    }


def _cell_is_mentioned(cell: Any, question: str) -> bool:
    """A text cell value that the question itself names (a filter value)."""
    if not isinstance(cell, str) or len(cell.strip()) < 3:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(cell.strip().lower())}(?![a-z0-9])"
    return bool(re.search(pattern, question.lower()))


def _columns_to_keep(
    columns: list[str],
    rows: list[list[Any]],
    question: str,
    question_words: set[str],
) -> list[int]:
    """Identifier and name columns, columns the question names, and columns
    holding a value the question mentions."""
    keep: list[int] = []
    for position, column in enumerate(columns):
        roots = _word_roots(column)
        is_key_or_identity = is_id_column(column) or "name" in roots
        is_named = bool(roots & question_words)
        has_mentioned_value = any(
            _cell_is_mentioned(row[position], question) for row in rows
        )
        if is_key_or_identity or is_named or has_mentioned_value:
            keep.append(position)
    return keep


def _project_columns(
    item: dict[str, Any], question: str, question_words: set[str]
) -> bool:
    """Narrow one table to the columns the question needs. Returns whether
    anything was removed. Every row stays; the removal is flagged."""
    content = item["content"]
    if content.get("query_type") is not None:
        return False
    columns = [str(c) for c in content["columns"]]
    rows = content["rows"]
    if not all(isinstance(row, list) and len(row) == len(columns) for row in rows):
        return False

    keep = _columns_to_keep(columns, rows, question, question_words)
    if not keep or len(keep) == len(columns):
        return False

    content["columns_omitted_from_prompt"] = [
        c for j, c in enumerate(columns) if j not in keep
    ]
    content["columns"] = [columns[j] for j in keep]
    content["rows"] = [[row[j] for j in keep] for row in rows]
    return True


def _project_largest_table(items: list[dict[str, Any]], question: str) -> bool:
    """One step of the budget loop: narrow the biggest table not yet narrowed."""
    question_words = _word_roots(question)
    if not question_words:
        return False
    candidates = [
        item
        for item in items
        if _is_table(item) and "columns_omitted_from_prompt" not in item["content"]
    ]
    for item in sorted(candidates, key=lambda i: -_size(i["content"]["rows"])):
        if _project_columns(item, question, question_words):
            return True
    return False


def _rows_that_fit(rows: list[Any], deficit: int) -> int:
    """How many rows to KEEP so the table shrinks by about `deficit` chars.
    Only what the budget needs is removed (never half a table by default);
    always at least one row goes, so the budget loop makes progress."""
    average_row = max(_size(rows) // max(len(rows), 1), 1)
    rows_to_drop = max(math.ceil(max(deficit, 1) / average_row), 1)
    return max(MIN_ROWS_WHEN_TRIMMED, len(rows) - rows_to_drop)


def _fit_to_budget(
    items: list[dict[str, Any]],
    budget_chars: int,
    anchors: set[str],
    query: str = "",
) -> list[dict[str, Any]]:
    """
    Last resort, only when the semantically compact evidence still exceeds the
    budget. Order of sacrifice: generic text passages (shortened), then
    product knowledge (whole-section selection for the current query -
    never a prefix cut), then COLUMNS the question does not need (every row
    is kept; flagged with columns_omitted_from_prompt), and only then rows of
    the largest table that are NOT anchored to the question. Row removal is
    always flagged (truncated + rows_omitted_from_prompt) so it is never
    mistaken for a complete population. Aggregates and notes are never trimmed.
    """
    guard = 0
    while _size(items) > budget_chars and guard < 200:
        guard += 1
        texts = [
            i
            for i in items
            if isinstance(i.get("content"), str)
            and len(i["content"]) > 300
            and i.get("source_type") not in _NOTE_TYPES
            and i.get("source_type") != "product_knowledge"
        ]
        if texts:
            biggest = max(texts, key=lambda i: len(i["content"]))
            biggest["content"] = biggest["content"][
                : int(len(biggest["content"]) * 0.7)
            ]
            continue

        products = [
            i
            for i in items
            if i.get("source_type") == "product_knowledge"
            and isinstance(i.get("content"), str)
            and len(i["content"]) > 600
        ]
        if products:
            biggest = max(products, key=lambda i: len(i["content"]))
            deficit = _size(items) - budget_chars

            target = max(len(biggest["content"]) - deficit - 160, 600)
            if target < len(biggest["content"]):
                reduced = _relevant_product_sections(
                    biggest["content"],
                    query,
                    target,
                )
                if reduced and len(reduced) < len(biggest["content"]):
                    biggest["content"] = reduced
                    continue
        if _project_largest_table(items, query):
            continue
        tables = [
            i
            for i in items
            if _is_table(i) and len(i["content"]["rows"]) > MIN_ROWS_WHEN_TRIMMED
        ]
        if not tables:
            break
        biggest = max(tables, key=lambda i: _size(i["content"]["rows"]))
        content = biggest["content"]
        rows = content["rows"]
        flags = _anchored_rows(rows, anchors)
        order = [i for i, f in enumerate(flags) if f] + [
            i for i, f in enumerate(flags) if not f
        ]
        keep_count = _rows_that_fit(rows, deficit=_size(items) - budget_chars)
        keep_idx = sorted(order[:keep_count])
        content["source_rows"] = content.get("source_rows", len(rows))
        content["rows_omitted_from_prompt"] = (
            int(content.get("rows_omitted_from_prompt", 0)) + len(rows) - len(keep_idx)
        )
        content["truncated"] = True
        content["rows"] = [rows[i] for i in keep_idx]
    if _size(items) > budget_chars:
        logger.warning(
            "answer evidence still over budget after compaction: chars=%d", _size(items)
        )
    return items


_NOTE_SOURCE_TYPES = _NOTE_TYPES | {
    "evidence_note",
    "document_coverage",
    "coverage_note",
    "coverage",
}
_NON_DATA_STATUSES = {
    "",
    "NOT_FOUND",
    "RETRIEVAL_ERROR",
    "PARTIAL",
    "INSUFFICIENT_EVIDENCE",
}


def _is_note(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    kind = str(item.get("source_type") or "").lower()
    return kind in _NOTE_SOURCE_TYPES or "coverage" in kind


def _available_source_titles(evidence: list[Any]) -> set[str]:
    """Titles of sources whose structured rows are actually in the evidence."""
    titles: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or _is_note(item):
            continue
        if str(item.get("result_status") or "SUCCESS").upper() != "SUCCESS":
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        title = content.get("document_title")
        if title and (content.get("rows") or content.get("aggregates")):
            titles.add(str(title).strip().lower())
    return {t for t in titles if len(t) >= 3}


def _contradicted_note(item: Any, titles: set[str]) -> bool:
    """
    A retrieval/coverage note that says a source has no passages / no match /
    nothing included, about a source whose rows ARE in the evidence, describes
    the text search only. Sent as evidence it reads as 'source unavailable'.
    """
    if not titles or not _is_note(item):
        return False
    if str(item.get("result_status") or "").upper() not in _NON_DATA_STATUSES:
        return False
    blob = json.dumps(item.get("content"), ensure_ascii=False, default=str).lower()
    return any(title in blob for title in titles)


def _source_inventory(items: list[dict[str, Any]]) -> list[str]:
    """
    Authoritative list of what IS available, built from the canonical items
    themselves (so the indices are exactly the citable ones). One short string
    per source: "n document (kind, rows)".
    """
    inventory: list[str] = []
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or item.get("result_status"):
            continue
        n = item.get("citation_index")
        document = content.get("document") or "source"
        if content.get("scope") == "selection_result":
            inventory.append(
                f"{n} {document} (deterministic selection over "
                f"{content.get('population_rows')} qualifying rows)"
            )
        elif content.get("scope") == "aggregate_result":
            population = content.get("population_rows")
            detail = "aggregates, complete population" + (
                f" of {population} rows" if population is not None else ""
            )
            inventory.append(f"{n} {document} ({detail})")
        elif isinstance(content.get("rows"), list):
            kind = content.get("scope") or "table"
            sent = len(content["rows"])
            total = content.get("source_rows")
            rows = (
                f"{sent} of {total} rows"
                if total not in (None, sent)
                else f"{sent} rows"
            )
            inventory.append(f"{n} {document} ({kind}, {rows})")
    return inventory


def _log_evidence_debug(stage: str, items: list[dict[str, Any]]) -> None:
    """One line per table: rows the answer model really receives."""
    for item in items:
        if not _is_table(item):
            continue
        content = item["content"]
        logger.info(
            "[ANSWER_EVIDENCE_DEBUG] stage=%s n=%s document=%r rows_sent=%d "
            "source_rows=%s rows_omitted=%s columns_omitted=%s truncated=%s",
            stage,
            item.get("citation_index"),
            content.get("document"),
            len(content["rows"]),
            content.get("source_rows"),
            content.get("rows_omitted_from_prompt", 0),
            content.get("columns_omitted_from_prompt", []),
            bool(content.get("truncated")),
        )


def _canonical_answer_evidence(
    evidence: list[Any],
    *,
    budget_chars: int,
    query: str = "",
) -> list[dict[str, Any]]:
    """
    Raw stored evidence -> compact canonical answer evidence.

    citation_index is the item's 1-based position in the stored evidence, set
    BEFORE any item is merged away, and never renumbered afterwards.
    """
    anchors = _anchor_strings(evidence)
    available = _available_source_titles(evidence)
    items: list[dict[str, Any]] = []
    for position, raw in enumerate(evidence, start=1):
        stored = raw.get("citation_index") if isinstance(raw, dict) else None
        if stored is not None and stored != position:
            logger.warning(
                "evidence citation_index=%r differs from stored position %d; "
                "using the stored position",
                stored,
                position,
            )
        if _contradicted_note(raw, available):
            logger.info(
                "answer evidence: dropped note contradicted by source rows (n=%d)",
                position,
            )
            continue
        compact = _compact_item(raw)
        items.append({"citation_index": position, **compact})
    items = _dedupe_notes(items)
    items = _dedupe_rows_across_items(items)

    if _size(items) > budget_chars:
        items = _reduce_covered_tables(items, anchors)
        items = _dedupe_rows_across_items(items)
    _log_evidence_debug("before_fit_to_budget", items)
    items = _fit_to_budget(items, budget_chars, anchors, query=query)
    _log_evidence_debug("after_fit_to_budget", items)
    return items


def _bound_evidence(evidence: list[Any]) -> list[dict[str, Any]]:
    return _canonical_answer_evidence(evidence, budget_chars=_payload_char_budget())


def _population_contract(*, evidence: list[Any]) -> dict[str, Any] | None:
    keys: list[str] = []
    where: dict[str, list[int]] = {}
    for position, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        metadata = item.get("metadata")
        content_keys = None
        if isinstance(content, dict):
            content_keys = content.get("matched_keys") or content.get("join_keys")
        metadata_keys = (
            metadata.get("matched_keys") if isinstance(metadata, dict) else None
        )
        matched_keys = content_keys if isinstance(content_keys, list) else metadata_keys
        if not isinstance(matched_keys, list):
            continue
        for value in matched_keys:
            key = str(value).strip()
            if not key:
                continue
            if key not in where:
                where[key] = []
                keys.append(key)
            if position not in where[key]:
                where[key].append(position)
    if len(keys) <= 1:
        return None
    return {
        "type": "matched_population",
        "record_count": len(keys),
        "matched_keys": keys,
        "evidence_indices_by_key": where,
    }


def _population_hint(
    contract: dict[str, Any] | None, has_aggregate: bool
) -> dict[str, Any] | None:
    """Compact payload form: no per-key index map (the rows already carry
    it); the key list only when no validated aggregate covers the group."""
    if not contract:
        return None
    hint: dict[str, Any] = {"record_count": contract["record_count"]}
    if not has_aggregate:
        hint["matched_keys"] = contract["matched_keys"]
        hint["rule"] = (
            "Each key is one record established across sources; when the "
            "question concerns the group, apply every condition to every key."
        )
    return hint


def _missing_population_keys(answer: str, contract: dict[str, Any] | None) -> list[str]:
    if not contract:
        return []
    text = str(answer or "")
    return [str(k) for k in contract.get("matched_keys", []) if str(k) not in text]


_FOREIGN_CURRENCY = re.compile(r"(?:USD|US\$|\$\s?\d|EUR|€|£|GBP)")
_CURRENCY_PREFIX = re.compile(r"(?<![A-Za-z])(?:US\$|\$|₹|Rs\.?|INR)\s?(?=\d)")


def _default_currency_symbol() -> str | None:
    try:
        from django.conf import settings

        symbol = getattr(
            settings, "QUERYPILOT_CURRENCY_SYMBOL", DEFAULT_CURRENCY_SYMBOL
        )
    except Exception:
        symbol = DEFAULT_CURRENCY_SYMBOL
    return symbol or None


def _currency_symbol_for(evidence: list[Any]) -> str | None:
    considered = [
        item
        for item in evidence
        if isinstance(item, dict) and item.get("source_type") != "conversation_memory"
    ]
    if _FOREIGN_CURRENCY.search(
        json.dumps(considered, ensure_ascii=False, default=str)
    ):
        return None
    return _default_currency_symbol()


def _normalize_currency(answer: str, symbol: str | None) -> str:
    return _CURRENCY_PREFIX.sub(symbol, answer) if symbol else answer


def _partial_evidence_notice(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Tables that reach the answer model with fewer rows than their source.
    Stated at the top of the payload so a list built from them is never
    presented as the complete population."""
    partial = [
        {
            "document": item["content"].get("document"),
            "rows_sent": len(item["content"]["rows"]),
            "source_rows": item["content"].get("source_rows"),
        }
        for item in items
        if _is_table(item) and item["content"].get("truncated")
    ]
    if not partial:
        return None
    return {
        "tables": partial,
        "rule": "Rows are missing from these tables. Say any list or count from them may be incomplete.",
    }


def _prepare_request(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """Returns (payload_json, canonical_items, population_contract)."""
    if not query.strip():
        raise ValueError("Answer generation query cannot be empty.")
    if not evidence:
        raise ValueError("Answer generation requires evidence.")
    extras: dict[str, Any] = {}
    history = _answer_history(conversation_history, query)
    if history:
        extras["conversation_history"] = history
    symbol = _currency_symbol_for(evidence)
    if symbol:
        extras["presentation"] = {"currency_symbol": symbol}
    evidence_budget = max(
        _payload_char_budget() - _size(extras) - len(query) - 200,
        2500,
    )
    items = _canonical_answer_evidence(
        evidence, budget_chars=evidence_budget, query=query
    )
    indices = [item["citation_index"] for item in items]
    has_aggregate = any(
        isinstance(i.get("content"), dict)
        and i["content"].get("scope") == "aggregate_result"
        for i in items
    )
    contract = _population_contract(evidence=evidence)
    payload: dict[str, Any] = {
        "user_query": query,
        "evidence": items,
        "citation_contract": {"available": indices},
    }
    inventory = _source_inventory(items)
    if inventory:
        payload["sources_available"] = inventory
    hint = _population_hint(contract, has_aggregate)
    if hint:
        payload["population"] = hint
    partial = _partial_evidence_notice(items)
    if partial:
        payload["partial_evidence"] = partial
    _log_evidence_debug("final_payload", items)
    payload.update(extras)
    return (
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
        items,
        contract,
    )


def _build_payload(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None,
) -> tuple[str, int]:
    """Returns (payload_json, number_of_evidence_items_sent)."""
    payload, items, _ = _prepare_request(
        query=query,
        evidence=evidence,
        conversation_history=conversation_history,
    )
    return payload, len(items)


_INTERNAL_MARKER = re.compile(r" ?【[^】]*】")


def _clean_answer(content: str) -> str:
    cleaned = _INTERNAL_MARKER.sub("", content)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_MARKER = re.compile(r" ?\[(\d{1,3})\]")
_GROUPED_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})+)\]")
_RANGE_MARKER = re.compile(r"\[(\d{1,3})\s*[-–]\s*(\d{1,3})\]")


def _expand_grouped_markers(text: str) -> str:
    def expand_group(match: re.Match[str]) -> str:
        return "".join(f"[{int(n)}]" for n in match.group(1).split(","))

    def expand_range(match: re.Match[str]) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and end - start <= 5:
            return "".join(f"[{n}]" for n in range(start, end + 1))
        return match.group(0)

    return _RANGE_MARKER.sub(expand_range, _GROUPED_MARKER.sub(expand_group, text))


def _salvage_answer(text: str) -> str | None:
    """Recover the answer string from truncated or slightly broken JSON."""
    match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.DOTALL)
    if not match:
        return None
    fragment = re.sub(r"\\$", "", match.group(1))
    try:
        return json.loads('"' + fragment + '"')
    except ValueError:
        return fragment.replace("\\\n", "\n").replace('\\"', '"')


_NUM_TOKEN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(\s?%)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[_-])")


def _value_index(
    items: list[dict[str, Any]],
) -> tuple[dict[float, set[int]], dict[str, set[int]]]:
    """Which canonical items hold which numeric / identifying string values."""
    numbers: dict[float, set[int]] = {}
    strings: dict[str, set[int]] = {}

    def visit(value: Any, n: int, depth: int = 0) -> None:
        if depth > 8 or value is None or isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numbers.setdefault(round(float(value), 4), set()).add(n)
        elif isinstance(value, str):
            text = value.strip()
            match = _NUM_TOKEN.fullmatch(text.replace("₹", "").replace("$", "").strip())
            if match:
                try:
                    number = float(re.sub(r"[^\d.+-]", "", text.replace("₹", "")))
                    numbers.setdefault(round(number, 4), set()).add(n)
                except ValueError:
                    pass
            elif 3 <= len(text) <= 80:
                strings.setdefault(text.lower(), set()).add(n)
        elif isinstance(value, dict):
            for inner in value.values():
                visit(inner, n, depth + 1)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                visit(inner, n, depth + 1)

    for item in items:
        if item.get("source_type") in _NOTE_SOURCE_TYPES or item.get("result_status"):
            continue
        content = item.get("content")
        if isinstance(content, dict):
            visit(
                {
                    k: v
                    for k, v in content.items()
                    if k not in ("columns", "document", "scope")
                },
                item["citation_index"],
            )
    return numbers, strings


def _attribute_citations(
    answer: str, items: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    """
    Used ONLY when the model returned an answer with no inline marker at all.
    A segment (line / sentence) is cited only by evidence items that actually
    contain a value printed in that segment. Segments with no matching value
    (pure narrative, or a calculated result not present in evidence) stay
    uncited; nothing is spread across the answer.
    """
    numbers, strings = _value_index(items)
    if not numbers and not strings:
        return answer, []
    out_lines: list[str] = []
    used: set[int] = set()
    for line in answer.split("\n"):
        segments = _SENTENCE_SPLIT.split(line) if len(line) > 160 else [line]
        rebuilt: list[str] = []
        for segment in segments:
            hits: Counter[int] = Counter()
            for match in _NUM_TOKEN.finditer(segment):
                raw = match.group(0)
                percent = bool(match.group(1))
                try:
                    number = float(re.sub(r"[^\d.+-]", "", raw))
                except ValueError:
                    continue
                if "." not in raw and not percent and abs(number) < 100:
                    continue
                candidates = {round(number, 4)}
                if percent:
                    candidates.add(round(number / 100, 4))
                for candidate in candidates:
                    for n in numbers.get(candidate, ()):
                        hits[n] += 1
            lowered = segment.lower()
            for text, owners in strings.items():
                if text in lowered:
                    for n in owners:
                        hits[n] += 1
            if hits and segment.strip():
                top = sorted(hits, key=lambda n: (-hits[n], n))[:3]
                marks = "".join(f"[{n}]" for n in sorted(top))
                used.update(top)
                stripped = segment.rstrip()
                rebuilt.append(stripped + marks + segment[len(stripped) :])
            else:
                rebuilt.append(segment)
        out_lines.append(" ".join(rebuilt) if len(rebuilt) > 1 else rebuilt[0])
    return "\n".join(out_lines), sorted(used)


def _parse_answer(
    raw: str,
    *,
    valid: set[int],
    items: list[dict[str, Any]] | None = None,
) -> tuple[str, list[int]]:
    """
    Deterministic, fail-closed citation handling.

    * Inline [n] markers in the text are the source of truth.
    * Markers that are not valid evidence indices are stripped, never remapped.
    * Declarations without a marker are never turned into markers.
    * If the text has NO marker at all, citations are re-derived from the
      evidence values each segment actually states (see _attribute_citations);
      if nothing can be matched the answer is returned uncited rather than
      with an invented or dangling citation.
    """
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except ValueError:
                data = None
    declared: set[int] = set()
    if data is None:
        salvaged = _salvage_answer(text)
        if salvaged is not None:
            logger.warning(
                "Answer LLM JSON was malformed/truncated; salvaged answer text."
            )
            answer_text = salvaged
        elif re.match(r'^\s*\{\s*"(?:answer|citations)"', text):
            raise ValueError("Answer LLM returned an unparseable structured response.")
        else:
            answer_text = text
    else:
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("answer"), str)
            or not data["answer"].strip()
        ):
            raise ValueError("Answer LLM returned an invalid structured response.")
        answer_text = data["answer"]
        for value in data.get("citations") or []:
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in valid
            ):
                declared.add(value)
            else:
                logger.warning("Rejected invalid citation index: %r", value)
    answer_text = _expand_grouped_markers(answer_text)
    in_text = {int(m.group(1)) for m in _MARKER.finditer(answer_text)}
    good = in_text & valid
    if in_text - good:
        logger.warning(
            "Stripped out-of-range citation markers: %s", sorted(in_text - good)
        )
    answer_text = _MARKER.sub(
        lambda m: m.group(0) if int(m.group(1)) in good else "",
        answer_text,
    )
    if declared and declared != good:
        logger.warning(
            "Citation mismatch: declared=%s markers_in_text=%s",
            sorted(declared),
            sorted(good),
        )
    if not good and items:
        answer_text, derived = _attribute_citations(answer_text, items)
        good = set(derived)
        logger.info("Citations re-derived from stated values: %s", sorted(good))
    return answer_text, sorted(good)


def _parse_answer_response(
    raw: str,
    *,
    evidence_count: int,
    valid_indices: Iterable[int] | None = None,
) -> str:
    """
    Returns answer text whose [n] markers are exactly valid evidence indices.
    valid_indices defaults to 1..evidence_count (stored evidence positions).
    """
    valid = (
        set(valid_indices)
        if valid_indices is not None
        else set(range(1, evidence_count + 1))
    )
    answer_text, _ = _parse_answer(raw, valid=valid)
    return answer_text


_EXHAUSTION_MARKERS = (
    "json_validate_failed",
    "max completion tokens reached",
    "max_tokens reached",
    "finish_reason=length",
)


def _is_generation_exhaustion(exc: BaseException) -> bool:
    """
    True when the provider ran out of COMPLETION tokens before finishing the
    output (the reasoning model spent the budget), as opposed to a request
    that was too large or a transport failure.
    """
    parts = [str(exc), repr(exc)]
    for name in ("body", "response", "failed_generation", "error", "detail"):
        value = getattr(exc, name, None)
        if value is not None:
            parts.append(str(value))
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        parts.append(str(cause))
    blob = " ".join(parts).lower()
    return any(marker in blob for marker in _EXHAUSTION_MARKERS)


def _failed_generation_text(exc: BaseException) -> str | None:
    """Text the provider reports as the failed generation, if it exposes it."""
    for holder in (exc, exc.__cause__, exc.__context__):
        if holder is None:
            continue
        value = getattr(holder, "failed_generation", None)
        if isinstance(value, str) and value.strip():
            return value
        body = getattr(holder, "body", None)
        if isinstance(body, dict):
            error = body.get("error") if isinstance(body.get("error"), dict) else body
            value = error.get("failed_generation") if isinstance(error, dict) else None
            if isinstance(value, str) and value.strip():
                return value
    return None


def _router_options(effort: str | None) -> dict[str, Any]:
    """Pass reasoning_effort only if router.generate declares it explicitly."""
    if not effort:
        return {}
    try:
        params = inspect.signature(router.generate).parameters
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": effort} if "reasoning_effort" in params else {}


def _request_answer(
    payload: str,
    *,
    max_tokens: int,
    concise: bool = False,
) -> tuple[str, str | None]:
    """One plain-text generation. Returns (content, finish_reason)."""
    system = SYSTEM_PROMPT
    if concise:
        system += (
            "\nThe previous attempt ran out of tokens. Answer directly and "
            "briefly: state each requested value once with its [n] marker."
        )
    response = router.generate(
        task="answer_generation",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        include_reasoning=False,
        **_router_options("low"),
    )
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError("Answer LLM returned an invalid response.") from exc
    finish = getattr(choice, "finish_reason", None)
    return (content or ""), (str(finish) if finish is not None else None)


def _retry_budget(payload: str) -> int:
    input_tokens = (
        _system_tokens() + _estimate_tokens(payload) + REQUEST_OVERHEAD_TOKENS + 60
    )
    return max(min(RETRY_MAX_OUTPUT_TOKENS, RETRY_TOTAL_CEILING - input_tokens), 0)


def generate_answer(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate the complete user-facing answer in one request."""
    payload, canon_items, contract = _prepare_request(
        query=query,
        evidence=evidence,
        conversation_history=conversation_history,
    )
    indices = [item["citation_index"] for item in canon_items]
    logger.info(
        "answer_payload: stored_items=%d sent_items=%d payload_chars=%d",
        len(evidence),
        len(indices),
        len(payload),
    )
    max_tokens, _ = _completion_budget(payload)
    content = ""
    exhausted = False
    try:
        content, finish = _request_answer(payload, max_tokens=max_tokens)
        exhausted = finish == "length"
    except Exception as exc:
        if not _is_generation_exhaustion(exc):
            raise
        exhausted = True
        logger.warning(
            "answer completion exhausted (%s); attempting recovery", type(exc).__name__
        )

        partial = _failed_generation_text(exc)
        if partial:
            try:
                candidate, _ = _parse_answer(
                    partial, valid=set(indices), items=canon_items
                )
                if _complete_json_answer(partial):
                    return _finish_answer(candidate, evidence, contract)
            except ValueError:
                pass
    if exhausted or not content.strip():
        retry_tokens = _retry_budget(payload)
        if retry_tokens < MIN_OUTPUT_TOKENS:
            raise ValueError("Answer generation exhausted its completion budget.")
        logger.warning(
            "answer_retry: completion=%d (single bounded retry, concise instruction)",
            retry_tokens,
        )
        content, finish = _request_answer(
            payload, max_tokens=retry_tokens, concise=True
        )
        if finish == "length" or not content.strip():
            raise ValueError("Answer generation exhausted its completion budget.")
    if not content.strip():
        raise ValueError("Answer LLM returned an empty answer.")
    answer, _cited = _parse_answer(content, valid=set(indices), items=canon_items)
    return _finish_answer(answer, evidence, contract)


def _complete_json_answer(text: str) -> bool:
    """True only for a fully parseable {"answer": "..."} document."""
    try:
        data = json.loads(text.strip())
    except ValueError:
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("answer"), str)
        and bool(data["answer"].strip())
    )


def _finish_answer(
    answer: str, evidence: list[Any], contract: dict[str, Any] | None
) -> str:
    missing = _missing_population_keys(answer, contract)
    if missing:
        logger.info(
            "Matched population keys not mentioned in the answer "
            "(may be legitimately filtered out): %s",
            missing,
        )
    return _normalize_currency(_clean_answer(answer), _currency_symbol_for(evidence))


def generate_answer_stream(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """
    True provider-level text stream. Deltas are plain Markdown with inline
    [n] markers; the caller accumulates them and passes the full text through
    _parse_answer_response (markers are validated there; legacy JSON output
    is still accepted).
    """
    payload, canon_items, _ = _prepare_request(
        query=query,
        evidence=evidence,
        conversation_history=conversation_history,
    )
    indices = [item["citation_index"] for item in canon_items]
    logger.info(
        "answer_payload(stream): stored_items=%d sent_items=%d payload_chars=%d",
        len(evidence),
        len(indices),
        len(payload),
    )
    max_tokens, _ = _completion_budget(payload)
    stream = router.generate_stream(
        task="answer_generation",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        include_reasoning=False,
    )
    yielded_any = False
    for delta in stream:
        if isinstance(delta, str) and delta:
            yielded_any = True
            yield delta
    if not yielded_any:
        raise ValueError("Answer LLM returned an empty streaming response.")
