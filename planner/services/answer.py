from __future__ import annotations

import json

import logging

import re

from collections.abc import Iterator

from typing import Any

from llm.router import router

from planner.prompts.answer import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12

MAX_HISTORY_CHARS = 12000

MAX_EVIDENCE_ITEMS = 8

MAX_CHARS_PER_EVIDENCE = 6000

MAX_TOTAL_EVIDENCE_CHARS = 24000

# Table results are structured, so they are not cut by the character limit

# above. Cap them explicitly so one wide 100-row result cannot dominate the

# prompt (or blow a per-minute token limit).

MAX_TABLE_ROWS_IN_PAYLOAD = 50

MAX_CELL_CHARS = 300

# GPT-OSS can spend completion budget on internal reasoning before emitting
# the strict JSON object. The previous 1536-token cap could terminate the
# response before a valid JSON document was produced.
ANSWER_MAX_OUTPUT_TOKENS = 4096
ANSWER_STREAM_MAX_OUTPUT_TOKENS = 2048

# Presentation only. QueryPilot is currency-agnostic: no symbol is applied

# unless settings.QUERYPILOT_CURRENCY_SYMBOL is set. Amounts are otherwise

# shown in whatever currency the evidence itself states.

DEFAULT_CURRENCY_SYMBOL = None





ANSWER_RESPONSE_FORMAT = {

    "type": "json_schema",

    "json_schema": {

        "name": "querypilot_answer",

        "strict": True,

        "schema": {

            "type": "object",

            "properties": {

                "answer": {"type": "string"},

                "citations": {

                    "type": "array",

                    "items": {"type": "integer"},

                },

            },

            "required": ["answer", "citations"],

            "additionalProperties": False,

        },

    },

}



def _bound_history(

    history: list[dict[str, Any]],

) -> list[dict[str, Any]]:

    """Keep recent conversation history within a hard size budget."""

    bounded: list[dict[str, Any]] = []

    total_chars = 0

    for message in history[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(message, dict):

            continue

        role = str(message.get("role", ""))

        content = str(message.get("content", ""))

        remaining = MAX_HISTORY_CHARS - total_chars

        if remaining <= 0:

            break

        content = content[:remaining]

        bounded.append({"role": role, "content": content})

        total_chars += len(content)

    return bounded





def _shrink_structured(content: Any) -> Any:

    """

    Cap the rows/cells of a table result for the prompt. Every other key

    (row_count, rows_returned, source_table_row_counts, interpretation, ...)

    is preserved untouched.

    """

    if not isinstance(content, dict) or not isinstance(content.get("rows"), list):

        return content

    rows = content["rows"]

    shrunk = dict(content)

    shrunk["rows"] = [

        {

            key: (value[:MAX_CELL_CHARS] if isinstance(value, str) else value)

            for key, value in row.items()

        }

        if isinstance(row, dict)

        else row

        for row in rows[:MAX_TABLE_ROWS_IN_PAYLOAD]

    ]

    if len(rows) > MAX_TABLE_ROWS_IN_PAYLOAD:

        shrunk["rows_omitted_from_prompt"] = len(rows) - MAX_TABLE_ROWS_IN_PAYLOAD

    return shrunk





def _bound_evidence(

    evidence: list[Any],

) -> list[dict[str, Any]]:

    """

    Keep evidence within a hard context budget.

    String content is truncated. Structured table/metadata content is

    preserved as structured data (rows capped) and is not converted to a

    Python string.

    Index stability: output position i always corresponds to input position

    i (non-dict items are wrapped, never skipped), and each item is tagged

    with an explicit 1-based "citation_index" equal to that position. Only

    the tail can be dropped by the item/char caps.

    """

    bounded: list[dict[str, Any]] = []

    total_chars = 0

    for position, item in enumerate(evidence[:MAX_EVIDENCE_ITEMS], start=1):

        remaining = MAX_TOTAL_EVIDENCE_CHARS - total_chars

        if remaining <= 0:

            break

        if isinstance(item, dict):

            bounded_item = dict(item)

        else:

            bounded_item = {"content": item}

        content = bounded_item.get("content", "")

        if isinstance(content, str):

            content_limit = min(MAX_CHARS_PER_EVIDENCE, remaining)

            bounded_item["content"] = content[:content_limit]

            total_chars += len(bounded_item["content"])

        elif isinstance(content, (dict, list)):

            content = _shrink_structured(content)

            bounded_item["content"] = content

            try:

                serialized = json.dumps(

                    content,

                    ensure_ascii=False,

                    default=str,

                )

                total_chars += min(len(serialized), remaining)

            except (TypeError, ValueError):

                total_chars += min(len(str(content)), remaining)

        else:

            stringified = str(content)

            content_limit = min(MAX_CHARS_PER_EVIDENCE, remaining)

            bounded_item["content"] = stringified[:content_limit]

            total_chars += len(bounded_item["content"])

        bounded_item["citation_index"] = position

        bounded.append(bounded_item)

    return bounded





# ---------------------------------------------------------------------------

# Currency presentation (never touches a number, only the symbol before it)

# ---------------------------------------------------------------------------

_FOREIGN_CURRENCY = re.compile(r"USD|US\$|\$\s?\d|EUR|€|£|GBP")

# A currency marker immediately before a digit. The lookbehind stops "Rs"

# matching inside a word.

_CURRENCY_PREFIX = re.compile(r"(?<![A-Za-z])(?:US\$|\$|₹|Rs\.?|INR)\s?(?=\d)")





def _default_currency_symbol() -> str | None:

    try:

        from django.conf import settings

        symbol = getattr(

            settings,

            "QUERYPILOT_CURRENCY_SYMBOL",

            DEFAULT_CURRENCY_SYMBOL,

        )

    except Exception:

        symbol = DEFAULT_CURRENCY_SYMBOL

    return symbol or None





def _currency_symbol_for(evidence: list[Any]) -> str | None:

    """

    The symbol to present money in, or None to leave the model alone.

    If the evidence itself states another currency ($, USD, EUR, ...), do

    not override it: never invent a presentation the source doesn't

    establish. Conversation-memory evidence is ignored here because it only

    replays earlier answers (which may contain a wrong symbol).

    """

    considered = [

        item

        for item in evidence

        if isinstance(item, dict)

        and item.get("source_type") != "conversation_memory"

    ]

    text = json.dumps(considered, ensure_ascii=False, default=str)

    if _FOREIGN_CURRENCY.search(text):

        return None

    return _default_currency_symbol()





def _normalize_currency(answer: str, symbol: str | None) -> str:

    """Make every money amount use one symbol. Digits are never modified."""

    if not symbol:

        return answer

    return _CURRENCY_PREFIX.sub(symbol, answer)





def _build_payload(

    *,

    query: str,

    evidence: list[Any],

    conversation_history: list[dict[str, Any]] | None,

) -> tuple[str, int]:

    """

    Build the bounded answer-generation payload.

    Returns (payload_json, number_of_evidence_items_actually_sent).

    The payload contains only the query, bounded history, and evidence.

    """

    if not query.strip():

        raise ValueError("Answer generation query cannot be empty.")

    if not evidence:

        raise ValueError("Answer generation requires evidence.")

    bounded_evidence = _bound_evidence(evidence)

    payload: dict[str, Any] = {

        "user_query": query,

        "conversation_history": _bound_history(conversation_history or []),

        "evidence": bounded_evidence,

        "citation_contract": {

            "required_for_evidence_based_factual_claims": True,

            "indexing": (

                "each evidence item's 'citation_index' (its 1-based "

                "position in the evidence array)"

            ),

            "available_evidence_indices": list(

                range(1, len(bounded_evidence) + 1)

            ),

            "format": "[n]",

        },

    }

    symbol = _currency_symbol_for(evidence)

    if symbol:

        payload["presentation"] = {"currency_symbol": symbol}

    return (

        json.dumps(payload, ensure_ascii=False, default=str),

        len(bounded_evidence),

    )





_INTERNAL_MARKER = re.compile(r" ?【[^】]*】")





def _clean_answer(

    content: str,

) -> str:

    """

    Remove internal/debug markers from a completed answer.

    Only the marker (and one preceding space) is touched. Newlines,

    indentation, tables, lists and code blocks are preserved.

    """

    cleaned = _INTERNAL_MARKER.sub("", content)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()





_MARKER = re.compile(r" ?\[(\d{1,3})\]")

# "[1, 2]" is not something the downstream validator recognises, so it is

# rewritten to "[1][2]" before validation. Numbers are never changed.

_GROUPED_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})+)\]")





def _expand_grouped_markers(text: str) -> str:

    def expand(match: re.Match[str]) -> str:

        return "".join(

            f"[{int(number)}]" for number in match.group(1).split(",")

        )

    return _GROUPED_MARKER.sub(expand, text)





def _parse_answer_response(raw: str, *, evidence_count: int) -> str:

    """

    Parse the structured {"answer", "citations"} response and return answer

    text whose [n] markers are exactly the valid, in-range citations.

    Deterministic and fail-closed:

- The markers in the answer text are the source of truth.

- Any marker outside 1..evidence_count is stripped (never remapped).

- Declared citations that never appear as a marker are ignored; an

        index is never added to the text or to the result on the model's

        say-so alone.

- Non-JSON output keeps the prose but loses every marker, so

        grounding fails downstream.

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

    if data is None:

        logger.warning("Answer LLM returned non-JSON output; citations dropped.")

        return _MARKER.sub("", text)

    if (

        not isinstance(data, dict)

        or not isinstance(data.get("answer"), str)

        or not data["answer"].strip()

    ):

        raise ValueError("Answer LLM returned an invalid structured response.")

    answer_text = _expand_grouped_markers(data["answer"])

    in_text = {int(m.group(1)) for m in _MARKER.finditer(answer_text)}

    valid = {n for n in in_text if 1 <= n <= evidence_count}

    invalid_in_text = in_text - valid

    if invalid_in_text:

        logger.warning(

            "Stripped out-of-range citation markers: %s",

            sorted(invalid_in_text),

        )

    declared_raw = data.get("citations")

    declared: set[int] = set()

    if isinstance(declared_raw, list):

        for value in declared_raw:

            if (

                isinstance(value, int)

                and not isinstance(value, bool)

                and 1 <= value <= evidence_count

            ):

                declared.add(value)

            else:

                logger.warning("Rejected invalid citation index: %r", value)

    if declared != valid:

        logger.warning(

            "Citation mismatch: declared=%s markers_in_text=%s",

            sorted(declared),

            sorted(valid),

        )

    def _keep(match: re.Match[str]) -> str:

        return match.group(0) if int(match.group(1)) in valid else ""

    return _MARKER.sub(_keep, answer_text)





def generate_answer(

    *,

    query: str,

    evidence: list[Any],

    conversation_history: list[dict[str, Any]] | None = None,

) -> str:

    """Generate the complete user-facing answer in one request."""

    payload, evidence_count = _build_payload(

        query=query,

        evidence=evidence,

        conversation_history=conversation_history,

    )

    response = router.generate(

        task="answer_generation",

        messages=[

            {"role": "system", "content": SYSTEM_PROMPT},

            {"role": "user", "content": payload},

        ],

        temperature=0.0,

        max_tokens=ANSWER_MAX_OUTPUT_TOKENS,

        response_format=ANSWER_RESPONSE_FORMAT,

        include_reasoning=False,

    )

    try:

        content = response.choices[0].message.content

    except (AttributeError, IndexError) as exc:

        raise ValueError("Answer LLM returned an invalid response.") from exc

    if not content or not content.strip():

        raise ValueError("Answer LLM returned an empty answer.")

    answer = _parse_answer_response(content, evidence_count=evidence_count)

    return _normalize_currency(

        _clean_answer(answer),

        _currency_symbol_for(evidence),

    )





def generate_answer_stream(

    *,

    query: str,

    evidence: list[Any],

    conversation_history: list[dict[str, Any]] | None = None,

) -> Iterator[str]:

    """

    Generate the answer using a true provider-level text stream.

    The yielded values are raw provider text deltas. Under the structured

    output contract these deltas are JSON text, so the caller must

    accumulate them and pass the full text through _parse_answer_response

    before persisting or displaying the answer.

    """

    payload, _ = _build_payload(

        query=query,

        evidence=evidence,

        conversation_history=conversation_history,

    )

    stream = router.generate_stream(

        task="answer_generation",

        messages=[

            {"role": "system", "content": SYSTEM_PROMPT},

            {"role": "user", "content": payload},

        ],

        temperature=0.0,

        max_tokens=ANSWER_STREAM_MAX_OUTPUT_TOKENS,

        include_reasoning=False,

    )

    yielded_any = False

    for delta in stream:

        if not isinstance(delta, str):

            continue

        if not delta:

            continue

        yielded_any = True

        yield delta

    if not yielded_any:

        raise ValueError("Answer LLM returned an empty streaming response.")