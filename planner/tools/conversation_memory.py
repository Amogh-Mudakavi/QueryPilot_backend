"""
planner/tools/conversation_memory.py

Bounded conversation-memory retrieval for QueryPilot.

Purpose:
    Retrieve the smallest useful slice of the current conversation
    needed to resolve the user's latest request.

The tool does NOT:
    - search documents
    - search Chroma
    - query SQLite
    - call the web
    - call an LLM

Design (v2):

    Conversation memory has exactly four jobs, in this order of
    precedence:

        1. "What did you say earlier?" -> return the last assistant
           message directly. This is not retrieval, it's a lookup.

        2. "What department is HE in?" / "What about THAT?" ->
           resolve a pronoun/demonstrative to whatever entity was
           most recently talked about, and return the turn(s) where
           that entity was introduced -- not the turn most lexically
           similar to the pronoun (a pronoun by definition shares no
           words with its antecedent).

        3. "What about the SECOND one?" -> resolve a positional/
           ordinal reference against an ordered trail of entities
           mentioned across the conversation (documents, named
           people, etc), not against word overlap.

        4. Anything else that still needs older context -> lexical
           overlap search, same idea as v1, kept as a fallback for
           genuinely topical lookups ("what did we say about the
           refund policy earlier") rather than as the primary
           mechanism for reference resolution, which it is
           structurally the wrong tool for.

    On top of all of that, the most recent few turns are always
    included, since most real follow-ups are answered by "the last
    thing that was said" without needing any resolution logic at
    all.

    This keeps follow-up handling useful without injecting the
    entire conversation into every agent iteration, and without
    ever calling an LLM to do it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from planner.services.history import split_history

RECENT_MESSAGES = 6
MAX_RELEVANT_OLDER_MESSAGES = 4
MAX_TOTAL_MESSAGES = 10

TOKEN_PATTERN = re.compile(
    r"\b\w+\b",
    re.UNICODE,
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "do",
    "does",
    "did",
    "i",
    "me",
    "my",
    "you",
    "your",
    "it",
    "this",
    "that",
    "he",
    "she",
    "they",
    "we",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
}


PREVIOUS_ANSWER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhat did (you|u) (say|tell me|mean)\b",
        r"\bwhat was (your|the) (previous|last|earlier) (answer|response|message)\b",
        r"\brepeat (that|it|yourself)\b",
        r"\bsay (that|it) again\b",
        r"\byour (previous|last|earlier) (answer|response|message)\b",
        r"\bwhat.*(you|u).*(just|earlier).*(say|said)\b",
        r"\bwhat was the answer (you|u) (gave|provided|said)\b",
        r"\bthe answer (you|u) (gave|provided)\b",
    ]
]


RECAP_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhat (did|have) we (discuss|discussed|talk|talked|cover|covered|say|said|do|done)\b",
        r"\bsummari[sz]e\b.{0,40}\b(conversation|chat|discussion|we|so far|earlier)\b",
        r"\b(recap|sum up)\b",
        r"\bwhat have we\b",
        r"\b(earlier|before) (in|during) (this|our) (chat|conversation)\b",
    ]
]


PRONOUN_PATTERN = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|it|its|that|this|"
    r"the other one|the other)\b",
    re.IGNORECASE,
)

ORDINAL_PATTERN = re.compile(
    r"\b(first|second|third|fourth|fifth|last|other|previous|"
    r"1st|2nd|3rd|4th|5th)\b.{0,15}\b"
    r"(one|file|doc|document|report|sheet|upload)\b",
    re.IGNORECASE,
)

ORDINAL_WORD_TO_INDEX = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}


ENTITY_PATTERNS = [
    re.compile(
        r"\b(document|file|doc|report|sheet|upload)s?\s*"
        r"([A-Za-z0-9]{1,3}|\"[^\"]{1,60}\"|'[^']{1,60}')",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\"']?([\w\-]{2,60}\.(?:xlsx|xls|csv|pdf|docx|doc|txt|md))[\"']?",
        re.IGNORECASE,
    ),
]

COMMON_CAPITALIZED_NON_ENTITIES = {
    "What",
    "How",
    "Why",
    "When",
    "Where",
    "Who",
    "Which",
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Can",
    "Could",
    "Would",
    "Should",
    "Please",
    "Thanks",
    "Hi",
    "Hello",
    "Okay",
    "Yes",
    "No",
    "Sure",
    "Here",
    "There",
    "It",
    "If",
    "So",
    "But",
    "And",
    "Also",
    "Note",
    "Currently",
    "Unfortunately",
    "Yes",
    "No",
    "Ok",
}

CAPITALIZED_WORD_PATTERN = re.compile(r"\b([A-Z][a-z]{2,20})\b")


@dataclass(frozen=True)
class MemoryItem:
    """One selected conversation message."""

    index: int
    role: str
    content: str
    score: float


@dataclass(frozen=True)
class EntityMention:
    """One occurrence of a candidate entity in the conversation."""

    index: int
    text: str


def _matches_recap_request(query: str) -> bool:
    return any(pattern.search(query) for pattern in RECAP_PATTERNS)


def conversation_memory(
    *,
    query: str,
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Public entry point.

    - The rolling summary (a leading system message) is split off first so
      it is never mistaken for an "older message" and never sliced away.
    - Recap questions ("what did we discuss?") get the whole recent
      conversation.
    - Everything else goes through the existing selection logic.
    - If a summary exists it is always returned first.
    """

    query = query.strip()

    if not query:
        raise ValueError("conversation_memory query cannot be empty.")

    summary_message, regular_history = split_history(conversation_history or [])

    if _matches_recap_request(query):
        normalized = _normalize_history(regular_history)
        selected = normalized[-MAX_TOTAL_MESSAGES:]
        result: dict[str, Any] = {
            "messages": [
                {"role": item["role"], "content": item["content"]} for item in selected
            ],
            "message_count": len(selected),
            "used_recent_messages": len(selected),
            "used_relevant_older_messages": 0,
            "resolution": "recap",
        }
    else:
        result = dict(
            _select_messages(
                query=query,
                conversation_history=regular_history,
            )
        )

    if summary_message is not None:
        summary_text = str(summary_message.get("content", "")).strip()
        if summary_text:
            result["messages"] = [{"role": "system", "content": summary_text}] + list(
                result.get("messages", [])
            )
            result["message_count"] = len(result["messages"])
            result["has_summary"] = True

    return result


def _select_messages(
    *,
    query: str,
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Retrieve the most useful previous conversation messages.

    Args:
        query:
            Current user message.

        conversation_history:
            Conversation messages in chronological order, each a
            dict of the form {"role": "user"|"assistant", "content": str}.

    Returns:
        A bounded memory packet containing selected messages, plus
        flags describing which resolution path(s) fired so the
        agent/caller can tell why these messages were chosen.
    """

    query = query.strip()

    if not query:
        raise ValueError("conversation_memory query cannot be empty.")

    if not conversation_history:
        return _empty_result()

    normalized = _normalize_history(conversation_history)

    if not normalized:
        return _empty_result()

    if _matches_previous_answer_request(query):
        last_assistant = _last_assistant_message(normalized)
        if last_assistant is not None:
            return {
                "messages": [
                    {
                        "role": last_assistant["role"],
                        "content": last_assistant["content"],
                    }
                ],
                "message_count": 1,
                "used_recent_messages": 0,
                "used_relevant_older_messages": 0,
                "resolution": "previous_answer",
            }

    recent = normalized[-RECENT_MESSAGES:]
    recent_indices = {item["index"] for item in recent}
    older = [item for item in normalized if item["index"] not in recent_indices]

    resolution_notes: list[str] = []
    anchor_indices: set[int] = set()

    if ORDINAL_PATTERN.search(query):
        trail = _build_entity_trail(normalized)
        ordinal_anchor = _resolve_ordinal(query, trail)
        if ordinal_anchor is not None:
            anchor_indices.add(ordinal_anchor)
            resolution_notes.append("ordinal_reference")

    if PRONOUN_PATTERN.search(query) and not anchor_indices:
        trail = _build_entity_trail(normalized)
        if trail:
            anchor_indices.add(trail[-1].index)
            resolution_notes.append("pronoun_reference")

    expanded_anchor_indices: set[int] = set()
    normalized_by_index = {item["index"]: item for item in normalized}
    for idx in anchor_indices:
        expanded_anchor_indices.add(idx)
        for neighbor_idx in (idx - 1, idx + 1):
            if neighbor_idx in normalized_by_index:
                expanded_anchor_indices.add(neighbor_idx)

    anchor_messages = [
        normalized_by_index[idx]
        for idx in expanded_anchor_indices
        if idx in normalized_by_index and idx not in recent_indices
    ]

    remaining_older = [
        item for item in older if item["index"] not in expanded_anchor_indices
    ]
    lexical_budget = max(0, MAX_RELEVANT_OLDER_MESSAGES - len(anchor_messages))
    selected_lexical = (
        _select_relevant_older_messages(
            query=query,
            messages=remaining_older,
            limit=lexical_budget,
        )
        if lexical_budget > 0
        else []
    )
    if selected_lexical:
        resolution_notes.append("lexical_fallback")

    selected_older = anchor_messages + selected_lexical

    selected = _merge_in_chronological_order(
        recent=recent,
        older=selected_older,
    )

    selected = _cap_total_messages(
        selected=selected,
        recent_indices=recent_indices,
        anchor_indices=expanded_anchor_indices,
        limit=MAX_TOTAL_MESSAGES,
    )

    return {
        "messages": [
            {"role": item["role"], "content": item["content"]} for item in selected
        ],
        "message_count": len(selected),
        "used_recent_messages": len(recent),
        "used_relevant_older_messages": len(selected_older),
        "resolution": ",".join(resolution_notes) if resolution_notes else "recent_only",
    }


def _empty_result() -> dict[str, Any]:
    return {
        "messages": [],
        "message_count": 0,
        "used_recent_messages": 0,
        "used_relevant_older_messages": 0,
        "resolution": "empty",
    }


def _normalize_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only valid conversation messages."""

    normalized: list[dict[str, Any]] = []

    for index, message in enumerate(history):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not isinstance(content, str):
            continue

        role = role.strip()
        content = content.strip()

        if role not in {"user", "assistant", "system"}:
            continue
        if not content:
            continue

        normalized.append({"index": index, "role": role, "content": content})

    return normalized


def _last_assistant_message(
    normalized: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for item in reversed(normalized):
        if item["role"] == "assistant":
            return item
    return None


def _matches_previous_answer_request(query: str) -> bool:
    return any(pattern.search(query) for pattern in PREVIOUS_ANSWER_PATTERNS)


def _build_entity_trail(
    normalized: list[dict[str, Any]],
) -> list[EntityMention]:
    """
    Build an ordered list of candidate entity mentions across the
    whole conversation, in first-seen-per-message order.

    This is deliberately simple pattern matching (proper nouns,
    "Document A"-style references, quoted filenames) -- not real
    NER. It only needs to be good enough to answer "what was the
    most recently mentioned thing" and "what was the Nth mentioned
    thing", which is what pronoun/ordinal resolution actually needs.
    """

    trail: list[EntityMention] = []

    for item in normalized:
        content = item["content"]
        found_in_message: set[str] = set()

        for pattern in ENTITY_PATTERNS:
            for match in pattern.finditer(content):
                text = match.group(0).strip()
                if not text or text.strip("\"'") in COMMON_CAPITALIZED_NON_ENTITIES:
                    continue
                key = text.lower()
                if key in found_in_message:
                    continue
                found_in_message.add(key)
                trail.append(EntityMention(index=item["index"], text=text))

        for text in _find_capitalized_entities(content):
            key = text.lower()
            if key in found_in_message:
                continue
            found_in_message.add(key)
            trail.append(EntityMention(index=item["index"], text=text))

    return trail


def _find_capitalized_entities(content: str) -> list[str]:
    """
    Find capitalized proper-noun-looking words in a message,
    excluding the first word of each sentence (a plain heuristic
    sentence split on '. ! ?'), since sentence-initial capitals are
    almost always just normal sentence-starters, not names.
    """

    found: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", content)

    for sentence in sentences:
        words = sentence.split()

        for word in words[1:]:
            match = CAPITALIZED_WORD_PATTERN.match(word.strip(".,!?;:\"'()"))
            if not match:
                continue
            text = match.group(1)
            if text in COMMON_CAPITALIZED_NON_ENTITIES:
                continue
            found.append(text)

    return found


def _resolve_ordinal(
    query: str,
    trail: list[EntityMention],
) -> int | None:
    """
    Resolve "the second one" / "the other file" style references
    against the entity trail. Returns the message index where that
    ordinal-position entity was introduced, or None if unresolved.
    """

    if not trail:
        return None

    lowered = query.lower()

    if "other" in lowered and "second" not in lowered:
        distinct = _distinct_entities_in_order(trail)
        if len(distinct) >= 2:
            return distinct[-2].index
        return None

    for word, position in ORDINAL_WORD_TO_INDEX.items():
        if re.search(rf"\b{word}\b", lowered):
            distinct = _distinct_entities_in_order(trail)
            if position < len(distinct):
                return distinct[position].index
            return None

    if "last" in lowered:
        distinct = _distinct_entities_in_order(trail)
        if distinct:
            return distinct[-1].index

    return None


def _distinct_entities_in_order(
    trail: list[EntityMention],
) -> list[EntityMention]:
    """Collapse the trail to first-occurrence-per-entity, in order."""

    seen: set[str] = set()
    distinct: list[EntityMention] = []
    for mention in trail:
        key = mention.text.lower()
        if key in seen:
            continue
        seen.add(key)
        distinct.append(mention)
    return distinct


def _select_relevant_older_messages(
    *,
    query: str,
    messages: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """
    Select older messages using lexical overlap plus recency.

    Fallback mechanism only -- used to fill remaining budget for
    genuinely topical lookups, not relied on for reference
    resolution (see fast paths above).
    """

    query_tokens = _meaningful_tokens(query)
    if not query_tokens or limit <= 0:
        return []

    scored: list[MemoryItem] = []
    total = len(messages)

    for item in messages:
        message_tokens = _meaningful_tokens(item["content"])
        overlap = len(query_tokens & message_tokens)
        if overlap == 0:
            continue

        overlap_ratio = overlap / len(query_tokens)
        recency_bonus = (item["index"] + 1) / max(total, 1)
        score = 0.80 * overlap_ratio + 0.20 * recency_bonus

        scored.append(
            MemoryItem(
                index=item["index"],
                role=item["role"],
                content=item["content"],
                score=score,
            )
        )

    scored.sort(key=lambda item: (item.score, item.index), reverse=True)
    selected_indices = {item.index for item in scored[:limit]}

    return [item for item in messages if item["index"] in selected_indices]


def _meaningful_tokens(text: str) -> set[str]:
    """Tokenize conversation text and remove common stopwords."""

    tokens = {token.lower() for token in TOKEN_PATTERN.findall(text)}
    return {token for token in tokens if token not in STOPWORDS}


def _merge_in_chronological_order(
    *,
    recent: list[dict[str, Any]],
    older: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge selected messages without duplicates while preserving order."""

    by_index: dict[int, dict[str, Any]] = {}

    for item in older:
        by_index[item["index"]] = item
    for item in recent:
        by_index[item["index"]] = item

    return [by_index[index] for index in sorted(by_index)]


def _cap_total_messages(
    *,
    selected: list[dict[str, Any]],
    recent_indices: set[int],
    anchor_indices: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    """
    Enforce the hard message cap without silently discarding
    reference-resolution anchors.

    Priority when something has to be dropped: keep recent messages
    and anchor messages first (these are the ones doing actual
    resolution work), drop lexical-fallback messages first if the
    budget is tight. Only if recent+anchors alone exceed the limit
    do we fall back to a chronological slice (oldest dropped first).
    """

    if len(selected) <= limit:
        return selected

    must_keep = [
        item
        for item in selected
        if item["index"] in recent_indices or item["index"] in anchor_indices
    ]
    fallback_only = [
        item
        for item in selected
        if item["index"] not in recent_indices and item["index"] not in anchor_indices
    ]

    if len(must_keep) >= limit:
        return sorted(must_keep, key=lambda i: i["index"])[-limit:]

    room_for_fallback = limit - len(must_keep)
    kept_fallback = fallback_only[-room_for_fallback:] if room_for_fallback > 0 else []

    merged = {item["index"]: item for item in must_keep + kept_fallback}
    return [merged[i] for i in sorted(merged)]
