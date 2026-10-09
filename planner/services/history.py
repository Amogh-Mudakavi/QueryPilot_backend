"""Conversation history helpers, including explicit correction replay."""

from __future__ import annotations

import re
from typing import Any

SUMMARY_KIND = "summary"
SUMMARY_PREFIX = "CONVERSATION MEMORY"


def is_summary_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("kind") == SUMMARY_KIND:
        return True
    return message.get("role") == "system" and str(
        message.get("content", "")
    ).startswith(SUMMARY_PREFIX)


def split_history(history: list[dict[str, Any]] | None):
    summary = None
    regular = []
    for message in history or []:
        if is_summary_message(message):
            if summary is None:
                summary = message
        else:
            regular.append(message)
    return summary, regular


def _summary_body(message: dict[str, Any]) -> str:
    content = str(message.get("content", ""))
    if content.startswith(SUMMARY_PREFIX) and "\n\n" in content:
        return content.split("\n\n", 1)[1].strip()
    return content.strip()


def _role(message: Any) -> str:
    return (
        str(message.get("role", "")).strip().lower()
        if isinstance(message, dict)
        else ""
    )


_CORRECTION_PHRASES = (
    "that's wrong",
    "that is wrong",
    "wrong answer",
    "incorrect answer",
    "that's incorrect",
    "that is incorrect",
    "this is incorrect",
    "it's incorrect",
    "it is incorrect",
    "you are incorrect",
    "your answer is wrong",
    "you are wrong",
    "you're wrong",
    "you got it wrong",
    "i asked for",
    "i asked you",
    "not what i asked",
    "that's not what i asked",
    "that is not what i asked",
    "give me the correct answer",
    "give the correct answer",
    "correct the answer",
    "correct your answer",
    "redo it",
    "redo this",
    "try again",
    "try it again",
    "recalculate",
    "recheck",
    "you misunderstood",
    "misunderstood my question",
)

_CORRECTION_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(p) for p in _CORRECTION_PHRASES) + r")(?!\w)"
)


CORRECTION_MAX_WORDS = 20

_GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "thx"}


def _normalize(text: Any) -> str:
    value = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(value.split())


def is_correction_query(query: str | None) -> bool:
    normalized = _normalize(query)
    if not normalized or len(normalized.split()) > CORRECTION_MAX_WORDS:
        return False
    return bool(_CORRECTION_RE.search(normalized))


def _is_greeting(text: Any) -> bool:
    return re.sub(r"[^\w\s]", "", _normalize(text)).strip() in _GREETINGS


def previous_user_query(history: list[dict[str, Any]] | None) -> str | None:
    """
    Most recent substantive USER message. Works whether or not the current
    message is already in history (the current correction is skipped because
    it is a correction). Earlier corrections and greetings are skipped too, so
    "try again" twice in a row still replays the original task.
    Assistant messages are never read.
    """
    _, regular = split_history(history)
    for message in reversed(regular):
        if _role(message) != "user":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if is_correction_query(content) or _is_greeting(content):
            continue
        return content
    return None


def resolve_replay(
    current_query: str,
    history: list[dict[str, Any]] | None,
) -> tuple[str, str] | None:
    """(previous_user_task, effective_query) for an explicit correction, else None."""
    current = str(current_query or "").strip()
    if not is_correction_query(current):
        return None
    previous = previous_user_query(history)
    if not previous:
        return None
    effective = f"{previous}\n\nCorrection from the user: {current}"
    return previous, effective





def bound_history_for_planner(
    history,
    *,
    max_recent=6,
    max_user_chars=400,
    max_assistant_chars=240,
    max_summary_chars=900,
    max_total_chars=3200,
):
    summary, regular = split_history(history)
    result = []
    budget = max_total_chars
    if summary is not None:
        body = _summary_body(summary)[:max_summary_chars]
        if body:
            result.append(
                {"role": "system", "content": "SUMMARY OF OLDER CONVERSATION: " + body}
            )
            budget -= len(body)
    kept = []
    for message in reversed(regular[-max_recent:]):
        if not isinstance(message, dict):
            continue
        role = _role(message)
        limit = max_assistant_chars if role == "assistant" else max_user_chars
        content = str(message.get("content", ""))[:limit]
        if not content:
            continue
        if kept and budget - len(content) < 0:
            break
        budget -= len(content)
        kept.append({"role": role, "content": content})
    kept.reverse()
    return result + kept


def bound_history_for_answer(
    history,
    *,
    current_query=None,
    max_recent_user=2,
    max_chars_per_message=250,
    max_summary_chars=400,
    max_total_chars=900,
):
    summary, regular = split_history(history)
    result = []
    budget = max_total_chars
    if summary is not None and budget > 0:
        body = _summary_body(summary)[: min(max_summary_chars, budget)]
        if body:
            result.append(
                {"role": "system", "content": "SUMMARY OF OLDER CONVERSATION: " + body}
            )
            budget -= len(body)
    kept = []
    users = [m for m in regular if _role(m) == "user"]
    if current_query is None:
        users = users[:-1]
    elif users and _normalize(users[-1].get("content")) == _normalize(current_query):
        users = users[:-1]
    for message in reversed(users[-max_recent_user:]):
        if budget <= 0:
            break
        content = str(message.get("content", ""))[: min(max_chars_per_message, budget)]
        if not content:
            continue
        kept.append({"role": "user", "content": content})
        budget -= len(content)
    kept.reverse()
    return result + kept
