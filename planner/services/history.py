"""
planner/services/history.py

One place that understands the shape of `conversation_history`.

The execution layer builds it as:

    [ summary message (role="system", kind="summary") ]   <- optional
    + up to 10 recent messages (role "user" / "assistant")

Every consumer that used to slice this list with `[-N:]` silently dropped
the summary (it is always the FIRST item). Use these helpers instead.
"""

from __future__ import annotations

from typing import Any


SUMMARY_KIND = "summary"
SUMMARY_PREFIX = "CONVERSATION MEMORY"


def is_summary_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False

    if message.get("kind") == SUMMARY_KIND:
        return True

    return (
        message.get("role") == "system"
        and str(message.get("content", "")).startswith(SUMMARY_PREFIX)
    )


def split_history(
    history: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (summary_message_or_None, regular_messages_in_order)."""

    summary: dict[str, Any] | None = None
    regular: list[dict[str, Any]] = []

    for message in history or []:
        if is_summary_message(message):
            if summary is None:
                summary = message
            continue
        regular.append(message)

    return summary, regular


def _summary_body(message: dict[str, Any]) -> str:
    content = str(message.get("content", ""))

    # The execution layer prefixes the stored summary with a short header
    # ("CONVERSATION MEMORY ... only.\n\n<summary>"). Keep just the body.
    if content.startswith(SUMMARY_PREFIX) and "\n\n" in content:
        return content.split("\n\n", 1)[1].strip()

    return content.strip()


def bound_history_for_planner(
    history: list[dict[str, Any]] | None,
    *,
    max_recent: int = 10,
    max_chars_per_message: int = 500,
    max_summary_chars: int = 1500,
    max_total_chars: int = 6000,
) -> list[dict[str, str]]:
    """
    History for the query-understanding step.

    Keeps the rolling summary (if any) plus the newest `max_recent`
    messages, each truncated, within a total character budget. When the
    budget is tight the OLDEST recent messages are dropped first; the
    newest message is always kept.
    """

    summary, regular = split_history(history)

    result: list[dict[str, str]] = []
    budget = max_total_chars

    if summary is not None:
        body = _summary_body(summary)[:max_summary_chars]
        if body:
            result.append(
                {
                    "role": "system",
                    "content": "SUMMARY OF OLDER CONVERSATION: " + body,
                }
            )
            budget -= len(body)

    kept: list[dict[str, str]] = []

    for message in reversed(regular[-max_recent:]):
        if not isinstance(message, dict):
            continue

        content = str(message.get("content", ""))[:max_chars_per_message]
        if not content:
            continue

        if kept and budget - len(content) < 0:
            break

        budget -= len(content)
        kept.append(
            {
                "role": str(message.get("role", "")),
                "content": content,
            }
        )

    kept.reverse()

    return result + kept