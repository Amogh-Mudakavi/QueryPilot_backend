"""
planner/services/answer.py

Final answer generation for QueryPilot.

This service is intentionally separate from the LangGraph reasoning
agent.

The agent gathers and finalizes evidence.
This service turns that evidence into a user-facing answer.

The context sent to the answer model is explicitly bounded.
"""

from __future__ import annotations

import json
from typing import Any

from llm.router import router

from planner.prompts.answer import SYSTEM_PROMPT
import re


MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12000

MAX_EVIDENCE_ITEMS = 8
MAX_CHARS_PER_EVIDENCE = 6000
MAX_TOTAL_EVIDENCE_CHARS = 24000


def _bound_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep recent conversation history within a hard size budget.
    """

    bounded: list[dict[str, Any]] = []
    total_chars = 0

    for message in history[
        -MAX_HISTORY_MESSAGES:
    ]:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        remaining = (
            MAX_HISTORY_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        content = content[:remaining]

        bounded.append(
            {
                "role": role,
                "content": content,
            }
        )

        total_chars += len(content)

    return bounded


def _bound_evidence(
    evidence: list[Any],
) -> list[dict[str, Any]]:
    """
    Keep evidence within a hard context budget.

    Source metadata is preserved.
    Only evidence content is truncated.
    """

    bounded: list[dict[str, Any]] = []
    total_chars = 0

    for item in evidence[
        :MAX_EVIDENCE_ITEMS
    ]:

        if not isinstance(
            item,
            dict,
        ):
            continue

        remaining = (
            MAX_TOTAL_EVIDENCE_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        bounded_item = dict(item)

        content = str(
            bounded_item.get(
                "content",
                "",
            )
        )

        content_limit = min(
            MAX_CHARS_PER_EVIDENCE,
            remaining,
        )

        bounded_item["content"] = (
            content[:content_limit]
        )

        total_chars += len(
            bounded_item["content"]
        )

        bounded.append(
            bounded_item
        )

    return bounded


def generate_answer(
    *,
    query: str,
    evidence: list[Any],
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Generate the final user-facing answer.

    Args:
        query:
            Original user question.

        evidence:
            Final evidence packet produced by the agent.

        conversation_history:
            Relevant conversation context.

    Returns:
        Final answer text.
    """

    if not query.strip():
        raise ValueError(
            "Answer generation query cannot be empty."
        )

    if not evidence:
        raise ValueError(
            "Answer generation requires evidence."
        )

    bounded_history = _bound_history(
        conversation_history or []
    )

    bounded_evidence = _bound_evidence(
        evidence
    )

    payload = {
        "user_query": query,
        "conversation_history": bounded_history,
        "evidence": bounded_evidence,
    }

    response = router.generate(
        task="answer_generation",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    try:
        content = (
            response
            .choices[0]
            .message
            .content
        )
    except (
        AttributeError,
        IndexError,
    ) as exc:
        raise ValueError(
            "Answer LLM returned an invalid response."
        ) from exc

    if not content or not content.strip():
        raise ValueError(
            "Answer LLM returned an empty answer."
        )

    # Remove any internal/debug markers that must never be shown
    # to end users. These markers use fullwidth brackets like
    # 【...】 and are introduced by internal tooling; strip them
    # from the final LLM output while preserving the answer text.
    cleaned = re.sub(r'【[^】]*】', '', content)
    # Also collapse excessive whitespace created by removals.
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()