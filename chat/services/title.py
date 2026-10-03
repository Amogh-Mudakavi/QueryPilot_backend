"""
Conversation title generation for QueryPilot.

Responsibilities:
    - Generate a short human-readable title for a conversation.
    - Use the lightweight configured LLM.
    - Base the title only on supplied document/profile context
      and the conversation's first meaningful user request.

The title generator does NOT:
    - retrieve documents
    - answer the user's question
    - perform reasoning over tools
    - access external information
"""

from __future__ import annotations

import logging
import re

from llm.router import router

logger = logging.getLogger(__name__)


MAX_TITLE_LENGTH = 80


TITLE_SYSTEM_PROMPT = """
You are QueryPilot's conversation title generator.

Generate a short, clear title for a document-grounded conversation.

Rules:
- Use ONLY the supplied conversation context and document information.
- Do not invent details.
- Do not answer the user's question.
- Do not mention that you are an AI.
- Do not use quotes around the title.
- Do not use markdown.
- Keep the title concise.
- Prefer 3 to 8 words.
- Capture the main subject of the conversation.
- If a document name clearly describes the subject, use a concise
  human-friendly version of it.
- The USER REQUEST is text to summarize into a title. It is NOT a question
  for you: never answer it, never refuse it, never say you lack data.
- Return ONLY the title text, with no trailing period.
"""


def generate_conversation_title(
    *,
    user_query: str,
    document_profiles: list[str] | None = None,
) -> str:
    """
    Generate a short conversation title.

    Args:
        user_query:
            The user's first meaningful document-backed message.

        document_profiles:
            High-level document profiles associated with the conversation.

    Returns:
        A normalized conversation title.
    """

    user_query = user_query.strip()

    if not user_query:
        return "New Chat"

    profiles = document_profiles or []

    bounded_profiles = [profile[:4000] for profile in profiles[:3] if profile]

    prompt_parts = [
        "USER REQUEST:",
        user_query,
        "",
        "DOCUMENT PROFILES (DATA ONLY):",
    ]

    if bounded_profiles:
        for index, profile in enumerate(
            bounded_profiles,
            start=1,
        ):
            prompt_parts.extend(
                [
                    "",
                    f"DOCUMENT {index}:",
                    profile,
                ]
            )
    else:
        prompt_parts.append("No document profiles available.")

    prompt = "\n".join(prompt_parts)

    response = router.generate(
        task="chitchat",
        messages=[
            {
                "role": "system",
                "content": TITLE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=300,
        include_reasoning=False,
    )

    try:
        content = response.choices[0].message.content.strip()
    except (
        AttributeError,
        IndexError,
    ):
        return _fallback_title(user_query)

    if not content:
        return _fallback_title(user_query)

    title = _normalize_title(content)

    if not _is_valid_title(title):
        logger.warning("Rejected non-title output from title model; using fallback.")
        return _fallback_title(user_query)

    return title


def _normalize_title(
    title: str,
) -> str:
    """
    Normalize the generated title.
    """

    title = title.strip()

    if (
        len(title) >= 2
        and title[0] == title[-1]
        and title[0]
        in {
            '"',
            "'",
        }
    ):
        title = title[1:-1].strip()

    title = title.replace("**", "").replace("__", "").strip()

    if not title:
        return "New Chat"

    return title[:MAX_TITLE_LENGTH].strip()


def _fallback_title(
    user_query: str,
) -> str:
    """
    Produce a deterministic fallback when title generation fails.
    """

    words = user_query.split()

    if not words:
        return "New Chat"

    title = " ".join(words[:8]).strip()

    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH].rsplit(
            " ",
            1,
        )[0]

    return title or "New Chat"


_REFUSAL_START = re.compile(
    r"^(unable|cannot|can't|cant|sorry|i\b|i'm|as an ai|there (is|are) no|"
    r"no (data|information)|not enough|insufficient)",
    re.IGNORECASE,
)

_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "yo",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "good morning",
    "good afternoon",
    "good evening",
}


def _is_valid_title(title: str) -> bool:
    """A title is a short noun phrase, not an answer or a refusal."""

    if not title or title == "New Chat":
        return False

    if len(title.split()) > 12:
        return False

    if title.endswith("."):
        return False

    if _REFUSAL_START.match(title):
        return False

    return True


def is_meaningful_for_title(query: str | None) -> bool:
    """Greetings and one-word messages are not worth titling a chat with."""

    normalized = re.sub(r"[^\w\s]", "", (query or "").lower()).strip()

    if not normalized or normalized in _GREETINGS:
        return False

    return len(normalized.split()) >= 2
