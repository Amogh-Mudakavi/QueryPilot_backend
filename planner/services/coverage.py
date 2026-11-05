"""Evidence coverage checks and earlier-answer context.

Two small, deterministic helpers that sit next to evidence collection. Neither
gates the answer; both only add a factual note that tells the answer stage
what the collection step did and did not produce.

* ``document_coverage_note``  which sources the plan identified as needed but
                              contributed no evidence at all;
* ``requirement_coverage_note`` which planner requirements share no wording
                              with any collected evidence;
* ``prior_conclusions_note``  what was stated earlier in the conversation, for
                              questions that refer back to it ("reconcile your
                              previous answers").

Nothing here knows any domain, column name or file format. The module has no
framework imports, so it can be tested on its own.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

# A question that points back at what the assistant said earlier. Longer
# questions are normally treated as self-contained and get no history, which
# is wrong for these: the earlier answer is the subject of the question.
EARLIER_ANSWER_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:your|the)\s+(?:previous|earlier|prior|last|above|first|second)\s+"
    r"(?:answers?|responses?|replies|results?|calculations?|figures|numbers|"
    r"findings|conclusions?|analysis|statements?)"
    r"|reconcil\w*"
    r"|previously\s+(?:identified|established|determined|stated|calculated|"
    r"computed|found|reported|listed|shown|said)"
    r"|(?:identified|established|determined|stated|calculated|computed|found|"
    r"reported|listed|shown)\s+(?:above|earlier|previously|before)"
    r"|as\s+you\s+(?:said|stated|calculated|mentioned)"
    r"|you\s+(?:said|stated|calculated|identified|reported)\s+(?:earlier|before|previously)"
    r"|(?:two|both)\s+(?:of\s+)?(?:your\s+)?answers"
    r"|discrepanc\w*\s+between\s+(?:your|the)"
    r")\b",
    re.IGNORECASE,
)

_CITATION = re.compile(r"\s?\[\d{1,3}\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-_/]{3,}", re.IGNORECASE)

# Words that appear in nearly every requirement and say nothing about its
# subject. Matching on them would make every requirement look covered.
_GENERIC_WORDS = frozenset(
    """
    about above after against all also amount amounts and any are because been
    before being below between both but calculate calculated calculation can
    check compare compared could data detail details determine does each find
    for from give given have into its list made make many more most must need
    needed not only other over per question rate rates required requirement
    result results same should show some state than that the their them then
    there these they this those total under using value values was were what
    when where which while whether will with within would
    """.split()
)


def references_earlier_answer(query: str) -> bool:
    """True when the query asks about what was said earlier in the chat."""
    return bool(EARLIER_ANSWER_REFERENCE.search(str(query or "")))


# --------------------------------------------------------------------------
# which documents an evidence item came from
# --------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def evidence_document_ids(item: Any) -> set[int]:
    """Document ids an evidence item (table result or text passage) drew on."""
    found: set[int] = set()
    if not isinstance(item, Mapping):
        return found

    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    content = item.get("content") if isinstance(item.get("content"), Mapping) else {}

    for value in (
        item.get("document_id"),
        metadata.get("document_id"),
        content.get("document_id"),
    ):
        parsed = _as_int(value)
        if parsed is not None:
            found.add(parsed)

    for source in item.get("sources") or []:
        if isinstance(source, Mapping):
            parsed = _as_int(source.get("document_id"))
            if parsed is not None:
                found.add(parsed)
    return found


def _has_data(item: Any) -> bool:
    """Whether an item carries source data (not a note or an error)."""
    if not isinstance(item, Mapping):
        return False
    if str(item.get("source_type") or "").lower() == "evidence_note":
        return False
    status = str(item.get("result_status") or "SUCCESS").upper()
    return status == "SUCCESS"


# --------------------------------------------------------------------------
# document coverage
# --------------------------------------------------------------------------


def document_coverage_note(
    needed_document_ids: Iterable[Any],
    titles: Mapping[int, str],
    evidence: Sequence[Any],
) -> dict[str, Any] | None:
    """Note naming needed sources that contributed no evidence.

    ``needed_document_ids`` are the sources the plan identified (resolved
    references and planner-declared supporting sources). A source that was
    needed but reached the answer stage with nothing is reported, so the
    answer says it could not be established instead of filling the gap.
    """
    needed = [doc for doc in (_as_int(v) for v in needed_document_ids or []) if doc is not None]
    if not needed:
        return None

    present: set[int] = set()
    for item in evidence or []:
        if _has_data(item):
            present |= evidence_document_ids(item)

    missing = [doc for doc in dict.fromkeys(needed) if doc not in present]
    if not missing:
        return None

    names = ", ".join(
        str(titles.get(doc) or f"source {doc}") for doc in missing[:8]
    )
    return {
        "source_type": "evidence_note",
        "note_kind": "coverage_ledger",
        "result_status": "NOT_FOUND",
        "content": (
            "Evidence check. These sources were identified as needed, but no "
            f"evidence from them reached the answer stage: {names}. Do not "
            "assume anything about their content. For any part of the "
            "question that depends on them, say it could not be established "
            "from the retrieved evidence."
        ),
    }


# --------------------------------------------------------------------------
# requirement coverage
# --------------------------------------------------------------------------


def _distinctive_tokens(requirement: Mapping[str, Any]) -> set[str]:
    parts: list[str] = [str(requirement.get("description") or "")]
    for key in ("entities", "queries"):
        value = requirement.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if v)
    tokens = {
        match.group(0).casefold().strip("-_/")
        for match in _WORD.finditer(" ".join(parts))
    }
    return {t for t in tokens if len(t) >= 4 and t not in _GENERIC_WORDS}


def _evidence_text(item: Any) -> str:
    if not isinstance(item, Mapping) or not _has_data(item):
        return ""
    content = item.get("content")
    if isinstance(content, str):
        return content.casefold()
    if isinstance(content, Mapping):
        pieces: list[str] = []
        for key in ("document_title", "table_title", "table_context", "columns"):
            value = content.get(key)
            if value:
                pieces.append(str(value))
        for row in content.get("rows") or []:
            pieces.append(str(row))
        for aggregate in content.get("aggregates") or []:
            pieces.append(str(aggregate))
        return " ".join(pieces).casefold()
    return str(content or "").casefold()


def requirement_coverage_note(
    requirements: Sequence[Any],
    evidence: Sequence[Any],
) -> dict[str, Any] | None:
    """Note listing planner requirements that share no wording with evidence.

    Deliberately conservative: a requirement is flagged only when NONE of its
    distinctive words appear anywhere in the collected evidence. It is a hint
    that retrieval may have missed something, never a verdict.
    """
    if not requirements:
        return None
    blob = " ".join(_evidence_text(item) for item in evidence or [])
    if not blob.strip():
        return None

    flagged: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            continue
        tokens = _distinctive_tokens(requirement)
        if len(tokens) < 2:
            continue
        if not any(token in blob for token in tokens):
            label = " ".join(str(requirement.get("description") or "").split())
            if label:
                flagged.append(label[:140])

    if not flagged:
        return None
    return {
        "source_type": "evidence_note",
        "note_kind": "coverage_ledger",
        "result_status": "SUCCESS",
        "content": (
            "Evidence check (automatic, keyword based, so it can miss "
            "matches). No collected evidence shares wording with these "
            "requested items: "
            + "; ".join(flagged[:6])
            + ". If the answer needs facts for them that are not in the "
            "evidence, say they could not be established."
        ),
    }


# --------------------------------------------------------------------------
# earlier answers
# --------------------------------------------------------------------------

_PRIOR_MAX_ANSWERS = 2
_PRIOR_MAX_CHARS = 2000


def prior_conclusions_note(
    query: str,
    history: Sequence[Any] | None,
) -> dict[str, Any] | None:
    """What the assistant said earlier, for questions about those answers.

    Without this the answer stage never sees its own earlier figures for a
    long question, so it cannot compare against them and tends to invent an
    explanation for any difference. The note is labelled as non-evidence.
    """
    if not references_earlier_answer(query) or not history:
        return None

    answers: list[tuple[str, str]] = []
    last_user = ""
    for message in history:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        text = _CITATION.sub("", str(message.get("content") or "")).strip()
        if not text:
            continue
        if role == "user":
            last_user = text
        elif role in {"assistant", "ai"}:
            answers.append((last_user, text))

    if not answers:
        return None

    parts: list[str] = []
    for question, answer in answers[-_PRIOR_MAX_ANSWERS:]:
        piece = f"Question: {' '.join(question.split())[:300]}\nAnswer given: {answer[:_PRIOR_MAX_CHARS]}"
        parts.append(piece)

    return {
        "source_type": "evidence_note",
        "note_kind": "prior_conclusions",
        "result_status": "SUCCESS",
        "content": (
            "prior_conclusions. Earlier in this conversation the assistant "
            "stated the following. These are earlier outputs, not source "
            "facts. Compare your result with them; where the figures differ, "
            "say so and explain only what the evidence supports.\n\n"
            + "\n\n".join(parts)
        ),
    }
