"""
retrieval/services/keys.py

Format-independent helpers for stable cross-document keys (Employee ID,
invoice number, SKU, ...). Pure functions, no Django / Chroma / SQLite.

A "key" is an identifier-looking token that is found NEXT TO a resolved
entity in whichever source it lives in (PDF chunk text or a SQLite row) and
that can then be used to locate the companion evidence in any other source.

Nothing here knows about employees, salaries or specific documents.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

MAX_KEY_LENGTH = 64

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
_SEP = rf"[-_/{_DASHES}]"

_HYPHEN_ID = re.compile(
    rf"(?<![A-Za-z0-9])[A-Za-z]{{1,10}}{_SEP}\d{{2,}}"
    rf"(?:{_SEP}[A-Za-z0-9]+)*(?![A-Za-z0-9])"
)

_COMPACT_ID = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,4}\d{2,}(?![A-Za-z0-9])")

_LABELLED_ID = re.compile(
    r"(?i)\b(?:id|code|key|ref|reference|number)\b\s*[:#=]\s*"
    r"(?P<v>(?=[A-Za-z0-9_\/\-]*\d)[A-Za-z0-9][A-Za-z0-9_\/\-]{2,})"
)

_ID_COLUMN_TOKENS = frozenset({"id", "code", "key", "ref", "reference"})
_FILE_LIKE = re.compile(r"\.(pdf|xlsx?|xlsm|csv|docx?|txt)\b", re.I)


def _clean_token(token: str) -> str:
    for dash in _DASHES:
        token = token.replace(dash, "-")
    return token.strip(" .,;:)(-_/")


def dedupe_keys(keys: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for key in keys:
        key = _clean_token(str(key or ""))
        norm = normalize_key(key)

        if not key or len(key) > MAX_KEY_LENGTH or len(norm) < 3 or norm in seen:
            continue

        seen.add(norm)
        out.append(key)

    return out


def normalize_key(key: str) -> str:
    """Separator/case-insensitive identity: 'EMP-1150' == 'emp1150'."""
    return re.sub(r"[^A-Za-z0-9]", "", str(key or "")).upper()


def extract_id_tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, token) for every identifier-looking token, by position."""
    if not text:
        return []

    found: dict[tuple[int, int], str] = {}

    for pattern in (_HYPHEN_ID, _COMPACT_ID):
        for m in pattern.finditer(text):
            found[(m.start(), m.end())] = _clean_token(m.group(0))

    for m in _LABELLED_ID.finditer(text):
        found[(m.start("v"), m.end("v"))] = _clean_token(m.group("v"))

    spans = sorted(found, key=lambda s: (s[0], -(s[1] - s[0])))
    result: list[tuple[int, int, str]] = []

    last_end = -1
    for start, end in spans:
        if start < last_end and end <= last_end:
            continue

        if found[(start, end)]:
            result.append((start, end, found[(start, end)]))

        last_end = max(last_end, end)

    return result


def key_pattern(key: str) -> re.Pattern[str]:
    """Regex that finds the key regardless of separators/case."""
    parts = re.findall(r"[A-Za-z]+|\d+", str(key))

    if not parts:
        return re.compile(r"(?!x)x")

    body = r"[\W_]{0,3}".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.I)


def text_contains_key(text: str, key: str) -> bool:
    return bool(text) and key_pattern(key).search(text) is not None


def anchor_pattern(anchor: str) -> re.Pattern[str] | None:
    """Whole-token, order-preserving, case-insensitive entity-name matcher."""
    cleaned = re.sub(r"['’]s\b", "", str(anchor or ""))
    tokens = re.findall(r"\w+", cleaned.lower())

    if not tokens:
        return None

    body = r"[\W_]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.I)


def is_anchor_reference(text: str) -> bool:
    """False for file names and ID-only strings (those are sources/keys)."""
    text = str(text or "").strip()
    return bool(text) and not _FILE_LIKE.search(text)


def keys_from_text(text: str, anchors: Iterable[str], window: int = 300) -> list[str]:
    """
    Identifiers that sit next to a mention of an anchor (entity name) in
    prose. The nearest identifier inside the same paragraph wins, so a chunk
    that holds several people's records still yields the right person's key.
    """
    anchors = [a for a in anchors if a]

    if not text or not anchors:
        return []

    tokens = extract_id_tokens(text)

    if not tokens:
        return []

    breaks = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)] + [len(text)]
    found: list[str] = []

    for anchor in anchors:
        pattern = anchor_pattern(anchor)

        if pattern is None:
            continue

        for m in pattern.finditer(text):
            s, e = m.span()

            ps = max(b for b in breaks if b <= s)
            pe = min(b for b in breaks if b >= e)

            lo, hi = max(ps, s - window), min(pe, e + window)
            cand = [t for t in tokens if t[0] >= lo and t[1] <= hi]

            if not cand:
                cand = [t for t in tokens if t[0] >= s - window and t[1] <= e + window]

            if not cand:
                continue

            def distance(t: tuple[int, int, str]) -> int:
                if t[0] < e and t[1] > s:
                    return 0
                return min(abs(t[0] - e), abs(s - t[1]))

            found.append(min(cand, key=distance)[2])

    return dedupe_keys(found)


def is_id_column(name: str) -> bool:
    words = re.findall(
        r"[a-z0-9]+",
        re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name)).lower(),
    )
    return any(w in _ID_COLUMN_TOKENS for w in words)


def _is_full_id(value: str) -> bool:
    value = value.strip()

    for pattern in (_HYPHEN_ID, _COMPACT_ID):
        m = pattern.fullmatch(value)
        if m:
            return True

    return False


def keys_from_rows(
    rows: Iterable[Mapping[str, Any]], anchors: Iterable[str]
) -> list[str]:
    """
    Identifier values from rows that mention an anchor. Values from columns
    named like an ID come first, then any cell that is shaped like an ID.
    """
    patterns = [p for p in (anchor_pattern(a) for a in anchors if a) if p is not None]

    if not patterns:
        return []

    id_cols: list[str] = []
    shaped: list[str] = []

    for row in rows or []:
        if not isinstance(row, Mapping):
            continue

        joined = " | ".join("" if v is None else str(v) for v in row.values())

        if not any(p.search(joined) for p in patterns):
            continue

        for column, value in row.items():
            text = "" if value is None else str(value).strip()

            if not text or len(text) > MAX_KEY_LENGTH:
                continue

            if is_id_column(column):
                id_cols.append(text)
            elif _is_full_id(text):
                shaped.append(text)

    return dedupe_keys(id_cols + shaped)
