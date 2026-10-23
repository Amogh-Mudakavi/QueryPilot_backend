"""
planner/services/query_interpretation.py

Turns the user's question into small, typed intents for deterministic evidence.

This module only reads the QUESTION TEXT. It never sees a table, a column or a
row, so nothing in it can depend on a dataset. Evidence computation lives in
evidence.py and consumes the intents produced here.

Interpreted
  * selection   earliest / latest / oldest / newest / first-to-join / last-to-join
  * aggregate   average / total, plus the phrase that names the measured field
  * numeric predicates   between, at least, above, "or higher", ...

Whatever is not recognised is simply not returned. The caller then leaves the
result unresolved and the existing higher-level flow handles the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXT_WORDS = 4
MAX_METRIC_WORDS = 4

# Include common currency prefixes in the numeric token so comparisons such
# as ``above Rs. 12,00,000`` retain their amount during parsing.
_NUMBER = (
    r"[-+]?(?:(?:[₹$€£]\s*|(?:rs\.?|inr|usd|eur|gbp)\s*))?"
    r"\d[\d,]*(?:\.\d+)?"
)
_WORD = re.compile(r"[A-Za-z0-9']+")


_SEGMENT_BREAK = re.compile(
    r"(?<=[?!;])\s+|(?<!Rs\.)(?<=[.])\s+|,?\s*\bthen\b,?\s*",
    re.IGNORECASE,
)
_LEADING_THEN = re.compile(r"^\s*then\b[,\s]*", re.IGNORECASE)

_LABELLED_LEVEL_PREFIX = re.compile(
    r"\b(?:tier|level|grade|stage|step|phase|chapter|section|part|version|"
    r"class|rank|priority|category|band|group|type|slab)\s*$",
    re.IGNORECASE,
)


def _is_labelled_level_predicate(text: str, match: re.Match[str]) -> bool:
    """Recognize ordinal labels such as ``Tier 4 or higher``.

    A labelled level is a category, not a scalar threshold to match against
    an arbitrary numeric column. This remains domain-neutral: it applies to
    common labelled stages, grades, versions, bands, and similar concepts.
    """
    return bool(_LABELLED_LEVEL_PREFIX.search(text[: match.start()].rstrip()))


def split_segments(query: str) -> list[str]:
    """Split a question into sentences / 'then' steps; each one owns its own
    filters, so a condition in one step never leaks into another."""
    parts = _SEGMENT_BREAK.split(str(query or ""))
    cleaned = [_LEADING_THEN.sub("", part).strip() for part in parts if part]
    return [part for part in cleaned if part]


@dataclass(frozen=True)
class NumericPredicate:
    operator: str
    lower: float
    upper: float | None
    context: str

    def matches(self, value: float) -> bool:
        if self.operator == "gte":
            return value >= self.lower
        if self.operator == "gt":
            return value > self.lower
        if self.operator == "lte":
            return value <= self.lower
        if self.operator == "lt":
            return value < self.lower
        if self.upper is None:
            return False
        if self.operator == "between_inclusive":
            return self.lower <= value <= self.upper
        if self.operator == "between_exclusive":
            return self.lower < value < self.upper
        return False


_BETWEEN = re.compile(
    rf"\bbetween\s+(?P<low>{_NUMBER})\s+(?:and|to)\s+(?P<high>{_NUMBER})"
    r"(?:\s+(?P<mode>inclusive|exclusive))?",
    re.IGNORECASE,
)


_ONE_SIDED = tuple(
    (operator, re.compile(pattern, re.IGNORECASE))
    for operator, pattern in (
        (
            "gte",
            rf"(?P<n>{_NUMBER})\s*(?:or|and)\s+(?:higher|above|more|greater|over|up)\b",
        ),
        ("lte", rf"(?P<n>{_NUMBER})\s*(?:or|and)\s+(?:lower|below|less|under|down)\b"),
        (
            "gte",
            rf"(?:\bat\s+least|\bno\s+less\s+than|\bminimum\s+of|>=|≥)\s*(?P<n>{_NUMBER})",
        ),
        (
            "lte",
            rf"(?:\bat\s+most|\bno\s+more\s+than|\bmaximum\s+of|<=|≤)\s*(?P<n>{_NUMBER})",
        ),
        (
            "gt",
            rf"(?:\babove|\bover|\bgreater\s+than|\bhigher\s+than|\bmore\s+than|\bexceeding|>)\s*(?P<n>{_NUMBER})",
        ),
        (
            "lt",
            rf"(?:\bbelow|\bunder|\bless\s+than|\blower\s+than|\bfewer\s+than|<)\s*(?P<n>{_NUMBER})",
        ),
    )
)


def _to_float(text: str) -> float:
    # Remove textual currency abbreviations before stripping punctuation. If
    # ``Rs.`` is stripped afterwards, its period is mistaken for a decimal
    # point (``Rs. 12,00,000`` would become ``.1200000``).
    without_currency = re.sub(
        r"(?i)\b(?:rs\.?|inr|usd|eur|gbp)", "", str(text or "")
    )
    return float(re.sub(r"[^\d.+-]", "", without_currency))


def _context_before(text: str, position: int) -> str:
    return " ".join(_WORD.findall(text[:position])[-CONTEXT_WORDS:])


def _between_predicate(match: re.Match[str], text: str) -> NumericPredicate:
    low, high = sorted((_to_float(match.group("low")), _to_float(match.group("high"))))
    exclusive = str(match.group("mode") or "").lower() == "exclusive"
    return NumericPredicate(
        operator="between_exclusive" if exclusive else "between_inclusive",
        lower=low,
        upper=high,
        context=_context_before(text, match.start()),
    )


def extract_numeric_predicates(text: str) -> list[NumericPredicate]:
    """Every numeric condition in `text`, in reading order. 'between X and Y'
    is inclusive unless the text says 'exclusive'."""
    found: list[tuple[int, int, NumericPredicate]] = []

    def is_claimed(start: int, end: int) -> bool:
        return any(
            start < other_end and other_start < end
            for other_start, other_end, _ in found
        )

    for match in _BETWEEN.finditer(text):
        found.append((match.start(), match.end(), _between_predicate(match, text)))

    for operator, pattern in _ONE_SIDED:
        for match in pattern.finditer(text):
            if is_claimed(match.start(), match.end()):
                continue
            # ``Stage 3 or higher`` is an ordinal category. Leave it to the
            # evidence/rule reasoning path rather than treating 3 as a value
            # that must exist in a source column.
            if _is_labelled_level_predicate(text, match):
                continue
            predicate = NumericPredicate(
                operator=operator,
                lower=_to_float(match.group("n")),
                upper=None,
                context=_context_before(text, match.start()),
            )
            found.append((match.start(), match.end(), predicate))

    found.sort(key=lambda entry: entry[0])
    return [predicate for _, _, predicate in found]


@dataclass(frozen=True)
class SelectionIntent:
    direction: str
    phrase: str
    scope_text: str
    segment_index: int


_JOIN_VERBS = r"(?:join|joined|hired|hire|started|start)"
_SELECTION_PATTERNS = (
    (
        "min",
        re.compile(
            rf"\b(?:earliest|oldest|first\s+to\s+(?:be\s+)?{_JOIN_VERBS}"
            rf"|{_JOIN_VERBS}\s+first|first\s+{_JOIN_VERBS})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "max",
        re.compile(
            rf"\b(?:latest|newest|most\s+recent(?:ly)?|last\s+to\s+(?:be\s+)?{_JOIN_VERBS}"
            rf"|{_JOIN_VERBS}\s+last|last\s+{_JOIN_VERBS})\b",
            re.IGNORECASE,
        ),
    ),
)


def _detect_selection(segments: list[str]) -> SelectionIntent | None:
    """One sentence with one direction. Anything murkier stays unresolved."""
    hits: list[tuple[int, str, list[tuple[str, str]]]] = []
    for index, segment in enumerate(segments):
        found = [
            (direction, match.group(0))
            for direction, pattern in _SELECTION_PATTERNS
            for match in pattern.finditer(segment)
        ]
        if found:
            hits.append((index, segment, found))

    if len(hits) != 1:
        return None
    index, segment, found = hits[0]
    if len({direction for direction, _ in found}) != 1:
        return None
    return SelectionIntent(
        direction=found[0][0],
        phrase=found[0][1],
        scope_text=segment,
        segment_index=index,
    )


@dataclass(frozen=True)
class AggregateIntent:
    function: str
    metric_phrase: str
    scope_text: str
    segment_index: int


_AGGREGATE_WORD = re.compile(r"\b(average|avg|mean|total|sum)\b", re.IGNORECASE)
_AGGREGATE_FUNCTION = {
    "average": "average",
    "avg": "average",
    "mean": "average",
    "total": "sum",
    "sum": "sum",
}
_METRIC_STOP = re.compile(
    r"\b(?:of|for|among|across|in|from|where|whose|who|with|between|by|that|having"
    r"|than|and|is|are|was|were)\b|[,.;?!()]",
    re.IGNORECASE,
)
_FILLER_WORDS = frozenset({"the", "their", "its", "his", "her", "our", "my", "a", "an"})
_POPULATION_MARKER = re.compile(
    r"\b(?:of|among|across|where|whose|with|between|at\s+least|at\s+most|above|below)\b"
    r"|or\s+(?:higher|lower)",
    re.IGNORECASE,
)


def _metric_phrase(segment: str, start: int) -> str:
    """'average base salary of Product ...' -> 'base salary'."""
    rest = re.sub(r"^\s+of\s+", " ", segment[start:], count=1, flags=re.IGNORECASE)
    head = _METRIC_STOP.split(rest, maxsplit=1)[0]
    words = [word for word in _WORD.findall(head) if word.lower() not in _FILLER_WORDS]
    return " ".join(words[:MAX_METRIC_WORDS])


def _detect_aggregate(segments: list[str]) -> AggregateIntent | None:
    """Pick the sentence that DEFINES a population for the aggregate (markers
    such as 'of', 'whose', 'between'), not a later sentence that merely lists
    the aggregate as an output field."""
    scored: list[tuple[int, AggregateIntent]] = []
    for index, segment in enumerate(segments):
        predicate_count = len(extract_numeric_predicates(segment))
        for match in _AGGREGATE_WORD.finditer(segment):
            phrase = _metric_phrase(segment, match.end())
            if not phrase:
                continue
            markers = len(_POPULATION_MARKER.findall(segment[match.start() :]))
            intent = AggregateIntent(
                function=_AGGREGATE_FUNCTION[match.group(1).lower()],
                metric_phrase=phrase,
                scope_text=segment,
                segment_index=index,
            )
            scored.append((markers + predicate_count, intent))

    if not scored:
        return None
    best = max(score for score, _ in scored)
    leaders = [intent for score, intent in scored if score == best]
    if best == 0 or len(leaders) != 1:
        return None
    return leaders[0]


_COMPARISON_WORDS = re.compile(
    r"\b(?:compare[sd]?|comparison|difference|differ(?:s|ence)?|versus|vs)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryInterpretation:
    selection: SelectionIntent | None = None
    aggregate: AggregateIntent | None = None
    wants_comparison: bool = False


def interpret_query(query: str) -> QueryInterpretation:
    """Parse the question once. The selection and the aggregate each get their
    own sentence, so their filters cannot contaminate each other."""
    segments = split_segments(query)
    selection = _detect_selection(segments)
    aggregate = _detect_aggregate(segments)

    if selection and aggregate and selection.segment_index == aggregate.segment_index:
        selection, aggregate = None, None

    return QueryInterpretation(
        selection=selection,
        aggregate=aggregate,
        wants_comparison=bool(_COMPARISON_WORDS.search(str(query or ""))),
    )
