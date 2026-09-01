"""
Structure-aware chunker for QueryPilot's ingestion pipeline.

Takes text_nodes produced by the Llama Adapter (one dict per page, with
LlamaParse's markdown formatting preserved) and turns them into chunks
that are safe to embed and store in Chroma.

Why structure-aware instead of plain fixed-size splitting:
Financial PDFs (10-Ks, annual reports) have real hierarchy - "Revenue
Recognition" is a subsection of "Notes to Financial Statements", and a
chunk that loses that context is a chunk that retrieves badly. So we
split on headings first, and only fall back to size-based splitting
inside a section when that section is too big to embed as one chunk.
"""

import re
from dataclasses import dataclass, field


# Rough chars-per-token ratio for English text. We deliberately do not
# pull in tiktoken here - the project rule is no external chunking
# frameworks, and this approximation is within ~10-15% of real GPT/Claude
# tokenizers, which is accurate enough for sizing chunks. Being off by a
# little just means a chunk is slightly bigger or smaller than the target;
# it never breaks correctness.
CHARS_PER_TOKEN = 4

# Target and hard-limit chunk sizes, in estimated tokens.
DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_SENTENCES = 2

# Matches markdown headings LlamaParse emits, e.g. "# Revenue Growth",
# "## Note 3. Income Taxes". Captures the heading level (number of #s)
# and the heading text separately.
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Fallback for text that reaches the chunker without markdown heading
# markers - e.g. if the adapter ever passes through raw paragraph_title
# text without prefixing it with '#'. Two patterns:
#   1. Numbered section headings: "1.", "1.1", "3.2.1 Income Taxes"
#   2. Short ALL-CAPS lines, which annual reports use heavily for
#      section titles: "CONSOLIDATED STATEMENTS OF OPERATIONS"
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\.?\s+\S.*$")
ALL_CAPS_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9 ,&/\-()]{3,80}$")

# Splits on sentence-ending punctuation followed by whitespace. Not
# perfect (won't handle "Mr. Smith" correctly), but good enough for
# chunk-boundary purposes where an occasional over-split sentence is
# harmless - it only affects where we cut, not what gets lost.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Section:
    """
    A logical piece of the document - all the text that falls under one
    heading, before it gets cut into embed-sized chunks.

    Pages is a list, not just start/end, because a section rarely lines
    up exactly on page boundaries - it can span several pages, and
    tracking every page it touches keeps citation back to the source
    PDF accurate even if a section skips content oddly.
    """

    heading: str
    pages: list[int] = field(default_factory=list)
    content: str = ""


class StructureAwareChunker:
    """
    Splits parsed PDF text into chunks that respect document structure.

    Two-stage design:
      1. build_sections() groups the raw per-page text into sections,
         one per heading, using the page's markdown headings as
         boundaries.
      2. chunk() takes those sections and, for any section too large to
         embed as a single chunk, recursively splits it (paragraph ->
         sentence) with a small sentence-based overlap between the
         resulting pieces.

    Splitting on headings first and only falling back to size-based
    cuts within a section is what keeps a chunk from mixing content
    from two unrelated sections - which is the main way naive fixed-size
    chunking hurts retrieval quality on structured documents.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ) -> None:
        """
        Args:
            max_tokens: Soft upper bound, in estimated tokens, for a
                single chunk. Sections under this size become one chunk
                each; larger sections get recursively split.
            overlap_sentences: Number of trailing sentences from a chunk
                to repeat at the start of the next chunk, so a query that
                lands near a cut point still gets full context.
        """
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, text_nodes: list[dict]) -> list[dict]:
        """
        Main entry point. Turns raw per-page text nodes into final chunks.

        Args:
            text_nodes: [{"page": 1, "text": "..."}, ...] from the Llama
                Adapter, one entry per PDF page, in page order.

        Returns:
            List of chunk dicts, each shaped as:
                {
                    "page_start": int,
                    "page_end": int,
                    "heading": str,
                    "chunk_index": int,
                    "text": str,
                }
            chunk_index is a running count across the whole document, so
            downstream code can restore document order without re-sorting.
        """
        for node in text_nodes:
            if "page" not in node or "text" not in node:
                raise ValueError(
                    "Each text_node must have 'page' and 'text' keys, "
                    f"got: {sorted(node.keys())}"
                )

        sections = self.build_sections(text_nodes)

        chunks: list[dict] = []
        chunk_index = 0
        for section in sections:
            section_chunks = self._split_section(section)
            for text in section_chunks:
                chunks.append(
                    {
                        "page_start": section.pages[0],
                        "page_end": section.pages[-1],
                        "heading": section.heading,
                        "chunk_index": chunk_index,
                        "text": text,
                    }
                )
                chunk_index += 1

        return chunks

    def build_sections(self, text_nodes: list[dict]) -> list[Section]:
        """
        Groups per-page text into sections, splitting wherever a new
        heading is found.

        Runs as one pass over every line of every page, in order. Each
        line is checked for whether it's a heading; if it is, we close
        off the current section and start a new one. Everything else
        gets appended to whatever section is currently open.

        A section can span multiple pages (a subsection that starts near
        the bottom of page 4 and continues onto page 5 is one section,
        not two) - that's why we track pages as a list and only append a
        page number when it's new.

        Args:
            text_nodes: [{"page": 1, "text": "..."}, ...] in page order.

        Returns:
            List of Section objects in document order. If the document
            has content before its first heading (e.g. a title block),
            that content becomes a section with heading "Untitled
            Section" rather than being dropped.
        """
        sections: list[Section] = []
        current: Section | None = None

        for node in text_nodes:
            page = node["page"]
            lines = node["text"].split("\n")

            for line in lines:
                heading_text = self._detect_heading(line)

                if heading_text is not None:
                    if current is not None and current.content.strip():
                        # Current section has real content under it -
                        # close it out and start a fresh section for the
                        # new heading.
                        sections.append(current)
                        current = Section(heading=heading_text, pages=[page])
                    elif current is not None:
                        # Current section is still empty (e.g. we just
                        # saw "# Apple Inc." and this line is the very
                        # next heading, "## CONSOLIDATED STATEMENTS OF
                        # OPERATIONS"). Two headings in a row with no
                        # content between them are really one combined
                        # heading, not an empty section followed by a
                        # new one - so fold the old heading text into
                        # the new heading instead of discarding it.
                        current = Section(
                            heading=f"{current.heading} - {heading_text}",
                            pages=current.pages,
                        )
                    else:
                        current = Section(heading=heading_text, pages=[page])
                    continue

                if not line.strip():
                    continue

                if current is None:
                    current = Section(heading="Untitled Section", pages=[page])

                if page not in current.pages:
                    current.pages.append(page)
                current.content += line + "\n"

        if current is not None:
            sections.append(current)

        return sections

    def _detect_heading(self, line: str) -> str | None:
        """
        Checks whether a single line is a heading, and if so returns its
        clean text (heading markers stripped).

        Detection order, most to least reliable:
          1. Markdown headings ("# ", "## ", ...) - this is the primary
             signal, since the Llama Adapter is expected to carry over
             LlamaParse's own paragraph_title / heading labels as
             markdown '#' prefixes. Trusting LlamaParse's own layout
             model beats re-guessing structure from plain text.
          2. Numbered section headings ("1.1 Revenue Recognition") -
             common in 10-Ks and policy documents even when markdown
             markup isn't present.
          3. Short ALL-CAPS lines ("CONSOLIDATED STATEMENTS OF
             OPERATIONS") - a heading style annual reports use heavily.

        Returns:
            The heading text with markers stripped, or None if the line
            is not a heading.
        """
        stripped = line.strip()
        if not stripped:
            return None

        md_match = MARKDOWN_HEADING_PATTERN.match(stripped)
        if md_match:
            return md_match.group(2).strip("*").strip()

        if NUMBERED_HEADING_PATTERN.match(stripped) and len(stripped) <= 100:
            return stripped

        if ALL_CAPS_HEADING_PATTERN.match(stripped):
            return stripped.title()

        return None

    def _split_section(self, section: Section) -> list[str]:
        """
        Turns one section's content into one or more chunk texts.

        If the section already fits under max_tokens, it becomes exactly
        one chunk - no point splitting further and losing context for no
        reason. Otherwise it's recursively split (paragraphs, then
        sentences if a paragraph is still too big) and reassembled into
        chunks with sentence overlap between them.

        Every chunk gets the section heading prepended before it's
        returned. This is deliberate: an embedded chunk of "...grew 12%
        year over year..." is ambiguous on its own, but "Revenue Growth:
        ...grew 12% year over year..." tells the embedding model, and
        later the retriever, what the number is actually about.

        Args:
            section: A single Section with its full content.

        Returns:
            List of chunk text strings, each already prefixed with the
            section heading.
        """
        content = section.content.strip()
        if not content:
            return []

        if self._estimate_tokens(content) <= self.max_tokens:
            return [self._with_heading(section.heading, content)]

        pieces = self._recursive_split(content)
        overlapped = self._add_sentence_overlap(pieces)
        return [self._with_heading(section.heading, piece) for piece in overlapped]

    def _recursive_split(self, text: str) -> list[str]:
        """
        Cuts text down to max_tokens-sized pieces, preferring to split on
        paragraph breaks and only falling back to sentence-level splits
        when a single paragraph is still too big.

        This mirrors how a person would actually shorten a document:
        keep whole paragraphs together where possible, and only cut
        mid-paragraph when there's no other option. Splitting on
        arbitrary character counts instead would risk cutting a sentence
        (or a number mid-figure, which is especially bad for financial
        text) right down the middle.

        Args:
            text: The full text of a section that's too big for one chunk.

        Returns:
            List of text pieces, each under (or close to) max_tokens.
        """
        paragraphs = [p for p in text.split("\n") if p.strip()]

        pieces: list[str] = []
        current_piece = ""

        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_tokens(paragraph)

            if paragraph_tokens > self.max_tokens:
                if current_piece:
                    pieces.append(current_piece.strip())
                    current_piece = ""
                pieces.extend(self._split_by_sentences(paragraph))
                continue

            candidate = f"{current_piece}\n{paragraph}" if current_piece else paragraph
            if self._estimate_tokens(candidate) > self.max_tokens:
                pieces.append(current_piece.strip())
                current_piece = paragraph
            else:
                current_piece = candidate

        if current_piece:
            pieces.append(current_piece.strip())

        return pieces

    def _split_by_sentences(self, paragraph: str) -> list[str]:
        """
        Splits a single oversized paragraph into max_tokens-sized pieces
        along sentence boundaries.

        Only called when a whole paragraph exceeds max_tokens on its own
        (long-winded legal/policy text, dense MD&A sections). Splitting
        by sentence here, instead of by raw character count, means we
        never end a chunk in the middle of a sentence even in the worst
        case.

        Args:
            paragraph: A single paragraph's text, too large for one chunk.

        Returns:
            List of text pieces built from whole sentences.
        """
        sentences = SENTENCE_SPLIT_PATTERN.split(paragraph)

        pieces: list[str] = []
        current_piece = ""

        for sentence in sentences:
            candidate = f"{current_piece} {sentence}".strip()
            if current_piece and self._estimate_tokens(candidate) > self.max_tokens:
                pieces.append(current_piece.strip())
                current_piece = sentence
            else:
                current_piece = candidate

        if current_piece:
            pieces.append(current_piece.strip())

        return pieces

    def _add_sentence_overlap(self, pieces: list[str]) -> list[str]:
        """
        Prepends the last few sentences of each piece onto the start of
        the next piece, so consecutive chunks share a bit of context.

        Sentence-based overlap instead of character-based: cutting the
        last N characters of a chunk to glue onto the next one will,
        most of the time, chop a sentence in half - so the "overlap"
        context is often unreadable and doesn't actually help retrieval.
        Carrying over whole sentences means the repeated context is
        always a complete thought, and a query that would have landed
        right at a cut boundary still retrieves a chunk with full
        context on both sides of that boundary.

        The number of sentences actually carried over is capped by an
        overlap token budget (a fraction of max_tokens), not just by
        overlap_sentences. Financial and legal text often has long
        sentences (40-60+ tokens each), so blindly prepending
        overlap_sentences full sentences can nearly double a chunk that
        was already sized right up to max_tokens. Capping by token
        budget means overlap shrinks automatically for long sentences
        instead of blowing past the size limit.

        Args:
            pieces: Ordered list of text pieces from _recursive_split.

        Returns:
            Same number of pieces, each (except the first) prefixed with
            as many trailing sentences from the piece before it as fit
            within the overlap budget.
        """
        if len(pieces) <= 1 or self.overlap_sentences <= 0:
            return pieces

        # Overlap should stay a minority of a chunk's budget - otherwise
        # it stops being "a bit of shared context" and starts being most
        # of the chunk. Capped at 25% of max_tokens.
        overlap_token_budget = self.max_tokens // 4

        overlapped = [pieces[0]]

        for i in range(1, len(pieces)):
            previous_sentences = SENTENCE_SPLIT_PATTERN.split(pieces[i - 1].strip())
            candidates = previous_sentences[-self.overlap_sentences :]

            carry_over: list[str] = []
            for sentence in reversed(candidates):
                trial = [sentence] + carry_over
                if self._estimate_tokens(" ".join(trial)) > overlap_token_budget:
                    break
                carry_over = trial

            overlap_text = " ".join(carry_over).strip()
            if overlap_text:
                overlapped.append(f"{overlap_text} {pieces[i]}")
            else:
                overlapped.append(pieces[i])

        return overlapped

    def _with_heading(self, heading: str, content: str) -> str:
        """
        Prefixes a chunk's content with its section heading.

        Kept as its own method (rather than inlined) because it's called
        from two places - the "fits in one chunk" path and the
        recursively-split path - and both need the heading attached the
        same way.

        Args:
            heading: The section heading this content belongs to.
            content: The chunk's body text.

        Returns:
            "{heading}: {content}", ready to embed.
        """
        return f"{heading}: {content.strip()}"

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count for a piece of text.

        Uses a character-count / 4 approximation rather than a real
        tokenizer. This project deliberately avoids pulling in an
        external tokenizer just for chunk sizing - the ~4 chars/token
        ratio is a well-known rule of thumb for English text and is
        accurate enough to keep chunks in the right ballpark. Being off
        by 10-15% only makes a chunk slightly bigger or smaller than
        max_tokens; it doesn't break anything downstream.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // CHARS_PER_TOKEN)