"""
retrieval/services/bm25.py

A small, dependency-free BM25 implementation for QueryPilot.

Why BM25 exists here:
    Dense retrieval is good at semantic similarity, but lexical retrieval
    is valuable for exact names, IDs, product codes, error codes, headings,
    and other terms where literal matching matters.

Design:
    - The BM25 corpus is built per workspace at query time.
    - The workspace is expected to be small (currently max 3 documents).
    - No persistent BM25 index is maintained.
"""

import math
import re
from dataclasses import dataclass
from typing import Sequence

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase lexical tokens.

    This intentionally uses a simple tokenizer because BM25 only needs
    consistent term boundaries. We are not trying to perform linguistic
    normalization or semantic interpretation here.
    """
    if not text:
        return []

    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class BM25Result:
    """
    Represents one BM25 match.

    Attributes:
        index:
            Position of the document in the original BM25 corpus.
        score:
            Raw BM25 score.
    """

    index: int
    score: float


class BM25Retriever:
    """
    Lightweight BM25 scorer.

    The implementation follows the standard BM25 structure:

        score(D, Q) =
            sum(
                IDF(term)
                * TF component
            )

    Parameters:
        k1:
            Controls term-frequency saturation.

        b:
            Controls document-length normalization.
    """

    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        self._tokenized_documents = [tokenize(document) for document in self.documents]

        self._document_lengths = [len(tokens) for tokens in self._tokenized_documents]

        self._avg_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

        self._term_frequencies = [
            self._build_term_frequency(tokens) for tokens in self._tokenized_documents
        ]

        self._document_frequency = self._build_document_frequency()

        self._idf = self._build_idf()

    @staticmethod
    def _build_term_frequency(
        tokens: Sequence[str],
    ) -> dict[str, int]:
        """
        Count how many times each term occurs in one document.
        """
        frequencies: dict[str, int] = {}

        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

        return frequencies

    def _build_document_frequency(self) -> dict[str, int]:
        """
        Count in how many documents each term appears.

        A term contributes at most once per document to document frequency.
        """
        frequencies: dict[str, int] = {}

        for tokens in self._tokenized_documents:
            for term in set(tokens):
                frequencies[term] = frequencies.get(term, 0) + 1

        return frequencies

    def _build_idf(self) -> dict[str, float]:
        """
        Compute BM25 IDF values.

        The +0.5 formulation keeps the score numerically stable for
        common terms and avoids undefined behavior for edge cases.
        """
        document_count = len(self.documents)

        if document_count == 0:
            return {}

        idf: dict[str, float] = {}

        for term, document_frequency in self._document_frequency.items():
            value = (document_count - document_frequency + 0.5) / (
                document_frequency + 0.5
            )

            idf[term] = math.log(1.0 + value)

        return idf

    def get_scores(
        self,
        query: str,
    ) -> list[float]:
        """
        Score every document against the query.

        Returns:
            A score for every corpus document, in corpus order.
        """
        query_terms = tokenize(query)

        if not self.documents or not query_terms:
            return [0.0] * len(self.documents)

        scores = [0.0] * len(self.documents)

        for document_index, term_frequencies in enumerate(self._term_frequencies):
            document_length = self._document_lengths[document_index]

            for term in query_terms:
                term_frequency = term_frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                idf = self._idf.get(term, 0.0)

                denominator = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * (
                        document_length / self._avg_document_length
                        if self._avg_document_length > 0
                        else 0.0
                    )
                )

                tf_component = (term_frequency * (self.k1 + 1.0)) / denominator

                scores[document_index] += idf * tf_component

        return scores

    def top_k(
        self,
        query: str,
        k: int = 10,
    ) -> list[BM25Result]:
        """
        Return the highest-scoring BM25 documents.

        Zero-score documents are excluded because they contain none of
        the query terms and therefore add no lexical evidence.
        """
        if k <= 0:
            return []

        scores = self.get_scores(query)

        ranked = [
            BM25Result(index=index, score=score)
            for index, score in enumerate(scores)
            if score > 0.0
        ]

        ranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked[:k]
