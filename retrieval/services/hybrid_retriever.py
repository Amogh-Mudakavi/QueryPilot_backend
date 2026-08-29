"""
retrieval/services/hybrid_retriever.py

QueryPilot's primary workspace evidence retriever.

Responsibilities:
    1. Dense semantic retrieval from Chroma.
    2. BM25 lexical retrieval over the current workspace.
    3. Rank-based fusion using Reciprocal Rank Fusion (RRF).
    4. Deduplicate candidates returned by both retrieval methods.
    5. Return the strongest 5-10 evidence candidates.
    6. Provide a retrieval-strength signal for the planner.

This class deliberately does NOT:
    - decide whether the question is in scope
    - decide whether SQL is required
    - execute SQLite queries
    - run LangGraph
    - generate answers

Those are planner/agent responsibilities.

Important:
    `confidence` is a retrieval-strength heuristic, NOT a calibrated
    probability that the final answer is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from documents.ingestion.vectors import VectorStore

from .bm25 import BM25Retriever
from .evidence import Evidence


@dataclass
class RetrievalResult:
    """
    Result returned by HybridRetriever.

    Attributes:
        evidence:
            Final top-K evidence items.

        confidence:
            Heuristic retrieval-strength signal in [0, 1].

        candidate_count:
            Number of unique candidates after dense/BM25 fusion.
    """

    evidence: list[Evidence]
    confidence: float
    candidate_count: int


class HybridRetriever:
    """
    Workspace-scoped hybrid retriever.

    Retrieval flow:

        Dense Top-N
              +
        BM25 Top-N
              ↓
        RRF Fusion
              ↓
        Deduplicate
              ↓
        Top-K Evidence

    RRF is used instead of directly combining dense and BM25 raw scores
    because those scores are not naturally comparable. RRF combines
    their ranks instead.

    Current QueryPilot choice:

        dense_k = 20
        bm25_k = 20
        top_k = 5
        rrf_k = 60

    The candidate pool is intentionally larger than the final context.
    This gives both retrieval methods room to contribute before the
    final evidence list is selected.
    """

    DEFAULT_DENSE_K = 20
    DEFAULT_BM25_K = 20
    DEFAULT_TOP_K = 5

    RRF_K = 60

    def __init__(
        self,
        *,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:

        if dense_k <= 0:
            raise ValueError(
                "dense_k must be greater than 0"
            )

        if bm25_k <= 0:
            raise ValueError(
                "bm25_k must be greater than 0"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.top_k = top_k

    def retrieve(
        self,
        *,
        query: str,
        workspace_id: int,
    ) -> RetrievalResult:
        """
        Retrieve the strongest evidence for a query within one workspace.

        Workspace isolation is mandatory. Every retrieval operation is
        filtered by workspace_id.

        Args:
            query:
                User or agent-generated search query.

            workspace_id:
                Current workspace.

        Returns:
            RetrievalResult containing final evidence and a heuristic
            retrieval-strength signal.

        Raises:
            ValueError:
                If query is empty.
        """
        query = query.strip()

        if not query:
            raise ValueError(
                "query must not be empty"
            )

        dense_candidates = self._dense_search(
            query=query,
            workspace_id=workspace_id,
        )

        bm25_candidates = self._bm25_search(
            query=query,
            workspace_id=workspace_id,
        )

        fused_candidates = self._rrf_fuse(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
        )

        final_candidates = fused_candidates[
            : self.top_k
        ]

        evidence = [
            candidate["evidence"]
            for candidate in final_candidates
        ]

        confidence = self._calculate_confidence(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            fused_candidates=fused_candidates,
        )

        return RetrievalResult(
            evidence=evidence,
            confidence=confidence,
            candidate_count=len(fused_candidates),
        )

    def _dense_search(
        self,
        *,
        query: str,
        workspace_id: int,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic retrieval using the existing Chroma knowledge
        collection.

        The collection already contains:
            - text chunks
            - table semantic representations

        Workspace filtering is performed by Chroma.
        """
        collection = (
            VectorStore.knowledge_collection()
        )

        query_embedding = VectorStore.embed(
            [query]
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.dense_k,
            where={
                "workspace_id": workspace_id,
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = (
            results.get("documents", [[]])[0]
        )

        metadatas = (
            results.get("metadatas", [[]])[0]
        )

        distances = (
            results.get("distances", [[]])[0]
        )

        ids = (
            results.get("ids", [[]])[0]
        )

        candidates: list[dict[str, Any]] = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            evidence_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            candidates.append(
                {
                    "id": evidence_id,
                    "text": document,
                    "metadata": metadata,
                    "dense_distance": distance,
                }
            )

        return candidates

    def _bm25_search(
        self,
        *,
        query: str,
        workspace_id: int,
    ) -> list[dict[str, Any]]:
        """
        Perform workspace-scoped BM25 retrieval.

        BM25 is rebuilt in memory for the current workspace.

        This is deliberate because QueryPilot currently limits a
        workspace to a small number of documents. Maintaining a
        persistent BM25 index would add synchronization and deletion
        complexity without a meaningful benefit at this scale.
        """
        collection = (
            VectorStore.knowledge_collection()
        )

        corpus = collection.get(
            where={
                "workspace_id": workspace_id,
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = corpus.get(
            "documents",
            [],
        )

        metadatas = corpus.get(
            "metadatas",
            [],
        )

        ids = corpus.get(
            "ids",
            [],
        )

        if not documents:
            return []

        bm25 = BM25Retriever(
            documents=documents
        )

        matches = bm25.top_k(
            query=query,
            k=self.bm25_k,
        )

        candidates: list[dict[str, Any]] = []

        for match in matches:

            index = match.index

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            evidence_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            candidates.append(
                {
                    "id": evidence_id,
                    "text": documents[index],
                    "metadata": metadata,
                    "bm25_score": match.score,
                }
            )

        return candidates

    def _rrf_fuse(
        self,
        *,
        dense_candidates: list[dict[str, Any]],
        bm25_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion.

        RRF score:

            sum(1 / (k + rank))

        where:
            k = RRF_K
            rank starts at 1

        A candidate appearing near the top of both retrieval lists gets
        support from both signals.

        A candidate appearing in only one list can still survive, which
        is important for exact lexical matches that dense retrieval may
        miss.
        """
        candidates: dict[str, dict[str, Any]] = {}

        for rank, candidate in enumerate(
            dense_candidates,
            start=1,
        ):
            evidence_id = candidate.get("id")

            if evidence_id is None:
                continue

            item = candidates.setdefault(
                evidence_id,
                {
                    "id": evidence_id,
                    "text": candidate.get(
                        "text",
                        "",
                    ),
                    "metadata": candidate.get(
                        "metadata",
                        {},
                    ),
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )

            item["dense_rank"] = rank
            item["dense_distance"] = (
                candidate.get(
                    "dense_distance"
                )
            )

            item["rrf_score"] += (
                1.0
                / (
                    self.RRF_K
                    + rank
                )
            )

        for rank, candidate in enumerate(
            bm25_candidates,
            start=1,
        ):
            evidence_id = candidate.get("id")

            if evidence_id is None:
                continue

            item = candidates.setdefault(
                evidence_id,
                {
                    "id": evidence_id,
                    "text": candidate.get(
                        "text",
                        "",
                    ),
                    "metadata": candidate.get(
                        "metadata",
                        {},
                    ),
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )

            item["bm25_rank"] = rank
            item["bm25_score"] = (
                candidate.get(
                    "bm25_score"
                )
            )

            # Preserve BM25 text/metadata if the candidate was not
            # already discovered by dense retrieval.
            if not item["text"]:
                item["text"] = candidate.get(
                    "text",
                    "",
                )

            if not item["metadata"]:
                item["metadata"] = candidate.get(
                    "metadata",
                    {},
                )

            item["rrf_score"] += (
                1.0
                / (
                    self.RRF_K
                    + rank
                )
            )

        fused = []

        for item in candidates.values():

            evidence = self._build_evidence(
                content=item["text"],
                score=item["rrf_score"],
                metadata=item["metadata"],
            )

            fused.append(
                {
                    **item,
                    "evidence": evidence,
                }
            )

        fused.sort(
            key=lambda candidate: (
                candidate["rrf_score"],
                self._rank_tiebreak(candidate),
            ),
            reverse=True,
        )

        return fused

    @staticmethod
    def _rank_tiebreak(
        candidate: dict[str, Any],
    ) -> float:
        """
        Small deterministic tiebreak value.

        Candidates present in both retrieval lists receive additional
        support because they have both ranks.
        """
        score = 0.0

        if candidate.get("dense_rank") is not None:
            score += (
                1.0
                / (
                    1.0
                    + candidate["dense_rank"]
                )
            )

        if candidate.get("bm25_rank") is not None:
            score += (
                1.0
                / (
                    1.0
                    + candidate["bm25_rank"]
                )
            )

        return score

    def _build_evidence(
        self,
        *,
        content: str,
        score: float,
        metadata: dict[str, Any],
    ) -> Evidence:
        """
        Convert a retrieved candidate into the common Evidence type.
        """
        source_type = metadata.get(
            "type",
            "unknown",
        )

        if source_type == "text":
            normalized_source_type = (
                "text_chunk"
            )
        elif source_type == "table":
            normalized_source_type = (
                "table_metadata"
            )
        else:
            normalized_source_type = (
                source_type
            )

        workspace_id = metadata.get(
            "workspace_id"
        )

        document_id = metadata.get(
            "document_id"
        )

        if workspace_id is None:
            raise ValueError(
                "Retrieved evidence is missing "
                "workspace_id"
            )

        if document_id is None:
            raise ValueError(
                "Retrieved evidence is missing "
                "document_id"
            )

        return Evidence(
            content=content,
            score=float(score),
            source_type=normalized_source_type,
            workspace_id=int(
                workspace_id
            ),
            document_id=int(
                document_id
            ),
            document_name=metadata.get(
                "document_name"
            ),
            metadata=metadata,
        )

    @classmethod
    def _calculate_confidence(
        cls,
        *,
        dense_candidates: list[
            dict[str, Any]
        ],
        bm25_candidates: list[
            dict[str, Any]
        ],
        fused_candidates: list[
            dict[str, Any]
        ],
    ) -> float:
        """
        Estimate retrieval strength without pretending it is a
        calibrated probability.

        Signals:

            1. Absolute dense similarity of the best candidate.
            2. Whether the best fused candidate is also a strong BM25 hit.
            3. Whether the two retrieval systems agree on multiple
               candidates.

        Because these signals come from different systems, this value is
        intentionally treated as a heuristic for the planner.

        The important behavior is:

            strong semantic match + lexical agreement
                -> high

            weak semantic match with no lexical agreement
                -> low

        This avoids the old behavior where min-max normalization forced
        the best candidate of every query to look like a "1.0".
        """
        if not fused_candidates:
            return 0.0

        best_dense_similarity = 0.0

        if dense_candidates:
            best_distance = (
                dense_candidates[0].get(
                    "dense_distance"
                )
            )

            best_dense_similarity = (
                cls._l2_distance_to_similarity(
                    best_distance
                )
            )

        top_candidate = (
            fused_candidates[0]
        )

        lexical_support = 0.0

        if (
            top_candidate.get(
                "bm25_rank"
            )
            is not None
        ):
            bm25_rank = top_candidate[
                "bm25_rank"
            ]

            if bm25_rank <= 3:
                lexical_support = 1.0
            elif bm25_rank <= 10:
                lexical_support = 0.6
            else:
                lexical_support = 0.3

        overlap_count = sum(
            1
            for candidate in fused_candidates
            if (
                candidate.get(
                    "dense_rank"
                )
                is not None
                and candidate.get(
                    "bm25_rank"
                )
                is not None
            )
        )

        agreement = min(
            overlap_count / 3.0,
            1.0,
        )

        confidence = (
            0.60
            * best_dense_similarity
            + 0.25
            * lexical_support
            + 0.15
            * agreement
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    @staticmethod
    def _l2_distance_to_similarity(
        distance: float | None,
    ) -> float:
        """
        Convert L2 distance between normalized vectors into cosine
        similarity.

        VectorStore normalizes embeddings before storing/querying them.
        For unit vectors:

            ||a-b||² = 2 - 2*cos(theta)

        therefore:

            cosine = 1 - distance² / 2

        The result is clamped to [0, 1] because retrieval confidence
        should not treat negative similarity as positive evidence.
        """
        if distance is None:
            return 0.0

        try:
            value = float(distance)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        similarity = 1.0 - (
            value * value / 2.0
        )

        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )