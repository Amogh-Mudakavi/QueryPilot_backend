from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from documents.ingestion.vectors import VectorStore

from .bm25 import BM25Retriever
from .evidence import Evidence


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result returned by HybridRetriever.
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

    When allowed_document_ids is supplied, BOTH retrieval paths
    are restricted to those documents before fusion.
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
        allowed_document_ids: list[int] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve evidence within one workspace.

        If allowed_document_ids is supplied, only those document IDs
        may contribute evidence.

        None:
            no additional document restriction beyond workspace.

        []:
            no documents are allowed.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "query must not be empty"
            )

        normalized_document_ids = (
            None
            if allowed_document_ids is None
            else {
                int(document_id)
                for document_id in allowed_document_ids
            }
        )

        logger.info(
            "HybridRetriever.retrieve: workspace_id=%s allowed_document_ids=%s",
            workspace_id,
            normalized_document_ids,
        )

        if normalized_document_ids == set():
            return RetrievalResult(
                evidence=[],
                confidence=0.0,
                candidate_count=0,
            )

        dense_candidates = self._dense_search(
            query=query,
            workspace_id=workspace_id,
            allowed_document_ids=normalized_document_ids,
        )

        bm25_candidates = self._bm25_search(
            query=query,
            workspace_id=workspace_id,
            allowed_document_ids=normalized_document_ids,
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

        logger.info(
            "HybridRetriever.retrieve: dense=%d bm25=%d fused=%d final=%d",
            len(dense_candidates),
            len(bm25_candidates),
            len(fused_candidates),
            len(final_candidates),
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
        allowed_document_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic retrieval.

        Document restrictions are applied directly in the Chroma query.
        """

        collection = (
            VectorStore.knowledge_collection()
        )

        query_embedding = VectorStore.embed(
            [query]
        )[0]

        where: dict[str, Any] = {
            "workspace_id": workspace_id,
        }

        if allowed_document_ids is not None:
            if not allowed_document_ids:
                return []

            where = {
                "$and": [
                    {
                        "workspace_id": workspace_id,
                    },
                    {
                        "document_id": {
                            "$in": sorted(
                                allowed_document_ids
                            ),
                        }
                    },
                ]
            }

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.dense_k,
            where=where,
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
        allowed_document_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform BM25 retrieval.

        When an allow-list exists, the corpus itself is restricted
        before BM25 ranking.
        """

        collection = (
            VectorStore.knowledge_collection()
        )

        where: dict[str, Any] = {
            "workspace_id": workspace_id,
        }

        if allowed_document_ids is not None:
            if not allowed_document_ids:
                return []

            where = {
                "$and": [
                    {
                        "workspace_id": workspace_id,
                    },
                    {
                        "document_id": {
                            "$in": sorted(
                                allowed_document_ids
                            ),
                        }
                    },
                ]
            }

        corpus = collection.get(
            where=where,
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
            item["dense_distance"] = candidate.get(
                "dense_distance"
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
            item["bm25_score"] = candidate.get(
                "bm25_score"
            )

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

        source_type = metadata.get(
            "type",
            "unknown",
        )

        if source_type == "text":
            normalized_source_type = "text_chunk"
        elif source_type == "table":
            normalized_source_type = "table_metadata"
        else:
            normalized_source_type = source_type

        workspace_id = metadata.get(
            "workspace_id"
        )

        document_id = metadata.get(
            "document_id"
        )

        if workspace_id is None:
            raise ValueError(
                "Retrieved evidence is missing workspace_id"
            )

        if document_id is None:
            raise ValueError(
                "Retrieved evidence is missing document_id"
            )

        return Evidence(
            content=content,
            score=float(score),
            source_type=normalized_source_type,
            workspace_id=int(workspace_id),
            document_id=int(document_id),
            document_name=metadata.get(
                "document_name"
            ),
            metadata=metadata,
        )

    @classmethod
    def _calculate_confidence(
        cls,
        *,
        dense_candidates: list[dict[str, Any]],
        bm25_candidates: list[dict[str, Any]],
        fused_candidates: list[dict[str, Any]],
    ) -> float:

        if not fused_candidates:
            return 0.0

        best_dense_similarity = 0.0

        if dense_candidates:
            best_distance = dense_candidates[0].get(
                "dense_distance"
            )

            best_dense_similarity = (
                cls._l2_distance_to_similarity(
                    best_distance
                )
            )

        top_candidate = fused_candidates[0]

        lexical_support = 0.0

        if top_candidate.get("bm25_rank") is not None:
            bm25_rank = top_candidate["bm25_rank"]

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
                candidate.get("dense_rank") is not None
                and candidate.get("bm25_rank") is not None
            )
        )

        agreement = min(
            overlap_count / 3.0,
            1.0,
        )

        confidence = (
            0.60 * best_dense_similarity
            + 0.25 * lexical_support
            + 0.15 * agreement
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