"""
retrieval/services/hybrid_retriever.py

QueryPilot's primary workspace evidence retriever.

Responsibilities:
    1. Dense semantic retrieval from Chroma.
    2. BM25 lexical retrieval over the current workspace.
    3. Optional source-type filtering (text/table).
    4. Merge and deduplicate candidates.
    5. Rank candidates with Reciprocal Rank Fusion (RRF).
    6. Return the strongest evidence candidates with a retrieval
       confidence estimate.

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
import logging
from typing import Any, Iterable

from documents.ingestion.vectors import VectorStore

from .bm25 import BM25Retriever
from .evidence import Evidence


logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """
    Typed error raised when a retrieval backend cannot complete an
    operation.

    The original backend exception remains available as `__cause__`
    for logging/debugging without leaking backend-specific text to
    the user-facing layer.
    """

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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

    Current QueryPilot choice:

        dense_k = 20
        bm25_k = 20
        top_k = 5
        rrf_k = 60

    Dense and BM25 are independent signals. A failure in one backend does
    not automatically destroy the other, which is important for exact
    lexical questions such as names, IDs, filenames, and codes.
    """

    DEFAULT_DENSE_K = 20
    DEFAULT_BM25_K = 20
    DEFAULT_TOP_K = 5

    RRF_K = 60

    TEXT_SOURCE_TYPE = "text"
    TABLE_SOURCE_TYPE = "table"

    def __init__(
        self,
        *,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if dense_k <= 0:
            raise ValueError("dense_k must be greater than 0")

        if bm25_k <= 0:
            raise ValueError("bm25_k must be greater than 0")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.top_k = top_k

    def retrieve(
        self,
        *,
        query: str,
        workspace_id: int,
        allowed_document_ids: Iterable[int] | None = None,
        source_types: Iterable[str] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the strongest evidence for a query inside one workspace.

        All available scope constraints are applied in the Chroma where
        clause before ranking. A defensive validation pass is still kept
        after retrieval so malformed/stale metadata cannot widen scope.
        """
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        if workspace_id <= 0:
            raise ValueError("workspace_id must be greater than zero")

        normalized_document_ids = self._normalize_document_ids(
            allowed_document_ids
        )
        normalized_source_types = self._normalize_source_types(source_types)

        logger.info(
            "HybridRetriever.retrieve: workspace_id=%s "
            "allowed_document_ids=%s source_types=%s",
            workspace_id,
            normalized_document_ids,
            normalized_source_types,
        )

        dense_candidates: list[dict[str, Any]] = []
        bm25_candidates: list[dict[str, Any]] = []
        dense_error: RetrievalError | None = None
        bm25_error: RetrievalError | None = None

        try:
            dense_candidates = self._dense_search(
                query=query,
                workspace_id=workspace_id,
                allowed_document_ids=normalized_document_ids,
                source_types=normalized_source_types,
            )
        except RetrievalError as exc:
            dense_error = exc
            logger.exception(
                "HybridRetriever.retrieve: dense retrieval failed "
                "code=%s workspace_id=%s",
                exc.code,
                workspace_id,
            )

        try:
            bm25_candidates = self._bm25_search(
                query=query,
                workspace_id=workspace_id,
                allowed_document_ids=normalized_document_ids,
                source_types=normalized_source_types,
            )
        except RetrievalError as exc:
            bm25_error = exc
            logger.exception(
                "HybridRetriever.retrieve: BM25 retrieval failed "
                "code=%s workspace_id=%s",
                exc.code,
                workspace_id,
            )

        # If either side returned anything, use the successful signal(s).
        # This prevents a transient/broken dense index from turning an
        # otherwise answerable exact-match query into a total failure.
        if dense_candidates or bm25_candidates:
            fused_candidates = self._rrf_fuse(
                dense_candidates=dense_candidates,
                bm25_candidates=bm25_candidates,
            )

            final_candidates = fused_candidates[: self.top_k]
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
                "HybridRetriever.retrieve: returning %d final "
                "candidates (fused=%d dense=%d bm25=%d)",
                len(final_candidates),
                len(fused_candidates),
                len(dense_candidates),
                len(bm25_candidates),
            )

            return RetrievalResult(
                evidence=evidence,
                confidence=confidence,
                candidate_count=len(fused_candidates),
            )

        # No candidates and at least one backend failed: surface the typed
        # backend failure instead of pretending the workspace is empty.
        if dense_error is not None:
            raise dense_error

        if bm25_error is not None:
            raise bm25_error

        return RetrievalResult(
            evidence=[],
            confidence=0.0,
            candidate_count=0,
        )

    @staticmethod
    def _normalize_document_ids(
        allowed_document_ids: Iterable[int] | None,
    ) -> list[int] | None:
        """
        Normalize a document allow-list.

        None means unrestricted at the document-ID level.
        An explicitly supplied empty list remains empty and is treated as
        matching no documents.
        """
        if allowed_document_ids is None:
            return None

        normalized: list[int] = []

        for value in allowed_document_ids:
            try:
                document_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "allowed_document_ids must contain integer IDs."
                ) from exc

            if document_id <= 0:
                raise ValueError(
                    "allowed_document_ids must contain positive IDs."
                )

            if document_id not in normalized:
                normalized.append(document_id)

        return normalized

    @classmethod
    def _normalize_source_types(
        cls,
        source_types: Iterable[str] | None,
    ) -> list[str] | None:
        """Normalize and validate Chroma source-type filters."""
        if source_types is None:
            return None

        allowed = {
            cls.TEXT_SOURCE_TYPE,
            cls.TABLE_SOURCE_TYPE,
        }

        normalized: list[str] = []

        for value in source_types:
            source_type = str(value).strip().lower()

            if source_type not in allowed:
                raise ValueError(
                    f"Unsupported source_type {value!r}; "
                    "expected 'text' or 'table'."
                )

            if source_type not in normalized:
                normalized.append(source_type)

        return normalized

    @staticmethod
    def _build_chroma_where(
        *,
        workspace_id: int,
        allowed_document_ids: list[int] | None,
        source_types: list[str] | None,
    ) -> dict[str, Any]:
        """
        Build the complete Chroma metadata predicate.

        Scope is enforced before ranking rather than by post-filtering a
        larger candidate pool.
        """
        clauses: list[dict[str, Any]] = [
            {"workspace_id": workspace_id},
        ]

        if allowed_document_ids is not None:
            clauses.append(
                {
                    "document_id": {
                        "$in": (
                            allowed_document_ids
                            if allowed_document_ids
                            else [-1]
                        )
                    }
                }
            )

        if source_types is not None:
            clauses.append(
                {
                    "type": {
                        "$in": (
                            source_types
                            if source_types
                            else ["__querypilot_no_source__"]
                        )
                    }
                }
            )

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    def _dense_search(
        self,
        *,
        query: str,
        workspace_id: int,
        allowed_document_ids: list[int] | None,
        source_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic retrieval from Chroma.

        Chroma receives the workspace, document and source-type boundaries
        directly so table metadata cannot contaminate text-only retrieval.
        """
        try:
            collection = VectorStore.knowledge_collection()
            query_embedding = VectorStore.embed([query])[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.dense_k,
                where=self._build_chroma_where(
                    workspace_id=workspace_id,
                    allowed_document_ids=allowed_document_ids,
                    source_types=source_types,
                ),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        except Exception as exc:
            raise RetrievalError(
                code="RETRIEVAL_ERROR",
                message="Dense retrieval failed.",
            ) from exc

        documents = (
            results.get("documents", [[]])[0]
            if isinstance(results, dict)
            else []
        )
        metadatas = (
            results.get("metadatas", [[]])[0]
            if isinstance(results, dict)
            else []
        )
        distances = (
            results.get("distances", [[]])[0]
            if isinstance(results, dict)
            else []
        )
        ids = (
            results.get("ids", [[]])[0]
            if isinstance(results, dict)
            else []
        )

        if not isinstance(documents, list):
            documents = []
        if not isinstance(metadatas, list):
            metadatas = []
        if not isinstance(distances, list):
            distances = []
        if not isinstance(ids, list):
            ids = []

        candidates: list[dict[str, Any]] = []

        for index, document in enumerate(documents):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and isinstance(metadatas[index], dict)
                else {}
            )

            candidates.append(
                {
                    "id": ids[index] if index < len(ids) else None,
                    "text": str(document or ""),
                    "metadata": metadata,
                    "dense_distance": (
                        distances[index]
                        if index < len(distances)
                        else None
                    ),
                }
            )

        return self._validate_candidates(
            candidates=candidates,
            workspace_id=workspace_id,
            allowed_document_ids=allowed_document_ids,
            source_types=source_types,
        )

    def _bm25_search(
        self,
        *,
        query: str,
        workspace_id: int,
        allowed_document_ids: list[int] | None,
        source_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """
        Perform workspace-scoped BM25 retrieval.

        The corpus is restricted by the same Chroma metadata predicate as
        dense retrieval before the lexical ranker sees it.
        """
        try:
            collection = VectorStore.knowledge_collection()
            corpus = collection.get(
                where=self._build_chroma_where(
                    workspace_id=workspace_id,
                    allowed_document_ids=allowed_document_ids,
                    source_types=source_types,
                ),
                include=[
                    "documents",
                    "metadatas",
                ],
            )
        except Exception as exc:
            raise RetrievalError(
                code="RETRIEVAL_ERROR",
                message="Lexical retrieval corpus could not be loaded.",
            ) from exc

        documents = (
            corpus.get("documents", [])
            if isinstance(corpus, dict)
            else []
        )
        metadatas = (
            corpus.get("metadatas", [])
            if isinstance(corpus, dict)
            else []
        )
        ids = (
            corpus.get("ids", [])
            if isinstance(corpus, dict)
            else []
        )

        if not isinstance(documents, list):
            documents = []
        if not isinstance(metadatas, list):
            metadatas = []
        if not isinstance(ids, list):
            ids = []

        if not documents:
            return []

        try:
            bm25 = BM25Retriever(documents=documents)
            matches = bm25.top_k(
                query=query,
                k=self.bm25_k,
            )
        except Exception as exc:
            raise RetrievalError(
                code="RETRIEVAL_ERROR",
                message="Lexical ranking failed.",
            ) from exc

        candidates: list[dict[str, Any]] = []

        for match in matches:
            index = getattr(match, "index", None)

            if not isinstance(index, int):
                continue

            if index < 0 or index >= len(documents):
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and isinstance(metadatas[index], dict)
                else {}
            )

            candidates.append(
                {
                    "id": ids[index] if index < len(ids) else None,
                    "text": str(documents[index] or ""),
                    "metadata": metadata,
                    "bm25_score": getattr(match, "score", 0.0),
                }
            )

        return self._validate_candidates(
            candidates=candidates,
            workspace_id=workspace_id,
            allowed_document_ids=allowed_document_ids,
            source_types=source_types,
        )

    @staticmethod
    def _validate_candidates(
        *,
        candidates: list[dict[str, Any]],
        workspace_id: int,
        allowed_document_ids: list[int] | None,
        source_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """
        Defensive post-validation.

        Chroma is the primary enforcement point. This second check is
        intentionally fail-closed: malformed candidates are discarded,
        never retained as an unscoped fallback.
        """
        allowed_document_set = (
            set(allowed_document_ids)
            if allowed_document_ids is not None
            else None
        )
        source_type_set = (
            set(source_types)
            if source_types is not None
            else None
        )

        validated: list[dict[str, Any]] = []

        for candidate in candidates:
            metadata = candidate.get("metadata")

            if not isinstance(metadata, dict):
                continue

            try:
                candidate_workspace_id = int(
                    metadata.get("workspace_id")
                )
                document_id = int(
                    metadata.get("document_id")
                )
            except (TypeError, ValueError):
                continue

            if candidate_workspace_id != workspace_id:
                continue

            if (
                allowed_document_set is not None
                and document_id not in allowed_document_set
            ):
                continue

            source_type = metadata.get("type")

            if (
                source_type_set is not None
                and source_type not in source_type_set
            ):
                continue

            candidate["document_id"] = document_id
            candidate["source_type"] = source_type
            candidate["metadata"] = metadata
            validated.append(candidate)

        return validated

    @staticmethod
    def _candidate_key(
        candidate: dict[str, Any],
    ) -> str:
        """
        Stable identity used by RRF fusion.

        Chroma IDs are preferred. Metadata identity is the fail-safe when
        a backend result lacks an ID.
        """
        evidence_id = candidate.get("id")

        if evidence_id is not None:
            return f"chroma:{evidence_id}"

        metadata = candidate.get("metadata") or {}

        return (
            "meta:"
            f"{metadata.get('workspace_id')}:"
            f"{metadata.get('document_id')}:"
            f"{metadata.get('type')}:"
            f"{metadata.get('chunk_index')}:"
            f"{metadata.get('table_name')}"
        )

    def _rrf_fuse(
        self,
        *,
        dense_candidates: list[dict[str, Any]],
        bm25_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion.
        """
        candidates: dict[str, dict[str, Any]] = {}

        for rank, candidate in enumerate(
            dense_candidates,
            start=1,
        ):
            key = self._candidate_key(candidate)

            item = candidates.setdefault(
                key,
                {
                    "id": candidate.get("id"),
                    "text": candidate.get("text", ""),
                    "metadata": candidate.get("metadata", {}),
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
            item["rrf_score"] += 1.0 / (self.RRF_K + rank)

        for rank, candidate in enumerate(
            bm25_candidates,
            start=1,
        ):
            key = self._candidate_key(candidate)

            item = candidates.setdefault(
                key,
                {
                    "id": candidate.get("id"),
                    "text": candidate.get("text", ""),
                    "metadata": candidate.get("metadata", {}),
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )

            item["bm25_rank"] = rank
            item["bm25_score"] = candidate.get("bm25_score")

            if not item["text"]:
                item["text"] = candidate.get("text", "")

            if not item["metadata"]:
                item["metadata"] = candidate.get("metadata", {})

            item["rrf_score"] += 1.0 / (self.RRF_K + rank)

        fused: list[dict[str, Any]] = []

        for item in candidates.values():
            try:
                evidence = self._build_evidence(
                    content=item["text"],
                    score=item["rrf_score"],
                    metadata=item["metadata"],
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed retrieval candidate: %s",
                    exc,
                )
                continue

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
        """Favor candidates supported near the top of both rank lists."""
        score = 0.0

        if candidate.get("dense_rank") is not None:
            score += 1.0 / (1.0 + candidate["dense_rank"])

        if candidate.get("bm25_rank") is not None:
            score += 1.0 / (1.0 + candidate["bm25_rank"])

        return score

    def _build_evidence(
        self,
        *,
        content: str,
        score: float,
        metadata: dict[str, Any],
    ) -> Evidence:
        """Convert a retrieved candidate into the common Evidence type."""
        source_type = metadata.get("type", "unknown")

        if source_type == "text":
            normalized_source_type = "text_chunk"
        elif source_type == "table":
            normalized_source_type = "table_metadata"
        else:
            normalized_source_type = source_type

        workspace_id = metadata.get("workspace_id")
        document_id = metadata.get("document_id")

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
            document_name=metadata.get("document_name"),
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
        """
        Estimate retrieval strength without treating it as calibrated
        probability.
        """
        if not fused_candidates:
            return 0.0

        best_dense_similarity = 0.0

        if dense_candidates:
            best_dense_similarity = cls._l2_distance_to_similarity(
                dense_candidates[0].get("dense_distance")
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

        agreement = min(overlap_count / 3.0, 1.0)

        confidence = (
            0.60 * best_dense_similarity
            + 0.25 * lexical_support
            + 0.15 * agreement
        )

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _l2_distance_to_similarity(
        distance: float | None,
    ) -> float:
        """
        Convert L2 distance between normalized vectors into cosine
        similarity.
        """
        if distance is None:
            return 0.0

        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0

        similarity = 1.0 - (value * value / 2.0)

        return max(0.0, min(1.0, similarity))
