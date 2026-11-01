"""Optional multilingual cross-encoder reranking for hybrid candidates.

Enable explicitly with RETRIEVAL_RERANKER_ENABLED=true. The model is lazy-loaded
so deployments that have not provisioned its memory/download budget retain the
existing RRF-only path. Ranking scores are ordering signals, not probabilities.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("RETRIEVAL_RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


class CrossEncoderReranker:
    DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    MAX_CANDIDATES = 60

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return _enabled()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    model_name = os.getenv("RETRIEVAL_RERANKER_MODEL", self.DEFAULT_MODEL).strip()
                    self._model = CrossEncoder(model_name)
                    logger.info("[RERANKER] loaded cross-encoder model=%s", model_name)
        return self._model

    def rerank(self, *, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not candidates:
            return candidates
        selected = candidates[: self.MAX_CANDIDATES]
        pairs = []
        for item in selected:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            title = str(metadata.get("document_title") or "")
            heading = str(metadata.get("heading_path") or metadata.get("heading") or "")
            content = str(item.get("text") or "")
            pairs.append((query, "\n".join(part for part in (f"Document: {title}" if title else "", f"Heading: {heading}" if heading else "", content) if part)))
        try:
            scores = self._get_model().predict(pairs, batch_size=16, show_progress_bar=False)
            for item, score in zip(selected, scores):
                item["reranker_score"] = float(score)
            selected.sort(key=lambda item: item.get("reranker_score", float("-inf")), reverse=True)
            return selected + candidates[len(selected):]
        except Exception:
            logger.exception("[RERANKER] failed; preserving RRF ordering")
            return candidates


reranker = CrossEncoderReranker()
