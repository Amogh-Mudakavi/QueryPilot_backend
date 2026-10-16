"""Implement the QueryPilot vector-storage interface using local ChromaDB.

This backend persists knowledge vectors and document-profile vectors on the
local filesystem. It is intended for development and is selected by the
shared VectorStore facade when VECTOR_BACKEND=chroma.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import chromadb
from django.conf import settings


logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "querypilot_knowledge"
DOCUMENT_COLLECTION = "querypilot_document_metadata"


def _chroma_path() -> str:
    configured = os.getenv("CHROMA_PATH", "").strip()
    if configured:
        return configured
    return str(Path(settings.BASE_DIR) / "chroma_data")


def _clean(metadata) -> dict:
    return {key: value for key, value in metadata.items() if value is not None}


class ChromaBackend:
    def __init__(self):
        path = _chroma_path()
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=path)
        self._knowledge = self._client.get_or_create_collection(
            name=KNOWLEDGE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._documents = self._client.get_or_create_collection(
            name=DOCUMENT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[CHROMA] Ready: path=%s.", path)

    @staticmethod
    def _filter(workspace_id, allowed_document_ids, source_types):
        clauses = [{"workspace_id": {"$eq": int(workspace_id)}}]
        if allowed_document_ids is not None:
            clauses.append(
                {"document_id": {"$in": [int(value) for value in allowed_document_ids]}}
            )
        if source_types is not None:
            clauses.append({"type": {"$in": [str(value) for value in source_types]}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _upsert(collection, records) -> None:
        collection.upsert(
            ids=[record["id"] for record in records],
            embeddings=[
                [float(value) for value in record["embedding"]]
                for record in records
            ],
            documents=[record["text"] for record in records],
            metadatas=[_clean(record["metadata"]) for record in records],
        )

    def upsert_knowledge(self, records) -> None:
        with self._lock:
            self._upsert(self._knowledge, records)

    def upsert_document_profile(self, record) -> None:
        with self._lock:
            self._upsert(self._documents, [record])

    def fetch_profile_rows(self, workspace_id, normalized_ids) -> dict:
        requested = set(normalized_ids)
        keys = [f"document_{document_id}" for document_id in normalized_ids]
        result = self._documents.get(
            ids=keys,
            include=["metadatas", "documents"],
        )
        rows = {}
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        for index in range(len(ids)):
            metadata = metadatas[index] if index < len(metadatas) else None
            if not isinstance(metadata, dict):
                continue
            try:
                profile_workspace_id = int(metadata.get("workspace_id"))
                document_id = int(metadata.get("document_id"))
            except (TypeError, ValueError):
                continue
            if profile_workspace_id != workspace_id or document_id not in requested:
                continue
            profile_text = documents[index] if index < len(documents) else None
            if isinstance(profile_text, str) and profile_text.strip():
                rows[document_id] = {
                    "document_id": document_id,
                    "document_name": metadata.get("document_name") or "",
                    "profile": profile_text,
                }
        return rows

    def query_knowledge(
        self,
        *,
        query_embedding,
        top_k,
        workspace_id,
        allowed_document_ids,
        source_types,
    ) -> list[dict]:
        total = self._knowledge.count()
        if total == 0:
            return []
        result = self._knowledge.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=max(1, min(int(top_k), total)),
            where=self._filter(workspace_id, allowed_document_ids, source_types),
            include=["metadatas", "documents", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, vector_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            hits.append(
                {
                    "id": vector_id,
                    "text": str((documents[index] if index < len(documents) else "") or ""),
                    "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return hits

    def list_knowledge(
        self,
        *,
        workspace_id,
        allowed_document_ids,
        source_types,
    ) -> dict:
        result = self._knowledge.get(
            where=self._filter(workspace_id, allowed_document_ids, source_types),
            include=["metadatas", "documents"],
        )
        ids = list(result.get("ids") or [])
        documents = [str(text or "") for text in (result.get("documents") or [])]
        metadatas = [
            dict(metadata) if isinstance(metadata, dict) else {}
            for metadata in (result.get("metadatas") or [])
        ]
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def delete_document(self, document_id, workspace_id=None) -> None:
        errors = []
        try:
            clauses = [{"document_id": {"$eq": int(document_id)}}]
            if workspace_id is not None:
                clauses.append({"workspace_id": {"$eq": int(workspace_id)}})
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
            with self._lock:
                self._knowledge.delete(where=where)
            logger.info(
                "[CHROMA] deleted knowledge vectors for document %s.", document_id
            )
        except Exception as exc:
            errors.append(exc)
        try:
            with self._lock:
                self._documents.delete(ids=[f"document_{int(document_id)}"])
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                f"Vector deletion failed for document {document_id}: {errors}"
            )
