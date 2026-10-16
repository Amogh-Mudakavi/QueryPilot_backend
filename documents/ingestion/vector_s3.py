"""Implement the QueryPilot vector-storage interface using AWS S3 Vectors.

This backend stores knowledge vectors and document-profile vectors in the
configured S3 Vectors bucket and indexes. It is selected by the shared
VectorStore facade when VECTOR_BACKEND=s3.
"""
from __future__ import annotations

import logging
import os
import threading

import boto3
from botocore.config import Config


logger = logging.getLogger(__name__)

DEFAULT_S3_VECTOR_REGION = "ap-south-1"

DEFAULT_S3_VECTOR_BUCKET = "querypilot-prod-vectors"

DEFAULT_KNOWLEDGE_INDEX = "querypilot-knowledge"

DEFAULT_DOCUMENT_INDEX = "querypilot-document-metadata"

S3_VECTORS_MAX_BATCH = 500

S3_VECTORS_MAX_GET_KEYS = 100

TEXT_METADATA_KEY = "text"


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


class S3VectorsBackend:
    def __init__(self):
        self._client = None
        self._client_lock = threading.Lock()

                                                                        
            
                                                                        

    @staticmethod
    def bucket_name() -> str:
        return _env("S3_VECTOR_BUCKET", DEFAULT_S3_VECTOR_BUCKET)

    @staticmethod
    def knowledge_index() -> str:
        return _env("S3_KNOWLEDGE_INDEX", DEFAULT_KNOWLEDGE_INDEX)

    @staticmethod
    def document_index() -> str:
        return _env("S3_DOCUMENT_INDEX", DEFAULT_DOCUMENT_INDEX)

    def get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    region = _env(
                        "S3_VECTOR_REGION",
                        DEFAULT_S3_VECTOR_REGION,
                    )
                    self._client = boto3.client(
                        "s3vectors",
                        region_name=region,
                        config=Config(
                            retries={
                                "max_attempts": 5,
                                "mode": "standard",
                            },
                        ),
                    )
                    logger.info(
                        "[S3VECTORS] Client ready: region=%s bucket=%s "
                        "knowledge_index=%s document_index=%s.",
                        region,
                        self.bucket_name(),
                        self.knowledge_index(),
                        self.document_index(),
                    )
        return self._client

                                                                        
                       
                                                                        

    @staticmethod
    def _vector(record) -> dict:
        metadata = {k: v for k, v in record["metadata"].items() if v is not None}
        metadata[TEXT_METADATA_KEY] = record["text"]
        return {
            "key": record["id"],
            "data": {"float32": [float(value) for value in record["embedding"]]},
            "metadata": metadata,
        }

    def _put_vectors(self, index_name, vectors) -> None:
        """PutVectors in batches of at most 500 (overwrites existing keys)."""
        client = self.get_client()
        for start in range(0, len(vectors), S3_VECTORS_MAX_BATCH):
            client.put_vectors(
                vectorBucketName=self.bucket_name(),
                indexName=index_name,
                vectors=vectors[start : start + S3_VECTORS_MAX_BATCH],
            )

    def _delete_keys(self, index_name, keys) -> None:
        """DeleteVectors in batches of at most 500 keys."""
        client = self.get_client()
        for start in range(0, len(keys), S3_VECTORS_MAX_BATCH):
            client.delete_vectors(
                vectorBucketName=self.bucket_name(),
                indexName=index_name,
                keys=keys[start : start + S3_VECTORS_MAX_BATCH],
            )

    def _iter_vectors(self, index_name):
        """
        Yield every vector (key + metadata, no vector data) of an index.

        ListVectors cannot filter, so callers filter in application code.
        Pagination follows nextToken until exhausted.
        """
        client = self.get_client()
        params = {
            "vectorBucketName": self.bucket_name(),
            "indexName": index_name,
            "maxResults": S3_VECTORS_MAX_BATCH,
            "returnData": False,
            "returnMetadata": True,
        }
        while True:
            response = client.list_vectors(**params)
            for vector in response.get("vectors", []) or []:
                yield vector
            next_token = response.get("nextToken")
            if not next_token:
                break
            params["nextToken"] = next_token

    @staticmethod
    def _knowledge_filter(
        workspace_id,
        allowed_document_ids,
        source_types,
    ) -> dict:
        """QueryVectors metadata filter: workspace + documents + types."""
        clauses = [{"workspace_id": {"$eq": int(workspace_id)}}]
        if allowed_document_ids is not None:
            clauses.append(
                {"document_id": {"$in": [int(i) for i in allowed_document_ids]}}
            )
        if source_types is not None:
            clauses.append({"type": {"$in": [str(t) for t in source_types]}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

                                                                        
                       
                                                                        

    def upsert_knowledge(self, records) -> None:
        self._put_vectors(
            self.knowledge_index(),
            [self._vector(record) for record in records],
        )

    def upsert_document_profile(self, record) -> None:
        self._put_vectors(
            self.document_index(),
            [self._vector(record)],
        )

    def fetch_profile_rows(self, workspace_id, normalized_ids) -> dict:
        """
        GetVectors for the deterministic keys document_<id>.

        Every row is re-validated against the requested workspace and
        document ids; anything else is dropped.
        """
        client = self.get_client()
        requested = set(normalized_ids)
        keys = [f"document_{document_id}" for document_id in normalized_ids]
        rows = {}
        for start in range(0, len(keys), S3_VECTORS_MAX_GET_KEYS):
            response = client.get_vectors(
                vectorBucketName=self.bucket_name(),
                indexName=self.document_index(),
                keys=keys[start : start + S3_VECTORS_MAX_GET_KEYS],
                returnData=False,
                returnMetadata=True,
            )
            for vector in response.get("vectors", []) or []:
                metadata = vector.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                try:
                    profile_workspace_id = int(metadata.get("workspace_id"))
                    document_id = int(metadata.get("document_id"))
                except (TypeError, ValueError):
                    continue
                if profile_workspace_id != workspace_id or document_id not in requested:
                    continue
                profile_text = metadata.get(TEXT_METADATA_KEY)
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
        response = self.get_client().query_vectors(
            vectorBucketName=self.bucket_name(),
            indexName=self.knowledge_index(),
            queryVector={"float32": [float(value) for value in query_embedding]},
            topK=top_k,
            filter=self._knowledge_filter(
                workspace_id,
                allowed_document_ids,
                source_types,
            ),
            returnMetadata=True,
            returnDistance=True,
        )
        hits = []
        for vector in response.get("vectors", []) or []:
            metadata = vector.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            hits.append(
                {
                    "id": vector.get("key"),
                    "text": str(metadata.get(TEXT_METADATA_KEY) or ""),
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key != TEXT_METADATA_KEY
                    },
                    "distance": vector.get("distance"),
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
        ids: list = []
        documents: list = []
        metadatas: list = []
        document_set = (
            set(allowed_document_ids) if allowed_document_ids is not None else None
        )
        type_set = set(source_types) if source_types is not None else None
        for vector in self._iter_vectors(self.knowledge_index()):
            metadata = vector.get("metadata")
            if not isinstance(metadata, dict):
                continue
            try:
                vector_workspace_id = int(metadata.get("workspace_id"))
                vector_document_id = int(metadata.get("document_id"))
            except (TypeError, ValueError):
                continue
            if vector_workspace_id != workspace_id:
                continue
            if document_set is not None and vector_document_id not in document_set:
                continue
            if type_set is not None and metadata.get("type") not in type_set:
                continue
            ids.append(vector.get("key"))
            documents.append(str(metadata.get(TEXT_METADATA_KEY) or ""))
            metadatas.append(
                {
                    key: value
                    for key, value in metadata.items()
                    if key != TEXT_METADATA_KEY
                }
            )
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def delete_document(self, document_id, workspace_id=None) -> None:
        """
        Delete every vector of a document.

        Knowledge vectors are found through their document_id metadata
        (and workspace_id when supplied); the profile vector is removed by
        its deterministic key document_<id>.
        """
        errors = []
        try:
            keys = []
            for vector in self._iter_vectors(self.knowledge_index()):
                metadata = vector.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                try:
                    vector_document_id = int(metadata.get("document_id"))
                except (TypeError, ValueError):
                    continue
                if vector_document_id != document_id:
                    continue
                if workspace_id is not None:
                    try:
                        vector_workspace_id = int(metadata.get("workspace_id"))
                    except (TypeError, ValueError):
                        continue
                    if vector_workspace_id != workspace_id:
                        continue
                keys.append(vector["key"])
            self._delete_keys(self.knowledge_index(), keys)
            logger.info(
                "[S3VECTORS] deleted %d knowledge vector(s) for document %s.",
                len(keys),
                document_id,
            )
        except Exception as exc:
            errors.append(exc)
        try:
            self._delete_keys(
                self.document_index(),
                [f"document_{document_id}"],
            )
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                f"Vector deletion failed for document {document_id}: {errors}"
            )