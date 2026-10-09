"""AWS S3 vector storage for QueryPilot.

This module stores text, table, and document metadata embeddings used by
retrieval and planning. It keeps the storage API separate from parsing and
chunking logic and exposes the minimal operations the rest of the backend
needs: embedding generation, vector upserts, document-scoped deletes, and lookups.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import boto3
import torch
from botocore.config import Config
from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


DEFAULT_S3_VECTOR_REGION = "ap-south-1"
DEFAULT_S3_VECTOR_BUCKET = "querypilot-prod-vectors"
DEFAULT_KNOWLEDGE_INDEX = "querypilot-knowledge"
DEFAULT_DOCUMENT_INDEX = "querypilot-document-metadata"

S3_VECTORS_MAX_BATCH = 500
S3_VECTORS_MAX_GET_KEYS = 100
S3_VECTORS_MAX_TOP_K = 100

TEXT_METADATA_KEY = "text"
MAX_HEADING_CHARS = 300
MAX_DOCUMENT_NAME_CHARS = 300
MAX_TABLE_NAME_CHARS = 200

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

EMBEDDING_BATCH_SIZE = getattr(
    settings,
    "EMBEDDING_BATCH_SIZE",
    32,
)

S3_VECTOR_BATCH_SIZE = max(
    1,
    min(
        S3_VECTORS_MAX_BATCH,
        int(
            getattr(
                settings,
                "S3_VECTOR_BATCH_SIZE",
                getattr(settings, "CHROMA_BATCH_SIZE", 128),
            )
        ),
    ),
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _clip(value, limit: int) -> str:
    return ("" if value is None else str(value))[:limit]


def _resolve_torch_threads() -> int:
    """
    Intra-op threads for CPU embedding inference.

    EMBEDDING_TORCH_THREADS > 0 wins. Otherwise default to at most 4:
    PyTorch defaults to every visible core, which oversubscribes small
    or shared machines (and containers that report host cores).
    """
    try:
        configured = int(
            os.getenv(
                "EMBEDDING_TORCH_THREADS",
                "0",
            )
        )
    except (TypeError, ValueError):
        configured = 0

    if configured > 0:
        return configured

    return max(
        1,
        min(
            4,
            os.cpu_count() or 1,
        ),
    )


class VectorStore:
    _client = None
    _embedder = None

    _client_lock = threading.Lock()
    _embedder_lock = threading.Lock()
    _embedding_lock = threading.Lock()

    @classmethod
    def get_client(cls):
        if cls._client is None:
            with cls._client_lock:
                if cls._client is None:
                    region = _env(
                        "S3_VECTOR_REGION",
                        DEFAULT_S3_VECTOR_REGION,
                    )

                    cls._client = boto3.client(
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
                        cls.bucket_name(),
                        cls.knowledge_index(),
                        cls.document_index(),
                    )

        return cls._client

    @staticmethod
    def bucket_name() -> str:
        return _env("S3_VECTOR_BUCKET", DEFAULT_S3_VECTOR_BUCKET)

    @staticmethod
    def knowledge_index() -> str:
        return _env("S3_KNOWLEDGE_INDEX", DEFAULT_KNOWLEDGE_INDEX)

    @staticmethod
    def document_index() -> str:
        return _env("S3_DOCUMENT_INDEX", DEFAULT_DOCUMENT_INDEX)

    @classmethod
    def get_embedder(cls):

        if cls._embedder is None:
            with cls._embedder_lock:
                if cls._embedder is None:
                    threads = _resolve_torch_threads()

                    try:
                        torch.set_num_threads(
                            threads,
                        )
                    except Exception:
                        logger.exception(
                            "[EMBEDDING] Could not set torch num_threads=%s.",
                            threads,
                        )

                    try:
                        cls._embedder = SentenceTransformer(
                            EMBEDDING_MODEL,
                            backend="onnx",
                            model_kwargs={
                                "file_name": "onnx/model_quantized.onnx",
                            },
                            local_files_only=True,
                            trust_remote_code=False,
                        )
                    except OSError:
                        cls._embedder = SentenceTransformer(
                            EMBEDDING_MODEL,
                            backend="onnx",
                            model_kwargs={
                                "file_name": "onnx/model_quantized.onnx",
                            },
                            trust_remote_code=False,
                        )

                    logger.info(
                        "[EMBEDDING] Model ready: %s "
                        "(backend=onnx, quantized=true, torch_threads=%s, "
                        "batch_size=%s).",
                        EMBEDDING_MODEL,
                        threads,
                        EMBEDDING_BATCH_SIZE,
                    )

        return cls._embedder

    @classmethod
    def embed(
        cls,
        texts,
    ):

        texts = list(texts)

        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        model = cls.get_embedder()

        started = time.monotonic()

        logger.info(
            "[EMBEDDING] start: %d text(s) (batch_size=%s).",
            len(texts),
            EMBEDDING_BATCH_SIZE,
        )

        with cls._embedding_lock:
            vectors = model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()

        logger.info(
            "[EMBEDDING] completed: %d text(s) in %.2fs.",
            len(texts),
            time.monotonic() - started,
        )

        return vectors

    @staticmethod
    def _vector(key, embedding, metadata) -> dict:
        return {
            "key": key,
            "data": {"float32": [float(value) for value in embedding]},
            "metadata": {k: v for k, v in metadata.items() if v is not None},
        }

    @classmethod
    def _put_vectors(cls, index_name, vectors) -> None:
        """PutVectors in batches of at most 500 (overwrites existing keys)."""
        client = cls.get_client()

        for start in range(0, len(vectors), S3_VECTORS_MAX_BATCH):
            client.put_vectors(
                vectorBucketName=cls.bucket_name(),
                indexName=index_name,
                vectors=vectors[start : start + S3_VECTORS_MAX_BATCH],
            )

    @classmethod
    def _delete_keys(cls, index_name, keys) -> None:
        """DeleteVectors in batches of at most 500 keys."""
        client = cls.get_client()

        for start in range(0, len(keys), S3_VECTORS_MAX_BATCH):
            client.delete_vectors(
                vectorBucketName=cls.bucket_name(),
                indexName=index_name,
                keys=keys[start : start + S3_VECTORS_MAX_BATCH],
            )

    @classmethod
    def _iter_vectors(cls, index_name):
        """
        Yield every vector (key + metadata, no vector data) of an index.

        ListVectors cannot filter, so callers filter in application code.
        Pagination follows nextToken until exhausted.
        """
        client = cls.get_client()

        params = {
            "vectorBucketName": cls.bucket_name(),
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

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @classmethod
    def store_chunks(
        cls,
        workspace_id,
        document_id,
        chunks,
    ):

        if not chunks:
            return

        started = time.monotonic()

        upserted = 0

        for start in range(
            0,
            len(chunks),
            S3_VECTOR_BATCH_SIZE,
        ):
            batch = chunks[start : start + S3_VECTOR_BATCH_SIZE]

            documents = [chunk["text"] for chunk in batch]

            embeddings = cls.embed(documents)

            vectors = []

            for chunk, embedding in zip(batch, embeddings):
                vectors.append(
                    cls._vector(
                        f"doc_{document_id}_chunk_{chunk['chunk_index']}",
                        embedding,
                        {
                            "workspace_id": workspace_id,
                            "document_id": document_id,
                            "type": "text",
                            "chunk_index": chunk["chunk_index"],
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "heading": _clip(
                                chunk.get("heading") or "",
                                MAX_HEADING_CHARS,
                            ),
                            TEXT_METADATA_KEY: chunk["text"],
                        },
                    )
                )

            cls._put_vectors(cls.knowledge_index(), vectors)

            upserted += len(batch)

        logger.info(
            "[S3VECTORS] put completed: %d chunk(s) for document %s in %.2fs.",
            upserted,
            document_id,
            time.monotonic() - started,
        )

    @classmethod
    def store_table_metadata(
        cls,
        workspace_id,
        document_id,
        table_name,
        summary,
    ):

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Table metadata summary cannot be empty.")

        table_id = f"doc_{document_id}_table_{table_name}_metadata"

        cls._put_vectors(
            cls.knowledge_index(),
            [
                cls._vector(
                    table_id,
                    cls.embed([summary])[0],
                    {
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "type": "table",
                        "table_name": _clip(table_name, MAX_TABLE_NAME_CHARS),
                        TEXT_METADATA_KEY: summary,
                    },
                )
            ],
        )

        logger.info(
            "[S3VECTORS] put completed: table metadata '%s' for document %s.",
            table_name,
            document_id,
        )

    @classmethod
    def store_document_profile(
        cls,
        workspace_id,
        document_id,
        document_name,
        profile_text,
    ):

        if not isinstance(profile_text, str) or not profile_text.strip():
            raise ValueError("Document profile cannot be empty.")

        cls._put_vectors(
            cls.document_index(),
            [
                cls._vector(
                    f"document_{document_id}",
                    cls.embed([profile_text])[0],
                    {
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "document_name": _clip(
                            document_name,
                            MAX_DOCUMENT_NAME_CHARS,
                        ),
                        "type": "document_profile",
                        TEXT_METADATA_KEY: profile_text,
                    },
                )
            ],
        )

        logger.info(
            "[S3VECTORS] put completed: document profile for document %s.",
            document_id,
        )

    # ------------------------------------------------------------------
    # Document profiles (exact-key GetVectors)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_profile_scope(workspace_id, document_ids):
        try:
            workspace_id = int(workspace_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("workspace_id must be an integer.") from exc

        if workspace_id <= 0:
            raise ValueError("workspace_id must be greater than zero.")

        normalized_ids = []
        for value in document_ids or []:
            try:
                document_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("document_ids must contain integer ids.") from exc

            if document_id <= 0:
                raise ValueError("document_ids must contain positive ids.")

            if document_id not in normalized_ids:
                normalized_ids.append(document_id)

        return workspace_id, normalized_ids

    @classmethod
    def _fetch_profile_rows(cls, workspace_id, normalized_ids) -> dict:
        """
        GetVectors for the deterministic keys document_<id>.

        Every row is re-validated against the requested workspace and
        document ids; anything else is dropped.
        """
        client = cls.get_client()

        requested = set(normalized_ids)
        keys = [f"document_{document_id}" for document_id in normalized_ids]
        rows = {}

        for start in range(0, len(keys), S3_VECTORS_MAX_GET_KEYS):
            response = client.get_vectors(
                vectorBucketName=cls.bucket_name(),
                indexName=cls.document_index(),
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

    @classmethod
    def get_document_profiles(
        cls,
        workspace_id,
        document_ids,
    ):
        """Retrieve planner document profiles for one workspace."""
        workspace_id, normalized_ids = cls._normalize_profile_scope(
            workspace_id,
            document_ids,
        )

        if not normalized_ids:
            return {}

        try:
            rows = cls._fetch_profile_rows(workspace_id, normalized_ids)
        except Exception:
            logger.exception(
                "[S3VECTORS] document profile retrieval failed for workspace %s.",
                workspace_id,
            )
            return {}

        profiles = {document_id: row["profile"] for document_id, row in rows.items()}

        logger.info(
            "[S3VECTORS] retrieved %d document profile(s) for workspace %s.",
            len(profiles),
            workspace_id,
        )
        return profiles

    @classmethod
    def get_document_catalog(
        cls,
        workspace_id,
        document_ids,
    ):
        """
        Document id + name + profile for the given documents of one workspace.

        Used by the Scope Guard. Raises on backend failure (the Scope Guard
        is fail-closed); rows outside the workspace / requested ids are
        never returned.
        """
        workspace_id, normalized_ids = cls._normalize_profile_scope(
            workspace_id,
            document_ids,
        )

        if not normalized_ids:
            return []

        rows = cls._fetch_profile_rows(workspace_id, normalized_ids)

        logger.info(
            "[S3VECTORS] retrieved %d document catalog row(s) for workspace %s.",
            len(rows),
            workspace_id,
        )

        return [rows[document_id] for document_id in sorted(rows)]

    # ------------------------------------------------------------------
    # Knowledge retrieval primitives (used by HybridRetriever)
    # ------------------------------------------------------------------

    @classmethod
    def query_knowledge(
        cls,
        *,
        query_embedding,
        top_k,
        workspace_id,
        allowed_document_ids=None,
        source_types=None,
    ) -> list[dict]:
        """
        Dense search (QueryVectors) in the knowledge index.

        Returns [{"id", "text", "metadata", "distance"}] ordered best first.
        `distance` is the S3 Vectors cosine distance (1 - cosine similarity).
        The "text" key is moved out of metadata. None for
        allowed_document_ids / source_types means unrestricted; an empty
        list matches nothing.
        """
        if allowed_document_ids is not None and not list(allowed_document_ids):
            return []

        if source_types is not None and not list(source_types):
            return []

        response = cls.get_client().query_vectors(
            vectorBucketName=cls.bucket_name(),
            indexName=cls.knowledge_index(),
            queryVector={"float32": [float(value) for value in query_embedding]},
            topK=max(1, min(int(top_k), S3_VECTORS_MAX_TOP_K)),
            filter=cls._knowledge_filter(
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

    @classmethod
    def list_knowledge(
        cls,
        *,
        workspace_id,
        allowed_document_ids=None,
        source_types=None,
    ) -> dict:
        """
        Enumerate knowledge vectors (ListVectors, paginated) and filter them
        in application code by workspace / document / type.

        Returns the Chroma-style shape {"ids", "documents", "metadatas"} so
        the retriever's downstream handling is unchanged. The text is moved
        out of metadata into "documents". None for allowed_document_ids /
        source_types means unrestricted; an empty list matches nothing.
        """
        ids: list = []
        documents: list = []
        metadatas: list = []

        if allowed_document_ids is not None and not list(allowed_document_ids):
            return {"ids": ids, "documents": documents, "metadatas": metadatas}

        if source_types is not None and not list(source_types):
            return {"ids": ids, "documents": documents, "metadatas": metadatas}

        workspace_id = int(workspace_id)
        document_set = (
            {int(i) for i in allowed_document_ids}
            if allowed_document_ids is not None
            else None
        )
        type_set = {str(t) for t in source_types} if source_types is not None else None

        for vector in cls._iter_vectors(cls.knowledge_index()):
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

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    @classmethod
    def delete_document(
        cls,
        document_id,
        workspace_id=None,
    ):
        """
        Delete every vector of a document.

        Knowledge vectors are found through their document_id metadata
        (and workspace_id when supplied); the profile vector is removed by
        its deterministic key document_<id>.
        """

        errors = []

        try:
            target_document_id = int(document_id)
            target_workspace_id = (
                int(workspace_id) if workspace_id is not None else None
            )

            keys = []

            for vector in cls._iter_vectors(cls.knowledge_index()):
                metadata = vector.get("metadata")

                if not isinstance(metadata, dict):
                    continue

                try:
                    vector_document_id = int(metadata.get("document_id"))
                except (TypeError, ValueError):
                    continue

                if vector_document_id != target_document_id:
                    continue

                if target_workspace_id is not None:
                    try:
                        vector_workspace_id = int(metadata.get("workspace_id"))
                    except (TypeError, ValueError):
                        continue

                    if vector_workspace_id != target_workspace_id:
                        continue

                keys.append(vector["key"])

            cls._delete_keys(cls.knowledge_index(), keys)

            logger.info(
                "[S3VECTORS] deleted %d knowledge vector(s) for document %s.",
                len(keys),
                document_id,
            )
        except Exception as exc:
            errors.append(exc)

        try:
            cls._delete_keys(
                cls.document_index(),
                [f"document_{int(document_id)}"],
            )
        except Exception as exc:
            errors.append(exc)

        if errors:
            raise RuntimeError(
                f"Vector deletion failed for document {document_id}: {errors}"
            )