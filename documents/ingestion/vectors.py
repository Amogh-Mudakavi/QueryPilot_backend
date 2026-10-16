"""Provide a shared vector-storage interface for ingestion and retrieval.

Embedding generation and backend selection live here. ChromaDB is used for
local development and AWS S3 Vectors is used for production. Configure the
backend with VECTOR_BACKEND=chroma or VECTOR_BACKEND=s3. When unset, DEBUG
selects the default: ChromaDB for DEBUG=True and S3 Vectors for DEBUG=False.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import onnxruntime as ort
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

MAX_TOP_K = 100

MAX_HEADING_CHARS = 300

MAX_DOCUMENT_NAME_CHARS = 300

MAX_TABLE_NAME_CHARS = 200

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

EMBEDDING_BATCH_SIZE = getattr(
    settings,
    "EMBEDDING_BATCH_SIZE",
    32,
)

VECTOR_BATCH_SIZE = max(
    1,
    int(
        getattr(
            settings,
            "VECTOR_BATCH_SIZE",
            getattr(
                settings,
                "S3_VECTOR_BATCH_SIZE",
                getattr(settings, "CHROMA_BATCH_SIZE", 128),
            ),
        )
    ),
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _clip(value, limit: int) -> str:
    return ("" if value is None else str(value))[:limit]


def _resolve_onnx_threads() -> int:
    """
    ONNX Runtime intra-op threads for CPU embedding inference.

    EMBEDDING_ONNX_THREADS > 0 wins. Otherwise default to at most 4
    threads to limit CPU oversubscription on small/shared instances.
    """
    try:
        configured = int(
            os.getenv(
                "EMBEDDING_ONNX_THREADS",
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


def _resolve_backend_name() -> str:
    default = "chroma" if settings.DEBUG else "s3"
    name = _env("VECTOR_BACKEND", default).lower()
    if name not in ("chroma", "s3"):
        raise ImproperlyConfigured(
            "VECTOR_BACKEND must be 'chroma' or 's3'."
        )
    if name == "chroma" and not settings.DEBUG:
        raise ImproperlyConfigured(
            "VECTOR_BACKEND=chroma is only allowed when DEBUG is True. "
            "Use VECTOR_BACKEND=s3 in production."
        )
    return name


class VectorStore:
    _backend = None
    _embedder = None
    _backend_lock = threading.Lock()
    _embedder_lock = threading.Lock()
    _embedding_lock = threading.Lock()





    @classmethod
    def get_backend(cls):
        if cls._backend is None:
            with cls._backend_lock:
                if cls._backend is None:
                    name = _resolve_backend_name()


                    if name == "s3":
                        from .vector_s3 import S3VectorsBackend

                        cls._backend = S3VectorsBackend()
                    else:
                        from .vector_chroma import ChromaBackend

                        cls._backend = ChromaBackend()
                    logger.info("[VECTORS] Backend selected: %s.", name)
        return cls._backend





    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            with cls._embedder_lock:
                if cls._embedder is None:
                    threads = _resolve_onnx_threads()
                    session_options = ort.SessionOptions()
                    session_options.intra_op_num_threads = threads
                    session_options.inter_op_num_threads = 1
                    model_kwargs = {
                        "file_name": "onnx/model_quantized.onnx",
                        "provider": "CPUExecutionProvider",
                        "session_options": session_options,
                    }
                    try:
                        cls._embedder = SentenceTransformer(
                            EMBEDDING_MODEL,
                            backend="onnx",
                            model_kwargs=model_kwargs,
                            local_files_only=True,
                            trust_remote_code=False,
                        )
                    except OSError:

                        cls._embedder = SentenceTransformer(
                            EMBEDDING_MODEL,
                            backend="onnx",
                            model_kwargs=model_kwargs,
                            trust_remote_code=False,
                        )
                    logger.info(
                        "[EMBEDDING] Model ready: %s "
                        "(backend=onnx, quantized=true, onnx_intra_threads=%s, "
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





    @classmethod
    def store_chunks(
        cls,
        workspace_id,
        document_id,
        chunks,
    ):
        if not chunks:
            return
        backend = cls.get_backend()
        started = time.monotonic()
        upserted = 0
        for start in range(0, len(chunks), VECTOR_BATCH_SIZE):
            batch = chunks[start : start + VECTOR_BATCH_SIZE]
            embeddings = cls.embed([chunk["text"] for chunk in batch])
            records = []
            for chunk, embedding in zip(batch, embeddings):
                records.append(
                    {
                        "id": f"doc_{document_id}_chunk_{chunk['chunk_index']}",
                        "embedding": embedding,
                        "text": chunk["text"],
                        "metadata": {
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
                        },
                    }
                )
            backend.upsert_knowledge(records)
            upserted += len(batch)
        logger.info(
            "[VECTORS] put completed: %d chunk(s) for document %s in %.2fs.",
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
        cls.get_backend().upsert_knowledge(
            [
                {
                    "id": f"doc_{document_id}_table_{table_name}_metadata",
                    "embedding": cls.embed([summary])[0],
                    "text": summary,
                    "metadata": {
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "type": "table",
                        "table_name": _clip(table_name, MAX_TABLE_NAME_CHARS),
                    },
                }
            ]
        )
        logger.info(
            "[VECTORS] put completed: table metadata '%s' for document %s.",
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
        cls.get_backend().upsert_document_profile(
            {
                "id": f"document_{document_id}",
                "embedding": cls.embed([profile_text])[0],
                "text": profile_text,
                "metadata": {
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "document_name": _clip(
                        document_name,
                        MAX_DOCUMENT_NAME_CHARS,
                    ),
                    "type": "document_profile",
                },
            }
        )
        logger.info(
            "[VECTORS] put completed: document profile for document %s.",
            document_id,
        )





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
            rows = cls.get_backend().fetch_profile_rows(
                workspace_id,
                normalized_ids,
            )
        except Exception:
            logger.exception(
                "[VECTORS] document profile retrieval failed for workspace %s.",
                workspace_id,
            )
            return {}
        profiles = {document_id: row["profile"] for document_id, row in rows.items()}
        logger.info(
            "[VECTORS] retrieved %d document profile(s) for workspace %s.",
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
        rows = cls.get_backend().fetch_profile_rows(workspace_id, normalized_ids)
        logger.info(
            "[VECTORS] retrieved %d document catalog row(s) for workspace %s.",
            len(rows),
            workspace_id,
        )
        return [rows[document_id] for document_id in sorted(rows)]





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
        Dense search in the knowledge index.

        Returns [{"id", "text", "metadata", "distance"}] ordered best first.
        `distance` is cosine distance (1 - cosine similarity) on both backends.
        None for allowed_document_ids / source_types means unrestricted;
        an empty list matches nothing.
        """
        if allowed_document_ids is not None and not list(allowed_document_ids):
            return []
        if source_types is not None and not list(source_types):
            return []
        return cls.get_backend().query_knowledge(
            query_embedding=query_embedding,
            top_k=max(1, min(int(top_k), MAX_TOP_K)),
            workspace_id=int(workspace_id),
            allowed_document_ids=(
                [int(i) for i in allowed_document_ids]
                if allowed_document_ids is not None
                else None
            ),
            source_types=(
                [str(t) for t in source_types] if source_types is not None else None
            ),
        )

    @classmethod
    def list_knowledge(
        cls,
        *,
        workspace_id,
        allowed_document_ids=None,
        source_types=None,
    ) -> dict:
        """
        Enumerate knowledge vectors filtered by workspace / document / type.

        Returns {"ids", "documents", "metadatas"}. None for
        allowed_document_ids / source_types means unrestricted;
        an empty list matches nothing.
        """
        empty = {"ids": [], "documents": [], "metadatas": []}
        if allowed_document_ids is not None and not list(allowed_document_ids):
            return empty
        if source_types is not None and not list(source_types):
            return empty
        return cls.get_backend().list_knowledge(
            workspace_id=int(workspace_id),
            allowed_document_ids=(
                [int(i) for i in allowed_document_ids]
                if allowed_document_ids is not None
                else None
            ),
            source_types=(
                [str(t) for t in source_types] if source_types is not None else None
            ),
        )





    @classmethod
    def delete_document(
        cls,
        document_id,
        workspace_id=None,
    ):
        """Delete every vector of a document (knowledge + profile)."""
        cls.get_backend().delete_document(
            int(document_id),
            int(workspace_id) if workspace_id is not None else None,
        )
