"""
chat/services/sources.py

Deterministic source extraction for QueryPilot.

This module builds an INDEXED source list from an agent's final
evidence. It performs no LLM calls and makes no decisions about what
the answer says -- it only describes, in a stable/reproducible way,
where each piece of evidence came from.

Indexing contract (this is the piece that makes citations work):

    Each source row's "index" is the 1-based POSITION of the evidence
    item it was extracted from, in the exact same `evidence` list/
    order that was supplied to answer generation
    (planner.services.answer.generate_answer, which serializes
    `evidence` as an ordered JSON array and does not reorder it).

    The answer prompt (planner/prompts/answer.py) instructs the model
    to cite a claim using [n] where n is that same 1-based evidence
    position. Because both sides read the same list in the same
    order, index N here always means the same thing as marker [N] in
    the answer text -- no separate renumbering, lookup table, or
    guessing is required downstream (see chat.services.citations).

This intentionally does NOT deduplicate source rows *across* evidence
items (two different evidence items that happen to reference the same
document/page keep their own distinct indices), because collapsing
them would break the 1:1 correspondence between citation markers and
evidence positions. Duplicate rows produced *within* a single
evidence item (e.g. the same table listed twice) are still deduplicated,
since those would otherwise be indistinguishable duplicates under one
citation marker.

Building on top of / preserving the field extraction that the
previous chat.services.execution._build_sources() implementation
used, so both normal document evidence (top-level or metadata-embedded
document_id, chunk_index, page_start/page_end, heading) and
table-query evidence (an embedded `sources` list of
{document_id, table_name} entries) continue to be supported.

Workspace scoping:

    Sources are resolved against the current workspace.

    A document can only appear in the returned source list when:

        Document.id == referenced document_id
        AND
        Document.workspace_id == workspace_id

    Evidence referring to a document outside the supplied workspace
    is ignored rather than returned with incomplete metadata.
"""

from __future__ import annotations

from typing import Any

from documents.models import Document


def build_sources(
    evidence: list[Any],
    *,
    workspace_id: int,
) -> list[dict[str, Any]]:
    """
    Build an indexed source list from final evidence.

    Parameters:
        evidence:
            Final evidence produced by the agent, in the exact order
            supplied to answer generation.

        workspace_id:
            Workspace that owns the active conversation. Every
            referenced document must belong to this workspace.

    Returns:
        A flat list of source dictionaries. Every returned source has
        an "index" field containing the 1-based position of the
        evidence item from which it was extracted.

    Workspace safety:
        Document references are resolved using both document ID and
        workspace ID. Foreign-workspace document references are
        excluded completely.
    """

    if not evidence:
        return []

    document_ids: set[int] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}

        document_id = (
            item.get("document_id")
            or metadata.get("document_id")
        )

        if document_id is not None:
            try:
                document_ids.add(int(document_id))
            except (TypeError, ValueError):
                pass

        sources_list = (
            item.get("sources")
            or metadata.get("sources")
        )

        if isinstance(sources_list, list):
            for src in sources_list:
                if not isinstance(src, dict):
                    continue

                did = src.get("document_id")

                if did is None:
                    continue

                try:
                    document_ids.add(int(did))
                except (TypeError, ValueError):
                    continue

    # ---------------------------------------------------------
    # Resolve only documents belonging to the active workspace.
    #
    # This is the defense-in-depth boundary. Even if malformed or
    # unexpected evidence contains a document ID from another
    # workspace, it will never enter the returned source list.
    # ---------------------------------------------------------

    document_names: dict[int, str] = {}

    if document_ids:
        documents = (
            Document.objects.filter(
                id__in=document_ids,
                workspace_id=workspace_id,
            )
            .only(
                "id",
                "title",
            )
        )

        document_names = {
            document.id: document.title
            for document in documents
        }

    # Only document IDs that were actually resolved inside the
    # current workspace are allowed to become source rows.
    valid_document_ids = set(
        document_names.keys()
    )

    sources: list[dict[str, Any]] = []

    for position, item in enumerate(
        evidence,
        start=1,
    ):
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}

        rows_for_item: list[dict[str, Any]] = []

        # ---------------------------------------------------------
        # Case A:
        # Normal document evidence.
        #
        # Supports top-level or metadata-embedded document_id.
        # ---------------------------------------------------------

        document_id = (
            item.get("document_id")
            or metadata.get("document_id")
        )

        if document_id is not None:
            try:
                document_id = int(document_id)
            except (TypeError, ValueError):
                document_id = None

        # A document reference is accepted only when the document
        # was successfully resolved inside the current workspace.
        if (
            document_id is not None
            and document_id in valid_document_ids
        ):
            row: dict[str, Any] = {
                "document_id": document_id,
                "document_name": document_names[
                    document_id
                ],
                "source_type": (
                    item.get("source_type")
                    or metadata.get("type")
                    or "document"
                ),
            }

            chunk_index = (
                item.get("chunk_index")
                or metadata.get("chunk_index")
            )

            page_start = (
                item.get("page_start")
                or metadata.get("page_start")
            )

            page_end = (
                item.get("page_end")
                or metadata.get("page_end")
            )

            heading = (
                item.get("heading")
                or metadata.get("heading")
            )

            table_name = (
                item.get("table_name")
                or metadata.get("table_name")
            )

            if chunk_index is not None:
                row["chunk_index"] = chunk_index

            if page_start is not None:
                row["page_start"] = page_start

            if page_end is not None:
                row["page_end"] = page_end

            if heading:
                row["heading"] = heading

            if table_name:
                row["table_name"] = table_name

            rows_for_item.append(row)

        # ---------------------------------------------------------
        # Case B:
        # Table-query style evidence with an embedded `sources`
        # list of per-table {document_id, table_name}.
        # ---------------------------------------------------------

        sources_list = (
            item.get("sources")
            or metadata.get("sources")
        )

        if isinstance(sources_list, list):
            for src in sources_list:
                if not isinstance(src, dict):
                    continue

                doc_id = src.get("document_id")

                if doc_id is None:
                    continue

                try:
                    doc_id = int(doc_id)
                except (TypeError, ValueError):
                    continue

                # Foreign-workspace document references are excluded.
                if doc_id not in valid_document_ids:
                    continue

                table_name = src.get("table_name")

                table_row: dict[str, Any] = {
                    "document_id": doc_id,
                    "document_name": document_names[
                        doc_id
                    ],
                    "source_type": "table",
                }

                if table_name:
                    table_row["table_name"] = table_name

                rows_for_item.append(table_row)

        # ---------------------------------------------------------
        # Dedupe rows produced within this one evidence item only.
        #
        # Evidence items retain their original citation index.
        # ---------------------------------------------------------

        seen_row_keys: set[tuple[Any, ...]] = set()

        for row in rows_for_item:
            row_key = (
                row.get("document_id"),
                row.get("chunk_index"),
                row.get("page_start"),
                row.get("page_end"),
                row.get("heading"),
                row.get("table_name"),
            )

            if row_key in seen_row_keys:
                continue

            seen_row_keys.add(row_key)

            row["index"] = position
            sources.append(row)

    return sources