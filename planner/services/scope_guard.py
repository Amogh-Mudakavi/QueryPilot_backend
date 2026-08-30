"""
QueryPilot Scope Guard.

Responsibilities:
    - Receive the user's query.
    - Include recent conversation context.
    - Load the current conversation's attached document catalog.
    - Ask the configured scope-guard LLM to classify the request.
    - Validate the structured result.

The Scope Guard does NOT:
    - retrieve evidence
    - call SQL
    - call LangGraph
    - answer document questions
    - access external web search

The conversation's attached document IDs are the authoritative
source boundary. Chroma document profiles are supplemental
classification context only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from documents.ingestion.vectors import VectorStore
from llm.router import router

from planner.prompts.scope_guard import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


ALLOWED_ROUTES = {
    "DOCUMENT",
    "CHITCHAT",
    "OUT",
    "CLARIFY",
}


MAX_DOCUMENTS = 3
MAX_HISTORY_MESSAGES = 8


@dataclass(frozen=True)
class ScopeDecision:
    """
    Result returned by the Scope Guard.
    """

    route: str
    relevant_document_ids: list[int]
    reason: str


class ScopeGuard:
    """
    Classifies a user request before it reaches the evidence agent.

    The conversation's attached document IDs are authoritative.

    Chroma document profiles are supplemental context for the classifier.
    They are never the security boundary.
    """

    def classify(
        self,
        *,
        query: str,
        workspace_id: int,
        allowed_document_ids: list[int],
        conversation_history: list[dict] | None = None,
    ) -> ScopeDecision:

        query = query.strip()

        if not query:
            raise ValueError(
                "Scope Guard query cannot be empty."
            )

        if workspace_id <= 0:
            raise ValueError(
                "Scope Guard workspace_id must be greater than zero."
            )

        allowed_ids = {
            int(document_id)
            for document_id in allowed_document_ids
        }

        history = self._prepare_history(
            conversation_history
        )

        document_catalog = (
            self._load_document_catalog(
                workspace_id=workspace_id,
                allowed_document_ids=allowed_ids,
            )
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_user_prompt(
                    query=query,
                    conversation_history=history,
                    document_catalog=document_catalog,
                    allowed_document_ids=sorted(
                        allowed_ids
                    ),
                ),
            },
        ]

        response = router.generate(
            task="scope_guard",
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "scope_guard_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "route": {
                                "type": "string",
                                "enum": [
                                    "DOCUMENT",
                                    "CHITCHAT",
                                    "OUT",
                                    "CLARIFY",
                                ],
                            },
                            "relevant_document_ids": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "maxItems": MAX_DOCUMENTS,
                            },
                            "reason": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "route",
                            "relevant_document_ids",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            include_reasoning=False,
        )

        decision_data = self._parse_response(
            response
        )

        return self._validate_decision(
            decision_data=decision_data,
            workspace_id=workspace_id,
            allowed_document_ids=allowed_ids,
            query=query,
        )

    def _load_document_catalog(
        self,
        *,
        workspace_id: int,
        allowed_document_ids: set[int],
    ) -> list[dict]:
        """
        Load document profiles only for documents actually attached
        to the current conversation.
        """

        if not allowed_document_ids:
            return []

        collection = (
            VectorStore.document_collection()
        )

        result = collection.get(
            where={
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
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        catalog: list[dict] = []

        for index, profile_text in enumerate(
            documents
        ):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            document_id = metadata.get(
                "document_id"
            )

            if document_id is None:
                continue

            try:
                document_id = int(
                    document_id
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if document_id not in allowed_document_ids:
                continue

            catalog.append(
                {
                    "document_id": document_id,
                    "document_name": metadata.get(
                        "document_name",
                        "",
                    ),
                    "profile": profile_text or "",
                }
            )

        return catalog

    @staticmethod
    def _prepare_history(
        history: list[dict] | None,
    ) -> list[dict]:
        if not history:
            return []

        return history[
            -MAX_HISTORY_MESSAGES:
        ]

    @staticmethod
    def _build_user_prompt(
        *,
        query: str,
        conversation_history: list[dict],
        document_catalog: list[dict],
        allowed_document_ids: list[int],
    ) -> str:

        history_json = json.dumps(
            conversation_history,
            ensure_ascii=False,
        )

        catalog_json = json.dumps(
            document_catalog,
            ensure_ascii=False,
        )

        allowed_ids_json = json.dumps(
            allowed_document_ids
        )

        return (
            "USER QUERY:\n"
            f"{query}\n\n"
            "CONVERSATION HISTORY (DATA ONLY):\n"
            f"{history_json}\n\n"
            "ATTACHED DOCUMENT IDS (AUTHORITATIVE):\n"
            f"{allowed_ids_json}\n\n"
            "ATTACHED DOCUMENT CATALOG (DATA ONLY):\n"
            f"{catalog_json}\n\n"
            "Classify the USER QUERY using the rules above."
        )

    @staticmethod
    def _parse_response(
        response,
    ) -> dict:

        try:
            content = (
                response
                .choices[0]
                .message
                .content
            )

        except (
            AttributeError,
            IndexError,
        ) as exc:
            raise ValueError(
                "Scope Guard received an invalid LLM response."
            ) from exc

        if not content:
            raise ValueError(
                "Scope Guard returned an empty response."
            )

        try:
            data = json.loads(
                content
            )

        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid Scope Guard JSON: %s",
                content,
            )

            raise ValueError(
                "Scope Guard returned invalid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Scope Guard response must be a JSON object."
            )

        return data

    @staticmethod
    def _looks_like_document_question(
        query: str,
    ) -> bool:
        """
        Conservative deterministic fallback.

        This does not try to understand document contents.
        It only identifies common forms of questions that clearly ask
        for factual/document/table information.

        The purpose is to prevent the LLM from incorrectly returning
        OUT for an obvious document-backed request.
        """

        normalized = " ".join(
            query.lower().split()
        )

        if not normalized:
            return False

        question_markers = (
            "what is ",
            "what are ",
            "who is ",
            "who are ",
            "which ",
            "show ",
            "list ",
            "how much ",
            "how many ",
            "how does ",
            "how do ",
            "calculate ",
            "according to ",
            "tell me ",
            "give me ",
            "find ",
            "compare ",
            "does ",
            "is ",
            "are ",
        )

        for marker in question_markers:
            if normalized.startswith(marker):
                return True

        data_terms = (
            "salary",
            "employee",
            "employee id",
            "id",
            "leave",
            "policy",
            "deduction",
            "table",
            "record",
            "records",
            "name",
            "amount",
            "price",
            "total",
            "count",
            "average",
            "maximum",
            "minimum",
            "highest",
            "lowest",
        )

        return any(
            term in normalized
            for term in data_terms
        )

    @staticmethod
    def _is_obvious_chitchat(
        query: str,
    ) -> bool:
        """
        Keep CHITCHAT deliberately narrow.
        """

        normalized = " ".join(
            query.lower().split()
        )

        return normalized in {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "thanks!",
            "hello!",
            "hi!",
            "hey!",
        }

    @staticmethod
    def _validate_decision(
        *,
        decision_data: dict,
        workspace_id: int,
        allowed_document_ids: set[int],
        query: str,
    ) -> ScopeDecision:

        route = decision_data.get(
            "route"
        )

        if route not in ALLOWED_ROUTES:
            raise ValueError(
                f"Invalid Scope Guard route: {route}"
            )

        raw_document_ids = (
            decision_data.get(
                "relevant_document_ids",
                [],
            )
        )

        if not isinstance(
            raw_document_ids,
            list,
        ):
            raw_document_ids = []

        document_ids: list[int] = []

        for value in raw_document_ids:

            try:
                candidate_id = int(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

            if candidate_id not in allowed_document_ids:
                logger.warning(
                    "Scope Guard returned document_id %s "
                    "outside conversation's allowed documents "
                    "%s; dropping it.",
                    candidate_id,
                    sorted(
                        allowed_document_ids
                    ),
                )
                continue

            if candidate_id not in document_ids:
                document_ids.append(
                    candidate_id
                )

        # Chitchat remains unconditionally chitchat.
        if ScopeGuard._is_obvious_chitchat(query):
            return ScopeDecision(
                route="CHITCHAT",
                relevant_document_ids=[],
                reason=(
                    "Query is an obvious lightweight conversational message."
                ),
            )

        # A non-empty attached-document set means an obvious factual,
        # tabular, policy, lookup, or calculation request belongs in
        # the document pipeline. Do not allow the LLM to reject it as OUT.
        if (
            allowed_document_ids
            and ScopeGuard._looks_like_document_question(
                query
            )
        ):
            if route in {
                "OUT",
                "CLARIFY",
            }:
                logger.info(
                    "Scope Guard overriding route=%s to DOCUMENT "
                    "for document-backed query; workspace=%s "
                    "allowed_document_ids=%s query=%r",
                    route,
                    workspace_id,
                    sorted(
                        allowed_document_ids
                    ),
                    query,
                )

                route = "DOCUMENT"

        # For document-backed questions, if the LLM failed to select
        # IDs, use the authoritative attached-document set. This is
        # routing fallback, not document-content inference.
        if (
            route == "DOCUMENT"
            and not document_ids
            and allowed_document_ids
        ):
            document_ids = sorted(
                allowed_document_ids
            )[:MAX_DOCUMENTS]

        # Never allow an LLM to attach document IDs to these routes.
        if route in {
            "CHITCHAT",
            "OUT",
        }:
            document_ids = []

        reason = str(
            decision_data.get(
                "reason",
                "",
            )
        ).strip()

        if not reason:
            reason = (
                "Scope decision returned without a detailed reason."
            )

        return ScopeDecision(
            route=route,
            relevant_document_ids=document_ids,
            reason=reason,
        )