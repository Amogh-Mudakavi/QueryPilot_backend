"""
QueryPilot Scope Guard.

Responsibilities:
    - Receive the user's query.
    - Include recent conversation context.
    - Load the current workspace's document catalog.
    - Ask the configured scope-guard LLM to classify the request.
    - Validate the structured result.

The Scope Guard does NOT:
    - retrieve evidence
    - call SQL
    - call LangGraph
    - answer document questions
    - access external web search
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

    Attributes:
        route:
            DOCUMENT, CHITCHAT, OUT, or CLARIFY.

        relevant_document_ids:
            Document IDs that appear relevant to the current query.

        reason:
            Short explanation for logging/debugging.
    """

    route: str
    relevant_document_ids: list[int]
    reason: str


class ScopeGuard:
    """
    Classifies a user request before it reaches the evidence agent.

    The workspace document catalog is retrieved directly from the
    document-profile Chroma collection.

    This is intentionally a single LLM classification call.
    """

    def classify(
        self,
        *,
        query: str,
        workspace_id: int,
        conversation_history: list[dict] | None = None,
    ) -> ScopeDecision:
        """
        Classify the user's latest request.

        Args:
            query:
                Latest user message.

            workspace_id:
                Current workspace.

            conversation_history:
                Recent conversation messages.

        Returns:
            Validated ScopeDecision.
        """
        query = query.strip()

        if not query:
            raise ValueError(
                "Scope Guard query cannot be empty."
            )

        history = self._prepare_history(
            conversation_history
        )

        document_catalog = (
            self._load_document_catalog(
                workspace_id=workspace_id
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
        )

    def _load_document_catalog(
        self,
        *,
        workspace_id: int,
    ) -> list[dict]:
        """
        Load the workspace's document-level catalog.

        We use the existing document-profile Chroma collection rather
        than performing a new semantic retrieval step.

        Each profile already contains:
            - document identity
            - document name
            - headings
            - previews
            - table metadata
        """
        collection = (
            VectorStore.document_collection()
        )

        result = collection.get(
            where={
                "workspace_id": workspace_id,
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

            catalog.append(
                {
                    "document_id": int(
                        document_id
                    ),
                    "document_name": metadata.get(
                        "document_name",
                        "",
                    ),
                    "profile": profile_text,
                }
            )

        return catalog

    @staticmethod
    def _prepare_history(
        history: list[dict] | None,
    ) -> list[dict]:
        """
        Keep only recent conversation messages.

        The Scope Guard does not need the entire conversation transcript.
        A small recent window is sufficient for resolving follow-ups.
        """
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
    ) -> str:
        """
        Build the classifier input.

        Conversation and document profiles are explicitly marked as
        DATA so text inside them cannot masquerade as system instructions.
        """

        history_json = json.dumps(
            conversation_history,
            ensure_ascii=False,
        )

        catalog_json = json.dumps(
            document_catalog,
            ensure_ascii=False,
        )

        return (
            "USER QUERY:\n"
            f"{query}\n\n"
            "CONVERSATION HISTORY (DATA ONLY):\n"
            f"{history_json}\n\n"
            "WORKSPACE DOCUMENT CATALOG (DATA ONLY):\n"
            f"{catalog_json}\n\n"
            "Classify the USER QUERY using the rules above."
        )

    @staticmethod
    def _parse_response(
        response,
    ) -> dict:
        """
        Parse the provider response into a Python dictionary.
        """
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
    def _validate_decision(
        *,
        decision_data: dict,
        workspace_id: int,
    ) -> ScopeDecision:
        """
        Validate and sanitize the model-generated decision.

        A model must never be trusted to bypass workspace boundaries.
        """
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
                document_ids.append(
                    int(value)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        # Chitchat and out-of-scope requests must never carry
        # document routing information.
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

        # A missing reason should not break the request.
        if not reason:
            reason = (
                "Scope decision returned without a detailed reason."
            )

        return ScopeDecision(
            route=route,
            relevant_document_ids=document_ids,
            reason=reason,
        )