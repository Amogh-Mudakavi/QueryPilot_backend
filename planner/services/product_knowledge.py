from __future__ import annotations

PRODUCT_KNOWLEDGE = (
    "QueryPilot is a workspace-based AI assistant that answers questions "
    "using uploaded documents and structured tables. It combines document "
    "retrieval, safe read-only table queries, conversation context, and "
    "grounded answer generation with sources."
)


def get_product_knowledge() -> dict:
    return {
        "source_type": "product_knowledge",
        "content": PRODUCT_KNOWLEDGE,
        "metadata": {"authoritative": True},
    }
