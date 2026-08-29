"""
retrieval/services/evidence.py

Standard evidence objects used throughout QueryPilot.

Every retrieval source (text chunks, tables, future web search,
knowledge graph, etc.) should be converted into one of these
objects before being passed to the planner.

This gives the planner a consistent interface regardless of
where the evidence originated.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Evidence:
    """
    Represents a single piece of retrieved evidence.

    Examples:
    ---------
    Text Chunk
        Refund requests are eligible within 7 days...

    Table Metadata
        Columns: Product Name, Price, Stock

    Table Row
        Rahul | 101 | 85000

    Future Web Result
        Official company refund policy...
    """

    # Actual evidence content
    content: str

    # Retrieval score after dense/BM25 fusion
    score: float

    # Source type
    # Examples:
    # text_chunk
    # table_metadata
    # table_row
    # web_result
    source_type: str

    # Workspace ownership
    workspace_id: int

    # Origin document
    document_id: int

    # Optional document name for debugging/UI
    document_name: Optional[str] = None

    # Additional metadata from Chroma / SQLite
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize evidence into a JSON-safe structure.
        """

        return {
            "content": self.content,
            "score": self.score,
            "source_type": self.source_type,
            "workspace_id": self.workspace_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "metadata": self.metadata,
        }

    @property
    def is_table(self) -> bool:
        """
        Returns True if evidence originated from a table.
        """

        return self.source_type in {
            "table_metadata",
            "table_row",
        }

    @property
    def is_text(self) -> bool:
        """
        Returns True if evidence originated from text chunks.
        """

        return self.source_type == "text_chunk"