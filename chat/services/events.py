"""
Execution events emitted by the QueryPilot chat pipeline.

These events are transport-neutral. The chat service produces them;
the HTTP layer decides how to stream them to the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatEvent:
    """
    One event emitted during a QueryPilot request.

    Attributes:
        event:
            Stable event name consumed by the frontend.

        message:
            Human-readable status text.

        data:
            Optional structured metadata for the frontend.
    """

    event: str
    message: str
    data: dict[str, Any] | None = None