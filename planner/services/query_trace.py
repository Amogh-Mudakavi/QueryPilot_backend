"""Request-scoped, privacy-conscious structured tracing for planner execution."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)
_CURRENT_TRACE: ContextVar["QueryTrace | None"] = ContextVar("querypilot_query_trace", default=None)


class QueryTrace:
    def __init__(self, *, query: str, workspace_id: Any = None) -> None:
        self.trace_id = uuid.uuid4().hex
        self.query_hash = hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()[:16]
        try:
            self.workspace_id = int(workspace_id) if workspace_id is not None else None
        except (TypeError, ValueError):
            self.workspace_id = None
        self.started = time.monotonic()
        self.stages: dict[str, dict[str, float | int]] = {}
        self.counts: dict[str, int] = {}
        self._token = None
        self._closed = False

    def activate(self) -> None:
        if self._token is None:
            self._token = _CURRENT_TRACE.set(self)

    def deactivate(self) -> None:
        self.deactivate()

    def record_stage(self, name: str, seconds: float) -> None:
        entry = self.stages.setdefault(name, {"calls": 0, "seconds": 0.0})
        entry["calls"] = int(entry["calls"]) + 1
        entry["seconds"] = round(float(entry["seconds"]) + max(0.0, seconds), 4)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + max(0, int(amount))

    def close(self, *, route: str = "unknown", error_code: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        payload = {
            "event": "QUERY_TRACE",
            "trace_id": self.trace_id,
            "query_hash": self.query_hash,
            "workspace_id": self.workspace_id,
            "duration_seconds": round(time.monotonic() - self.started, 4),
            "route": str(route or "unknown")[:40],
            "error_code": str(error_code or "")[:64],
            "counts": self.counts,
            "stages": self.stages,
        }
        logger.info("QUERY_TRACE %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self.deactivate()


def current_trace() -> QueryTrace | None:
    return _CURRENT_TRACE.get()
