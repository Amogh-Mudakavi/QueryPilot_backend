"""Request-scoped, privacy-conscious structured tracing for planner execution.

A trace may move between LangGraph worker contexts. ContextVar tokens therefore
belong to the current execution context, not to the QueryTrace instance. The
trace's mutable aggregates are lock-protected because parallel graph work may
record stages concurrently.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)
_CURRENT_TRACE: ContextVar["QueryTrace | None"] = ContextVar(
    "querypilot_query_trace", default=None
)
_BINDING_STACK: ContextVar[tuple[tuple[str, Token], ...]] = ContextVar(
    "querypilot_query_trace_bindings", default=()
)


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
        self._closed = False
        self._lock = threading.Lock()

    def activate(self) -> bool:
        """Bind this trace in the current context; return whether bound here."""
        if _CURRENT_TRACE.get() is self:
            return False
        token = _CURRENT_TRACE.set(self)
        stack = _BINDING_STACK.get()
        _BINDING_STACK.set(stack + ((self.trace_id, token),))
        return True

    def deactivate(self) -> None:
        """Restore the previous trace binding in this execution context.

        This method is idempotent. It never calls itself recursively and never
        resets a token created in another context.
        """
        stack = _BINDING_STACK.get()
        if not stack or stack[-1][0] != self.trace_id:
            # The trace may not have been activated in this context, or a nested
            # scope may still own the top binding. Do not corrupt ContextVar state.
            return
        _, token = stack[-1]
        _BINDING_STACK.set(stack[:-1])
        try:
            _CURRENT_TRACE.reset(token)
        except (ValueError, RuntimeError):
            # Defensive recovery for an already-invalidated context token.
            logger.warning("Query trace binding could not be reset; clearing current context")
            if _CURRENT_TRACE.get() is self:
                _CURRENT_TRACE.set(None)

    def record_stage(self, name: str, seconds: float) -> None:
        safe_name = str(name or "unknown")[:100]
        elapsed = max(0.0, float(seconds))
        with self._lock:
            entry = self.stages.setdefault(safe_name, {"calls": 0, "seconds": 0.0})
            entry["calls"] = int(entry["calls"]) + 1
            entry["seconds"] = round(float(entry["seconds"]) + elapsed, 4)

    def increment(self, name: str, amount: int = 1) -> None:
        safe_name = str(name or "unknown")[:100]
        increment = max(0, int(amount))
        with self._lock:
            self.counts[safe_name] = self.counts.get(safe_name, 0) + increment

    def close(self, *, route: str = "unknown", error_code: str = "") -> None:
        """Emit one immutable snapshot; safe to call more than once."""
        with self._lock:
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
                "counts": dict(self.counts),
                "stages": {
                    name: dict(values) for name, values in self.stages.items()
                },
            }
        logger.info(
            "QUERY_TRACE %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


def current_trace() -> QueryTrace | None:
    return _CURRENT_TRACE.get()
