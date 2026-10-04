"""Passive token-usage recording for observability.

This answers "how many tokens did this user message actually use, and which
key/model handled each call?" and nothing else. It is write-only from the
provider's point of view: no provider, router or scheduler code ever reads it
to decide anything. It cannot refuse a call, cap a completion, or change a
retry, and it never raises into the caller.

Usage (chat entry point):

    with track_usage("conv=42 msg=7") as usage:
        ... run the whole pipeline ...
    # on exit it logs one [LLM_REQUEST_SUMMARY] line; usage.total_tokens is
    # also available inside the block.

Recording follows the active ``contextvars`` context. Plain worker threads do
not inherit it; start them with ``contextvars.copy_context().run(...)`` if their
calls should be included.
"""

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallUsage:
    task: str
    model: str
    key_index: int
    attempt: int
    status: str  # "success", "failed", or "stream_incomplete"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    latency_seconds: float | None = None


class UsageLog:
    def __init__(self, label: str = ""):
        self.label = label
        self._calls: list[CallUsage] = []
        self._lock = threading.Lock()

    def add(self, call: CallUsage) -> None:
        with self._lock:
            self._calls.append(call)

    @property
    def calls(self) -> list[CallUsage]:
        with self._lock:
            return list(self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens or 0 for c in self.calls)

    def by_task(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            totals[call.task] = totals.get(call.task, 0) + (call.total_tokens or 0)
        return totals

    def summary(self) -> str:
        calls = self.calls
        placements = ",".join(
            f"{c.task}@{c.model.rsplit('/', 1)[-1]}/k{c.key_index}" for c in calls
        )
        return (
            f"[LLM_REQUEST_SUMMARY] label={self.label} calls={len(calls)} "
            f"total_tokens={self.total_tokens} by_task={self.by_task()} "
            f"placements={placements}"
        )


_current: ContextVar[UsageLog | None] = ContextVar("llm_usage_log", default=None)


@contextmanager
def track_usage(label: str = ""):
    log = UsageLog(label)
    token = _current.set(log)
    try:
        yield log
    finally:
        _current.reset(token)
        try:
            logger.info(log.summary())
        except Exception:  # observability must never break a request
            pass


def record(call: CallUsage) -> None:
    """Called by the provider after each attempt; no-op outside track_usage."""
    log = _current.get()
    if log is not None:
        try:
            log.add(call)
        except Exception:
            pass