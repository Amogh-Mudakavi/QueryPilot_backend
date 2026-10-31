"""Router module for QueryPilot.

Resolves a logical task to its profile (model, output cap, timeout, reasoning
effort) and hands the call to the provider. The profile is the only source of
generation settings: the router never looks at earlier calls in the same user
message, so a late stage such as answer generation always gets its full,
task-defined allowance.

The router does NOT know about SSE, conversations or persistence, does not
parse provider stream chunks, and does not schedule keys (the provider's quota
scheduler does that, per key and model).
"""

import logging
from collections.abc import Callable, Iterator  # QP-FIX-5
from typing import Any

from .config import (
    DEFAULT_TEMPERATURE,
    TASK_FALLBACK_MODELS,
    TASK_PROFILES,
    TaskProfile,
)
from .exceptions import (
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from .exceptions import TimeoutError as LLMTimeoutError
from .providers import get_provider

# QP-FIX-5: failures that justify trying the next model of the task.
_FALLBACK_ERRORS = (
    RateLimitError,
    InvalidResponseError,
    ProviderUnavailableError,
    LLMTimeoutError,
)


def _finished_by_length(response: Any) -> bool:
    try:
        return getattr(response.choices[0], "finish_reason", None) == "length"
    except (AttributeError, IndexError, TypeError):
        return False

logger = logging.getLogger(__name__)

VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


class LLMRouter:
    def __init__(self):
        self.provider = get_provider()

    @staticmethod
    def _profile(task: str) -> TaskProfile:
        profile = TASK_PROFILES.get(task)
        if profile is None:
            raise ValueError(f"Unknown LLM task: {task}")
        return profile

    def _get_model(self, task: str) -> str:
        return self._profile(task).model

    def _models_for(self, task: str) -> list[str]:  # QP-FIX-5
        """The task's own model first, then its configured fallbacks."""
        chain = [self._profile(task).model]
        for model in TASK_FALLBACK_MODELS.get(task, ()):
            if model and model not in chain:
                chain.append(model)
        return chain

    def _stream_with_fallback(  # QP-FIX-5
        self, *, task: str, chain: list[str], call: dict[str, Any]
    ) -> Iterator[str]:
        """Try each model while NOTHING has been yielded yet.

        Once the first text delta is out, a retry would duplicate output, so
        later failures are raised as they are.
        """
        for position, model in enumerate(chain):
            started = False
            inner = self.provider.generate_text_stream(model=model, **call)
            try:
                for delta in inner:
                    started = True
                    yield delta
                return
            except _FALLBACK_ERRORS as exc:
                if started or position + 1 >= len(chain):
                    raise
                logger.warning(
                    "[LLM_FALLBACK] task=%s model=%s failed before output "
                    "(%s: %s); trying %s",
                    task,
                    model,
                    type(exc).__name__,
                    str(exc)[:160],
                    chain[position + 1],
                )
            finally:
                close = getattr(inner, "close", None)
                if callable(close):
                    close()

    @staticmethod
    def _max_tokens(profile: TaskProfile, requested: int | None) -> int:
        """Apply the centrally configured task ceiling to every request."""
        value = profile.max_tokens if requested is None else int(requested)
        if value <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if value > profile.max_tokens:
            logger.warning(
                "Clamping requested max_tokens=%d to task cap=%d",
                value,
                profile.max_tokens,
            )
        return min(value, profile.max_tokens)

    @staticmethod
    def _reasoning_effort(
        profile: TaskProfile,
        include_reasoning: bool | None,
        reasoning_effort: str | None = None,
    ):
        if reasoning_effort is not None:
            if reasoning_effort in VALID_REASONING_EFFORTS:
                return reasoning_effort
            logger.warning(
                "Ignoring unsupported reasoning_effort=%r for task profile",
                reasoning_effort,
            )
        return "low" if include_reasoning is False else profile.reasoning_effort

    def generate(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        include_reasoning: bool | None = None,
        reasoning_effort: str | None = None,
    ):
        """Execute a normal (or raw-streaming) LLM request.

        `max_tokens` may lower the task cap for complexity-aware calls, but the
        router always clamps it to the centrally configured task ceiling.
        `reasoning_effort` is an optional per-call override.
        """
        profile = self._profile(task)
        chain = self._models_for(task)  # QP-FIX-5
        call: dict[str, Any] = dict(
            task=task,
            messages=messages,
            max_tokens=self._max_tokens(profile, max_tokens),
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            response_format=response_format,
            stream=stream,
            reasoning_effort=self._reasoning_effort(
                profile, include_reasoning, reasoning_effort
            ),
            timeout=profile.timeout,
            input_hint=profile.input_hint,
        )
        for position, model in enumerate(chain):
            is_last = position + 1 >= len(chain)
            try:
                response = self.provider.generate(model=model, **call)
            except _FALLBACK_ERRORS as exc:
                if is_last:
                    raise
                logger.warning(
                    "[LLM_FALLBACK] task=%s model=%s failed (%s: %s); trying %s",
                    task,
                    model,
                    type(exc).__name__,
                    str(exc)[:160],
                    chain[position + 1],
                )
                continue
            if not stream and not is_last and _finished_by_length(response):
                logger.warning(
                    "[LLM_FALLBACK] task=%s model=%s hit max_tokens "
                    "(finish=length); trying %s",
                    task,
                    model,
                    chain[position + 1],
                )
                continue
            return response
        raise ProviderUnavailableError("No model produced a response.")

    def generate_stream(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        on_finish: Callable[[str | None], None] | None = None,  # QP-FIX-1
    ) -> Iterator[str]:
        """Start a text stream; the application receives plain text deltas.

        on_finish(finish_reason) is called once when the stream ends normally.
        """
        profile = self._profile(task)
        chain = self._models_for(task)  # QP-FIX-5
        call: dict[str, Any] = dict(
            task=task,
            messages=messages,
            max_tokens=self._max_tokens(profile, max_tokens),
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            reasoning_effort=self._reasoning_effort(
                profile, include_reasoning, reasoning_effort
            ),
            timeout=profile.timeout,
            input_hint=profile.input_hint,
            on_finish=on_finish,
        )
        return self._stream_with_fallback(task=task, chain=chain, call=call)


router = LLMRouter()