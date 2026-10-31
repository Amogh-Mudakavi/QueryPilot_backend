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
from collections.abc import Iterator
from typing import Any

from .config import DEFAULT_TEMPERATURE, TASK_PROFILES, TaskProfile
from .providers import get_provider

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
        return self.provider.generate(
            task=task,
            model=profile.model,
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

    def generate_stream(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_reasoning: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Start a text stream; the application receives plain text deltas."""
        profile = self._profile(task)
        return self.provider.generate_text_stream(
            task=task,
            model=profile.model,
            messages=messages,
            max_tokens=self._max_tokens(profile, max_tokens),
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            reasoning_effort=self._reasoning_effort(
                profile, include_reasoning, reasoning_effort
            ),
            timeout=profile.timeout,
            input_hint=profile.input_hint,
        )


router = LLMRouter()