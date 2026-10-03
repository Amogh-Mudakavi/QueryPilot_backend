"""Router module for QueryPilot."""

import logging
from collections.abc import Iterator
from typing import Any

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TASK_MODELS,
    TASK_TOKEN_BUDGETS,
)
from .providers import get_provider

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Central routing layer for all LLM tasks.

    Responsibilities:
        - Resolve the model associated with a logical task.
        - Apply shared defaults.
        - Expose normal LLM generation.
        - Expose provider-independent text streaming.

    The router does NOT:
        - know about SSE
        - know about chat conversations
        - persist application data
        - parse provider-specific stream chunks
    """

    def __init__(self):
        self.provider = get_provider()

    def _get_model(
        self,
        task: str,
    ) -> str:
        model = TASK_MODELS.get(task)

        if model is None:
            raise ValueError(f"Unknown LLM task: {task}")

        return model

    @staticmethod
    def _resolve_max_tokens(task: str, max_tokens: int | None) -> int:
        if max_tokens is not None:
            return max_tokens
        budget = TASK_TOKEN_BUDGETS.get(task, {})
        return int(budget.get("completion", DEFAULT_MAX_TOKENS))

    @staticmethod
    def _check_input_budget(task: str, messages: list[dict[str, str]]) -> None:
        """Log (never truncate) when a request exceeds its input target."""
        target = TASK_TOKEN_BUDGETS.get(task, {}).get("input")
        if not target:
            return
        chars = sum(len(str(m.get("content", ""))) for m in messages or [])
        estimate = (chars + 3) // 4
        if estimate > target:
            logger.warning(
                "[LLM_BUDGET] task=%s estimated_input_tokens=%d exceeds "
                "target=%d; reduce evidence/history instead of raising limits",
                task,
                estimate,
                target,
            )

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
    ):
        """
        Execute a normal LLM request.

        Existing callers continue to use this method unchanged.
        """

        model = self._get_model(task)
        self._check_input_budget(task, messages)

        return self.provider.generate(
            task=task,
            model=model,
            messages=messages,
            temperature=(DEFAULT_TEMPERATURE if temperature is None else temperature),
            max_tokens=self._resolve_max_tokens(task, max_tokens),
            response_format=response_format,
            stream=stream,
            include_reasoning=include_reasoning,
        )

    def generate_stream(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_reasoning: bool | None = None,
    ) -> Iterator[str]:
        """
        Start a true text streaming request.

        The provider owns all provider-specific stream parsing.
        The application receives only plain text deltas.
        """

        model = self._get_model(task)
        self._check_input_budget(task, messages)

        return self.provider.generate_text_stream(
            task=task,
            model=model,
            messages=messages,
            temperature=(DEFAULT_TEMPERATURE if temperature is None else temperature),
            max_tokens=self._resolve_max_tokens(task, max_tokens),
            include_reasoning=include_reasoning,
        )


router = LLMRouter()
