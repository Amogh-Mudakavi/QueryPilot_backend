from collections.abc import Iterator
from typing import Any

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TASK_MODELS,
)

from .providers import get_provider


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
            raise ValueError(
                f"Unknown LLM task: {task}"
            )

        return model

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

        return self.provider.generate(
            task=task,
            model=model,
            messages=messages,
            temperature=(
                DEFAULT_TEMPERATURE
                if temperature is None
                else temperature
            ),
            max_tokens=(
                DEFAULT_MAX_TOKENS
                if max_tokens is None
                else max_tokens
            ),
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

        return self.provider.generate_text_stream(
            task=task,
            model=model,
            messages=messages,
            temperature=(
                DEFAULT_TEMPERATURE
                if temperature is None
                else temperature
            ),
            max_tokens=(
                DEFAULT_MAX_TOKENS
                if max_tokens is None
                else max_tokens
            ),
            include_reasoning=include_reasoning,
        )


router = LLMRouter()
LLMRouter = LLMRouter