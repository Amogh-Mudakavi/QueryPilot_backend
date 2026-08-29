from typing import Any

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TASK_MODELS,
)

from .providers import get_provider


class LLMRouter:

    def __init__(self):
        self.provider = get_provider()

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
        model = TASK_MODELS.get(task)

        if model is None:
            raise ValueError(f"Unknown LLM task: {task}")

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


router = LLMRouter()
LLMRouter = LLMRouter