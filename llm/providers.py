import logging
import time

from groq import Groq

from .config import (
    GROQ_API_KEYS,
    MAX_RETRIES,
    TASK_KEY_OFFSET,
)
from .exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class LLMResponse:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class GroqProvider:

    def __init__(self):
        self.clients = [
            Groq(
                api_key=key
            )
            for key in GROQ_API_KEYS
        ]

        self.counters = {}

    def _get_client(self, task):
        counter = self.counters.get(task, 0)

        offset = TASK_KEY_OFFSET.get(
            task,
            0,
        )

        index = (
            counter + offset
        ) % len(self.clients)

        self.counters[task] = counter + 1

        return self.clients[index]

    def generate(
        self,
        *,
        task,
        model,
        messages,
        temperature=0.0,
        max_tokens=None,
        response_format=None,
        stream=False,
        include_reasoning=None,
    ):
        extra_params = {}

        # Reasoning models (e.g. gpt-oss-20b) spend completion
        # tokens on internal reasoning before writing output.
        # include_reasoning=False tells Groq to keep that low so
        # max_tokens isn't consumed before the JSON is produced.
        if include_reasoning is False:
            extra_params["reasoning_effort"] = "low"

        last_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            client = self._get_client(task)

            try:

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    stream=stream,
                    **extra_params,
                )

                if stream:
                    return response

                if not response.choices:
                    raise InvalidResponseError(
                        "Groq returned no choices."
                    )

                return response

            except Exception as exc:

                last_error = exc

                status = getattr(
                    exc,
                    "status_code",
                    None,
                )

                logger.warning(
                    "[LLM] task=%s model=%s "
                    "attempt=%s status=%s error=%s",
                    task,
                    model,
                    attempt,
                    status,
                    exc,
                )

                if status in (401, 403):
                    raise AuthenticationError(
                        str(exc)
                    ) from exc

                if status == 429:
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue

                    raise RateLimitError(
                        str(exc)
                    ) from exc

                if status == 400:
                    raise InvalidResponseError(
                        str(exc)
                    ) from exc

                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue

        raise ProviderUnavailableError(
            f"Groq failed: {last_error}"
        )


_provider = None


def get_provider():
    global _provider

    if _provider is None:
        _provider = GroqProvider()

    return _provider