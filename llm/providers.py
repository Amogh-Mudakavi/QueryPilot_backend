import logging
import random
import time

from groq import Groq

from .config import (
    GROQ_API_KEYS,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
    TASK_KEY_OFFSET,
)
from .exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
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
                api_key=key,
                timeout=REQUEST_TIMEOUT,
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

    def _get_retry_delay(self, attempt: int) -> float:
        """
        Small exponential backoff with jitter.

        attempt=1 -> about 1s
        attempt=2 -> about 2s
        """
        base_delay = (
            RETRY_BACKOFF_SECONDS
            * (2 ** (attempt - 1))
        )

        jitter = random.uniform(0, 0.25)

        return base_delay + jitter

    @staticmethod
    def _get_status_code(exc):
        return getattr(
            exc,
            "status_code",
            None,
        )

    @staticmethod
    def _is_retryable_status(status):
        return status in {
            429,
            500,
            502,
            503,
            504,
        }

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

        # Reasoning models can spend completion tokens on internal
        # reasoning. Keep reasoning low when explicitly requested.
        if include_reasoning is False:
            extra_params["reasoning_effort"] = "low"

        last_error = None

        for attempt in range(
            1,
            MAX_ATTEMPTS + 1,
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

                status = self._get_status_code(exc)

                logger.warning(
                    "[LLM] task=%s model=%s attempt=%s/%s "
                    "status=%s error=%s",
                    task,
                    model,
                    attempt,
                    MAX_ATTEMPTS,
                    status,
                    exc,
                )

                # Invalid key / forbidden request:
                # do not blindly keep retrying the same problem.
                if status in (401, 403):
                    raise AuthenticationError(
                        str(exc)
                    ) from exc

                # Timeouts should be retryable.
                exception_name = exc.__class__.__name__.lower()

                if "timeout" in exception_name:
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(
                            self._get_retry_delay(attempt)
                        )
                        continue

                    raise TimeoutError(
                        "Groq request timed out."
                    ) from exc

                # Rate limits and transient provider failures:
                # retry with the next key + backoff.
                if self._is_retryable_status(status):
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(
                            self._get_retry_delay(attempt)
                        )
                        continue

                    if status == 429:
                        raise RateLimitError(
                            str(exc)
                        ) from exc

                    raise ProviderUnavailableError(
                        str(exc)
                    ) from exc

                # Other connection/network-style failures usually don't
                # expose a useful HTTP status. Give them a bounded retry.
                if status is None:
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(
                            self._get_retry_delay(attempt)
                        )
                        continue

                    raise ProviderUnavailableError(
                        f"Groq request failed: {exc}"
                    ) from exc

                # HTTP 400 is intentionally NOT retried here.
                #
                # A bad request and a model-generated malformed response
                # are different problems. Structured-output recovery
                # belongs at the caller/planner layer.
                if status == 400:
                    raise InvalidResponseError(
                        str(exc)
                    ) from exc

                # Any other unexpected provider error:
                # bounded retry, then fail cleanly.
                if attempt < MAX_ATTEMPTS:
                    time.sleep(
                        self._get_retry_delay(attempt)
                    )
                    continue

                raise ProviderUnavailableError(
                    f"Groq failed: {exc}"
                ) from exc

        raise ProviderUnavailableError(
            f"Groq failed: {last_error}"
        )


_provider = None


def get_provider():
    global _provider

    if _provider is None:
        _provider = GroqProvider()

    return _provider