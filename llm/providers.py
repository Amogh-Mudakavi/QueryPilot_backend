from collections.abc import Iterator
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
    """
    Groq provider adapter.

    Responsibilities:
        - Own Groq clients.
        - Rotate configured API keys.
        - Execute normal and streaming requests.
        - Apply bounded retry handling around request creation.

    This layer intentionally does not:
        - emit SSE
        - persist application data
        - know about chat conversations
        - assemble frontend state

    For streaming requests, retry decisions are made while establishing
    the provider stream. Once a stream has successfully started, the
    provider returns that live iterator to the caller unchanged.
    """

    def __init__(self):
        self.clients = [
            Groq(
                api_key=key,
                timeout=REQUEST_TIMEOUT,
                # The Groq SDK retries on its own by default (same key, up to
                # 2 extra tries). That multiplies with the MAX_ATTEMPTS loop
                # below (worst case 3 x 3 attempts per call) and never rotates
                # keys. Keep ONE retry layer: ours.
                max_retries=0,
            )
            for key in GROQ_API_KEYS
        ]

        self.counters: dict[str, int] = {}

    def _get_client(self, task: str) -> Groq:
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

    @staticmethod
    def _log_usage(
        *,
        task: str,
        model: str,
        response,
        max_tokens,
    ) -> None:
        """
        Log how close each task runs to its completion cap, and warn when
        output was cut off (finish_reason == "length"). Reasoning models
        spend completion tokens on hidden reasoning, so a cap that looks
        generous can silently truncate answers, summaries or JSON.
        Diagnostics only: it must never affect the request.
        """

        try:
            finish = getattr(response.choices[0], "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None)

            logger.info(
                "[LLM] task=%s model=%s finish=%s completion_tokens=%s "
                "reasoning_tokens=%s max_tokens=%s",
                task,
                model,
                finish,
                completion,
                reasoning,
                max_tokens,
            )

            if finish == "length":
                logger.warning(
                    "[LLM] task=%s hit max_tokens=%s; output was truncated "
                    "(reasoning_tokens=%s)",
                    task,
                    max_tokens,
                    reasoning,
                )
        except Exception:
            pass

    def _get_retry_delay(
        self,
        attempt: int,
    ) -> float:
        """
        Small exponential backoff with jitter.

        attempt=1 -> about 1s
        attempt=2 -> about 2s
        """

        base_delay = (
            RETRY_BACKOFF_SECONDS
            * (2 ** (attempt - 1))
        )

        jitter = random.uniform(
            0,
            0.25,
        )

        return base_delay + jitter

    @staticmethod
    def _get_status_code(
        exc,
    ):
        return getattr(
            exc,
            "status_code",
            None,
        )

    @staticmethod
    def _is_retryable_status(
        status,
    ):
        return status in {
            429,
            500,
            502,
            503,
            504,
        }

    @staticmethod
    def _is_timeout_exception(
        exc,
    ) -> bool:
        return (
            "timeout"
            in exc.__class__.__name__.lower()
        )

    def _handle_failure(
        self,
        *,
        exc,
        task: str,
        model: str,
        attempt: int,
    ) -> bool:
        """
        Log a provider failure and determine whether the caller should
        retry the request.

        Returns:
            True  -> retry
            False -> caller must raise the mapped terminal exception
        """

        status = self._get_status_code(
            exc
        )

        logger.warning(
            "[LLM] task=%s model=%s attempt=%s/%s "
            "status=%s type=%s retry_after=%s error=%s",
            task,
            model,
            attempt,
            MAX_ATTEMPTS,
            status,
            type(exc).__name__,
            getattr(
                getattr(getattr(exc, "response", None), "headers", None) or {},
                "get",
                lambda *_: None,
            )("retry-after"),
            exc,
        )

        if status in (
            401,
            403,
        ):
            raise AuthenticationError(
                str(exc)
            ) from exc

        if self._is_timeout_exception(
            exc
        ):
            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    self._get_retry_delay(
                        attempt
                    )
                )
                return True

            raise TimeoutError(
                "Groq request timed out."
            ) from exc

        if self._is_retryable_status(
            status
        ):
            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    self._get_retry_delay(
                        attempt
                    )
                )
                return True

            if status == 429:
                raise RateLimitError(
                    str(exc)
                ) from exc

            raise ProviderUnavailableError(
                str(exc)
            ) from exc

        # Network/connection style failures often have no HTTP status.
        if status is None:
            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    self._get_retry_delay(
                        attempt
                    )
                )
                return True

            raise ProviderUnavailableError(
                f"Groq request failed: {exc}"
            ) from exc

        # Bad request errors are deterministic and should not be retried.
        if status == 400:
            raise InvalidResponseError(
                str(exc)
            ) from exc

        if attempt < MAX_ATTEMPTS:
            time.sleep(
                self._get_retry_delay(
                    attempt
                )
            )
            return True

        raise ProviderUnavailableError(
            f"Groq failed: {exc}"
        ) from exc

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
        """
        Execute a Groq request.

        For stream=False:
            Returns the normal Groq completion response.

        For stream=True:
            Returns Groq's live streaming iterator.

        Important:
            Once Groq successfully returns a streaming iterator, this
            method does not consume it. The caller owns the stream from
            that point onward.
        """

        extra_params = {}

        # Reasoning models can spend completion tokens on internal
        # reasoning. Keep reasoning low when explicitly requested.
        if include_reasoning is False:
            extra_params[
                "reasoning_effort"
            ] = "low"

        last_error = None

        for attempt in range(
            1,
            MAX_ATTEMPTS + 1,
        ):
            client = self._get_client(
                task
            )

            try:
                response = (
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        stream=stream,
                        **extra_params,
                    )
                )

                if stream:
                    # Do not inspect or consume the iterator here.
                    # The stream must remain live for the caller.
                    return response

                if not response.choices:
                    raise InvalidResponseError(
                        "Groq returned no choices."
                    )

                self._log_usage(
                    task=task,
                    model=model,
                    response=response,
                    max_tokens=max_tokens,
                )

                return response

            except Exception as exc:
                last_error = exc

                self._handle_failure(
                    exc=exc,
                    task=task,
                    model=model,
                    attempt=attempt,
                )

        raise ProviderUnavailableError(
            f"Groq failed: {last_error}"
        )

    def generate_text_stream(
        self,
        *,
        task,
        model,
        messages,
        temperature=0.0,
        max_tokens=None,
        include_reasoning=None,
    ) -> Iterator[str]:
        """
        Yield plain text deltas from a Groq streaming response.

        This is the clean provider contract consumed by the application
        layer. Provider-specific chunk structures stay inside this file.

        The method deliberately does not retry after the stream has
        already yielded data. At that point the request has begun and a
        transparent retry could duplicate part of the answer.
        """

        response = self.generate(
            task=task,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            stream=True,
            include_reasoning=include_reasoning,
        )

        yielded_any = False

        try:
            for chunk in response:
                choices = getattr(
                    chunk,
                    "choices",
                    None,
                )

                if not choices:
                    continue

                for choice in choices:
                    delta = getattr(
                        choice,
                        "delta",
                        None,
                    )

                    if delta is None:
                        continue

                    content = getattr(
                        delta,
                        "content",
                        None,
                    )

                    if not content:
                        continue

                    yielded_any = True

                    yield content

        except Exception as exc:
            # Do not retry here. The stream has already been handed to
            # the caller and may already have produced partial output.
            logger.exception(
                "[LLM] Streaming response failed "
                "after stream start: task=%s model=%s",
                task,
                model,
            )
            raise ProviderUnavailableError(
                "Groq streaming request failed."
            ) from exc

        # A successful stream may legitimately produce no text. Leave
        # the final empty-result decision to the answer service.
        if not yielded_any:
            return


_provider = None


def get_provider():
    global _provider

    if _provider is None:
        _provider = GroqProvider()

    return _provider