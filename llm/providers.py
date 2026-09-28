from collections.abc import Iterator
import logging
import random
import threading
import time

from groq import Groq

from .config import (
    DEFAULT_COOLDOWN_SECONDS,
    GROQ_API_KEYS,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
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
        - Own Groq clients (one per API key).
        - Distribute load using a per-model round-robin counter.
        - Track per-(key_index, model) cooldowns to skip rate-limited buckets.
        - Execute normal and streaming requests.
        - Apply bounded retry handling around request creation.

    Rate limit model:
        Groq enforces limits per key per model, giving 6 independent buckets
        across 3 keys and 2 model tiers (20b, 120b). A 429 on key0+20b says
        nothing about key0+120b or key1+20b. Two independent counters — one
        per model — distribute each tier's traffic evenly across all 3 keys
        without the tiers interfering with each other.

    429 handling:
        On a 429 the (key, model) bucket is cooled immediately and the retry
        selects the next available key with zero sleep. Sleep only occurs if
        all 3 keys are simultaneously cooling for the requested model, and
        only until the soonest cooldown expires.

    This layer intentionally does not:
        - emit SSE
        - persist application data
        - know about chat conversations
        - assemble frontend state

    For streaming requests, retry decisions are made while establishing the
    provider stream. Once a stream has successfully started the provider
    returns that live iterator unchanged.
    """

    def __init__(self):
        self.clients = [
            Groq(
                api_key=key,
                timeout=REQUEST_TIMEOUT,
                # The Groq SDK retries on its own by default (same key, up to
                # 2 extra tries). That multiplies with the MAX_ATTEMPTS loop
                # below and never rotates keys. Keep ONE retry layer: ours.
                max_retries=0,
            )
            for key in GROQ_API_KEYS
        ]

        self._lock = threading.Lock()

        # Per-model round-robin counter: model_string -> next starting index.
        # Each model tier advances independently so heavy 20b traffic does
        # not skew the 120b distribution and vice-versa.
        self._counters: dict[str, int] = {}

        # Per-(key_index, model) cooldown expiry as a monotonic timestamp.
        # A bucket is available when its expiry is <= time.monotonic().
        self._cooldowns: dict[tuple[int, str], float] = {}

    # ------------------------------------------------------------------
    # Key selection
    # ------------------------------------------------------------------

    def _get_client(self, model: str) -> tuple[Groq, int]:
        """
        Return the next available (client, key_index) for this model.

        Normal path (O(n)):
            Scan keys in round-robin order, return the first whose
            (key_index, model) bucket is not cooling. The counter advances
            past the chosen key so the next call starts from the following
            slot.

        All-cooled path:
            If every key is cooling, sleep outside the lock until the
            soonest cooldown expires, then return that key. This is the
            ONLY place a rate-limit-induced sleep can occur.
        """
        now = time.monotonic()
        n = len(self.clients)

        with self._lock:
            counter = self._counters.get(model, 0)

            for i in range(n):
                index = (counter + i) % n
                if self._cooldowns.get((index, model), 0.0) <= now:
                    # Advance counter past this key so the next call
                    # starts from the following slot.
                    self._counters[model] = (index + 1) % n
                    return self.clients[index], index

            # All keys are cooling — find the one that recovers soonest.
            soonest = min(
                range(n),
                key=lambda i: self._cooldowns.get((i, model), 0.0),
            )
            wait = self._cooldowns[(soonest, model)] - now

        # Sleep outside the lock so other threads can make progress
        # on different models or tasks while we wait.
        if wait > 0:
            logger.warning(
                "[LLM] All keys cooling for model=%s — sleeping %.1fs",
                model,
                wait,
            )
            time.sleep(wait)

        with self._lock:
            self._counters[model] = (soonest + 1) % n

        return self.clients[soonest], soonest

    def _mark_cooldown(
        self,
        key_index: int,
        model: str,
        seconds: float,
    ) -> None:
        """Mark a (key_index, model) bucket as unavailable for `seconds`."""
        with self._lock:
            self._cooldowns[(key_index, model)] = (
                time.monotonic() + seconds
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    def _get_retry_delay(self, attempt: int) -> float:
        """
        Small exponential backoff with jitter for non-429 failures.

        attempt=1 -> ~1s
        attempt=2 -> ~2s
        """
        base_delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.25)
        return base_delay + jitter

    @staticmethod
    def _get_status_code(exc):
        return getattr(exc, "status_code", None)

    @staticmethod
    def _is_retryable_status(status) -> bool:
        return status in {500, 502, 503, 504}

    @staticmethod
    def _is_timeout_exception(exc) -> bool:
        return "timeout" in exc.__class__.__name__.lower()

    @staticmethod
    def _parse_retry_after(exc) -> float | None:
        """
        Extract the Retry-After value (seconds) from a 429 response header.
        Returns None if the header is absent or unparseable.
        """
        try:
            headers = getattr(
                getattr(exc, "response", None), "headers", None
            )
            if headers:
                value = headers.get("retry-after")
                if value is not None:
                    return float(value)
        except (TypeError, ValueError):
            pass
        return None

    def _handle_failure(
        self,
        *,
        exc,
        task: str,
        model: str,
        attempt: int,
        key_index: int,
    ) -> None:
        """
        Log a provider failure and either raise a terminal exception or
        return to allow the caller to retry.

        429 path:
            Marks (key_index, model) as cooled using the Retry-After header
            or DEFAULT_COOLDOWN_SECONDS as a fallback, then returns
            immediately with no sleep. The next _get_client call will skip
            the cooled bucket and pick the next healthy key.

        All other retryable failures (timeouts, 5xx, no-status network
        errors) use a short exponential backoff before the next attempt.
        These errors are not key-specific so no cooldown is set.
        """
        status = self._get_status_code(exc)

        logger.warning(
            "[LLM] task=%s model=%s key=%s attempt=%s/%s "
            "status=%s type=%s error=%s",
            task,
            model,
            key_index,
            attempt,
            MAX_ATTEMPTS,
            status,
            type(exc).__name__,
            exc,
        )

        # --- terminal: bad credentials ---
        if status in (401, 403):
            raise AuthenticationError(str(exc)) from exc

        # --- 429: cool this bucket, retry immediately on next key ---
        if status == 429:
            retry_after = self._parse_retry_after(exc)
            cooldown = (
                retry_after
                if retry_after is not None
                else DEFAULT_COOLDOWN_SECONDS
            )
            self._mark_cooldown(key_index, model, cooldown)
            logger.info(
                "[LLM] key=%s model=%s rate-limited — cooled for %.1fs",
                key_index,
                model,
                cooldown,
            )
            if attempt >= MAX_ATTEMPTS:
                raise RateLimitError(str(exc)) from exc
            # No sleep — _get_client will skip this key on the next attempt.
            return

        # --- timeout ---
        if self._is_timeout_exception(exc):
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))
                return
            raise TimeoutError("Groq request timed out.") from exc

        # --- retryable 5xx ---
        if self._is_retryable_status(status):
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))
                return
            raise ProviderUnavailableError(str(exc)) from exc

        # --- network / connection errors (no HTTP status) ---
        if status is None:
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))
                return
            raise ProviderUnavailableError(
                f"Groq request failed: {exc}"
            ) from exc

        # --- bad request: deterministic, do not retry ---
        if status == 400:
            raise InvalidResponseError(str(exc)) from exc

        # --- any other unexpected status ---
        if attempt < MAX_ATTEMPTS:
            time.sleep(self._get_retry_delay(attempt))
            return

        raise ProviderUnavailableError(f"Groq failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            Returns Groq's live streaming iterator. Once the stream is
            successfully started this method does not consume it — the
            caller owns it from that point onward.
        """
        extra_params = {}
        if include_reasoning is False:
            extra_params["reasoning_effort"] = "low"

        last_error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            client, key_index = self._get_client(model)

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
                    key_index=key_index,
                )

        raise ProviderUnavailableError(f"Groq failed: {last_error}")

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
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue

                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    content = getattr(delta, "content", None)
                    if not content:
                        continue

                    yielded_any = True
                    yield content

        except Exception as exc:
            # Do not retry here. The stream has already been handed to
            # the caller and may already have produced partial output.
            logger.exception(
                "[LLM] Streaming response failed after stream start: "
                "task=%s model=%s",
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


def get_provider() -> GroqProvider:
    global _provider

    if _provider is None:
        _provider = GroqProvider()

    return _provider