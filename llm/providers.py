import logging
import random
import re
import threading
import time
from collections import deque
from collections.abc import Iterator

from groq import Groq

from .config import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_TPM_LIMIT,
    GROQ_API_KEYS,
    MAX_ADMISSION_WAIT_SECONDS,
    MAX_ATTEMPTS,
    MODEL_TPM_LIMITS,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
    TPM_ADMISSION_RATIO,
    TPM_WINDOW_SECONDS,
)
from .exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTooLargeError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


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

    Scheduling (usage-aware, not blind round-robin):

        Every key is an independent rolling-60s TPM pool. Each request

        reserves its requested tokens (input estimate + reserved output) on

        the chosen key; the reservation is replaced by the actual usage when

        the response arrives. A key is admitted only if used + requested

        stays under limit * TPM_ADMISSION_RATIO. Among admissible keys the

        least-loaded wins. If none is admissible the scheduler waits (bounded)

        for the earliest window to free up.

    Failure classes (never conflated):

        request_too_large  -> deterministic; NEVER retried on another key

        rate_limit (429)   -> transient; key cooled using the provider's own

                              retry hint, request moves to another quota pool

        invalid_response / invalid_request / authentication -> terminal

        timeout / connection / provider_5xx -> bounded backoff retry

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
                max_retries=0,
            )
            for key in GROQ_API_KEYS
        ]

        self._lock = threading.Lock()

        self._counters: dict[str, int] = {}

        self._cooldowns: dict[tuple[int, str], float] = {}

        self._windows: dict[tuple[int, str], deque] = {}

    @staticmethod
    def _tpm_limit(model: str) -> int:

        return int(MODEL_TPM_LIMITS.get(model, DEFAULT_TPM_LIMIT))

    def _used_tokens(self, index: int, model: str, now: float) -> int:
        """Tokens charged to this key in the rolling window. Lock must be held."""

        window = self._windows.get((index, model))

        if not window:
            return 0

        while window and window[0][0] <= now - TPM_WINDOW_SECONDS:
            window.popleft()

        return sum(entry[1] for entry in window)

    def _admission_wait(
        self, model: str, requested: int, ceiling: float, now: float
    ) -> float | None:
        """Seconds until some key can admit `requested`, or None if never. Lock held."""

        if requested > ceiling:
            return None

        best: float | None = None

        for index in range(len(self.clients)):
            cooling = max(self._cooldowns.get((index, model), 0.0) - now, 0.0)

            used = self._used_tokens(index, model, now)

            wait = 0.0

            if used + requested > ceiling:
                remaining = used

                wait = None

                for ts, tokens in list(self._windows.get((index, model), [])):
                    remaining -= tokens

                    if remaining + requested <= ceiling:
                        wait = max(ts + TPM_WINDOW_SECONDS - now, 0.0)

                        break

                if wait is None:
                    continue

            total = max(wait, cooling)

            best = total if best is None else min(best, total)

        return best

    def _acquire_key(self, model: str, requested_tokens: int):
        """

        Choose (client, key_index, reservation_entry) for a request.

        Least-loaded admissible key wins (ties broken round-robin). When no

        key is admissible, wait (bounded) for the earliest window to free;

        beyond the bound, fall back to the least-loaded non-cooling key and

        let the provider be the authority.

        """

        ceiling = self._tpm_limit(model) * TPM_ADMISSION_RATIO

        n = len(self.clients)

        waited = 0.0

        while True:
            now = time.monotonic()

            with self._lock:
                counter = self._counters.get(model, 0)

                fit = []

                for index in range(n):
                    if self._cooldowns.get((index, model), 0.0) > now:
                        continue

                    used = self._used_tokens(index, model, now)

                    if used + requested_tokens <= ceiling:
                        fit.append((used, (index - counter) % n, index))

                if fit:
                    fit.sort()

                    index = fit[0][2]

                    return self._reserve(index, model, requested_tokens, now)

                wait = self._admission_wait(model, requested_tokens, ceiling, now)

                if wait is None or waited + wait > MAX_ADMISSION_WAIT_SECONDS:
                    index = min(
                        range(n),
                        key=lambda i: (
                            self._cooldowns.get((i, model), 0.0) > now,
                            self._used_tokens(i, model, now),
                        ),
                    )

                    logger.warning(
                        "[LLM] no key admissible for model=%s requested=%s; "
                        "using least-loaded key=%s",
                        model,
                        requested_tokens,
                        index,
                    )

                    return self._reserve(index, model, requested_tokens, now)

            logger.info(
                "[LLM] all keys near TPM ceiling for model=%s; waiting %.1fs",
                model,
                wait,
            )

            time.sleep(wait + 0.05)

            waited += wait + 0.05

    def _reserve(self, index: int, model: str, tokens: int, now: float):

        entry = [now, int(tokens)]

        self._windows.setdefault((index, model), deque()).append(entry)

        self._counters[model] = (index + 1) % len(self.clients)

        return self.clients[index], index, entry

    def _settle(self, entry, tokens: int | None) -> None:
        """Replace a reservation with the actual (or zero) token charge."""

        if entry is None or tokens is None:
            return

        with self._lock:
            entry[1] = max(int(tokens or 0), 0)

    def _sync_used_from_provider(
        self, index: int, model: str, provider_used: int | None
    ) -> None:
        """If the provider says the key has used more than we tracked, adopt it."""

        if not provider_used:
            return

        now = time.monotonic()

        with self._lock:
            tracked = self._used_tokens(index, model, now)

            if provider_used > tracked:
                self._windows.setdefault((index, model), deque()).append(
                    [now, provider_used - tracked]
                )

    def _get_client(self, model: str) -> tuple[Groq, int]:
        """Compatibility wrapper: acquire a key without a token reservation."""

        client, index, entry = self._acquire_key(model, 0)

        self._settle(entry, 0)

        return client, index

    def _mark_cooldown(
        self,
        key_index: int,
        model: str,
        seconds: float,
    ) -> None:
        """Mark a (key_index, model) bucket as unavailable for `seconds`."""

        with self._lock:
            self._cooldowns[(key_index, model)] = time.monotonic() + seconds

    @staticmethod
    def _extract_usage(usage):
        """Extract token accounting fields without depending on SDK internals."""

        if usage is None:
            return {
                "prompt_tokens": None,
                "cached_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "rate_limit_tokens": None,
            }

        prompt_tokens = getattr(usage, "prompt_tokens", None)

        completion_tokens = getattr(usage, "completion_tokens", None)

        total_tokens = getattr(usage, "total_tokens", None)

        prompt_details = getattr(usage, "prompt_tokens_details", None)

        cached_tokens = getattr(prompt_details, "cached_tokens", None)

        completion_details = getattr(usage, "completion_tokens_details", None)

        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)

        rate_limit_tokens = None

        if prompt_tokens is not None and completion_tokens is not None:
            rate_limit_input = prompt_tokens

            if cached_tokens is not None:
                rate_limit_input = max(prompt_tokens - cached_tokens, 0)

            rate_limit_tokens = rate_limit_input + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "rate_limit_tokens": rate_limit_tokens,
        }

    @staticmethod
    def _request_text(messages) -> str:

        try:
            return "\n".join(
                str(message.get("content", ""))
                for message in messages or []
                if isinstance(message, dict)
            )

        except Exception:
            return ""

    @staticmethod
    def _estimate_input_tokens(messages) -> int | None:
        """Cheap diagnostics-only estimate; never used for admission decisions."""

        text = GroqProvider._request_text(messages)

        if not text:
            return 0

        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _exception_text(exc) -> str:

        parts = [str(exc)]

        for attr in ("message", "body", "code", "type"):
            try:
                value = getattr(exc, attr, None)

                if value is not None:
                    parts.append(str(value))

            except Exception:
                pass

        return " ".join(parts).lower()

    @classmethod
    def _is_intrinsically_too_large(cls, exc) -> bool:
        """Recognise provider errors where retrying the same request is futile."""

        status = cls._get_status_code(exc)

        text = cls._exception_text(exc)

        tpm_markers = (
            "tokens per minute",
            "token per minute",
            " tpm",
            "tpm limit",
            "tpm:",
        )

        request_markers = (
            "requested",
            "request too large",
            "prompt is too long",
            "context length",
            "maximum context",
            "max tokens",
        )

        if any(marker in text for marker in tpm_markers) and "requested" in text:
            limit_match = re.search(r"\blimit\D{0,24}(\d+)", text)

            requested_match = re.search(r"\brequested\D{0,24}(\d+)", text)

            if limit_match and requested_match:
                try:
                    if int(requested_match.group(1)) > int(limit_match.group(1)):
                        return True

                except ValueError:
                    pass

        if status in {413, 422} and any(marker in text for marker in request_markers):
            return True

        if status == 400 and any(
            marker in text
            for marker in (
                "context length",
                "too many tokens",
                "request too large",
                "prompt is too long",
                "maximum tokens",
            )
        ):
            return True

        return False

    @classmethod
    def _record_usage_totals(
        cls,
        key_index: int,
        model: str,
        token_usage: dict,
    ) -> dict[str, int]:

        with cls._provider_usage_lock:
            totals = cls._provider_usage_totals.setdefault(
                (key_index, model),
                {
                    "requests": 0,
                    "rate_limit_tokens": 0,
                    "completion_tokens": 0,
                },
            )

            totals["requests"] += 1

            for field in ("rate_limit_tokens", "completion_tokens"):
                value = token_usage.get(field)

                if isinstance(value, int):
                    totals[field] += value

            return dict(totals)

    _provider_usage_totals: dict[tuple[int, str], dict[str, int]] = {}

    _provider_usage_lock = threading.Lock()

    @classmethod
    def _log_usage(
        cls,
        *,
        task: str,
        model: str,
        key_index: int,
        response,
        max_tokens,
        latency_seconds: float | None = None,
        finish: str | None = None,
    ) -> None:
        """

        Log complete token accounting for one provider request.

        Diagnostics only: this must never affect the request.

        """

        try:
            if finish is None:
                choices = getattr(response, "choices", None) or []

                if choices:
                    finish = getattr(choices[0], "finish_reason", None)

            usage = getattr(response, "usage", None)

            token_usage = cls._extract_usage(usage)

            input_tokens = token_usage["prompt_tokens"]

            actual_completion_tokens = token_usage["completion_tokens"]

            requested_total_tokens = (
                input_tokens + max_tokens
                if isinstance(input_tokens, int) and isinstance(max_tokens, int)
                else None
            )

            actual_total_tokens = token_usage["total_tokens"]

            totals = cls._record_usage_totals(
                key_index=key_index,
                model=model,
                token_usage=token_usage,
            )

            logger.info(
                "[LLM_USAGE] task=%s model=%s key=%s finish=%s "
                "input_tokens=%s reserved_output_tokens=%s requested_total_tokens=%s "
                "actual_completion_tokens=%s reasoning_tokens=%s actual_total_tokens=%s "
                "cached_tokens=%s rate_limit_tokens=%s key_request_count=%s "
                "key_rate_limit_tokens_total=%s key_completion_tokens_total=%s "
                "latency_seconds=%s status=success failure_type=none",
                task,
                model,
                key_index,
                finish,
                input_tokens,
                max_tokens,
                requested_total_tokens,
                actual_completion_tokens,
                token_usage["reasoning_tokens"],
                actual_total_tokens,
                token_usage["cached_tokens"],
                token_usage["rate_limit_tokens"],
                totals["requests"],
                totals["rate_limit_tokens"],
                totals["completion_tokens"],
                (f"{latency_seconds:.3f}" if latency_seconds is not None else None),
            )

            if finish == "length":
                logger.warning(
                    "[LLM] task=%s model=%s key=%s hit max_tokens=%s; "
                    "output was truncated (reasoning_tokens=%s)",
                    task,
                    model,
                    key_index,
                    max_tokens,
                    token_usage["reasoning_tokens"],
                )

        except Exception:
            pass

    @staticmethod
    def _log_stream_usage(
        *,
        task: str,
        model: str,
        key_index: int,
        usage,
        finish: str | None,
        max_tokens,
        latency_seconds: float | None,
    ) -> None:
        """Log usage collected from the final usage chunk of a stream."""

        class _UsageHolder:
            pass

        holder = _UsageHolder()

        holder.usage = usage

        holder.choices = [type("_Choice", (), {"finish_reason": finish})()]

        GroqProvider._log_usage(
            task=task,
            model=model,
            key_index=key_index,
            response=holder,
            max_tokens=max_tokens,
            latency_seconds=latency_seconds,
            finish=finish,
        )

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
            headers = getattr(getattr(exc, "response", None), "headers", None)

            if headers:
                value = headers.get("retry-after")

                if value is not None:
                    return float(value)

        except (TypeError, ValueError):
            pass

        return None

    @classmethod
    def _parse_rate_limit_details(cls, exc) -> dict[str, float | int]:
        """

        Extract limit / used / requested / retry hint from a Groq 429 body,

        e.g. "Limit 8000, Used 5807, Requested 4026. Please try again in 2.1s".

        """

        text = cls._exception_text(exc)

        details: dict[str, float | int] = {}

        for name in ("limit", "used", "requested"):
            match = re.search(rf"\b{name}\D{0, 12}(\d+)", text)

            if match:
                details[name] = int(match.group(1))

        hint = re.search(r"try again in\s+([0-9hms.\s]{1,24})", text)

        if hint:
            seconds = 0.0

            found = False

            for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", hint.group(1)):
                found = True

                seconds += (
                    float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
                )

            if found:
                details["retry_after"] = seconds

        return details

    def _handle_failure(
        self,
        *,
        exc,
        task: str,
        model: str,
        attempt: int,
        key_index: int,
        entry=None,
        requested_tokens: int | None = None,
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

        intrinsically_too_large = self._is_intrinsically_too_large(exc)

        if isinstance(exc, InvalidResponseError):
            failure_type = "invalid_response"

        elif isinstance(exc, RequestTooLargeError) or intrinsically_too_large:
            failure_type = "request_too_large"

        elif status in (401, 403):
            failure_type = "authentication"

        elif status == 429:
            failure_type = "rate_limit"

        elif self._is_timeout_exception(exc):
            failure_type = "timeout"

        elif self._is_retryable_status(status):
            failure_type = "provider_5xx"

        elif status is None:
            failure_type = "connection"

        elif status == 400:
            failure_type = "invalid_request"

        elif status is not None and 400 <= int(status) < 500:
            failure_type = f"non_retryable_http_{status}"

        else:
            failure_type = "provider_error"

        self._settle(entry, 0)

        logger.warning(
            "[LLM_FAILURE] task=%s model=%s key=%s attempt=%s/%s "
            "status=%s type=%s failure_type=%s requested_total_tokens=%s error=%s",
            task,
            model,
            key_index,
            attempt,
            MAX_ATTEMPTS,
            status,
            type(exc).__name__,
            failure_type,
            requested_tokens,
            exc,
        )

        if isinstance(exc, InvalidResponseError):
            logger.error(
                "[LLM_FAILURE] task=%s model=%s key=%s failure_type=invalid_response",
                task,
                model,
                key_index,
            )

            raise exc

        if isinstance(exc, RequestTooLargeError) or intrinsically_too_large:
            logger.error(
                "[LLM_FAILURE] task=%s model=%s key=%s failure_type=request_too_large; no key retry",
                task,
                model,
                key_index,
            )

            if isinstance(exc, RequestTooLargeError):
                raise exc

            raise RequestTooLargeError(
                f"Groq rejected the request as too large for this provider/model: {exc}"
            ) from exc

        if status in (401, 403):
            raise AuthenticationError(str(exc)) from exc

        if status == 429:
            details = self._parse_rate_limit_details(exc)

            retry_after = self._parse_retry_after(exc)

            if retry_after is None:
                retry_after = details.get("retry_after")

            cooldown = (
                float(retry_after) + 0.25
                if retry_after is not None
                else DEFAULT_COOLDOWN_SECONDS
            )

            self._sync_used_from_provider(
                key_index, model, int(details["used"]) if "used" in details else None
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

            return

        if self._is_timeout_exception(exc):
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))

                return

            raise TimeoutError("Groq request timed out.") from exc

        if self._is_retryable_status(status):
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))

                return

            raise ProviderUnavailableError(str(exc)) from exc

        if status is None:
            if attempt < MAX_ATTEMPTS:
                time.sleep(self._get_retry_delay(attempt))

                return

            raise ProviderUnavailableError(f"Groq request failed: {exc}") from exc

        if status == 400:
            logger.error(
                "[LLM_FAILURE] task=%s model=%s key=%s failure_type=invalid_request",
                task,
                model,
                key_index,
            )

            raise InvalidResponseError(str(exc)) from exc

        if status is not None and 400 <= int(status) < 500:
            logger.error(
                "[LLM_FAILURE] task=%s model=%s key=%s failure_type=non_retryable_http_%s",
                task,
                model,
                key_index,
                status,
            )

            raise InvalidResponseError(str(exc)) from exc

        if attempt < MAX_ATTEMPTS:
            time.sleep(self._get_retry_delay(attempt))

            return

        raise ProviderUnavailableError(f"Groq failed: {exc}") from exc

    def _wrap_stream(
        self,
        *,
        response,
        task: str,
        model: str,
        key_index: int,
        max_tokens,
        started_at: float,
        entry=None,
    ) -> Iterator:
        """

        Forward a live Groq stream while capturing its terminal usage chunk.

        Groq may emit a final chunk containing usage and no choices. That chunk

        is consumed by this wrapper only for diagnostics; normal text chunks

        pass through unchanged.

        """

        usage = None

        finish = None

        try:
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []

                if choices:
                    for choice in choices:
                        value = getattr(choice, "finish_reason", None)

                        if value is not None:
                            finish = value

                chunk_usage = getattr(chunk, "usage", None)

                if chunk_usage is not None:
                    usage = chunk_usage

                yield chunk

        finally:
            if usage is not None:
                self._settle(entry, self._extract_usage(usage)["rate_limit_tokens"])

            self._log_stream_usage(
                task=task,
                model=model,
                key_index=key_index,
                usage=usage,
                finish=finish,
                max_tokens=max_tokens,
                latency_seconds=time.monotonic() - started_at,
            )

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

        estimated_input_tokens = self._estimate_input_tokens(messages) or 0

        requested_tokens = estimated_input_tokens + (
            max_tokens if isinstance(max_tokens, int) else 0
        )

        limit = self._tpm_limit(model)

        if requested_tokens > limit:
            logger.error(
                "[LLM_FAILURE] task=%s model=%s failure_type=request_too_large "
                "requested_total_tokens=%s tpm_limit=%s; not sent",
                task,
                model,
                requested_tokens,
                limit,
            )

            raise RequestTooLargeError(
                f"Estimated request of {requested_tokens} tokens exceeds the "
                f"{limit} tokens-per-minute limit for {model}."
            )

        for attempt in range(1, MAX_ATTEMPTS + 1):
            client, key_index, entry = self._acquire_key(model, requested_tokens)

            request_started_at = time.monotonic()

            logger.info(
                "[LLM_REQUEST] task=%s model=%s key=%s attempt=%s/%s "
                "estimated_input_tokens=%s reserved_output_tokens=%s "
                "estimated_requested_total_tokens=%s status=started",
                task,
                model,
                key_index,
                attempt,
                MAX_ATTEMPTS,
                estimated_input_tokens,
                max_tokens,
                (
                    estimated_input_tokens + max_tokens
                    if isinstance(estimated_input_tokens, int)
                    and isinstance(max_tokens, int)
                    else None
                ),
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    stream=stream,
                    **({"stream_options": {"include_usage": True}} if stream else {}),
                    **extra_params,
                )

                if stream:
                    return self._wrap_stream(
                        response=response,
                        task=task,
                        model=model,
                        key_index=key_index,
                        max_tokens=max_tokens,
                        started_at=request_started_at,
                        entry=entry,
                    )

                if not response.choices:
                    raise InvalidResponseError("Groq returned no choices.")

                self._settle(
                    entry,
                    self._extract_usage(getattr(response, "usage", None))[
                        "rate_limit_tokens"
                    ],
                )

                self._log_usage(
                    task=task,
                    model=model,
                    key_index=key_index,
                    response=response,
                    max_tokens=max_tokens,
                    latency_seconds=time.monotonic() - request_started_at,
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
                    entry=entry,
                    requested_tokens=requested_tokens,
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
            logger.exception(
                "[LLM] Streaming response failed after stream start: task=%s model=%s",
                task,
                model,
            )

            raise ProviderUnavailableError("Groq streaming request failed.") from exc

        if not yielded_any:
            return


_provider = None


def get_provider() -> GroqProvider:

    global _provider

    if _provider is None:
        _provider = GroqProvider()

    return _provider
