"""Groq provider adapter for QueryPilot.

Owns the Groq clients, talks to the quota scheduler, classifies failures and
applies the retry policy. It does not know about SSE, conversations, prompts or
user requests, and it holds no request-wide state: every call is sized and
retried on its own merits.

Flow of one call:

    estimate input -> scheduler.acquire(model, input + max_tokens)   # picks a bucket
        -> Groq call
        -> success: settle reservation with actual usage, log, record
        -> failure: release reservation, classify, then one of
             rate_limit      cool THAT bucket, retry at once on another key
             authentication  disable THAT key everywhere, retry on another key
             timeout / connection / 5xx
                             back off, retry preferring a different key
             too large / invalid request / invalid response / unexpected
                             raise immediately, no retry
    Retries are capped per failure kind and overall (config.RETRY_LIMITS,
    config.MAX_ATTEMPTS).

Streaming: the retry loop only covers *establishing* the stream. Once Groq has
accepted it, the live iterator is returned and never retried, so partial output
cannot be duplicated. Usage is settled and logged when the stream ends, is
closed early, or fails.
"""

import logging
import random
import re
import threading
import time
from collections.abc import Iterator

from groq import Groq

from . import usage as usage_log
from .config import (
    ADMISSION_MAX_BYPASS,
    AUTH_COOLDOWN_SECONDS,
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_TPM_LIMIT,
    GROQ_API_KEYS,
    MAX_ADMISSION_WAIT_SECONDS,
    MAX_ATTEMPTS,
    MAX_COOLDOWN_SECONDS,
    MESSAGE_OVERHEAD_TOKENS,
    MODEL_TPM_LIMITS,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
    RETRY_LIMITS,
    TPM_ADMISSION_RATIO,
    TPM_WINDOW_SECONDS,
)
from .exceptions import (
    AuthenticationError,
    InvalidResponseError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTooLargeError,
    TimeoutError,
)
from .scheduler import Lease, QuotaScheduler

logger = logging.getLogger(__name__)

_TRANSIENT_KINDS = frozenset({"timeout", "connection", "provider_5xx"})

_TOO_LARGE_MARKERS = (
    "context length",
    "maximum context",
    "too many tokens",
    "request too large",
    "prompt is too long",
    "reduce the length",
)


# ---------------------------------------------------------------------------
# Pure helpers: usage extraction, error parsing, classification
# ---------------------------------------------------------------------------


def _tpm_limit(model: str) -> int:
    return int(MODEL_TPM_LIMITS.get(model, DEFAULT_TPM_LIMIT))


def _estimate_input_tokens(messages) -> int:
    """Cheap pre-flight estimate used only for admission and diagnostics."""
    messages = messages or []
    chars = sum(
        len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)
    )
    return (
        int(chars / CHARS_PER_TOKEN_ESTIMATE)
        + 1
        + MESSAGE_OVERHEAD_TOKENS * len(messages)
    )


def _extract_usage(usage) -> dict:
    """Read token accounting without depending on SDK internals."""
    empty = dict.fromkeys(
        (
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "rate_limit_tokens",
        )
    )
    if usage is None:
        return empty

    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    reasoning = getattr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None
    )

    rate_limit_tokens = None
    if prompt is not None and completion is not None:
        billed_prompt = max(prompt - cached, 0) if cached is not None else prompt
        rate_limit_tokens = billed_prompt + completion

    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": getattr(usage, "total_tokens", None),
        "rate_limit_tokens": rate_limit_tokens,
    }


def _status_code(exc) -> int | None:
    status = getattr(exc, "status_code", None)
    return int(status) if isinstance(status, int) else None


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


def _parse_rate_limit_details(text: str) -> dict[str, float | int]:
    """Pull limit / used / requested / retry hint out of a Groq 429 body.

    Example body: "... Limit 8000, Used 5807, Requested 4026. Please try again
    in 2.1s". `text` must already be lower-cased.
    """
    details: dict[str, float | int] = {}
    for name in ("limit", "used", "requested"):
        match = re.search(rf"\b{name}\s*[:=]?\s*(\d+)", text)
        if match:
            details[name] = int(match.group(1))

    hint = re.search(r"try again in\s+([0-9hms.\s]{1,24})", text)
    if hint:
        units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        found = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", hint.group(1))
        if found:
            details["retry_after"] = sum(float(v) * units[u] for v, u in found)
    return details


def _retry_after_header(exc) -> float | None:
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            value = headers.get("retry-after")
            if value is not None:
                return float(value)
    except (TypeError, ValueError):
        pass
    return None


def _is_too_large(exc, status: int | None, text: str) -> bool:
    """True when the request itself can never fit, so retrying is futile."""
    if isinstance(exc, RequestTooLargeError) or status == 413:
        return True
    details = _parse_rate_limit_details(text)
    if (
        "limit" in details
        and "requested" in details
        and details["requested"] > details["limit"]
    ):
        return True
    return status in (400, 422) and any(m in text for m in _TOO_LARGE_MARKERS)


def _classify(exc) -> str:
    """Map an exception to a failure kind; each kind has its own policy."""
    if isinstance(exc, InvalidResponseError):
        return "invalid_response"
    status = _status_code(exc)
    text = _exception_text(exc)
    if _is_too_large(exc, status, text):
        return "request_too_large"
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    name = type(exc).__name__.lower()
    if "timeout" in name or status == 408:
        return "timeout"
    if status is not None:
        if status >= 500:
            return "provider_5xx"
        if 400 <= status < 500:
            return "invalid_request"
        return "unexpected"
    if any(marker in name for marker in ("connect", "network", "protocol", "transport")):
        return "connection"
    return "unexpected"


def _terminal_error(kind: str, exc) -> Exception:
    """The exception raised once a failure will not be retried."""
    if isinstance(exc, LLMError) and kind in ("invalid_response", "request_too_large"):
        return exc
    if kind == "request_too_large":
        return RequestTooLargeError(
            f"Groq rejected the request as too large for this model: {exc}"
        )
    if kind == "rate_limit":
        return RateLimitError(str(exc))
    if kind == "authentication":
        return AuthenticationError(str(exc))
    if kind == "timeout":
        return TimeoutError("Groq request timed out.")
    if kind in ("invalid_request", "invalid_response"):
        return InvalidResponseError(str(exc))
    return ProviderUnavailableError(f"Groq request failed: {exc}")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GroqProvider:
    """Groq adapter: one client per key, scheduling delegated to QuotaScheduler."""

    def __init__(self, clients=None, scheduler: QuotaScheduler | None = None):
        self.clients = clients or [
            Groq(api_key=key, timeout=REQUEST_TIMEOUT, max_retries=0)
            for key in GROQ_API_KEYS
        ]
        self.scheduler = scheduler or QuotaScheduler(
            key_count=len(self.clients),
            limit_for=_tpm_limit,
            admission_ratio=TPM_ADMISSION_RATIO,
            window_seconds=TPM_WINDOW_SECONDS,
            max_wait=MAX_ADMISSION_WAIT_SECONDS,
            max_bypass=ADMISSION_MAX_BYPASS,
        )
        self._sleep = time.sleep  # replaceable in tests

    # -- Groq call -----------------------------------------------------------

    @staticmethod
    def _create(client, kwargs: dict):
        """Call Groq, returning (response, headers).

        Uses the SDK's raw-response wrapper when present so Groq's own
        rate-limit headers can correct the scheduler's view.
        """
        completions = client.chat.completions
        raw = getattr(completions, "with_raw_response", None)
        if raw is not None:
            response = raw.create(**kwargs)
            return response.parse(), getattr(response, "headers", None) or {}
        return completions.create(**kwargs), {}

    def _sync_from_headers(self, lease: Lease, headers) -> None:
        """Adopt Groq's remaining-TPM figure if it is fuller than our estimate."""
        try:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            if remaining is None:
                return
            limit = headers.get("x-ratelimit-limit-tokens")
            limit = int(float(limit)) if limit is not None else _tpm_limit(lease.model)
            self.scheduler.sync_provider_used(
                lease.model, lease.key_index, limit - int(float(remaining))
            )
        except Exception:
            pass

    # -- failure policy ------------------------------------------------------

    def _backoff(self, n: int) -> float:
        return RETRY_BACKOFF_SECONDS * (2 ** (n - 1)) + random.uniform(0, 0.25)

    def _after_failure(
        self,
        *,
        exc,
        task: str,
        model: str,
        lease: Lease,
        attempt: int,
        retries: dict[str, int],
        avoid: set[int],
        started: float,
    ) -> float:
        """Log, apply side effects, then return a sleep time or raise."""
        kind = _classify(exc)
        status = _status_code(exc)
        text = _exception_text(exc)

        logger.warning(
            "[LLM_FAILURE] task=%s model=%s key=%s attempt=%s/%s status=%s "
            "failure_type=%s reserved_tokens=%s latency_seconds=%.3f error=%s",
            task,
            model,
            lease.key_index,
            attempt,
            MAX_ATTEMPTS,
            status,
            kind,
            lease.reserved,
            time.monotonic() - started,
            exc,
            exc_info=(kind == "unexpected"),
        )
        usage_log.record(
            usage_log.CallUsage(
                task=task,
                model=model,
                key_index=lease.key_index,
                attempt=attempt,
                status="failed",
                latency_seconds=time.monotonic() - started,
            )
        )

        if kind == "rate_limit":
            details = _parse_rate_limit_details(text)
            hint = _retry_after_header(exc)
            if hint is None:
                hint = details.get("retry_after")
            cooldown = (
                min(float(hint) + 0.25, MAX_COOLDOWN_SECONDS)
                if hint is not None
                else DEFAULT_COOLDOWN_SECONDS
            )
            self.scheduler.sync_provider_used(model, lease.key_index, details.get("used"))
            self.scheduler.cool(model, lease.key_index, cooldown)
            logger.info(
                "[LLM_RATE_LIMIT] task=%s model=%s key=%s cooldown_seconds=%.2f "
                "provider_limit=%s provider_used=%s provider_requested=%s",
                task,
                model,
                lease.key_index,
                cooldown,
                details.get("limit"),
                details.get("used"),
                details.get("requested"),
            )
        elif kind == "authentication":
            self.scheduler.disable_key(lease.key_index, AUTH_COOLDOWN_SECONDS)
            logger.error(
                "[LLM_AUTH] key=%s rejected (status=%s); disabled for %.0fs on all models",
                lease.key_index,
                status,
                AUTH_COOLDOWN_SECONDS,
            )

        used = retries.get(kind, 0)
        if used >= RETRY_LIMITS.get(kind, 0) or attempt >= MAX_ATTEMPTS:
            raise _terminal_error(kind, exc) from exc

        retries[kind] = used + 1
        delay = 0.0
        if kind in _TRANSIENT_KINDS:
            avoid.add(lease.key_index)
            delay = self._backoff(retries[kind])
        logger.info(
            "[LLM_RETRY] task=%s model=%s failed_key=%s reason=%s next_attempt=%s/%s "
            "delay_seconds=%.2f",
            task,
            model,
            lease.key_index,
            kind,
            attempt + 1,
            MAX_ATTEMPTS,
            delay,
        )
        return delay

    # -- accounting ------------------------------------------------------------

    def _account(
        self,
        *,
        task: str,
        model: str,
        lease: Lease,
        attempt: int,
        usage,
        finish: str | None,
        max_tokens: int,
        estimated_input: int,
        started: float,
        status: str,
    ) -> None:
        """Settle the reservation with actual usage, then log and record it.

        Observability only: nothing here can change a request.
        """
        tokens = _extract_usage(usage)
        bucket_used = self.scheduler.settle(lease, tokens["rate_limit_tokens"])
        latency = time.monotonic() - started
        try:
            logger.info(
                "[LLM_USAGE] task=%s model=%s key=%s attempt=%s status=%s finish=%s "
                "estimated_input=%s input_tokens=%s max_tokens=%s "
                "completion_tokens=%s reasoning_tokens=%s total_tokens=%s "
                "cached_tokens=%s rate_limit_tokens=%s bucket_window_used=%s "
                "queue_wait_seconds=%.3f latency_seconds=%.3f",
                task,
                model,
                lease.key_index,
                attempt,
                status,
                finish,
                estimated_input,
                tokens["prompt_tokens"],
                max_tokens,
                tokens["completion_tokens"],
                tokens["reasoning_tokens"],
                tokens["total_tokens"],
                tokens["cached_tokens"],
                tokens["rate_limit_tokens"],
                bucket_used,
                lease.waited,
                latency,
            )
            if finish == "length":
                logger.warning(
                    "[LLM] task=%s model=%s key=%s hit max_tokens=%s (reasoning_tokens=%s); "
                    "output was cut off",
                    task,
                    model,
                    lease.key_index,
                    max_tokens,
                    tokens["reasoning_tokens"],
                )
            usage_log.record(
                usage_log.CallUsage(
                    task=task,
                    model=model,
                    key_index=lease.key_index,
                    attempt=attempt,
                    status=status,
                    prompt_tokens=tokens["prompt_tokens"],
                    completion_tokens=tokens["completion_tokens"],
                    reasoning_tokens=tokens["reasoning_tokens"],
                    total_tokens=tokens["total_tokens"],
                    finish_reason=finish,
                    latency_seconds=latency,
                )
            )
        except Exception:
            pass

    # -- public API ------------------------------------------------------------

    def generate(
        self,
        *,
        task: str,
        model: str,
        messages,
        max_tokens: int,
        temperature: float = 0.0,
        response_format=None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        input_hint: int | None = None,
    ):
        """Run one Groq request with bounded, per-failure-kind retries.

        stream=False returns the Groq completion. stream=True returns a live
        iterator of Groq chunks (see module docstring for retry semantics).
        `max_tokens` is used exactly as given; this layer never lowers it.
        """
        estimated_input = _estimate_input_tokens(messages)
        reserved = estimated_input + max_tokens

        if input_hint and estimated_input > input_hint:
            logger.warning(
                "[LLM_BUDGET] task=%s estimated_input=%s exceeds input_hint=%s; "
                "compact the prompt/evidence upstream",
                task,
                estimated_input,
                input_hint,
            )

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout if timeout is not None else REQUEST_TIMEOUT,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if stream:
            kwargs["stream"] = True

        retries: dict[str, int] = {}
        avoid: set[int] = set()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                lease = self.scheduler.acquire(model, reserved, avoid)
            except LLMError as exc:
                logger.error(
                    "[LLM_FAILURE] task=%s model=%s attempt=%s/%s failure_type=%s "
                    "reserved_tokens=%s not sent: %s",
                    task,
                    model,
                    attempt,
                    MAX_ATTEMPTS,
                    type(exc).__name__,
                    reserved,
                    exc,
                )
                raise
            started = time.monotonic()
            logger.info(
                "[LLM_REQUEST] task=%s model=%s key=%s attempt=%s/%s "
                "estimated_input=%s max_tokens=%s reserved=%s bucket_window_used=%s "
                "queue_wait_seconds=%.3f stream=%s",
                task,
                model,
                lease.key_index,
                attempt,
                MAX_ATTEMPTS,
                estimated_input,
                max_tokens,
                reserved,
                lease.bucket_used,
                lease.waited,
                stream,
            )

            try:
                response, headers = self._create(self.clients[lease.key_index], kwargs)
                if not stream and not getattr(response, "choices", None):
                    raise InvalidResponseError("Groq returned no choices.")
            except Exception as exc:
                self.scheduler.release(lease)
                delay = self._after_failure(
                    exc=exc,
                    task=task,
                    model=model,
                    lease=lease,
                    attempt=attempt,
                    retries=retries,
                    avoid=avoid,
                    started=started,
                )
                if delay:
                    self._sleep(delay)
                continue

            if stream:
                # Headers arrive with the response head, so they already count
                # this request's reservation; sync now.
                self._sync_from_headers(lease, headers)
                return self._wrap_stream(
                    stream=response,
                    task=task,
                    model=model,
                    lease=lease,
                    attempt=attempt,
                    max_tokens=max_tokens,
                    estimated_input=estimated_input,
                    started=started,
                )

            self._account(
                task=task,
                model=model,
                lease=lease,
                attempt=attempt,
                usage=getattr(response, "usage", None),
                finish=getattr(response.choices[0], "finish_reason", None),
                max_tokens=max_tokens,
                estimated_input=estimated_input,
                started=started,
                status="success",
            )
            # After settling: the bucket's tracked total now holds actual usage,
            # so the provider's figure can only raise it to the true level.
            self._sync_from_headers(lease, headers)
            return response

        # Unreachable: the final attempt always returns or raises above.
        raise ProviderUnavailableError("Groq request failed.")

    def _wrap_stream(
        self,
        *,
        stream,
        task: str,
        model: str,
        lease: Lease,
        attempt: int,
        max_tokens: int,
        estimated_input: int,
        started: float,
    ) -> Iterator:
        """Forward a live Groq stream, capturing the terminal usage chunk.

        Text chunks pass through untouched. The reservation is settled and usage
        logged exactly once, in `finally`, whether the stream finishes, the
        consumer closes it early, or it fails midway. If Groq never sent usage
        the reservation is kept as the conservative charge.
        """
        usage = None
        finish = None
        completed = False
        try:
            for chunk in stream:
                for choice in getattr(chunk, "choices", None) or []:
                    reason = getattr(choice, "finish_reason", None)
                    if reason is not None:
                        finish = reason
                chunk_usage = getattr(chunk, "usage", None) or getattr(
                    getattr(chunk, "x_groq", None), "usage", None
                )
                if chunk_usage is not None:
                    usage = chunk_usage
                yield chunk
            completed = True
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._account(
                task=task,
                model=model,
                lease=lease,
                attempt=attempt,
                usage=usage,
                finish=finish,
                max_tokens=max_tokens,
                estimated_input=estimated_input,
                started=started,
                status="success" if completed else "stream_incomplete",
            )

    def generate_text_stream(
        self,
        *,
        task: str,
        model: str,
        messages,
        max_tokens: int,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        input_hint: int | None = None,
    ) -> Iterator[str]:
        """Yield plain text deltas; Groq chunk structures stay in this file.

        Nothing is retried once data has been yielded, since a transparent retry
        would duplicate part of the answer. Failures while establishing the
        stream are handled by `generate`; failures after it started surface as
        ProviderUnavailableError.
        """
        response = self.generate(
            task=task,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            input_hint=input_hint,
        )
        try:
            for chunk in response:
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        yield content
        except Exception as exc:
            logger.exception(
                "[LLM] stream failed after start: task=%s model=%s", task, model
            )
            raise ProviderUnavailableError("Groq streaming request failed.") from exc
        finally:
            response.close()  # runs _wrap_stream's finally if we stopped early


_provider: GroqProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> GroqProvider:
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = GroqProvider()
    return _provider