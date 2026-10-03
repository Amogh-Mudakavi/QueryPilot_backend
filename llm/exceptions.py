"""
Custom exceptions used by QueryPilot's LLM layer.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when a provider returns HTTP 429."""

    pass


class AuthenticationError(LLMError):
    """Raised when an API key is invalid."""

    pass


class ProviderUnavailableError(LLMError):
    """Raised when all provider attempts are unavailable."""

    pass


class InvalidResponseError(LLMError):
    """Raised when the provider response is invalid."""

    pass


class RequestTooLargeError(LLMError):
    """Raised when the request itself exceeds a provider request-size limit."""

    pass


class TimeoutError(LLMError):
    """Raised when an LLM request times out."""

    pass
