"""Asynchronous Python client for OpenRouter."""

from __future__ import annotations

from functools import partial
from typing import Any


class OpenRouterError(Exception):
    """Generic exception."""


class OpenRouterConnectionError(OpenRouterError):
    """OpenRouter connection exception."""


class OpenRouterAuthenticationError(OpenRouterError):
    """OpenRouter authentication exception."""


class OpenRouterResponseError(OpenRouterConnectionError):
    """OpenRouter returned an error response."""

    def __init__(
        self,
        msg: str,
        data: dict[str, Any],
        *,
        status: int,
        code: int | str | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the exception."""
        super().__init__(msg, data)
        self.status = status
        self.code = code
        self.message = message

    def __reduce__(self) -> tuple[Any, ...]:
        """Support pickling and copying despite the keyword-only arguments."""
        return (
            partial(
                type(self), status=self.status, code=self.code, message=self.message
            ),
            self.args,
        )


class OpenRouterBadRequestError(OpenRouterResponseError):
    """OpenRouter rejected the request parameters."""


class OpenRouterInsufficientCreditsError(OpenRouterResponseError):
    """OpenRouter account has insufficient credits."""


class OpenRouterPayloadTooLargeError(OpenRouterResponseError):
    """OpenRouter rejected the request because it is too large."""


class OpenRouterRateLimitError(OpenRouterResponseError):
    """OpenRouter rate limit exceeded."""
