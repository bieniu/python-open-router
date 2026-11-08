"""Asynchronous Python client for Open Router."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import metadata
import socket
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientResponseError, ClientSession
from aiohttp.hdrs import METH_DELETE, METH_GET, METH_PATCH, METH_POST
import orjson
from yarl import URL

from python_open_router.decisions import DecisionResponse
from python_open_router.exceptions import (
    OpenRouterAuthenticationError,
    OpenRouterBadRequestError,
    OpenRouterConnectionError,
    OpenRouterInsufficientCreditsError,
    OpenRouterPayloadTooLargeError,
    OpenRouterRateLimitError,
    OpenRouterResponseError,
)
from python_open_router.models import (
    DeletedKeyResponse,
    Key,
    KeyCreateResponse,
    KeyData,
    KeyDataWrapper,
    KeyResponseWrapper,
    KeysDataWrapper,
    Model,
    ModelResponseWrapper,
    ModelsDataWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from python_open_router.decisions import DecisionQuestion, DecisionState, ImageAsset


VERSION = metadata.version(__package__)
HOST = "openrouter.ai"

_ERRORS_BY_STATUS: dict[int, type[OpenRouterResponseError]] = {
    400: OpenRouterBadRequestError,
    402: OpenRouterInsufficientCreditsError,
    413: OpenRouterPayloadTooLargeError,
    429: OpenRouterRateLimitError,
}


def _response_error(
    status: int, content_type: str, text: str
) -> OpenRouterResponseError:
    """Build the exception for an error response."""
    code: int | str | None = None
    message: str | None = None
    try:
        body = orjson.loads(text)  # pylint: disable=no-member
    except orjson.JSONDecodeError:  # pylint: disable=no-member
        body = None
    if isinstance(body, dict) and isinstance(error := body.get("error"), dict):
        code = error.get("code")
        message = error.get("message")
    msg = f"Unexpected response from OpenRouter ({status})"
    if message:
        msg = f"{msg}: {message}"
    return _ERRORS_BY_STATUS.get(status, OpenRouterResponseError)(
        msg,
        {"Content-Type": content_type, "response": text},
        status=status,
        code=code,
        message=message,
    )


@dataclass
class OpenRouterClient:
    """Main class for handling connections with OpenRouter."""

    api_key: str
    session: ClientSession | None = None
    request_timeout: int = 10
    _close_session: bool = False

    async def _request(
        self,
        method: str,
        uri: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        api_version: str = "v1",
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> str:
        """Handle a request to OpenRouter."""
        url = URL.build(host=HOST, scheme="https").joinpath(f"api/{api_version}/{uri}")

        headers = {
            "User-Agent": f"PythonOpenRouter/{VERSION}",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if self.session is None:
            self.session = ClientSession()
            self._close_session = True

        try:
            async with asyncio.timeout(
                self.request_timeout if timeout is None else timeout
            ):
                response = await self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=data,
                )
        except TimeoutError as exception:
            msg = "Timeout occurred while connecting to the service"
            raise OpenRouterConnectionError(msg) from exception
        except (
            ClientError,
            ClientResponseError,
            socket.gaierror,
        ) as exception:
            msg = "Error occurred while communicating with the service"
            raise OpenRouterConnectionError(msg) from exception

        if response.status == 401:
            msg = "Authentication failed"
            raise OpenRouterAuthenticationError(msg)

        if response.status >= 400:
            content_type = response.headers.get("Content-Type", "")
            text = await response.text()
            raise _response_error(response.status, content_type, text)

        return await response.text()

    async def get_key_data(self) -> KeyData:
        """Get key data for API key."""
        response = await self._request(METH_GET, "key")
        return KeyDataWrapper.from_json(response).data

    async def get_keys(
        self,
        *,
        include_disabled: bool | None = None,
        offset: int | None = None,
        workspace_id: str | None = None,
    ) -> list[Key]:
        """Get all keys."""
        params: dict[str, Any] = {}
        if include_disabled is not None:
            params["include_disabled"] = str(include_disabled).lower()
        if offset is not None:
            params["offset"] = str(offset)
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        response = await self._request(METH_GET, "keys", params=params or None)
        return KeysDataWrapper.from_json(response).data

    async def create_key(
        self,
        name: str,
        limit: float | None = None,
        *,
        limit_reset: str | None = None,
        workspace_id: str | None = None,
        expires_at: str | None = None,
        include_byok_in_limit: bool | None = None,
    ) -> Key:
        """Create a new key."""
        data: dict[str, Any] = {"name": name}
        if limit is not None:
            data["limit"] = limit
        if limit_reset is not None:
            data["limit_reset"] = limit_reset
        if workspace_id is not None:
            data["workspace_id"] = workspace_id
        if expires_at is not None:
            data["expires_at"] = expires_at
        if include_byok_in_limit is not None:
            data["include_byok_in_limit"] = include_byok_in_limit

        response = await self._request(METH_POST, "keys", data=data)
        return KeyCreateResponse.from_json(response).data

    async def get_key(self, key_hash: str) -> Key:
        """Get a single key by hash."""
        response = await self._request(METH_GET, f"keys/{key_hash}")
        return KeyResponseWrapper.from_json(response).data

    async def update_key(
        self,
        key_hash: str,
        *,
        name: str | None = None,
        disabled: bool | None = None,
        limit: float | None = None,
        limit_reset: str | None = None,
        include_byok_in_limit: bool | None = None,
    ) -> Key:
        """Update an existing key."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if disabled is not None:
            data["disabled"] = disabled
        if limit is not None:
            data["limit"] = limit
        if limit_reset is not None:
            data["limit_reset"] = limit_reset
        if include_byok_in_limit is not None:
            data["include_byok_in_limit"] = include_byok_in_limit

        response = await self._request(METH_PATCH, f"keys/{key_hash}", data=data)
        return KeyResponseWrapper.from_json(response).data

    async def delete_key(self, key_hash: str) -> bool:
        """Delete a key by hash."""
        response = await self._request(METH_DELETE, f"keys/{key_hash}")
        return DeletedKeyResponse.from_json(response).deleted

    async def get_model(self, author: str, slug: str) -> Model:
        """Get a single model by author and slug."""
        response = await self._request(METH_GET, f"model/{author}/{slug}")
        return ModelResponseWrapper.from_json(response).data

    @staticmethod
    def _filter_model_params(
        kwargs: dict[str, Any],
        *,
        numeric_keys: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Filter None values and convert numeric params to strings."""
        params = {k: v for k, v in kwargs.items() if v is not None}
        for key in numeric_keys:
            if key in params:
                params[key] = str(params[key])
        return params

    async def get_models(  # pylint: disable=too-many-locals
        self,
        *,
        category: str | None = None,
        supported_parameters: str | None = None,
        output_modalities: str | None = None,
        sort: str | None = None,
        q: str | None = None,
        input_modalities: str | None = None,
        context: int | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        arch: str | None = None,
        model_authors: str | None = None,
        providers: str | None = None,
        distillable: str | None = None,
        zdr: str | None = None,
        region: str | None = None,
    ) -> list[Model]:
        """Get all available models."""
        numeric = frozenset({"context", "min_price", "max_price"})
        params = self._filter_model_params(
            {
                "category": category,
                "supported_parameters": supported_parameters,
                "output_modalities": output_modalities,
                "sort": sort,
                "q": q,
                "input_modalities": input_modalities,
                "context": context,
                "min_price": min_price,
                "max_price": max_price,
                "arch": arch,
                "model_authors": model_authors,
                "providers": providers,
                "distillable": distillable,
                "zdr": zdr,
                "region": region,
            },
            numeric_keys=numeric,
        )
        response = await self._request(METH_GET, "models", params=params or None)
        return ModelsDataWrapper.from_json(response).data

    async def count_models(
        self,
        *,
        output_modalities: str | None = None,
    ) -> int:
        """Get total count of available models."""
        params: dict[str, Any] = {}
        if output_modalities is not None:
            params["output_modalities"] = output_modalities
        response = await self._request(METH_GET, "models/count", params=params or None)
        result: dict[str, Any] = orjson.loads(response)  # pylint: disable=no-member
        count: int = result.get("data", {}).get("count", 0)
        return count

    async def create_decision(
        self,
        *,
        model: str,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion],
        assets: Mapping[str, ImageAsset] | None = None,
        provider: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        user: str | None = None,
        trace: Mapping[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> DecisionResponse:
        """Answer questions about a state with a decision model.

        This uses the alpha Decisions API, whose shape may still change.
        """
        data: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": questions,
        }
        optional = {
            "assets": assets,
            "provider": provider,
            "session_id": session_id,
            "user": user,
            "trace": trace,
        }
        data.update({k: v for k, v in optional.items() if v is not None})
        response = await self._request(
            METH_POST,
            "decisions",
            data=data,
            api_version="alpha",
            timeout=timeout,
        )
        return DecisionResponse.from_json(response)

    async def close(self) -> None:
        """Close open client session."""
        if self.session and self._close_session:
            await self.session.close()

    async def __aenter__(self) -> Self:
        """Async enter.

        Returns
        -------
            The OpenRouterClient object.

        """
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        """Async exit.

        Args:
        ----
            _exc_info: Exec type.

        """
        await self.close()
