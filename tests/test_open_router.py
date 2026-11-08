"""Tests for the client."""

from __future__ import annotations

import asyncio
import json
import pickle
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import ClientError
from aiohttp.hdrs import METH_DELETE, METH_GET, METH_PATCH, METH_POST
from aiointercept import CallbackResult, aiointercept
from mashumaro.exceptions import MissingField
import pytest

from python_open_router import (
    DecisionResponse,
    DecisionUsage,
    Modality,
    ModelReasoning,
    NoulAnswer,
    OpenRouterAuthenticationError,
    OpenRouterBadRequestError,
    OpenRouterClient,
    OpenRouterConnectionError,
    OpenRouterError,
    OpenRouterInsufficientCreditsError,
    OpenRouterPayloadTooLargeError,
    OpenRouterRateLimitError,
    OpenRouterResponseError,
    ReasoningEffort,
    RefusalAnswer,
    UnknownAnswer,
)
from tests import load_fixture
from tests.const import DECISIONS_URL, HEADERS, MOCK_URL

if TYPE_CHECKING:
    from syrupy import SnapshotAssertion


async def test_putting_in_own_session(
    responses: aiointercept,
) -> None:
    """Test putting in own session."""
    responses.get(
        f"{MOCK_URL}/key",
        status=200,
        body=load_fixture("key.json"),
    )
    async with aiohttp.ClientSession() as session:
        open_router = OpenRouterClient("abc", session=session)
        await open_router.get_key_data()
        assert open_router.session is not None
        assert not open_router.session.closed
        await open_router.close()
        assert not open_router.session.closed


async def test_creating_own_session(
    responses: aiointercept,
) -> None:
    """Test creating own session."""
    responses.get(
        f"{MOCK_URL}/key",
        status=200,
        body=load_fixture("key.json"),
    )
    open_router = OpenRouterClient("abc")
    await open_router.get_key_data()
    assert open_router.session is not None
    assert not open_router.session.closed
    await open_router.close()
    assert open_router.session.closed


async def test_unexpected_server_response(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test handling unexpected response."""
    responses.get(
        f"{MOCK_URL}/key",
        status=404,
        headers={"Content-Type": "plain/text"},
        body="Yes",
    )
    with pytest.raises(OpenRouterError):
        await client.get_key_data()


async def test_timeout(
    responses: aiointercept,
) -> None:
    """Test request timeout."""

    # Faking a timeout by sleeping
    async def response_handler(_: str, **_kwargs: Any) -> CallbackResult:
        """Response handler for this test."""
        await asyncio.sleep(2)
        return CallbackResult(body="Goodmorning!")

    responses.get(
        f"{MOCK_URL}/key",
        callback=response_handler,
    )
    async with OpenRouterClient(
        "abc",
        request_timeout=1,
    ) as open_router:
        with pytest.raises(OpenRouterConnectionError):
            await open_router.get_key_data()


async def test_client_error(
    client: OpenRouterClient,
    responses: aiointercept,
) -> None:
    """Test client error."""

    async def response_handler(_: str, **_kwargs: Any) -> CallbackResult:
        """Response handler for this test."""
        raise ClientError

    responses.get(
        f"{MOCK_URL}/request/count",
        callback=response_handler,
    )
    with pytest.raises(OpenRouterConnectionError):
        await client.get_key_data()


@pytest.mark.parametrize(
    ("endpoint", "fixture", "method"),
    [
        ("key", "key.json", "get_key_data"),
        ("keys", "keys.json", "get_keys"),
        ("models", "models.json", "get_models"),
    ],
    ids=[
        "get_key_data",
        "get_keys",
        "get_models",
    ],
)
async def test_data_retrieval(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
    endpoint: str,
    fixture: str,
    method: str,
) -> None:
    """Test data retrieval."""
    responses.get(
        f"{MOCK_URL}/{endpoint}",
        status=200,
        body=load_fixture(fixture),
    )
    assert await getattr(client, method)() == snapshot
    responses.assert_called_once_with(
        f"{MOCK_URL}/{endpoint}",
        METH_GET,
        headers=HEADERS,
        params=None,
        json=None,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "name", "limit": 5.0},
        {"name": "name"},
    ],
)
async def test_create_key(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
    kwargs: dict[str, Any],
) -> None:
    """Test creating a key."""
    responses.post(
        f"{MOCK_URL}/keys",
        status=200,
        body=load_fixture("create_key.json"),
    )
    assert await client.create_key(**kwargs) == snapshot
    responses.assert_called_once_with(
        f"{MOCK_URL}/keys",
        METH_POST,
        headers=HEADERS,
        params=None,
        json=kwargs,
    )


async def test_delete_key(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test deleting a key."""
    responses.delete(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        status=200,
        body=load_fixture("delete_key.json"),
    )
    result = await client.delete_key("abcabcabcbababcbabcabcabc")
    assert result is True
    responses.assert_called_once_with(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        METH_DELETE,
        headers=HEADERS,
        params=None,
        json=None,
    )


async def test_get_key(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test getting a single key."""
    responses.get(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        status=200,
        body=load_fixture("get_key.json"),
    )
    assert await client.get_key("abcabcabcbababcbabcabcabc") == snapshot
    responses.assert_called_once_with(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        METH_GET,
        headers=HEADERS,
        params=None,
        json=None,
    )


async def test_update_key(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test updating a key."""
    responses.patch(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        status=200,
        body=load_fixture("get_key.json"),
    )
    assert (
        await client.update_key(
            "abcabcabcbababcbabcabcabc", name="New Name", disabled=True
        )
        == snapshot
    )
    responses.assert_called_once_with(
        f"{MOCK_URL}/keys/abcabcabcbababcbabcabcabc",
        METH_PATCH,
        headers=HEADERS,
        params=None,
        json={"name": "New Name", "disabled": True},
    )


async def test_get_model(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test getting a single model."""
    responses.get(
        f"{MOCK_URL}/model/amazon/nova-premier-v1",
        status=200,
        body=load_fixture("single_model.json"),
    )
    assert await client.get_model("amazon", "nova-premier-v1") == snapshot
    responses.assert_called_once_with(
        f"{MOCK_URL}/model/amazon/nova-premier-v1",
        METH_GET,
        headers=HEADERS,
        params=None,
        json=None,
    )


async def test_get_model_with_reasoning(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test getting a model with reasoning options."""
    responses.get(
        f"{MOCK_URL}/model/anthropic/claude-opus-4.6",
        status=200,
        body=load_fixture("reasoning_model.json"),
    )
    assert await client.get_model("anthropic", "claude-opus-4.6") == snapshot


async def test_unsupported_reasoning_effort(
    responses: aiointercept,
    client: OpenRouterClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an unknown reasoning effort is dropped instead of failing."""
    fixture = json.loads(load_fixture("reasoning_model.json"))
    fixture["data"]["reasoning"]["supported_efforts"].append("extreme")
    fixture["data"]["reasoning"]["default_effort"] = "extreme"
    responses.get(
        f"{MOCK_URL}/model/anthropic/claude-opus-4.6",
        status=200,
        body=json.dumps(fixture),
    )
    model = await client.get_model("anthropic", "claude-opus-4.6")
    assert model.reasoning is not None
    assert model.reasoning.supported_efforts == [
        ReasoningEffort.MAX,
        ReasoningEffort.HIGH,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.LOW,
        ReasoningEffort.NONE,
    ]
    assert model.reasoning.default_effort is None
    assert "Unsupported reasoning effort: extreme" in caplog.text


@pytest.mark.parametrize(
    ("reasoning", "efforts"),
    [
        ({"mandatory": True}, None),
        ({"mandatory": True, "supported_efforts": None}, list(ReasoningEffort)),
    ],
    ids=["omitted", "null"],
)
def test_reasoning_supported_efforts(
    reasoning: dict[str, Any],
    efforts: list[ReasoningEffort] | None,
) -> None:
    """Test omitted and null supported efforts stay distinct."""
    assert ModelReasoning.from_dict(reasoning).supported_efforts == efforts


async def test_count_models(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test counting models."""
    responses.get(
        f"{MOCK_URL}/models/count",
        status=200,
        body='{"data": {"count": 100}}',
    )
    result = await client.count_models()
    assert result == 100
    responses.assert_called_once_with(
        f"{MOCK_URL}/models/count",
        METH_GET,
        headers=HEADERS,
        params=None,
        json=None,
    )


async def test_get_models_with_filters(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test getting models with filter parameters."""
    responses.get(
        re.compile(r"https://openrouter\.ai/api/v1/models\?.*"),
        status=200,
        body=load_fixture("models.json"),
    )
    await client.get_models(
        sort="most-popular",
        q="gpt",
        min_price=0.0,
        max_price=10.0,
    )
    responses.assert_called_once()


async def test_get_keys_with_filters(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test getting keys with filter parameters."""
    responses.get(
        re.compile(r"https://openrouter\.ai/api/v1/keys\?.*"),
        status=200,
        body=load_fixture("keys.json"),
    )
    await client.get_keys(include_disabled=True, offset=10)
    responses.assert_called_once()


async def test_get_decision_models(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test getting decision models."""
    responses.get(
        f"{MOCK_URL}/models?output_modalities=decisions",
        status=200,
        body=load_fixture("decision_models.json"),
    )
    models = await client.get_models(output_modalities="decisions")
    assert all(m.architecture.output_modalities == [Modality.DECISIONS] for m in models)
    assert models == snapshot


async def test_unsupported_modality(
    responses: aiointercept,
    client: OpenRouterClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an unknown modality is dropped instead of failing."""
    fixture = json.loads(load_fixture("decision_models.json"))
    architecture = fixture["data"][0]["architecture"]
    architecture["input_modalities"].append("smell")
    architecture["output_modalities"].append("hologram")
    responses.get(
        f"{MOCK_URL}/models",
        status=200,
        body=json.dumps(fixture),
    )
    models = await client.get_models()
    assert models[0].architecture.input_modalities == [Modality.TEXT]
    assert models[0].architecture.output_modalities == [Modality.DECISIONS]
    assert "Unsupported modality: smell" in caplog.text
    assert "Unsupported modality: hologram" in caplog.text


DECISION_QUESTIONS: dict[str, Any] = {
    "too_hot": {
        "type": "noul",
        "instructions": "Is the living room uncomfortably hot?",
        "criteria": {"true": "Above 26 C.", "false": "26 C or below."},
    },
    "action": {
        "type": "choice",
        "instructions": "What should the house do?",
        "criteria": {
            "open_windows": "Open windows to ventilate.",
            "turn_on_ac": "Turn on air conditioning.",
            "nothing": "Do nothing.",
        },
    },
    "comfort": {
        "type": "score",
        "instructions": "How comfortable is the room?",
        "criteria": ["Very uncomfortable", "Acceptable", "Comfortable"],
    },
}


async def test_create_decision(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test creating a decision with noul, choice and score questions."""
    responses.post(
        DECISIONS_URL,
        status=200,
        body=load_fixture("decision_text.json"),
    )
    state = {"sensor": "living room", "temperature_c": 29.5}
    assert (
        await client.create_decision(
            model="parallel/decisions-fast:free",
            state=state,
            questions=DECISION_QUESTIONS,
        )
        == snapshot
    )
    responses.assert_called_once_with(
        DECISIONS_URL,
        METH_POST,
        headers=HEADERS,
        params=None,
        json={
            "model": "parallel/decisions-fast:free",
            "state": state,
            "questions": DECISION_QUESTIONS,
        },
    )


async def test_create_decision_with_image(
    responses: aiointercept,
    client: OpenRouterClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test creating a decision with an image asset and every option."""
    responses.post(
        DECISIONS_URL,
        status=200,
        body=load_fixture("decision_image.json"),
    )
    state = ["`camera` is the image below.", "img-5d1c2f0e"]
    questions: dict[str, Any] = {
        "has_square": {
            "type": "noul",
            "instructions": "Is there a red square in `camera`?",
        },
        "square": {
            "type": "box",
            "instructions": "Where is the red square in `camera`?",
        },
    }
    assets: dict[str, Any] = {
        "img-5d1c2f0e": {"type": "image", "format": "png", "data_base64": "iVBORw0K"}
    }
    assert (
        await client.create_decision(
            model="typesafe/jev-1.14",
            state=state,
            questions=questions,
            assets=assets,
            provider={"sort": "price"},
            session_id="session",
            user="user",
            trace={"trace_id": "trace"},
        )
        == snapshot
    )
    responses.assert_called_once_with(
        DECISIONS_URL,
        METH_POST,
        headers=HEADERS,
        params=None,
        json={
            "model": "typesafe/jev-1.14",
            "state": state,
            "questions": questions,
            "assets": assets,
            "provider": {"sort": "price"},
            "session_id": "session",
            "user": "user",
            "trace": {"trace_id": "trace"},
        },
    )


async def test_create_decision_refusal_and_unknown_answer(
    responses: aiointercept,
    client: OpenRouterClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test refusals and unknown answer types do not fail the response."""
    fixture = json.loads(load_fixture("decision_text.json"))
    fixture["answers"]["action"] = {"type": "refusal"}
    fixture["answers"]["comfort"] = {"type": "ranking", "ranking": ["a", "b"]}
    responses.post(
        DECISIONS_URL,
        status=200,
        body=json.dumps(fixture),
    )
    response = await client.create_decision(
        model="parallel/decisions-fast:free",
        state="state",
        questions=DECISION_QUESTIONS,
    )
    assert response.answers["action"] == RefusalAnswer()
    assert response.answers["comfort"] == UnknownAnswer(raw_type="ranking")
    assert response.answers["too_hot"] == NoulAnswer(noul=0.9732037100253741)
    assert "Unsupported answer type: ranking" in caplog.text


@pytest.mark.parametrize(
    ("status", "body", "exception", "code", "message"),
    [
        (
            400,
            '{"error":{"message":"Model nope/nope does not exist","code":400}}',
            OpenRouterBadRequestError,
            400,
            "Model nope/nope does not exist",
        ),
        (
            402,
            '{"error":{"message":"Insufficient credits","code":402}}',
            OpenRouterInsufficientCreditsError,
            402,
            "Insufficient credits",
        ),
        (
            413,
            '{"error":{"message":"Payload too large","code":413}}',
            OpenRouterPayloadTooLargeError,
            413,
            "Payload too large",
        ),
        (
            429,
            '{"error":{"message":"Rate limit exceeded","code":429}}',
            OpenRouterRateLimitError,
            429,
            "Rate limit exceeded",
        ),
        (
            403,
            '{"error":{"message":"Management key required","code":403}}',
            OpenRouterResponseError,
            403,
            "Management key required",
        ),
        (503, "Service Unavailable", OpenRouterResponseError, None, None),
        (529, '{"error": "overloaded"}', OpenRouterResponseError, None, None),
    ],
)
async def test_error_response(
    responses: aiointercept,
    client: OpenRouterClient,
    status: int,
    body: str,
    exception: type[OpenRouterResponseError],
    code: int | None,
    message: str | None,
) -> None:
    """Test error responses raise typed, backwards compatible exceptions."""
    responses.post(
        DECISIONS_URL,
        status=status,
        headers={"Content-Type": "application/json"},
        body=body,
    )
    with pytest.raises(exception) as exc_info:
        await client.create_decision(
            model="parallel/decisions-fast:free",
            state="state",
            questions=DECISION_QUESTIONS,
        )
    assert exc_info.type is exception
    assert isinstance(exc_info.value, OpenRouterConnectionError)
    assert exc_info.value.status == status
    assert exc_info.value.code == code
    assert exc_info.value.message == message
    assert exc_info.value.args[1] == {
        "Content-Type": "application/json",
        "response": body,
    }


async def test_authentication_error(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test 401 still raises an authentication error."""
    responses.post(
        DECISIONS_URL,
        status=401,
        body='{"error":{"message":"User not found.","code":401}}',
    )
    with pytest.raises(OpenRouterAuthenticationError):
        await client.create_decision(
            model="parallel/decisions-fast:free",
            state="state",
            questions=DECISION_QUESTIONS,
        )


async def _slow_decision(_: str, **_kwargs: Any) -> CallbackResult:
    """Respond to a decisions request after a delay."""
    await asyncio.sleep(0.2)
    return CallbackResult(body=load_fixture("decision_text.json"))


async def test_create_decision_timeout(
    responses: aiointercept,
    client: OpenRouterClient,
) -> None:
    """Test a shorter per-call timeout overrides the client timeout."""
    responses.post(DECISIONS_URL, callback=_slow_decision)
    with pytest.raises(OpenRouterConnectionError):
        await client.create_decision(
            model="parallel/decisions-fast:free",
            state="state",
            questions=DECISION_QUESTIONS,
            timeout=0.05,
        )


async def test_create_decision_longer_timeout(
    responses: aiointercept,
) -> None:
    """Test a longer per-call timeout overrides the client timeout."""
    responses.post(DECISIONS_URL, callback=_slow_decision)
    async with OpenRouterClient("abc", request_timeout=0) as open_router:
        await open_router.create_decision(
            model="parallel/decisions-fast:free",
            state="state",
            questions=DECISION_QUESTIONS,
            timeout=5,
        )


def test_decision_response_from_dict_before_from_json() -> None:
    """Test from_dict before the first from_json does not break from_json.

    Runs in a fresh interpreter, as mashumaro compiles the unpackers lazily.
    """
    code = (
        "import json; from pathlib import Path;"
        "from python_open_router import DecisionResponse;"
        "s = Path('tests/fixtures/decision_image.json').read_text();"
        "DecisionResponse.from_dict(json.loads(s));"
        "DecisionResponse.from_json(s)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


async def test_decision_response_round_trip() -> None:
    """Test a decision response survives to_dict and from_dict."""
    response = DecisionResponse.from_json(load_fixture("decision_text.json"))
    assert response.annotations
    assert DecisionResponse.from_dict(response.to_dict()) == response


@pytest.mark.parametrize(
    ("usage", "cached_tokens"),
    [
        ({"input_tokens_details": {"cached_tokens": 12}}, 12),
        ({"input_tokens_details": None}, None),
        ({"cached_tokens": 7}, 7),
        ({}, None),
    ],
)
def test_decision_usage_cached_tokens(
    usage: dict[str, Any],
    cached_tokens: int | None,
) -> None:
    """Test cached tokens are read from the input token details."""
    usage = {"input_tokens": 10, "output_tokens": 0, **usage}
    assert DecisionUsage.from_dict(usage).cached_tokens == cached_tokens


@pytest.mark.parametrize(
    ("answer", "raw_type"),
    [(None, None), ({"noul": 0.5}, None), ({"type": "ranking"}, "ranking")],
)
def test_malformed_decision_answer(
    answer: dict[str, Any] | None,
    raw_type: str | None,
) -> None:
    """Test malformed answers become unknown answers."""
    data = json.loads(load_fixture("decision_text.json"))
    data["answers"]["action"] = answer
    data["annotations"].extend(["text", {"type": "file_citation"}])
    response = DecisionResponse.from_dict(data)
    assert response.answers["action"] == UnknownAnswer(raw_type=raw_type)
    assert len(response.annotations) == 2
    assert data["answers"]["action"] == answer


def test_decision_response_missing_answers() -> None:
    """Test a response without answers is rejected."""
    data = json.loads(load_fixture("decision_text.json"))
    del data["answers"]
    with pytest.raises(MissingField):
        DecisionResponse.from_dict(data)


def test_response_error_pickle() -> None:
    """Test response errors survive pickling."""
    error = OpenRouterRateLimitError(
        "msg",
        {"Content-Type": "application/json", "response": "{}"},
        status=429,
        code=429,
        message="Rate limit exceeded",
    )
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301
    assert restored.__class__ is OpenRouterRateLimitError
    assert restored.args == error.args
    assert (restored.status, restored.code, restored.message) == (
        429,
        429,
        "Rate limit exceeded",
    )
