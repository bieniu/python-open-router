"""Models for the OpenRouter Decisions API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

_LOGGER = logging.getLogger(__package__)

type DecisionState = str | Mapping[str, Any] | Sequence[Any]


class NoulQuestion(TypedDict):
    """Question answered with a probability that the statement is true."""

    type: Literal["noul"]
    instructions: Any
    criteria: NotRequired[Mapping[Literal["true", "false"], Any]]


class ChoiceQuestion(TypedDict):
    """Question answered by picking one of the options."""

    type: Literal["choice"]
    instructions: Any
    criteria: Mapping[str, Any]


class ScoreQuestion(TypedDict):
    """Question answered with a level, lowest level first."""

    type: Literal["score"]
    instructions: Any
    criteria: Sequence[Any]


class BoxQuestion(TypedDict):
    """Question answered with the location of a target in an image."""

    type: Literal["box"]
    instructions: Any


type DecisionQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion | BoxQuestion


class ImageAsset(TypedDict):
    """Image referenced by its key from the state, instructions or criteria."""

    type: Literal["image"]
    format: Literal["png", "jpg", "jpeg"]
    data_base64: str


@dataclass(kw_only=True)
class DecisionAnswer(DataClassORJSONMixin):
    """Answer to a decision question."""


@dataclass(kw_only=True)
class NoulAnswer(DecisionAnswer):
    """Answer to a noul question."""

    type: Literal["noul"] = "noul"
    noul: float
    confidence: float | None = None


@dataclass(kw_only=True)
class ChoiceAnswer(DecisionAnswer):
    """Answer to a choice question."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


@dataclass(kw_only=True)
class ScoreAnswer(DecisionAnswer):
    """Answer to a score question.

    The legend and probabilities are keyed by level index ("0", "1", ...).
    """

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Any] | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


@dataclass(kw_only=True)
class BoundingBox(DataClassORJSONMixin):
    """Bounding box, each edge a fraction of the image from the top left."""

    top: float
    bottom: float
    left: float
    right: float


@dataclass(kw_only=True)
class BoxAnswer(DecisionAnswer):
    """Answer to a box question."""

    type: Literal["box"] = "box"
    box: BoundingBox
    confidence: float | None = None


@dataclass(kw_only=True)
class RefusalAnswer(DecisionAnswer):
    """The model refused to answer the question."""

    type: Literal["refusal"] = "refusal"


@dataclass(kw_only=True)
class UnknownAnswer(DecisionAnswer):
    """Answer of a type this library does not know yet."""

    type: Literal["unknown"] = "unknown"
    raw_type: str | None = None


_ANSWER_TYPES = {"noul", "choice", "score", "box", "refusal"}

# A discriminator on the union, not on the base class: with include_subtypes
# on the base, from_json recursed forever once from_dict had run first.
type _AnyAnswer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer | RefusalAnswer | UnknownAnswer,
    Discriminator(field="type", include_supertypes=True),
]


@dataclass(kw_only=True)
class UrlCitation(DataClassORJSONMixin):
    """URL citation annotation."""

    url: str
    title: str | None = None
    content: str | None = None


@dataclass(kw_only=True)
class DecisionUsage(DataClassORJSONMixin):
    """Usage of a decisions request."""

    input_tokens: int
    output_tokens: int
    cost: float | None = None
    is_byok: bool | None = None
    cached_tokens: int | None = None

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Pre deserialize hook."""
        if "input_tokens_details" in d:
            details = d["input_tokens_details"] or {}
            d = {**d, "cached_tokens": details.get("cached_tokens")}
        return d


@dataclass(kw_only=True)
class DecisionResponse(DataClassORJSONMixin):
    """Response of a decisions request."""

    model: str
    answers: dict[str, _AnyAnswer]
    usage: DecisionUsage
    id: str | None = None
    provider: str | None = None
    annotations: list[UrlCitation] = field(default_factory=list)

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Pre deserialize hook."""
        if "answers" not in d:
            return d
        answers = {}
        for key, answer in (d["answers"] or {}).items():
            answer_type = answer.get("type") if isinstance(answer, dict) else None
            if answer_type not in _ANSWER_TYPES:
                _LOGGER.warning(
                    "Unsupported answer type: %s. Please report at https://github.com/joostlek/python-open-router/issues.",
                    answer_type,
                )
                answers[key] = {"type": "unknown", "raw_type": answer_type}
            else:
                answers[key] = answer
        # Citations are nested under a "url_citation" key, like chat completions.
        # Flat citations are what to_dict produces.
        annotations = [
            a.get("url_citation", a)
            for a in d.get("annotations") or []
            if isinstance(a, dict) and a.get("type", "url_citation") == "url_citation"
        ]
        return {**d, "answers": answers, "annotations": annotations}
