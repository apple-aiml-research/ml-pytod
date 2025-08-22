#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from typing_extensions import Annotated, Literal

from pytod.parser.expressions import ExpressionList
from pytod.pytod_types.aliases import IntentName, ServiceName
from pytod.utils import camel_to_snake_case


class UserTurn(BaseModel):
    author: Literal["User"]
    query: str


class SystemTurn(BaseModel):
    author: Literal["System"]
    expression: ExpressionList
    index: int
    novelty: str | None = None

    @field_validator("expression", mode="before")
    def create_from_string(cls, v: str) -> ExpressionList:
        return ExpressionList.from_string(v)

    def get_tool_name(self) -> str:
        return self.expression.get_tool_name()

    @field_serializer("expression")
    def dump_to_string(self, expression: ExpressionList) -> str:
        return str(expression)


class HintTurn(BaseModel):
    author: Literal["Hint"]
    dialog: str
    index: int
    # hints displayed for action
    # that are not in the ground
    # truth have 'None' origin
    origin: Optional[int] = None


class ResponseTurn(BaseModel):
    author: Literal["Response"]
    text: str


class SignalTurn(BaseModel):
    author: Literal["Signal"]
    dialog: str
    index: int
    origin: int


AnyTurn = Annotated[
    Union[UserTurn, SystemTurn, HintTurn, ResponseTurn, SignalTurn],
    Field(discriminator="author"),
]


class PyTODConversationMedatada(BaseModel):
    conversation_structure: dict[str, list[str]]
    truncated: bool
    single_turn: bool
    additional_labels: list[str]


class Slot(BaseModel):
    name: str
    description: str
    type: str
    choices: Optional[list[str]] = None

    @field_validator("name")
    @classmethod
    def check_casing(cls, v: str) -> str:
        snaked = camel_to_snake_case(v)
        assert snaked is not None
        return snaked


class Result(BaseModel):
    type: str
    properties: list[Slot]


class Intent(BaseModel):
    name: str
    domain_description: str | None = None
    description: str
    result: Result | None = None
    slots: list[Slot]
    novelty: str | None = None
    example_invocations: list[dict] | None = None

    @field_validator("name")
    @classmethod
    def check_casing(cls, v: str) -> str:
        snaked = camel_to_snake_case(v)
        assert snaked is not None
        return snaked


class ServiceInfo(BaseModel):
    next_service: ServiceName
    active_services: list[ServiceName]
    active_intent: IntentName
    previous_intent: IntentName
    none_intent: bool | None = False


class PyTODConversation(BaseModel):
    id: str = Field(alias="_id")
    turns: list[AnyTurn]
    intents: list[Intent] | None = None
    service_info: list[ServiceInfo] | None = None
    metadata: PyTODConversationMedatada | dict[str, Any] | None = None
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("intents")
    @classmethod
    def check_lengths_match(cls, field_value: str, info: ValidationInfo):
        if field_value is not None:
            assert len(field_value) == len(info.data["turns"])
        if "service_info" in info.data and info.data["service_info"] is not None:
            assert len(field_value) == len("service_info")
        return field_value

    @field_validator("service_info")
    @classmethod
    def check_lengths_match_(cls, field_value: str, info: ValidationInfo):
        if field_value is not None:
            assert len(field_value) == len(info.data["turns"])
        if "intents" in info.data and info.data["intents"] is not None:
            assert len(field_value) == len("intents")
        return field_value

    @field_validator("turns")
    @classmethod
    def check_turn_indices(cls, field_value: list[AnyTurn]):
        indices = []
        for turn in field_value:
            try:
                index = getattr(turn, "index")
                indices.append(index)
            except AttributeError:
                pass
        for first, next_ in zip(indices, indices[1:]):
            if next_ != first + 1:
                raise AssertionError("Transcript indices must be consecutive")

        return field_value
