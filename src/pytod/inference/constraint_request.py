#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import re

from pydantic import BaseModel, field_validator
from typing_extensions import Self

from pytod.inference.assistant import Request
from pytod.pytod_types.aliases import ServiceName
from pytod.utils import snake_case

logger = logging.getLogger(__name__)


class MemorisationInfo(BaseModel):
    services: list[ServiceName]


class SlotConstraintRequest(Request, frozen=True):
    """
    predicted_slot, predicted_slot_value
    known_slots
        The members of the schema that have correctly been
        predicted.
    memorisation_info
        This is set to a list of services from the same
        domain of which the hallucinated slot value is a
        member, if this exists. In this case, the prompt
        is formulated as a paraphrase identification task
        between slot descriptions.
    is_value_object_reference
        If `True`, all slots in the schema (including slots
        which can only be requested by the user) are included
        in the prompt.
    categorical
        If `True`, the assistant will prompt a language model
        with the definitions of bool and enum data slots along
        with an enumeration of all possible slot-value pairs.
    predicted_slot_description
        This is only populated if we are constraining a value
        object reference and `predicted_slot` is a member of
        the schema for which we are looking for slot similarity.
    select_kwarg
        Indicates that the constraint is applied to keyword
        that is part of a `select` statement.

    Notes
    -----
        05.04.2024 predicted_slot_description not in use,
        we use paraphrase matching instead.
    """

    predicted_slot: str
    predicted_slot_value: str | None = None
    known_slots: tuple[str, ...]
    memorisation_info: MemorisationInfo | None = None
    is_value_object_reference: bool = False
    categorical: bool = False
    predicted_slot_description: str | None = None
    select_kwarg: bool = False

    def __hash__(self):
        memo_info = tuple()
        if self.memorisation_info is not None:
            memo_info = tuple(self.memorisation_info.services)
        h = hash(
            tuple(
                sorted(
                    (self.predicted_slot, self.service) + self.known_slots + memo_info
                )
                + [self.is_value_object_reference, self.categorical, self.select_kwarg]
            )
        )
        return h

    def _memorisation_info__eq__(self, other: Self) -> bool:
        if self.memorisation_info is None:
            return other.memorisation_info is None
        try:
            return sorted(self.memorisation_info.services) == sorted(
                other.memorisation_info.services
            )
        except AttributeError:
            return False

    def __eq__(self, other: Self) -> bool:
        return (
            self.service == other.service
            and self.predicted_slot == other.predicted_slot
            and sorted(self.known_slots) == sorted(other.known_slots)
            and self._memorisation_info__eq__(other)
            and self.is_value_object_reference == other.is_value_object_reference
            and self.categorical == other.categorical
            and self.select_kwarg == other.select_kwarg
        )

    def __str__(self) -> str:
        known = "|".join(self.known_slots) if self.known_slots else "null"
        slot_disp = f"{self.predicted_slot}"
        service = snake_case(self.service)
        memo_disp = None
        if self.memorisation_info is not None:
            memo_disp = "|".join(self.memorisation_info.services)
        return (
            f"{service}::{slot_disp}"
            f"::known::{known}"
            f"::memo::{memo_disp}"
            f"::val_ref::{self.is_value_object_reference}"
            f"::categorical::{self.categorical}"
        )


class ValueConstraintRequest(Request, frozen=True):
    argument: str
    value: str

    @field_validator("value")
    @classmethod
    def ensure_no_object_reference(cls, v: str):
        try:
            assert not bool(re.match(r"^x\d{1,2}\.", v))
        except AssertionError:
            sanitised = v.split(".")[1]
            logger.warning(f"Invalid value: {v}. Sanitised: {sanitised}")
            return sanitised
        return v

    def __hash__(self):
        return hash((self.service, self.argument, self.value))

    def __eq__(self, other: Self) -> bool:
        return (
            self.service == other.service
            and self.argument == other.argument
            and self.value == other.value
        )

    def __str__(self) -> str:
        return (
            f"value_constraint::{self.service}"
            f"::argument:{self.argument}"
            f"::value::{self.value}"
        )


ConstraintRequest = SlotConstraintRequest | ValueConstraintRequest
