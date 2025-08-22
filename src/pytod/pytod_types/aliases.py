#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Contains aliases for type annotations used in this library"""
from collections import defaultdict
from typing import Literal, TypedDict, Union

from pydantic import StrictStr

SPLITS: list[Literal["train", "test", "dev"]] = ["train", "dev", "test"]
ShardName = str  # name of an SGD dialogue file (dialogues_*.json)
DialogueID = str  # SGD dialogue ID (eg 1_00000)
TaskSequence = str
"""A string representing the sequence of service-intent types
in a dialogue. These are the keys of index conversation.json nodes."""
SGDAuthor = Literal[
    "USER", "SYSTEM"
]  # SGD authors name, as annotated in dialogue files
SGDDialogueDict = dict
# dictionary in the same format as the raw SGD dialogue
ServiceName = str  # service name as it appears in the SGD schema
IntentName = str  # intent name as it appears in the SGD schema
# {ServiceName}.{IntentName} formatted string. Used to avoid code
#  errors that can occur due to cross-service intent name sharing
ServiceIntent = str
SlotName = str  # slot name as it appears in the SGD schema
ValueList = list[SlotName]  # list of strings containing slot values
# map containing slots carried over at a given turn and their values
CarriedOverSlotMap = dict[SlotName, ValueList]
SGDServiceSchema = dict
CanonicalValue = str  # a value, as appears in the "values" annotation of an action
SlotValue = str  # value, as appears in the "values" annotation of an action
EnumValues = list[str]  # categorical slot values, as listed in the schema
DefaultValue = str  # default value of slot, as listed in the schema
ValueParaphrase = (
    str  # a value, as appears in the "canonical_values" annotation of an action
)
Entity = dict[SlotName, CanonicalValue]
CanonicalValueCollector = dict[CanonicalValue, set[SlotValue]]
ParaphraseValueCollector = defaultdict[
    ServiceName,
    defaultdict[SlotName, defaultdict[CanonicalValue, set[ValueParaphrase]]],
]
EntityName = str  # the name of an object returned by one of the commands
# command.py
ArgumentValue = Union[str, StrictStr, None]  # type of command argument value

# index_utils.py
NodeName = str
"""Type alias to identify SGD corpus index node names."""

# pytod specific terminology
VariableName = str
"""A PyTOD variable name. The naming convention is 'x' followed
by any number of digits."""
VariableIdx = int
"""The index (subscript) of a variable in a PyTOD program transcript.
This is different to the index predicted by the model because parts
of the transcript (eg hints, say calls) are hidden from the model
during inference."""
ToolName = str
"""Refers to a snake cased form of the user intent.
May be prefixed with the service name (eg music_2_find_song)"""


SlotValueDict = dict[SlotName, list[str]]
"""The format of the `slot_values` field in the user
frame state annotations"""


class StateDict(TypedDict):
    """The format of the state annotations
    in the user frame."""

    slot_values: SlotValueDict
    active_intent: str
    requested_slots: list[str]


APIDescription = str
"""Description of the API, possibly including usage examples."""
