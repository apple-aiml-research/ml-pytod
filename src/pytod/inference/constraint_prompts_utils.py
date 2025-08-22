#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.command import ArgumentDefinition, CommandCollection
from pytod.inference.constraint_request import SlotConstraintRequest
from pytod.parser.expresion_validation_utils import ConstrainedKeyword

MQAChoice = str  # a, b, c ...
INVALID_SLOT = "invalid"


def int2alpha(index: int) -> MQAChoice:
    return chr(ord("a") + int(index) - 1)


def _get_slot_map(
    arg_def_map: dict[ArgumentDefinition, MQAChoice]
) -> dict[MQAChoice, ConstrainedKeyword]:
    """Return a mapping from question answer choice to slot names. Use to
    convert question answer to a constrained slot name."""
    choices = {
        choice: ConstrainedKeyword(arg.name, None)
        for arg, choice in arg_def_map.items()
    }
    invalid_choice = chr(ord(sorted(choices.keys())[-1]) + 1)
    choices[invalid_choice] = ConstrainedKeyword(INVALID_SLOT, None)
    return choices


def _collect_arg_defs(
    schema: CommandCollection,
    request: SlotConstraintRequest,
    optimise: bool = True,
) -> dict[ArgumentDefinition, MQAChoice]:
    all_args = schema.get_service_arg_definitions(request.service)

    if optimise:
        slot_schemas = [arg for arg in all_args if arg.name not in request.known_slots]
    else:
        slot_schemas = all_args
    return_args = slot_schemas if slot_schemas else all_args
    if request.is_value_object_reference:
        req_args = schema.get_requested_arg_schemas(request.service) or []
        return_args.extend([r for r in req_args if r not in return_args])
    return {arg: int2alpha(index + 1) for index, arg in enumerate(return_args)}


def build_categorical_choice_map(
    categ_arg_defs: list[ArgumentDefinition], include_none_choice: bool = True
) -> dict[MQAChoice, ConstrainedKeyword]:
    """Return a mapping from an index to a valid argument-value pair."""

    def preprocess_value(value: str) -> str:
        """Quote values appropriately to ensure constrained
        expressions can be parsed."""
        if value in {"true", "false"}:
            return value
        try:
            assert not value[0] in {'"', "'"}, value[-1] in {"'", '"'}
            return f"'{value}'"
        except AssertionError:
            raise AssertionError(f"Could not process value {value}")

    cnt = 0
    map_ = {}
    for slot in categ_arg_defs:
        for value in slot.possible_values:
            map_[int2alpha(cnt + 1)] = ConstrainedKeyword(
                slot.name, preprocess_value(value.lower())
            )
            cnt += 1
    if include_none_choice:
        map_[int2alpha(cnt + 1)] = ConstrainedKeyword(INVALID_SLOT, None)
    return map_
