#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from copy import deepcopy
from typing import Any, Optional, Type, Union

from omegaconf import DictConfig
from pydantic import TypeAdapter

from pytod.command import CommandCollection
from pytod.pytod_types.pytod import (
    AnyTurn,
    HintTurn,
    Intent,
    PyTODConversation,
    ResponseTurn,
    ServiceInfo,
    SignalTurn,
    Slot,
    SystemTurn,
    UserTurn,
)
from pytod.toolbox.toolbox import get_command_name
from pytod.toolbox.toolbox_utils import AppEntity, Command, ToolBox

logger = logging.getLogger(__name__)


def maybe_update_intents(intents: list[Intent], unique_intents: list[dict[str, Any]]):
    """Add a new intent to `unique_intents`."""
    for intent in intents:
        intent = intent.model_dump()
        if intent not in unique_intents:
            unique_intents.append(intent)


def remove_example_invocations_key_name(
    field_name: str, pytod_conversation_transcript: dict[str, Any]
):
    """Avoid null `example_invocations` keys inside conversation intent annotations."""
    if pytod_conversation_transcript["intents"] is None:
        return
    for intent_dict in pytod_conversation_transcript["intents"]:
        try:
            intent_dict.pop(field_name)
        except KeyError:
            # raised when metadata_type is ServiceInfo
            pass


def parse_as_pytod_conversation(
    transcript: dict[str, Any],
    toolbox: ToolBox,
    unique_intents: list[dict[str, Any]],
    toolbox_config: DictConfig,
    command_collection: CommandCollection,
    metadata_type: Optional[Type[Intent] | Type[ServiceInfo]] = Intent,
    **kwargs,
) -> PyTODConversation:
    def parse_intent(
        api_info: dict[str, Union[str, bool]],
        toolbox: ToolBox,
        toolbox_config,
        command_collection: CommandCollection,
    ) -> Intent:
        def get_result_properties(app_entity: AppEntity) -> list[Slot]:
            prop_as_dict = [p.model_dump() for p in app_entity.properties]
            return TypeAdapter(list[Slot]).validate_python(prop_as_dict)

        def get_slots(command: Command) -> list[Slot]:
            prop_as_dict = [p.model_dump() for p in command.properties]
            return TypeAdapter(list[Slot]).validate_python(prop_as_dict)

        service = api_info["service"]
        sgd_cmd = api_info["function"]
        command_name = get_command_name(
            service,
            sgd_cmd,
            toolbox_config.skip_service_variations,
            toolbox_config.use_snake_case,
            # anonymize_service=toolbox_config.anonymize_service
        )
        command_obj = toolbox.commands[command_name]
        assert len(command_obj.app_entities) == 1
        app_entity = command_obj.app_entities[0]
        result_dict = {
            "type": command_obj.return_type,
            "properties": get_result_properties(app_entity),
        }
        domain_description = command_collection.get(
            service, sgd_cmd
        ).service_description
        intent_dict = {
            "name": command_name,
            "description": command_obj.description,
            "result": result_dict,
            "slots": get_slots(command_obj),
            "domain_description": domain_description,
            "novelty": api_info["novelty"],
        }
        return Intent.model_validate(intent_dict)

    transcript = deepcopy(transcript)
    transcript.pop("services")
    conversation_dict = {
        "metadata": transcript.pop("metadata"),
        "_id": transcript.pop("id"),
    }
    api_annotation = transcript.pop("apis")
    turn_service_info = transcript.pop("service_info")
    transcript_turns = transcript.pop("turns")
    intents, parsed_turns, service_info = [], [], []
    for i, (turn, api) in enumerate(zip(transcript_turns, api_annotation)):
        intents.append(parse_intent(api, toolbox, toolbox_config, command_collection))
        service_info.append(ServiceInfo.model_validate(turn_service_info[i]))
        parsed_turns.append(parse_turn(turn))

    conversation_dict["turns"] = parsed_turns
    if metadata_type is Intent:
        conversation_dict["intents"] = intents
    conversation_dict["service_info"] = service_info
    pytod_conversation = PyTODConversation.model_validate(conversation_dict)
    # update the collection of intents for the corpus with any new
    # intents from this conversation
    maybe_update_intents(intents, unique_intents)
    return pytod_conversation


def parse_turn(turn: dict) -> AnyTurn:
    match turn["author"]:
        case "User":
            return UserTurn.model_validate(turn)
        case "System":
            turn = SystemTurn.model_validate(turn)
            assert not turn.expression.feedback
            return turn
        case "Hint":
            return HintTurn.model_validate(turn)
        case "Signal":
            return SignalTurn.model_validate(turn)
        case "Response":
            return ResponseTurn.model_validate(turn)
        case _:
            raise ValueError(f"Unknown author: {turn['author']}")


def pytod_transcript_builder(dialogue: list[dict], **kwargs) -> PyTODConversation:
    """Parses a PyTOD transcript into a structured object."""
    return PyTODConversation.model_validate(dialogue)
