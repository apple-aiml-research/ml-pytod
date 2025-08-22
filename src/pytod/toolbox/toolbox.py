#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from pytod.toolbox.toolbox_utils import (
    AppEntity,
    BuiltInTypes,
    Command,
    Property,
    ToolBox,
)
from pytod.utils import load_json, save_json, snake_case

logger = logging.getLogger(__name__)


def get_command_name(
    service_name: str,
    intent_name: str,
    skip_service_variations: bool,
    use_snake_case: bool,
    anonymize_service: bool = False,
) -> str:
    domain_name = service_name.split("_")[0]

    if skip_service_variations:
        name = domain_name + intent_name
        if use_snake_case:
            name = snake_case(name)
    else:
        if anonymize_service:
            service_name = domain_name
        if use_snake_case:
            name = f"{service_name}{intent_name}"
            name = snake_case(name)
        else:
            name = f"{service_name}.{intent_name}"
    return name


def get_entity_name(name: str, use_snake_case: bool) -> str:
    if use_snake_case:
        return snake_case(name)
    return name


def get_app_entity_name(
    intent_name: str, entity_suffix: str, use_snake_case: bool
) -> str:
    app_entity_name = (
        f"{intent_name}_{snake_case(entity_suffix)}"
        if use_snake_case
        else f"{intent_name}{entity_suffix}"
    )
    return app_entity_name


def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def should_skip(tool: dict[str, Any]) -> bool:
    """Services that are very similar to a given schema may be skipped."""
    return tool["service_name"].split("_")[1] != "1"


def get_toolbox(
    schema_path: Path,
    refine: bool = False,
    skip_service_variations: bool = False,
    use_snake_case: bool = True,
    entity_suffix: str = "",
    output_path: Optional[Path] = None,
) -> ToolBox:
    converted_tools = {}
    logger.info(
        f"Converting schema to toolbox with return types option: refined={refine}"
    )
    for tool in load_json(schema_path):
        if skip_service_variations and should_skip(tool):
            continue
        slots = tool["slots"]
        converted_properties = []
        for slot in slots:
            name = slot["name"]
            description = slot["description"]
            choices = None
            if set(slot["possible_values"]) == {"True", "False"}:
                type_ = BuiltInTypes.bool
            elif slot["is_categorical"]:
                if all([v.isdigit() for v in slot["possible_values"]]):
                    type_ = BuiltInTypes.int
                elif all([is_float(v) for v in slot["possible_values"]]):
                    type_ = BuiltInTypes.float
                else:
                    type_ = BuiltInTypes.enum
                    choices = slot["possible_values"]
            else:
                type_ = BuiltInTypes.str
            converted_properties.append(
                Property(
                    name=name,
                    description=description,
                    type=type_.value,
                    choices=choices,
                )
            )
        for intent in tool["intents"]:
            required_slot_names = intent["required_slots"]
            all_slot_names = required_slot_names + list(intent["optional_slots"].keys())
            properties = [
                property
                for property in converted_properties
                if property.name in all_slot_names
            ]
            return_app_entity_properties = properties
            if refine:
                try:
                    return_app_entity_properties = [
                        property
                        for property in converted_properties
                        if property.name in intent["api_returns"]
                    ]
                except KeyError:
                    logger.warning(
                        f"Could not find api_returns for {tool[intent]}. "
                        f"Entity will be the union of required and optional args"
                    )
                    return_app_entity_properties = properties

            name = get_command_name(
                tool["service_name"],
                intent["name"],
                skip_service_variations,
                use_snake_case,
            )
            app_entity_name = get_app_entity_name(name, entity_suffix, use_snake_case)
            return_app_entity = AppEntity(
                name=app_entity_name,
                description="",
                properties=return_app_entity_properties,
            )
            command = Command(
                name=name,
                description=intent["description"],
                required=required_slot_names,
                properties=properties,
                return_type=return_app_entity.name,
                app_entities=[return_app_entity],
            )
            converted_tools[command.name] = command

    toolbox = ToolBox(commands=converted_tools, app_entities={})
    if output_path is not None:
        toolbox.dump(output_path)
    return toolbox


def save_toolbox(
    toolbox_cfg: DictConfig,
    o_path: Path,
    toolbox: ToolBox,
    conversation_intents: list[dict[str, Any]],
):
    """
    Parameters
    -----------
    toolbox_cfg
        Toolbox configuration
    o_path
        Output path for the schema (should be release directory).
    toolbox
        An enriched schema representation derived from the SGD schema, containing
        additional information such as the slot data type.
    conversation_intents
        A list of all the intents active in the data.
    """

    # in this format, the toolbox contains a list of intents parseable with
    # `pytod.pytod_types.pytod.Intent` class
    if toolbox_cfg.save_refined_toolbox:
        save_json(conversation_intents, o_path / "toolbox_intent.json")

    conversation_intents = {intent["name"] for intent in conversation_intents}
    never_active_intents = {
        tool_name
        for tool_name in toolbox.commands
        if tool_name not in conversation_intents
    }
    if never_active_intents:
        logger.warning(
            f"The intents: {never_active_intents} were never active in conversation. "
            f"They will be removed from the toolbox."
        )
    while never_active_intents:
        next_intent = never_active_intents.pop()
        toolbox.commands.pop(next_intent)
    toolbox.dump(o_path / "toolbox.json")
