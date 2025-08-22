#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
from dataclasses import Field, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel


class Property(BaseModel):
    name: str
    type: str
    is_list: bool = False
    description: Optional[str] = ""
    choices: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.type == "enum":
            assert self.choices is not None and len(self.choices) >= 1


PropertyName = str


class BuiltInTypes(Enum):
    int = "int"
    str = "str"
    bool = "bool"
    float = "float"
    enum = "enum"


class AppEntityCommand(BaseModel):
    description: str
    required: list[str]
    properties: list[Property]


class AppEntity(BaseModel):
    name: str
    description: str
    properties: list[Property] = []
    commands: list[AppEntityCommand] = []


CommandCall = dict[str, Any]


class ExampleInvocation(BaseModel):
    utterance: str
    command_call: CommandCall


class Command(BaseModel):
    name: str
    description: str
    required: list[str]
    properties: list[Property]
    app_entities: list[AppEntity] = []
    app_names: Optional[list[str]] = None
    return_type: Optional[str] = None

    def get_property_dict(self) -> dict[PropertyName, Property]:
        return {property.name: property for property in self.properties}


CommandName = str
AppEntityName = str


@dataclass
class ToolBox:
    commands: dict[CommandName, Command]
    app_entities: dict[AppEntityName, AppEntity]

    def __post_init__(self) -> None:
        app_entity_names: set[str] = set()
        for app_entity in self.app_entities.values():
            if app_entity.name in app_entity_names:
                raise Exception(f"Trying to define {app_entity.name} more than once.")
            else:
                app_entity_names.update({app_entity.name})
        for command in self.commands.values():
            for app_entity in command.app_entities:
                if app_entity.name in app_entity_names:
                    raise Exception(
                        f"Trying to define {app_entity.name} more than once."
                    )
                else:
                    app_entity_names.update({app_entity.name})
            if command.return_type is not None and command.return_type not in [
                type.value for type in BuiltInTypes
            ]:
                if command.return_type not in app_entity_names:
                    raise Exception(
                        f"{command.name} has been declared to return {command.return_type} "
                        f"but {command.return_type} has not been defined."
                    )
            for property in command.properties:
                if property.type not in [type.value for type in BuiltInTypes]:
                    if property.type not in app_entity_names:
                        raise Exception(
                            f"{property.name} in {command.name} has been declared to "
                            f"have type {property.type} but {property.type} has not been defined."
                        )

    @classmethod
    def load(cls, file: Path) -> "ToolBox":
        with open(file, "r") as f:
            toolbox_dict = json.load(f)
        commands = [
            Command.parse_obj(command_dict)
            for command_dict in toolbox_dict.get("commands", [])
        ]
        app_entities = [
            AppEntity.parse_obj(app_entity_dict)
            for app_entity_dict in toolbox_dict.get("app_entities", [])
        ]
        return cls(
            commands={command.name: command for command in commands},
            app_entities={app_entity.name: app_entity for app_entity in app_entities},
        )

    def dump(self, file: Union[str, Path]) -> Path:
        if isinstance(file, str):
            file = Path(file)

        json.dump(
            {
                "commands": [
                    command.dict(exclude_none=True)
                    for command in self.commands.values()
                ]
            },
            file.with_suffix(".json").open("w"),
            indent=2,
        )
        return file

    def __len__(self) -> int:
        return len(self.commands)
