#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Union, cast

from pydantic import BaseModel
from typing_extensions import Self

from pytod.pytod_types.aliases import (
    ArgumentValue,
    IntentName,
    ServiceName,
    SlotName,
    SlotValue,
)
from pytod.sgd_metadata import requested_slots
from pytod.utils import default_to_regular, load_json, nested_defaultdict, snake_case

SlotDataType = Literal["int", "str", "bool", "float", "enum"]


class ServiceCall(BaseModel):
    """Representation of an SGD service call."""

    method: str
    parameters: dict[str, ArgumentValue]
    service: ServiceName


class ArgumentDefinition(BaseModel, frozen=True):
    """Definition of an SGD command argument.

    Parameters
    ----------
    name
        The slot name as defined in the SGD schema.
    description
        The slot description
    is_categorical
        Whether the slot takes values from an open (`False`)
        or closed (`True`) set
    possible_values
        List of possible values the slot can take (non-empty
        if `is_categorical = True`
    default_value
        Only defined if the argument is a keyword.
    data_type
        The data type of the slot
    accepts_wildcard
        Whether the slot can take the `dontcare` value or not.
    """

    name: str
    description: str
    is_categorical: bool
    possible_values: tuple[str, ...]
    default_value: str | None = None
    data_type: SlotDataType | None = None
    accepts_wildcard: bool | None = None

    def __eq__(self, other: Self) -> bool:
        # nb: the default value can be different across intents (see RentalCars_3)
        return (
            self.name == other.name
            and self.description == other.description
            and self.is_categorical == other.is_categorical
        )

    def __hash__(self):
        return hash(f"{self.name}{self.description}{self.is_categorical}")


class FollowupArguments(BaseModel):
    """Arguments that are shared with other
    APIs.

    Parameters
    ----------
    required, optional
        Required/optional arguments shared with
        APIs from the same service.
    public_members
        Entity properties made visible to
        all other services.
    """

    required: list[SlotName] | None = None
    optional: list[SlotName] | None = None
    public_members: list[SlotName] | None = None

    def get_args(self) -> list[SlotName]:
        args = []
        for val in self.__dict__.values():
            if val is not None:
                args.extend(val)
        return args


class ServiceCommand(BaseModel):
    """Definition of an SGD command."""

    name: IntentName  # eg FindRestaurant
    service: ServiceName  # Restaurants_1
    # lower-cased `service` followed by "_" and snake-cased `name`
    tool_name: str | None = None
    # a command implemented by the same service
    # the PyTOD agent might suggest after
    # execution of the current command
    followup_command: IntentName | None = None
    # metadata encoding arguments shared with followup-commands
    # and object properties exposed to other services
    followup_metadata: FollowupArguments | None = None
    # indicates whether the user should provide confirmation
    # before command execution
    is_transactional: bool
    # the name of the conversational entity
    # this command returns
    entity_name: str | None = None
    # populated if `is_transactional = False`
    # a key used to establish entity uniqueness
    entity_cmp_key: list[str] | None = None
    # populated if `is_transactional = False`
    # a key used to sort the database results.
    result_sort_key: str | None = None
    # natural language description of the command functionality
    description: str
    # natural language description of the functionality of the service
    # implementing the command
    service_description: str
    # very short summary of what the user did by invoking this command
    task_completion_description: str
    required_slots: tuple[ArgumentDefinition, ...]
    optional_slots: tuple[ArgumentDefinition, ...]
    # populated if `is_transactional = False`. Slots that are
    # automatically tracked when the user selects a search result
    system_tracked_slots: tuple[SlotName, ...] | None = None
    # populated if `is_transactional = True`. Slots for which
    # confirmation is requested before command execution
    system_confirmed_slots: list[SlotName] | None = None
    # list of properties exposed by this API to other services
    shared_properties: list[SlotName] | None = None
    # populated if `is_transactional = True`. Alternative slot
    # values offered by the agent in the event of a failed
    # transaction execution
    alternative_transaction: tuple[SlotName, ...] | None = None
    # lists what slots are included in the command output
    result_slots: tuple[str, ...]
    # a subset of result_slots, excluding command arg
    # that cannot be requested
    api_returns: tuple[ArgumentDefinition, ...] | None = None

    def get_argument_definition(self, argument: str) -> ArgumentDefinition | None:
        """Return the schema of a command argument."""
        all_defs = list(self.required_slots) + list(self.optional_slots)
        for arg_def in all_defs:
            if arg_def.name == argument:
                return arg_def
        for arg_def in self.api_returns or []:
            if arg_def.name == argument:
                return arg_def
        return


class CommandCollection:
    """Converts the SGD schema into a collection of ServiceCommand objects the agent can issue.

    The collection is hierarchical, with commands organised by the service they implement.
    This is necessary as the same command names are used across different services.


    Args
    ----
    schema_path (Path | str): path to the SGD schema JSON file defining supported services
        and commands.
    """

    def __init__(self, schema_path: Path | str):
        assistant_schema = load_json(schema_path)
        # descriptions for all command arguments in the schema
        arg_defs: dict[
            ServiceName, dict[SlotName, dict[str, Any]]
        ] = self._build_arg_def_lookup(assistant_schema)
        commands = {service["service_name"]: {} for service in assistant_schema}
        for service_schema in assistant_schema:
            service_name = service_schema["service_name"]
            for command_def in service_schema["intents"]:
                command_name = command_def["name"]
                service_command = self._parse_command(
                    command_def,
                    service_name,
                    service_schema["description"],
                    arg_defs[service_name],
                )
                commands[service_name][command_name] = service_command
        self._commands: dict[ServiceName, dict[IntentName, ServiceCommand]] = commands
        # schema of all argument definitions, including entity properties
        self._all_arg_defs = {service: {} for service in arg_defs}
        for service, service_args in arg_defs.items():
            for arg, arg_schema in service_args.items():
                self._all_arg_defs[service][arg] = ArgumentDefinition.model_validate(
                    arg_schema
                )
        if isinstance(schema_path, str):
            schema_path = Path(schema_path)
        self._split = cast(Literal["train", "dev", "test"], schema_path.parent.name)
        assert self._split in ["train", "dev", "test"]

    @property
    def split(self) -> Literal["train", "dev", "test"]:
        return self._split

    @staticmethod
    def _build_arg_def_lookup(
        schema: dict,
    ) -> dict[ServiceName, dict[SlotName, dict[str, Any]]]:
        args_def = nested_defaultdict(dict, depth=2)
        for service in schema:
            service_name = service["service_name"]
            for arg_def in service["slots"]:
                arg_name = arg_def["name"]
                args_def[service_name][arg_name] = arg_def
        return default_to_regular(args_def)

    @staticmethod
    def _parse_command(
        sgd_intent_schema: dict[str, Any],
        service: ServiceName,
        service_description: str,
        argument_defs: dict[str, dict[str, Any]],
    ) -> ServiceCommand:
        """Converts the raw SGD intent schema into a ServiceCommand."""
        required_slots = sgd_intent_schema.pop("required_slots")
        req_arg_defs = [
            ArgumentDefinition.model_validate(argument_defs[slot])
            for slot in required_slots
        ]

        opt_arg_defs = []
        optional_slots = sgd_intent_schema.pop("optional_slots")
        for slot_name, default_val in optional_slots.items():
            arg_def = deepcopy(argument_defs[slot_name])
            arg_def["default_value"] = default_val
            opt_arg_defs.append(arg_def)

        try:
            api_returns = sgd_intent_schema.pop("api_returns")
            api_return_defs = [
                ArgumentDefinition.model_validate(argument_defs[slot])
                for slot in api_returns
            ]
        except KeyError:
            api_return_defs = []
        tool_name = f"{snake_case(service)}_{snake_case(sgd_intent_schema['name'])}"
        sgd_intent_schema.update(
            required_slots=req_arg_defs,
            optional_slots=opt_arg_defs,
            api_returns=api_return_defs if api_return_defs else None,
            service=service,
            service_description=service_description,
            tool_name=tool_name,
        )

        return ServiceCommand.model_validate(sgd_intent_schema)

    def get(
        self, service: ServiceName, command_name: IntentName
    ) -> ServiceCommand | None:
        """Get the definition of `command_name` from service `service`."""
        return self._commands.get(service).get(command_name)

    def get_return_types(
        self, service: ServiceName, command_name: IntentName, return_objs: bool = True
    ) -> list[Union[str, ArgumentDefinition]] | None:
        """Get the return types of `command_name` from `service`.

        Parameters
        ----------
        service, command_name
        return_objs
            Set to `False` to return arguments returned as string literals.
        """
        command = self._commands.get(service).get(command_name)
        if return_objs:
            return command.api_returns
        return [arg.name for arg in command.api_returns]

    def get_service_commands(self, service: ServiceName) -> list[ServiceCommand]:
        """Get all the commands implementing a service."""
        cmds = self._commands.get(service)
        return list(cmds.values())

    def get_service_arg_definitions(
        self, service: ServiceName
    ) -> list[ArgumentDefinition]:
        """Get all schemata for all required and optional arguments of `service`."""
        args = []
        for command in self.get_service_commands(service):
            for arg_def in command.required_slots + command.optional_slots:
                if arg_def not in args:
                    args.append(arg_def)
        return args

    def get_services(self) -> list[ServiceName]:
        """Return the services that this command collection implements."""
        return list(self._commands.keys())

    def get_arg_services(self, arg: SlotName) -> list[ServiceName] | None:
        """Return the list of services `arg` is a member of."""
        services = []
        for service in self.services:
            if self.in_service_schema(service, arg):
                services.append(service)
        return services or None

    def get_arg_schema(
        self, service: ServiceName, arg: SlotName
    ) -> ArgumentDefinition | None:
        """Return the schema of `arg` from service `service`."""
        try:
            return self._all_arg_defs[service][arg]
        except KeyError:
            assert service in self._all_arg_defs
            return

    def get_requested_arg_schemas(
        self, service: ServiceName
    ) -> list[ArgumentDefinition] | None:
        """Return the schema of arguments which the user can request."""
        requested_arg_names = requested_slots.info[service].requestable
        return [self._all_arg_defs[service][arg] for arg in requested_arg_names] or None

    def in_service_schema(self, service: ServiceName, arg: SlotName) -> bool:
        """Returns `True` if `arg` is a member of the service schema of `service`."""
        return arg in self._all_arg_defs[service]

    @property
    def services(self) -> set[ServiceName]:
        return set(self._commands.keys())


class ServiceCallResult(BaseModel):
    # the actual keys are defined in ServiceCommand result_slots field
    service_results: list[dict[SlotName, SlotValue]]
    # set to `True` if the call returned no result
    is_error: bool
    # set for calls which return no results.
    error_message: str | None = None


def get_all_args(cmd: ServiceCommand) -> set[SlotName]:
    """Return the union of the `cmd` required and optional args."""
    arg_names = set()
    arg_types = ("required", "optional")
    for arg_type in arg_types:
        this_type_args = getattr(cmd, f"{arg_type}_slots")
        for arg in this_type_args:
            arg_names.add(arg.name)
    return arg_names


def get_required_arg_names(cmd: ServiceCommand) -> set[str]:
    """Return the names of the required args."""
    return {arg.name for arg in cmd.required_slots}


def get_shared_args(c1: ServiceCommand, c2: ServiceCommand) -> set[str]:
    """Return the set of shared args between commands `c1` and `c2`."""
    return get_all_args(c1).intersection(get_all_args(c2))


def is_arg(slot: str, command: ServiceCommand) -> bool:
    arg_types = ("required", "optional")
    for type_ in arg_types:
        this_type_args = getattr(command, f"{type_}_slots")
        assert isinstance(this_type_args, tuple)
        for arg_def in this_type_args:
            if slot == arg_def.name:
                return True
    return False


def is_api_return_property(slot: str, command: ServiceCommand) -> bool:
    return slot in getattr(command, "result_slots")


def get_bool_and_enum_arg_schemas(
    schema: CommandCollection, service: ServiceName
) -> list[ArgumentDefinition]:
    args = [
        arg
        for arg in schema.get_service_arg_definitions(service)
        if arg.data_type in {"bool", "enum"}
    ]
    return args


def get_arg_type(argument: SlotName, schema: ServiceCommand) -> SlotDataType:
    """Return the `argument`'s data type."""
    arg_def = schema.get_argument_definition(argument)
    return arg_def.data_type


def get_arg_description(argument: SlotName, schema: ServiceCommand) -> str:
    """Return the `argument`'s description."""
    arg_def = schema.get_argument_definition(argument)
    return arg_def.description.lower()
