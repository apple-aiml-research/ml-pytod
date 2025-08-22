#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import itertools
import logging
from enum import Enum
from typing import Mapping, MutableMapping, Optional, Type, TypeVar

from pytod.command import ArgumentDefinition, ServiceCommand, is_arg
from pytod.evaluation.helpers import ValueProcessor
from pytod.execution.policy_utils import ActionResult, HintGenerator, RecommendedAction
from pytod.pytod_types.aliases import (
    CanonicalValue,
    DefaultValue,
    DialogueID,
    EnumValues,
    IntentName,
    ServiceName,
    SlotName,
    StateDict,
)
from pytod.simulation._command_utils import _CommandSetattrControl
from pytod.simulation.api_driver import APIDriver
from pytod.simulation.database import MongoDBCollection
from pytod.simulation.database_utils import NormalisedCallParameters, Normalizer
from pytod.utils import cast_vals_to_string, listify_values, snake_case

logger = logging.getLogger(__name__)

WILDCARD_VALUE = "dontcare"

CommandT = TypeVar("CommandT", bound="Command")
SlotValue = TypeVar("SlotValue", bound=str)


class SlotState(Enum):
    SET = "set"
    NOTSET = "notset"


class State:
    """Descriptor base class for setting and accessing a Command's state."""

    def __get__(self, instance: CommandT, owner: Type[CommandT]) -> StateDict:
        """Returns the dialogue state in a format compatible
        with the SGD evaluation script."""

        # state always contains the slots actually mentioned in conversation
        arg_mentions: dict[SlotName, str] = cast_vals_to_string(
            instance.get_mentioned_args()
        )
        # restore case of categorical slots
        self._maybe_restore_case(arg_mentions, instance)
        state: StateDict = {
            "slot_values": listify_values(arg_mentions),
            "active_intent": instance.name,
            # requested slots are usually entity properties, so we update the
            # dialogue state directly during execution
            # (see pytod_agent_state::update_user_requests)
            "requested_slots": [],
        }
        self._append_canonical_form(state["slot_values"], instance)
        return state

    def _maybe_restore_case(
        self, slot_values: dict[SlotName, str | list[str]], instance: CommandT
    ):
        """Ensure the case of the categorical variables in the state is correct."""
        command_schema = instance.schema

        for slot, value in slot_values.items():
            arg_def = command_schema.get_argument_definition(slot)
            # can be None as some slots in the command are result properties (eg hotel_name)
            if arg_def is not None and arg_def.is_categorical:
                # note we only recase the first value because this is the only one
                # that matters for evaluation - subsequent values are not checked
                new_value = ValueProcessor.restore_case(
                    value[0] if isinstance(value, list) else value,
                    command_schema.service,
                )
                slot_values[slot] = new_value if isinstance(value, str) else [new_value]

    def _append_canonical_form(
        self, slot_values: dict[SlotName, list[str]], instance: CommandT
    ):
        """Add canonical values of a slot value to the state to ensure evaluation does not
        unfairly penalise value references."""
        for slot, value_list in slot_values.items():
            extended_values = []
            for value in value_list:
                extended_values.append(value)
                norm_result = instance._normalizer.normalise(
                    instance.service, {slot: str(value)}
                )
                if slot in norm_result.result:
                    n_value = norm_result.result[slot]
                    if n_value not in extended_values:
                        extended_values.append(n_value)
            slot_values[slot] = extended_values

    def __set__(self, instance: CommandT, state: Mapping[SlotName, SlotValue]):
        """Set the command arguments to specified values."""
        for slot, value in state.items():
            setattr(instance, slot, value)


class Command(metaclass=_CommandSetattrControl):
    _instance_attributes = set()
    state = State()

    def __init__(self, dialogue_id: DialogueID):
        self._dialogue_id: DialogueID = dialogue_id
        self._intent: IntentName = self.__class__.__name__
        self._service: ServiceName = ""
        # command positional params (required)
        self._positional_args: tuple[SlotName] = ()
        # command keyword params (optional)
        self._keyword_args: dict[SlotName, DefaultValue] = {}
        # mapping for cmd arguments that can be enumerated (categorical slots)
        self._enum_args: dict[SlotName, EnumValues] = {}
        # name of the entity type returned by the command
        self._entity_name = ""
        # list of keys available of each DB record
        self._entity_attributes: tuple[SlotName] = ()
        self._system_tracked_slots: tuple[SlotName] = ()
        # track whether optional slots were said by the user
        self._keywords_state: dict[SlotName, SlotState] = {}
        # properties set at runtime that are not instance attributes
        self._unknown_properties_set: dict[str, str] = {}
        # properties accessed at runtime that are not instance attributes
        self._unknown_properties_accessed: list[str] = []
        # at least one call to the DB or transaction API call was made
        self._executed: bool = False
        self._hint_generator: Optional[HintGenerator] = None
        # maps slot values extracted from text to canonical forms
        # for the purposes of calling the DB _only_
        self._normalizer = Normalizer()
        # a task that the agent may suggest the to user if the current
        # command is executed and the user has not ended the conversation
        self._followup_command: Optional[IntentName] = None
        # schema of the command
        self.schema: Optional[ServiceCommand] = None
        # schema of other commands implemented by the same service
        self.service_schemata: list[ServiceCommand] = []
        # include arguments mentioned by the user that are part of the
        # service schema but are not part positional/keyword args for
        # the command in mentioned arguments
        self.include_service_args = True

    def perform(self, command_context: MongoDBCollection | APIDriver) -> ActionResult:
        """Perform this command."""
        raise NotImplementedError

    def recommend_action(self) -> Optional[list[RecommendedAction]]:
        """Endpoint for action recommendations from the command."""
        missing_arguments = self.get_missing_positional_args()
        if missing_arguments:
            hints = [
                self._hint_generator.get_slot_filling_hint(slot)
                for slot in missing_arguments
            ]
            return [RecommendedAction(dialog=hint) for hint in hints]
        return

    @property
    def service(self) -> str:
        """The name of the service implementing this command."""
        return self._service

    @property
    def name(self) -> str:
        """The name of the command, excluding service information."""
        return self._intent

    @property
    def uuid(self) -> str:
        """Unique identifier of a command instance."""
        return self._dialogue_id

    @property
    def followup_command(self) -> Optional[IntentName]:
        """A property that the PyTOD agent can query to find out
        any suggested tasks that should be recommended to the user
        after the command is executed."""
        return self._followup_command

    @property
    def entity_name(self) -> str:
        """Return the name of the object that is retrieved
        by the command from an external database."""
        if self._entity_name:
            return self._entity_name
        raise NotImplementedError("Automatic entity naming is not implemented")

    @property
    def positional_args(self) -> tuple[SlotName]:
        """The command positional arguments, which must always be
        specified by the user."""
        return self._positional_args

    @property
    def keyword_args(self) -> dict[SlotName, DefaultValue]:
        """The command keyword arguments, which are sometimes
        explicitly set and other times selected or confirmed by
        the user."""
        return self._keyword_args

    @property
    def keyword_args_state(self) -> dict[SlotName, SlotState]:
        return self._keywords_state

    @property
    def system_tracked_slots(self) -> tuple[SlotName]:
        """System-side slots that are tracked before
        (for confirmed commands) or after (for search commands)
        the call to the database.

        Notes
        -----
        1. Some are entity properties, other are default optional
        slots that the system requests the user to confirm
        and are tracked upon confirmation.
        2. Confirmation patterns are command-dependent: not all
        optional slot defaults are confirmed.
        """
        return self._system_tracked_slots

    @property
    def entity_attributes(self) -> tuple[SlotName]:
        """The properties of the entity this command operates with."""
        return self._entity_attributes

    @property
    def executed(self) -> bool:
        """If `True`, at least one call to the underlying database
        or transaction was attempted."""
        return self._executed

    @property
    def hint_generator(self):
        """Provide runtime access to hint generation."""
        return self._hint_generator

    def get_full_command_name(self, snake_cased: bool = False) -> str:
        """Returns the full command name, including a service identifier."""
        if not snake_cased:
            return f"{self._service}.{self._intent}"
        return snake_case(f"{self._service}{self._intent}")

    def get_missing_positional_args(self) -> list[SlotName]:
        """Returns a list of positional arguments not yet specified."""
        missing_positional_args = []
        for arg in self.positional_args:
            arg_value = getattr(self, arg)
            if arg_value is None:
                missing_positional_args.append(arg)
        return missing_positional_args

    def get_mentioned_args(self) -> dict[SlotName, SlotValue]:
        """Returns a mapping containing the command arguments specified
        by the user so far."""
        mentioned_args = self.get_self_args()
        # include other properties set for other slots in the schema that
        # are not properties of the current command. In this way,
        # state tracking metrics are less sensitive to noisy intent utterances
        if self.include_service_args:
            for arg, value in self._unknown_properties_set.items():
                if any(is_arg(arg, s) for s in self.service_schemata):
                    mentioned_args[arg] = value
        return mentioned_args

    def get_self_args(self):
        """Returns the positional and keyword arguments that have been
        set for the current command."""
        mentioned_args = {}
        keys = (self.positional_args, self.keyword_args)
        for key in keys:
            for arg in key:
                value = getattr(self, arg)
                if value is not None:
                    mentioned_args[arg] = value
        return mentioned_args

    def get_default_parameters(self) -> dict[SlotName, DefaultValue]:
        """Returns a mapping from keyword argument name to default value
        for keyword arguments not mentioned by the user."""

        defaults = {}
        for arg, default_value in self.keyword_args.items():
            if (
                self._keywords_state[arg] == SlotState.NOTSET
                and default_value != WILDCARD_VALUE
            ):
                defaults[arg] = default_value
        return defaults

    def get_api_call_parameters(self) -> dict[SlotName, SlotValue | DefaultValue]:
        """Returns a mapping with parameters for calling the API."""
        # nb: these arguments include other relevant argument from the service
        #  schema
        parameters = self.get_mentioned_args()
        parameters.update(self.get_default_parameters())
        return parameters

    @staticmethod
    def _get_wildcards(
        parameters: MutableMapping[SlotValue, CanonicalValue]
    ) -> set[SlotName]:
        return {p for p in parameters if parameters[p] == WILDCARD_VALUE}

    def resolve_wildcard_values(
        self, parameters: MutableMapping[SlotValue, CanonicalValue]
    ) -> Optional[list[dict[SlotName, SlotValue]]]:
        """Resolve wildcard values to possible values for enum-type arguments."""

        def create_param_subsets(
            enum_vals: dict[SlotName, EnumValues]
        ) -> list[dict[SlotName, SlotValue]]:
            """Create all possible combinations of enum parameters."""
            subsets = []
            for combo in itertools.product(*enum_vals.values()):
                param_subset = {
                    param: val for param, val in zip(enum_vals.keys(), combo)
                }
                subsets.append(param_subset)
            return subsets

        wildcards = self._get_wildcards(parameters)
        enum_vals: dict[SlotName, EnumValues] = {}
        for p in wildcards:
            parameters.pop(p)
            if p in self._enum_args:
                enum_vals[p] = self._enum_args[p]
        return create_param_subsets(enum_vals) or None

    def log_normalisation(self, normalised_params: NormalisedCallParameters):
        """Logs predictions which could not be normalised."""
        if normalised_params.failed_normalisation is not None:
            for slot, pred_value in normalised_params.failed_normalisation.items():
                cmd_name = self.get_full_command_name(snake_cased=False)
                logger.warning(
                    f"{self._dialogue_id} || {cmd_name} || Value: {pred_value}. "
                    f"Could not normalise: {slot}."
                )

    def is_argument(self, maybe_arg: str) -> bool:
        """Check if `maybe_arg` is specified as an argument of the current command."""
        return maybe_arg in self._keyword_args or maybe_arg in self._positional_args

    def is_system_tracked(self, maybe_tracked: str) -> bool:
        """Check if `maybe_tracked` is a slot tracked by the agent in the current command."""
        return maybe_tracked in self._system_tracked_slots

    def __setattr__(self, maybe_command_arg: str, value: str):
        """Hook to detect attempts to set properties that are not instance
        members or entity attributes."""
        if maybe_command_arg in self._instance_attributes:
            # defer to the normal attribute setting machinery if we are trying to set
            # the value of an instance attribute
            super().__setattr__(maybe_command_arg, value)
        else:
            norm = self._normalizer.normalise(
                self.service, {maybe_command_arg: value}
            ).result
            value = norm[maybe_command_arg]
            cmd_name = self.get_full_command_name(snake_cased=False)
            logger.warning(
                f"{self._dialogue_id} || {cmd_name} || "
                f"Attempting to set undefined attribute: {maybe_command_arg}."
            )
            self.__dict__["_unknown_properties_set"][maybe_command_arg] = value

    def __getattr__(self, item: str) -> Optional[SlotValue]:
        """Handle access to undefined attributes."""
        dial_id = self._dialogue_id
        cmd = self.get_full_command_name(snake_cased=False)
        logger.warning(
            f"{dial_id} || {cmd} || Attempted reading unknown attribute: {item}."
        )
        self._unknown_properties_accessed.append(item)
        return

    @classmethod
    def build(
        cls: Type[CommandT],
        dialogue_id: DialogueID,
        command_schema: Optional[ServiceCommand] = None,
        **kwargs,
    ):
        """Build a Command given its API schema."""
        raise NotImplementedError


def _init_args(
    command: CommandT,
    service_command: ServiceCommand,
    service_commands: list[ServiceCommand] | None,
):
    """Initialise Command attributes relating to arguments."""

    def get_enum_values_map(
        service_command: Optional[ServiceCommand],
    ) -> Optional[dict[SlotName, EnumValues]]:
        enums = {}
        keys = ("required_slots", "optional_slots")
        for key in keys:
            this_type_args: list[ArgumentDefinition] = getattr(service_command, key)
            for arg_def in this_type_args:
                if enum_vals := arg_def.possible_values:
                    enums[arg_def.name] = enum_vals
        return enums or None

    command.schema = service_command
    command.service_schemata = service_commands or []
    command._positional_args = tuple(arg.name for arg in service_command.required_slots)
    command._keyword_args = {
        arg.name: arg.default_value for arg in service_command.optional_slots
    }
    enum_args = get_enum_values_map(service_command)
    if enum_args is not None:
        command._enum_args = enum_args
    for kw, value in command.keyword_args.items():
        command._keywords_state[kw] = SlotState.NOTSET
