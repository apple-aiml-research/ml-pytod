#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import NamedTuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from pytod.command import ArgumentDefinition, CommandCollection, ServiceCommand
from pytod.inference.utils import map_tool_name_to_service_intent
from pytod.interpreter.metadata import (
    PUBLIC_PROPERTY_MARKER,
    RESOLUTION_ERR,
    SYSTEM_ARG_MARKER,
    WILDCARD_VALUE,
)
from pytod.prompting.pytod_text2text_formatters_utils import VariableMapper
from pytod.pytod_types.aliases import (
    EntityName,
    IntentName,
    ServiceName,
    SlotName,
    ToolName,
    VariableName,
)
from pytod.utils import nested_defaultdict, snake_case

logger = logging.getLogger(__name__)

CompletedTaskStackT = TypeVar("CompletedTaskStackT", bound="CompletedTaskStack")


class EntityInfo(BaseModel):
    """
    Parameters
    ----------
    turn_or_api_call_idx
        An index to the current variable (for selected entities) or
        to the API call turn (for transactions). Currently, `perform`
        is used to mark successful task completion and does not bound
        the entities returned by the transactional API call. As a
        result, the model is instructed to refer to the API call instead.
    transaction
        A flag to help distinguish between entities returned by queries
        and transaction results.
    """

    turn_or_api_call_idx: int
    command_collection: CommandCollection
    task_schema: ServiceCommand
    properties_displayed: list[SlotName]
    transaction: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other: Self) -> bool:
        return (
            self.task_schema.name == other.task_schema.name
            and self.properties_displayed == other.properties_displayed
        )

    def __str__(self) -> str:
        prop_disp = ",".join(self.properties_displayed)
        return f"{self.task_schema.tool_name}-{prop_disp}"


class ServiceEntities(BaseModel):
    entities: list[EntityInfo]
    common_properties: list[SlotName]


class ValueType(Enum):
    WILDCARD = WILDCARD_VALUE
    NON_CATEGORICAL = "non_categorical"
    CATEGORICAL = "categorical"
    RESOLUTION_ERROR = RESOLUTION_ERR
    PUBLIC_VARIABLE = PUBLIC_PROPERTY_MARKER
    SYSTEM_ARGUMENT = SYSTEM_ARG_MARKER


class ObjectReferenceInfo(BaseModel):
    r"""An object containing information about
    entity attributes that may be referenced by the
    agent in follow-up tasks.

    Parameters
    ----------
    reference
        An code snippet in the format ^x\d{1,2}.[arg_name]
        describing the object reference (x2.time)
    current_task_reference
        The variable bound to the current user task.
    value
        The resolved value of the reference.
    value_type
        Indicates whether the value is from an open set (non-categorical),
        closed set (categorical) or wildcard.
    argument_definition
        The schema of the referenced argument.
    entity
        The entity name (camel cased).
    active_intent
        The last intent the user mentioned.
    system_arg
        Indicates the slot may be parsed from the system utterance.
    """

    reference: str
    current_task_reference: str
    value: str
    value_type: ValueType | None
    argument_definition: ArgumentDefinition | None = None
    entity: str
    active_intent: ToolName | None = None
    system_arg: bool

    @property
    def arg_name(self) -> (ServiceName, SlotName):
        service, _ = map_tool_name_to_service_intent(self.active_intent)
        # (Buses_1, city)
        return f"{service}", self.reference.split(".")[1]


class RequestableInfo(BaseModel):
    r"""An object containing entity properties which may
    be requested by the user during the conversation.

    Parameters
    ----------
    properties
        Properties the user may request.
    randomise
        If true, order in which schema elements are displayed
        in the prompt is randomised.
    entity
        Name of the entity about which information is requested.
    entity_var
        The variable to which the entity is bound.
    active_tool
        The active user intent.
    """

    properties: list[ArgumentDefinition]
    randomise: bool = False
    entity: EntityName | None = None
    entity_var: VariableName | None = None
    active_tool: ToolName | None = None


class StackItem(BaseModel):
    """An object containing information about a
    task the user has completed.

    Parameters
    ----------
    command_collection, task_schema
    variables
        The variables bound to the entities returned by
        the task. The list contains more than one item
        if the user completes the task multiple times.
    displayed_documentation
        Arguments for which the definitions are to
        be displayed.
    hidden_documentation
        Tasks implementing the same service often have
        common arguments. To avoid unnecessarily repeating
        the same definition, some definitions are hidden
        for earlier tasks. A reference to the documentation
        of a more recent task where the argument is defined
        is made instead.
    task_referenced
        Hidden documentation references a task where the
        definitions are displayed.
    randomise
        Whether the order in which the documentation is
        displayed is randomised.
    """

    variables: list[VariableName]
    command_collection: CommandCollection
    task_schema: ServiceCommand
    displayed_documentation: list[SlotName]
    hidden_documentation: list[SlotName] | None = None
    task_referenced: str | None = None
    randomise: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def tool_name(self) -> str:
        return self.task_schema.tool_name

    @property
    def task_completion_summary(self) -> str:
        return self.task_schema.task_completion_description

    @property
    def entity_name(self) -> str:
        return snake_case(self.task_schema.entity_name)

    @property
    def requires_property_display(self) -> bool:
        hidden_doc = self.hidden_documentation or []
        all_doc = self.displayed_documentation + hidden_doc
        return bool(all_doc)

    @property
    def entity_variables(self) -> list[VariableName]:
        return self.variables

    def shuffle(self):
        for attr in ["hidden_documentation", "displayed_documentation"]:
            v = getattr(self, attr)
            if v:
                random.shuffle(v)


StackPos = int


@dataclass
class CompletedTaskStack:
    items: list[StackItem] = field(default_factory=list)

    @classmethod
    def build(
        cls: Type[CompletedTaskStackT],
        entities: list[EntityInfo],
        mapper: VariableMapper,
        randomise: bool = False,
    ) -> CompletedTaskStackT:
        items = []
        stack_positions: dict[
            ServiceName, dict[IntentName : list[StackPos]]
        ] = nested_defaultdict(list, depth=2)
        # if one task is completed multiple times, we add
        #  the variable reference to the entity/command in the
        #  prompt as opposed to creating a new stack entry, so
        #  the number of entities may be greater than the number
        #  of stack entries
        position_offset = 0
        for i, entity in enumerate(entities):
            variable = mapper.mapping[f"x{entity.turn_or_api_call_idx}"]
            extend_stack = cls._update_predecessors(
                items, entity, stack_positions, variable
            )
            task_schema = entity.task_schema
            if extend_stack:
                new_item = StackItem.model_validate(
                    {
                        "variables": [variable],
                        "command_collection": entity.command_collection,
                        "task_schema": task_schema,
                        "displayed_documentation": entity.properties_displayed,
                        "randomise": randomise,
                    }
                )
                items.append(new_item)
            else:
                logger.info(
                    f"Task {task_schema.tool_name} was completed multiple times, "
                    "only one entry will be displayed on the stack."
                )
                position_offset += 1
                continue
            stack_positions[task_schema.service][task_schema.name].append(
                i - position_offset
            )
        return cls(items)

    @staticmethod
    def _update_predecessors(
        items: list[StackItem],
        entity: EntityInfo,
        stack_positions: dict[ServiceName, dict[IntentName, list[StackPos]]],
        entity_variable: VariableName,
    ) -> bool:
        """
        Modify completed task representations to optimise prompt length and complexity.

            1. If the same task is completed multiple times, then we just enumerate the
            variables bound to the entity in a single prompt entry as opposed to having
            to identical stack items

            2. When tasks have common arguments, their definition is displayed only
            once, for the latest task and referenced in earlier tasks docs.

        Parameters
        ----------
        entity_variable
            The variable bound to the current entity


        Returns
        -------
        A boolean indicating whether the entity should be appended to the
        task completion stack.

        """
        # no task from the same service was completed, we just extend the
        # completed tasks stack
        if not items or entity.task_schema.service not in stack_positions:
            return True
        curr_task = entity.task_schema.name
        related_items = stack_positions[entity.task_schema.service]
        # if the user completes the same task multiple times, we display
        # one entry which refers to both variables bound to the task
        if curr_task in related_items:
            prev_task = items[related_items[curr_task][-1]]
            assert prev_task.task_schema.name == curr_task
            prev_task.variables.append(entity_variable)
            return False
        # if tasks previously completed have arg definitions common with the most
        # recent task, we hide them in the documentation of those tasks and
        # display them just for the most recent task
        for _, stack_pos in related_items.items():
            stack_item = items[stack_pos[-1]]
            displayed_docs = [
                arg
                for arg in stack_item.displayed_documentation
                if arg not in entity.properties_displayed
            ]
            hidden_docs = [
                arg
                for arg in stack_item.displayed_documentation
                if arg in entity.properties_displayed
            ]
            stack_item.displayed_documentation = displayed_docs
            stack_item.hidden_documentation = hidden_docs
            stack_item.task_referenced = str(len(items) + 1)

        return True

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def triple_quote() -> str:
    return '"""'


class DisplayedProperties(NamedTuple):
    optional: list[SlotName]
    permanent: list[SlotName]


def naturalise(date_str: str) -> tuple[str, str] | None:
    """Naturalise the canonical form of a value. Currently,
    only dates in the form YYYY-MM-DD are naturalised to
    an expression such as March 5th or 5th of March."""

    def get_ordinal_suffix(day: int) -> str:
        """Return the ordinal suffix for a given day."""
        if 11 <= day <= 13:
            return "th"
        else:
            return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    pattern = r"^\d{4}-(\d{2})-(\d{2})$"
    match = re.match(pattern, date_str)
    if match:
        # Extract the month and day
        month = int(match.group(1))
        day = int(match.group(2))

        # Determine the ordinal suffix for the day
        ordinal_suffix = get_ordinal_suffix(day)

        # Create a datetime object to get the month name
        date_obj = datetime(
            1900, month, day
        )  # Year is set arbitrarily as it is not important

        # Format the date as "Month day<ordinal_suffix>" or "day<ordinal_suffix> of Month"
        formatted_date1 = date_obj.strftime(f"%B {day}{ordinal_suffix}")
        formatted_date2 = date_obj.strftime(f"{day}{ordinal_suffix} of %B")

        return formatted_date1, formatted_date2
    return None
