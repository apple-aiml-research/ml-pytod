#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import random
from collections import defaultdict
from copy import deepcopy
from functools import singledispatchmethod
from operator import attrgetter
from typing import Literal, Optional, Type

from omegaconf import DictConfig

from pytod.command import CommandCollection, ServiceCommand, is_arg
from pytod.interpreter.interpreter import TemplatesCollection
from pytod.interpreter.ontology_v3 import NLG_TAGS
from pytod.iterators import IndexMetadata
from pytod.pytod_types.aliases import IntentName, ServiceIntent, ServiceName
from pytod.pytod_types.sgd_conversation import APISlotCarryOverInfo
from pytod.pytod_types.transcript import (
    AttributeAccessTemplate,
    BackendHint,
    BackendHintTemplate,
    BackendNotification,
    BackendNotificationTemplate,
    NaturalLanguageTypes,
    ProgramStatement,
    ProgramStatementTemplate,
    PyTODInstruction,
    PyTODInstructionTemplate,
    Response,
    TemplateField,
    UserQuery,
)
from pytod.toolbox.toolbox import get_command_name
from pytod.transcript import APIInfo
from pytod.utils import dispatch_on_value, split_camel_case

logger = logging.getLogger(__name__)

_NO_ACTIVE_INTENT = "NONE"


class AssignmentError(Exception):
    pass


class DisambiguationError(Exception):
    pass


NO_ASSIGNMENT_TAGS = {"end_conversation"}
DISAMBIGUATION_MAP = {
    "Events_2.BuyEventTickets": {
        "time": "Events_2.GetEventDates",
        "venue_address": "Events_2.GetEventDates",
    },
    "Flights_1.ReserveRoundtripFlights": {
        "refundable": "Flights_1.SearchRoundtripFlights",
        "inbound_arrival_time": "Flights_1.SearchRoundtripFlights",
        "origin_airport": "Flights_1.SearchRoundtripFlights",
        "destination_airport": "Flights_1.SearchRoundtripFlights",
        "outbound_arrival_time": "Flights_1.SearchRoundtripFlights",
    },
}


def has_origin(instruction: PyTODInstructionTemplate | PyTODInstruction) -> bool:
    return (
        hasattr(instruction, "origin_fields") and instruction.origin_fields is not None
    )


def needs_rendering(instruction: PyTODInstruction) -> bool:
    return hasattr(instruction, "fields") or hasattr(instruction, "origin_fields")


def check_for_syntax_errors(expression: str):
    try:
        _ = ast.parse(expression)
    except SyntaxError:
        logger.error(f"Cannot parse: {expression}")
        raise SyntaxError


def assert_on_instruction_type(instruction: PyTODInstruction):
    assert any(
        (
            isinstance(instruction, ProgramStatement),
            isinstance(instruction, BackendNotification),
            isinstance(instruction, BackendHint),
        )
    )


def get_index(var_name: str) -> int:
    return int(var_name.replace("x", ""))


def format_tool_name(formatter: Type[get_command_name], api_info: APIInfo):
    return formatter(api_info.service, api_info.function)


class Assignments:
    def __init__(
        self,
        interpreter_cfg: DictConfig,
        dialog_id: str,
        metadata: IndexMetadata,
        tool_name_formatter: Type[get_command_name],
        start_index: int = 0,
        command_collection: Optional[CommandCollection] = None,
    ):
        # containers holding variable indices used to turn program instruction
        # templates into PyTOD program instructions
        self._call: defaultdict[ServiceIntent, list[int]] = defaultdict(list)
        # the index of the last call of each service
        self._last_call: dict[ServiceName, int] = {}
        self._entity: defaultdict[ServiceIntent, list[int]] = defaultdict(list)
        self._last_entity: Optional[defaultdict[ServiceIntent, list[int]]] = None
        self._hint: defaultdict[ServiceIntent, list[int]] = defaultdict(list)
        self._results_list: defaultdict[ServiceIntent, list[int]] = defaultdict(list)
        self._transaction: defaultdict[ServiceName, list[int]] = defaultdict(list)
        self._suggestions: defaultdict[ServiceName, list[int]] = defaultdict(list)
        self._suspended_tasks: defaultdict[ServiceName, list[int]] = defaultdict(list)

        self._current_index: int = start_index
        if interpreter_cfg.start_index == "random":
            self._current_index = random.randint(1, 10)
        else:
            self._current_index = interpreter_cfg.start_index
        self._grammar = interpreter_cfg.grammar
        self._interpreter_cfg: DictConfig = interpreter_cfg.backend
        self._dialog_id: str = dialog_id
        self._current_task: Optional[str] = None
        self._previous_task: Optional[str] = None
        self._current_task_transactional: bool = False
        self._metadata: IndexMetadata = metadata
        self._task_history: defaultdict[ServiceName, list[str]] = defaultdict(list)
        self._search_task_history: defaultdict[ServiceName, list[str]] = defaultdict(
            list
        )
        self._current_service: Optional[ServiceName] = None
        self._current_intent: Optional[IntentName] = None
        self._tool_name_formatter = tool_name_formatter
        # if this is true, the variable assigned to the `select()` instruction
        # is remembered as the last entity of the search intent, even if the
        # selection happened as the intent changed
        self._copy_selected_entity_to_search = (
            interpreter_cfg.copy_selected_entities_to_search
        )
        self._command_collection = command_collection

    @property
    def current_index(self) -> int:
        """Tracks the number of PyTOD instructions so far."""
        return self._current_index

    @current_index.setter
    def current_index(self, value):
        self._current_index = value

    @property
    def last_call(self) -> int:
        """The index of the last call of the active service."""
        service, intent = self._current_task.split(".")
        return self._last_call[service]

    @property
    def current_call(self) -> int:
        """Tracks variables bound to calls of the current tasks.

        Note
        ----
        Multiple calls to the same task appear:

            - in adjacent dialogue spans when transactions/queries fail

            - non-adjacent spans in Banks_* and Payment_* services
        """
        try:
            return self._call[self._current_task][-1]
        except IndexError:
            raise AssignmentError(
                f"No call is yet registered for task {self._current_task}"
            )

    @current_call.setter
    def current_call(self, index: int):
        for task, indices in self._call.items():
            service, intent = task.split(".")
            self._last_call[service] = max(indices)
        self._call[self._current_task].append(index)

    @property
    def current_entity(self) -> int:
        """Tracks the variables bound to an entity. This is either the output of a
        `next` or `select` call."""
        try:
            return self._entity[self._current_task][-1]
        except IndexError:
            raise AssignmentError("There are no entity assignments")

    @property
    def last_entity(self) -> int:
        """This property is automatically set when we increment the entity index.
        It is used to correctly implement entity selection (eg. select(x5))."""
        try:
            return self._last_entity[self._current_task][-1]
        except IndexError:
            raise AssignmentError("There are no entity assignments")

    @current_entity.setter
    def current_entity(self, index: int):
        if self._current_task in self._entity:
            if self._last_entity is None:
                self._last_entity = defaultdict(list)
            if self._entity[self._current_task]:
                self._last_entity[self._current_task].append(
                    self._entity[self._current_task][-1]
                )
        self._entity[self._current_task].append(index)

    @property
    def current_hint(self) -> int:
        """Tracks the variables bound to hints in the current task."""
        try:
            return self._hint[self._current_task][-1]
        except IndexError:
            raise AssignmentError(
                f"Dialogue: {self._dialog_id} (Task {self._current_task}). "
                f"Attempted to access hint index but no hint was found"
            )

    @current_hint.setter
    def current_hint(self, index: int):
        self._hint[self._current_task].append(index)

    @property
    def current_transaction_index(self) -> int:
        """Tracks variables bound to calls to transactional intent calls.

        Notes
        -----
        Transaction indices are referenced inside `perform` statements. In
        SGD, the entities returned by these transactions can be queried for
        properties (eg user: What is the ticket cost?) or provide slot values
        for related tasks. These reference the variables corresponding to the
        entity the user performs a transaction on (if it exists) or the variable
        bound to the draft (or performed) call.

        The current PyTOD version treats the variable bound to the `perform` statement
        as a mark of task/success or failure and communicates it to the NLG via `say`
        to ground agent utterances like "agent: You're booked in!".
        """
        try:
            return self._transaction[self._current_service][-1]
        except IndexError:
            raise AssignmentError(
                f"No transactions have been registered for service {self._current_service}"
            )

    @current_transaction_index.setter
    def current_transaction_index(self, index: int):
        self._transaction[self._current_service].append(index)

    @property
    def current_results_list(self) -> int:
        """Tracks variables bound to the calls to search intents, which have dual
        meaning as results lists after all the API required args have been specified."""
        try:
            return self._results_list[self._current_task][-1]
        except IndexError:
            raise AssignmentError(
                f"No results have been offered or carried over to the current task, "
                f"{self._current_task}"
            )

    @current_results_list.setter
    def current_results_list(self, index: int):
        self._results_list[self._current_task].append(index)

    @property
    def current_suggestion(self) -> int:
        """Tracks variables bound to task the agent has suggested.

        Notes
        -----
        The agent selects a related transaction after the user
        selects an entity, not a random task at a random point in
        the dialogue.
        """
        try:
            return self._suggestions[self._current_service][-1]
        except IndexError:
            raise AssignmentError(
                f"No suggestion has been made for service {self._current_service}"
            )

    @current_suggestion.setter
    def current_suggestion(self, index: int):
        self._suggestions[self._current_service].append(index)

    @property
    def last_suspended_task(self) -> int:
        """Tracks variables bound to tasks the user has put on hold.

        Notes
        -----
        The user suspends tasks suggested by the agent. They do not
        put on hold the current task and resume it later.
        """
        try:
            return self._suspended_tasks[self._current_service][-1]
        except IndexError:
            raise AssignmentError(
                f"No task has been suspended in service {self._current_service}"
            )

    @last_suspended_task.setter
    def last_suspended_task(self, index: int):
        """Tracks variables bound to tasks the agent has suggested but which
        the user has put on hold."""
        self._suspended_tasks[self._current_service].append(index)

    @singledispatchmethod
    def resolve_metadata(
        self, metadata_type: APISlotCarryOverInfo, field: TemplateField
    ) -> str:
        raise ValueError(f"Metadata of type {metadata_type} could not be resolved")

    @resolve_metadata.register(APISlotCarryOverInfo)
    def _(self, metadata: APISlotCarryOverInfo, field: TemplateField) -> str:
        def maybe_resolve_entity_call_ambiguity(
            metadata: APISlotCarryOverInfo,
            cmd: ServiceCommand,
            source_entities: list[int],
        ) -> Optional[tuple[Literal["entity"]]]:
            """Check if the metadata annotation points to a source slot that is
            not"""
            source_slot = metadata.slot
            # the source slot is a parameter, so
            # we follow the most recent rule
            if is_arg(source_slot, cmd):
                return
            assert source_slot in cmd.result_slots
            if source_entities:
                return ("_entity",)
            return

        service_intent = f"{metadata.service}.{metadata.function}"
        current_service = self._current_task.split(".")[0]
        possible_refs = ("_call", "_entity")
        if self._command_collection is not None:
            source_entities = self._entity.get(service_intent, [])
            resolution = maybe_resolve_entity_call_ambiguity(
                metadata,
                self._command_collection.get(metadata.service, metadata.function),
                source_entities,
            )
            if resolution is not None:
                possible_refs = resolution
        call_entity_indices = []
        # when slot/values are carried over across intents we reference
        # the last entity or call, whichever is highest
        for attr in possible_refs:
            try:
                call_entity_indices.append(getattr(self, attr)[service_intent][-1])
            except IndexError:
                pass
        # the user may select the entity and start a related task
        if metadata.service == current_service:
            try:
                call_entity_indices.append(
                    getattr(self, "_entity")[self._current_task][-1]
                )
            except IndexError:
                pass

        # in transactional-only dialogues, there is no entity. If the call fails and user
        # retries than the current call index is already set to self._current_index,
        # and we should not refer to it as a variable in the unresolved slot values in the call
        # signature. Hence, we exclude it and backtrack to the previous call if necessary
        call_entity_indices = [
            idx for idx in call_entity_indices if idx < self._current_index
        ]
        if not call_entity_indices:
            call_entity_indices.append((self._call[self._current_task][-2]))
        return f"x{max(call_entity_indices)}"

    def _get_container(self, attribute: str) -> dict[str, list[int]]:
        """Return the objects where all the call, entity and
        result list variables are stored."""

        match attribute:
            case "current_results_list":
                return dict(self._results_list)
            case "current_suggestion":
                return dict(self._suggestions)
            case "current_entity":
                return dict(self._entity)
            case "last_entity":
                return dict(self._last_entity)
            case _:
                raise ValueError(f"Could not get container for attribute {attribute}")

    def get_variable(
        self,
        field: TemplateField,
        instruction: PyTODInstruction | PyTODInstructionTemplate,
    ) -> str:
        if field.resolve_with_metadata:
            return self.resolve_metadata(field.metadata, field)
        # see dev/17_00074: the user can decline a task in a different service
        # and return to a service where a task was suggested
        match field.field:
            case "current_suggestion":
                try:
                    suggestion_idx = self._suggestions[instruction.service][-1]
                    return f"x{suggestion_idx}"
                except IndexError:
                    pass
            case "last_suspended_task":
                last_index = self.last_suspended_task
                resume_service = instruction.service
                assert last_index in self._suspended_tasks[resume_service]
        try:
            # this attempts to retrieve the information from the current
            # task - if the task changes and the user selected an entity
            # for the task that changes, this will fail
            return f"x{getattr(self, field.field)}"
        except AttributeError:
            try:
                method = getattr(self, f"get_{field.field}")
                return f"x{method(field.field)}"
            except AttributeError:
                raise ValueError(
                    f"Dialogue ({self._dialog_id}). "
                    f"Field {field} is marked as a variable but is not a known "
                    "Assignments property or method"
                )
        except AssignmentError:
            service = instruction.service
            intent = instruction.intent
            if service is not None and intent is not None:
                task = f"{service}.{intent}"
                if intent == _NO_ACTIVE_INTENT:
                    task = service
                try:
                    container = self._get_container(field.field)
                    index = container[task][-1]
                    return f"x{index}"
                except IndexError:
                    raise AssignmentError(
                        f"Attempted to access unassigned  _{field.field}_ variable in task {task}."
                        f"Falling back to the current call, {self.current_call}"
                    )
                except KeyError:
                    raise AssignmentError(
                        f"Attempted to access _{field.field}_ for task {task} "
                        f"but {field.field} is undefined for this task. "
                        f"Falling back to the current call, {self.current_call}"
                    )
            else:
                raise AssignmentError

    def _get_previous_service_tasks(self) -> list[IntentName]:
        """Return a list of previous tasks from the same service that are different
        to the current task."""
        current_function = self._current_task.split(".")[1]
        current_service = self._current_task.split(".")[0]
        possible_sources = [
            task
            for task in self._task_history[current_service]
            if task != current_function
        ]
        return possible_sources

    def maybe_disambiguate(
        self,
        instruction_template: PyTODInstructionTemplate | PyTODInstruction,
    ) -> PyTODInstruction | PyTODInstructionTemplate:
        """Templates may not be renderable because their fields can refer to
        multiple variables. An example is the template generated inside the
        `entity_query` interpreter routine, which adds a field called
        `current_entity_or_call`. Here we use information about the entities and
        tasks to decide what the variable referenced should be.


        Notes
        -----
        The function can be extended to allow customized referencing of entity/call
        variables at service/intent level."""

        if (
            not hasattr(instruction_template, "requires_disambiguation")
            or not instruction_template.requires_disambiguation
        ):
            return instruction_template

        for field in instruction_template.fields:
            if not field.requires_disambiguation:
                continue
            match field.field:
                case "current_entity_or_call":
                    try:
                        # task must have succeeded because we have an entity
                        entity_idx = self.current_entity
                        resolved_field = f"{{current_entity}}"  # noqa
                        override = self._maybe_override_entity_reference(
                            entity_idx, instruction_template
                        )
                        if override:
                            resolved_field = f"{{current_call}}"  # noqa
                        for elem_attribute in instruction_template.disambiguate:
                            attirb = getattr(instruction_template, elem_attribute)
                            setattr(
                                instruction_template,
                                elem_attribute,
                                attirb.format(**{f"{field.field}": resolved_field}),
                            )
                        field.field = "current_call" if override else "current_entity"
                    except AssignmentError:
                        assert self._current_task_transactional
                        source_tasks = self._get_previous_service_tasks()
                        if len(source_tasks) == 0:
                            # this is the only task we talked about, we must refer to the call
                            resolved_field = f"{{current_call}}"  # noqa
                            for elem_attribute in instruction_template.disambiguate:
                                attirb = getattr(instruction_template, elem_attribute)
                                setattr(
                                    instruction_template,
                                    elem_attribute,
                                    attirb.format(**{f"{field.field}": resolved_field}),
                                )
                            field.field = "current_call"
                        elif len(source_tasks) == 1:
                            source_task_service = (
                                f"{self._current_service}.{source_tasks[0]}"
                            )
                            assert (
                                source_task_service in self._entity
                                and self._entity[source_task_service]
                            )
                            source_task_entity_index = self._entity[
                                source_task_service
                            ][-1]
                            self._entity[self._current_task].append(
                                source_task_entity_index
                            )
                            resolved_field = f"{{current_entity}}"  # noqa
                            unresolved_field = field.field
                            # ensure that the attribute is actually a property
                            # of the entity it refers to
                            match instruction_template.tag:
                                case "entity_query":
                                    source_cmd = self._command_collection.get(
                                        self._current_service, source_tasks[0]
                                    )
                                    if (
                                        instruction_template.attribute
                                        not in source_cmd.result_slots
                                    ):
                                        resolved_field = f"{{current_call}}"  # noqa
                            # multiple fields of the instruction template (eg expression_template,
                            # variable) may have to be replaced with the resolved variable name
                            for elem_attribute in instruction_template.disambiguate:
                                attirb = getattr(instruction_template, elem_attribute)
                                setattr(
                                    instruction_template,
                                    elem_attribute,
                                    attirb.format(**{unresolved_field: resolved_field}),
                                )
                            field.field = resolved_field[
                                1:-1
                            ]  # {current_call} -> current_call
                        elif len(source_tasks) == 2:
                            assert isinstance(
                                instruction_template, AttributeAccessTemplate
                            )
                            attrib = instruction_template.attribute
                            try:
                                source_task_service = DISAMBIGUATION_MAP[
                                    self._current_task
                                ][attrib]
                            except KeyError:
                                raise DisambiguationError(
                                    f"Dialogue {self._dialog_id}, task {self._current_task}. "
                                    f"Could not disambiguate {field.field}."
                                    " There are two source tasks which "
                                    f"can be used for entity resolution: {source_tasks}."
                                    f"Disambiguation not defined for attribute {attrib}"
                                )
                            source_task_entity_index = self._entity[
                                source_task_service
                            ][-1]
                            self._entity[self._current_task].append(
                                source_task_entity_index
                            )
                            resolved_field = f"{{current_entity}}"  # noqa
                            for elem_attribute in instruction_template.disambiguate:
                                attirb = getattr(instruction_template, elem_attribute)
                                setattr(
                                    instruction_template,
                                    elem_attribute,
                                    attirb.format(**{f"{field.field}": resolved_field}),
                                )
                            field.field = "current_entity"
                        else:
                            # could back track through related search,
                            # but this should not be happening
                            raise DisambiguationError(
                                f"Cannot disambiguate: {field.field}"
                            )
                case _:
                    raise DisambiguationError(f"Cannot disambiguate: {field.field}")

    def _maybe_override_entity_reference(
        self, entity_idx: int, instruction_template: PyTODInstructionTemplate
    ) -> bool:
        """When entity selection co-occurs with task change, the entity of the query
        is copied to the new task so that it can be referenced in the command. However,
        in some services, the properties requested by the user uniquely refer to the
        transaction the user is performing on the entity (eg user is asking how long
        a funds transfer will take in `Banks_2`. Because we do not bind entities for
        transactions, we ensure that the copied entity is overridden if it does not
        have the property the user is requesting."""
        if self._current_task_transactional:
            # there ought to be a task whihc returned an entity
            prev_tasks = self._get_previous_service_tasks()
            if entity_idx in self._entity[f"{self._current_service}.{prev_tasks[-1]}"]:
                schema = self._command_collection.get(
                    self._current_service, prev_tasks[-1]
                )
                try:
                    assert instruction_template.attribute in schema.result_slots
                except AssertionError:
                    attr = instruction_template.attribute
                    logger.warning(
                        f"Resolving entity query to command {self._current_task} "
                        f"for attribute {attr}. Entity returned by {prev_tasks[-1]} "
                        "does not have this attribute."
                    )
                    return True
        return False

    def maybe_copy_entity_reference(self, tag: str):
        """Copies the entity reference from the current task to a
        compatible source task entity list. This is necessary so that an
        entity selected at the same time with initiating a new task is
        referenced in the call parameters of the new task. In v2/v3 data,
        we reference either the source task entity or the source task call,
        whichever is greatest."""
        if tag in {
            "result_selection_by_indexing_with_new_task",
            "entity_selection_with_new_task",
        }:
            previous_service = self._previous_task.split(".")[0]
            current_service, current_intent = self._current_task.split(".")
            possible_sources = self._get_previous_service_tasks()
            assert current_service == previous_service
            try:
                assert len(possible_sources) == 1
            except AssertionError:
                assert current_service in {"Events_2", "Flights_1"}
                if current_service == "Flights_1":
                    assert (
                        split_camel_case(current_intent)[1]
                        == split_camel_case(possible_sources[-1])[1]
                    )
            # carry-over the entity referenced in the task that just
            # finished to the current task. This "virtual" entity is copied
            # to the current task entity list so that we can resolve
            # the "previous_entity" tag to the list of results the system
            # provided during the query task.
            # we assume that the last intent provides the entity the user refers
            # to
            if tag == "result_selection_by_indexing_with_new_task":
                last_res_list_index = self._results_list[
                    f"{current_service}.{possible_sources[-1]}"
                ][-1]
                self.current_results_list = last_res_list_index
            if tag == "entity_selection_with_new_task":
                last_entity_index = self._entity[
                    f"{current_service}.{possible_sources[-1]}"
                ][-1]
                self.current_entity = last_entity_index
                self.current_results_list = self._results_list[
                    f"{current_service}.{possible_sources[-1]}"
                ][-1]

    def maybe_increment_references(
        self, instruction_template: PyTODInstruction | PyTODInstructionTemplate
    ):
        tag = instruction_template.tag
        if tag in {
            "call",
            "call_with_params",
            "call_with_carry_over",
            "call_with_selected_params",
        }:
            self.current_call = self._current_index
        if tag in {
            "assign_query_result",
            "results_iterator",
            "entity_selection",
            "entity_selection_with_new_task",
            "result_selection_by_indexing",
            # bind a new entity so that the call to the new task can reference it,
            # if required. This is not bound if:
            #   a)  the user states the value :we use the string value in this case.
            #   The 'result_selection_by_indexing_with_new_task' in
            #   `user_routines_registry_v3.py` does not return a template,
            #   deferring functionality to the `call_with_selected_param` routine)
            #   b) If the entity was already selected, in the turn before the intent change.
            #   In this case, we simply reference the variable assigned in the source intent
            # Similarly, the `result_selection_by_indexing_with_new_task` routine in
            # `user_routines_registry_v3.py` does not return an element
            # when we only have one entity in the results that was selected without
            # explicit name mention. Instead, we use the entity we bound already during
            # the previous task and use the `call_with_selected_params` routine to reference
            # the `current_entity` correctly
            "result_selection_by_indexing_with_new_task",
        }:
            current_service, current_intent = self._current_task.split(".")
            service, intent = instruction_template.service, instruction_template.intent
            # for these tags, we do not expect an intent change: we don't need intent information
            if tag not in {"assign_query_result", "results_iterator"}:
                assert (
                    intent is not None
                ), "Entity or slot selection requires active intent for correct assignment."
            assert intent != _NO_ACTIVE_INTENT, f"Unexpected intent: {intent}"
            if tag in {
                "entity_selection_with_new_task",
                "result_selection_by_indexing_with_new_task",
                "entity_selection",
            }:
                # rendering happens after binding the entity, so we need to remember
                # the variables
                self._last_entity = deepcopy(self._entity)
                # we don't use the setter because entities can be selected at the same time with a
                # task change to a new domain and the assigment would be wrong. This also shows
                # an example of how the framework can be updated to represent
                # multi-intent conversations as python programs
                if current_service == service:
                    # the intent changed but the service has not, so
                    # we can record the entity for the new task
                    self.current_entity = self._current_index
                    if (
                        tag
                        in {
                            "entity_selection_with_new_task",
                            "result_selection_by_indexing_with_new_task",
                        }
                        and self._copy_selected_entity_to_search
                    ):
                        for task in self._last_entity:
                            if (
                                task.startswith(current_service)
                                and current_intent not in task
                            ):
                                self._last_entity[task].append(self._current_index)
                                self._entity[task].append(self._current_index)
                else:
                    # the service changed, so we assign the entity to the task that the user
                    # just completed
                    self._entity[f"{service}.{intent}"].append(self._current_index)
            # the user selected an entity or among offered slot values,
            # but has not yet stated a new task
            elif tag in {"result_selection_by_indexing"}:
                if service == current_service:
                    assert intent == current_intent
                    self.current_entity = self._current_index
                else:
                    self._entity[f"{service}.{intent}"].append(self._current_index)

            else:
                # for regular assignments/results iteration we have a single frame in the semantic
                # annotation and the current task is known
                self.current_entity = self._current_index
        if tag in {"slot_filling_hint"}:
            self.current_hint = self._current_index
        if tag in {"task_performed"}:
            self.current_transaction_index = self._current_index
        if tag in {"inform_entity_count_signal"}:
            assert not self._current_task_transactional
            self.current_results_list = self.current_call
        if tag in {"task_suggestion"}:
            assert instruction_template.service == self._current_service
            self.current_suggestion = self._current_index
        if tag in {"declines_suggestion"}:
            self._suspended_tasks[instruction_template.service].append(
                self._current_index
            )

    def maybe_render(
        self,
        instruction_template: PyTODInstruction | PyTODInstructionTemplate,
        api_info: APIInfo,
    ) -> PyTODInstruction:
        if not needs_rendering(instruction_template):
            return instruction_template
        # render templates
        template_vals = {}
        for field in instruction_template.fields:
            placeholder = field.field
            if field.is_variable:
                template_variable = self.get_variable(field, instruction_template)
                template_vals[placeholder] = template_variable
            else:
                try:
                    template_value = getattr(api_info, placeholder)
                    template_vals[placeholder] = template_value
                except AttributeError:
                    assert instruction_template.tag == "task_status_signal"
                    template = self._interpreter_cfg.notifications.task_status_template
                    maybe_value = getattr(
                        getattr(template, placeholder), api_info.api_type
                    ).template
                    if "{task}" in maybe_value:
                        template_value = maybe_value.format(
                            task=format_tool_name(self._tool_name_formatter, api_info)
                        )
                    else:
                        template_value = maybe_value
                    template_vals[placeholder] = template_value
        # render origin
        origin_template_vals = {}

        if has_origin(instruction_template):
            for field in instruction_template.origin_fields:
                placeholder = field.field
                assert (
                    field.is_variable
                ), f"Origins are bound to variable indices but {field} is not marked as variable"
                try:
                    template_variable = self.get_variable(field, instruction_template)
                except AttributeError:
                    raise AttributeError(
                        f"Dialogue ({self._dialog_id}): "
                        f"Attempted to access undefined variable {field.field} "
                        f"while rendering field {field}"
                    )
                origin_template_vals[placeholder] = get_index(template_variable)
        match instruction_template:
            case ProgramStatementTemplate(
                expression_template=expression_template, tag=tag
            ):
                expression = expression_template.format(**template_vals)
                check_for_syntax_errors(expression)
                pass_to_nlg = tag in NLG_TAGS or instruction_template.pass_to_nlg
                # this happens if we want to pass a variable reference to
                # nlg - we need var_index to sort the `say` arguments
                is_inlined = instruction_template.is_inlined
                var_index, variable, pass_expression = None, None, False
                if is_inlined:
                    var_index = get_index(expression)
                    variable = expression
                    pass_expression = True
                return ProgramStatement(
                    expression=expression,
                    tag=instruction_template.tag,
                    is_inlined=instruction_template.is_inlined,
                    pass_to_nlg=pass_to_nlg,
                    var_index=var_index,
                    variable=variable,
                    pass_expression=pass_expression,
                )
            case BackendNotificationTemplate(
                dialog_template=dialog_template,
            ):
                origin = None
                if has_origin(instruction_template):
                    origin = instruction_template.origin_template.format(
                        **origin_template_vals
                    )
                return BackendNotification(
                    dialog=dialog_template.format(**template_vals),
                    is_inlined=instruction_template.is_inlined,
                    tag=instruction_template.tag,
                    is_assignable=True,
                    pass_to_nlg=instruction_template.pass_to_nlg,
                    origin=int(origin),
                )
            case BackendHintTemplate(dialog_template=dialog_template):  # noqa
                origin = None
                if has_origin(instruction_template):
                    origin = instruction_template.origin_template.format(
                        **origin_template_vals
                    )
                return BackendHint(
                    dialog=dialog_template.format(**template_vals),
                    tag=instruction_template.tag,
                    is_assignable=True,
                    pass_to_nlg=instruction_template.pass_to_nlg,
                    origin=int(origin) if origin is not None else origin,
                )
            case AttributeAccessTemplate(
                expression_template=expression_template, variable=variable
            ):
                variable = variable.format(**template_vals)
                var_index = get_index(variable)
                return ProgramStatement(
                    tag=instruction_template.tag,
                    expression=expression_template.format(**template_vals),
                    is_inlined=instruction_template.is_inlined,
                    is_assignable=instruction_template.is_assignable,
                    pass_to_nlg=instruction_template.pass_to_nlg,
                    pass_expression=True,
                    var_index=var_index,
                )
            case _:
                raise ValueError(
                    f"Unknown PyTOD program instruction: {type(instruction_template)}"
                )

    def update_tasks(self, api_info: APIInfo):
        function = api_info.function
        service = api_info.service
        self._current_service = service
        self._current_intent = function
        self._previous_task = self._current_task
        if function not in self._task_history[service]:
            self._task_history[service].append(function)
        if function in self._metadata.search_intents:
            self._search_task_history[service].append(function)
        self._current_task = f"{service}.{function}"
        self._current_task_transactional = (
            function in self._metadata.transactional_intents
        )

    def render_and_assign(
        self,
        user_query: UserQuery,
        instruction_templates: TemplatesCollection,
        api_info: APIInfo,
        system_response: Optional[Response],
    ) -> list[tuple[Optional[int], PyTODInstruction | NaturalLanguageTypes]]:
        self.update_tasks(api_info)
        instruction_templates: list[
            PyTODInstruction | PyTODInstructionTemplate
        ] = instruction_templates.expressions_and_templates
        transcript_draft = [(None, user_query)]
        nlg_call_variables = []
        for instruction in instruction_templates:
            # some templates require info about active/complete tasks
            # and variables bound to entities to be renderable
            self.maybe_disambiguate(instruction)
            if instruction.is_assignable:
                self._current_index += 1
                # copy references to/from previous task to handle
                # user entity selection
                self.maybe_copy_entity_reference(instruction.tag)
                self.maybe_increment_references(instruction)
                instruction: PyTODInstruction = self.maybe_render(instruction, api_info)
                transcript_draft.append((self._current_index, instruction))
                if instruction.var_index is None and instruction.pass_to_nlg:
                    instruction.var_index = self._current_index
            else:
                instruction: PyTODInstruction = self.maybe_render(instruction, api_info)
                if not instruction.is_inlined:
                    transcript_draft.append((None, instruction))
            assert_on_instruction_type(instruction)
            if instruction.pass_to_nlg:
                nlg_call_variables.append(instruction)

        self.append_nlg_call(transcript_draft, nlg_call_variables)

        if system_response is not None:
            self.append_system_response(transcript_draft, system_response)
        return transcript_draft

    def append_system_response(
        self,
        transcript_pieces: list[
            tuple[Optional[int], PyTODInstruction | NaturalLanguageTypes]
        ],
        system_response: Response,
    ):
        """Add the system response to the transcript elements."""

        @dispatch_on_value
        def update_transcript_with_response(grammar: str):
            raise ValueError(
                f"Unknown grammar version: {grammar}. Only v2 and v3 are supported."
            )

        @update_transcript_with_response.register("v3")
        def _(grammar: str):
            transcript_pieces.append((None, Response(response=sanitised_response)))

        @update_transcript_with_response.register("v2")
        def _(grammar: str):
            say_cmd = (
                f'{self._interpreter_cfg.actions.say_cmd_name}("{sanitised_response}")'
            )
            self._current_index += 1
            system_response.response = say_cmd
            transcript_pieces.append(
                (
                    self._current_index,
                    ProgramStatement(expression=say_cmd, tag="nlg_response"),
                )
            )

        say_cmd = f'{self._interpreter_cfg.actions.say_cmd_name}("{system_response.response}")'
        sanitised_response = sanitise_say_cmd(say_cmd, system_response)
        update_transcript_with_response(self._grammar)

    def append_nlg_call(
        self,
        transcript_pieces: list[
            tuple[Optional[int], PyTODInstruction | NaturalLanguageTypes]
        ],
        nlg_call_variables: list[PyTODInstruction],
    ):
        # this should happen at the end of the conversation
        if not nlg_call_variables:
            return
        nlg_function = self._interpreter_cfg.nlg_calls.cmd_name
        assert all((el.pass_to_nlg for el in nlg_call_variables))
        assert all(
            (
                (el.var_index is not None and isinstance(el.var_index, int))
                for el in nlg_call_variables
            )
        )
        nlg_call_variables.sort(key=attrgetter("var_index"))
        vars = [
            f"x{el.var_index}" if not el.pass_expression else el.expression
            for el in nlg_call_variables
        ]
        args = ", ".join(vars)
        cmd = ProgramStatement(
            expression=f"{nlg_function}({args})", tag="nlg_call", is_assignable=True
        )
        self._current_index += 1
        transcript_pieces.append((self._current_index, cmd))


def sanitise_say_cmd(say_cmd: str, system_response: Response) -> str:
    """Ensure responses do not contain quotes or any symbols that
    would render `say(response)` unparseable."""

    if """ " 8 Sushi " """ in say_cmd:
        say_cmd = say_cmd.replace(""" " 8 Sushi " """, " 8 Sushi ")
        system_response.response = system_response.response.replace(
            """ " 8 Sushi " """, " 8 Sushi "
        )
    if """\nfor""" in say_cmd:
        say_cmd = say_cmd.replace("""\nfor""", " For ")
        system_response.response = system_response.response.replace("""\nfor""", " For")
    try:
        _ = ast.parse(say_cmd)
    except SyntaxError as e:
        logger.error(f"Could not parse {say_cmd}...")
        assert "clock" not in say_cmd
        if "426 Brannan" in say_cmd:
            formatted_response = system_response.response.replace('"', ":")
        else:
            formatted_response = system_response.response.replace('"', "")
        formatted_response = formatted_response.replace("\n", " ")
        if any(s in formatted_response for s in ("Dumbo", "some examples are")):
            formatted_response = formatted_response.replace(" - ", ":")
            if "recommendations-" in formatted_response:
                formatted_response = formatted_response.replace("-", ":")
        if "booze" in formatted_response:
            formatted_response = system_response.response.replace('"', "")

        logger.info(f"Reformatted sentence to {formatted_response}")
        try:
            assert any(
                s in formatted_response
                for s in (
                    "To confirm",
                    "Funeral",
                    "Dumbo",
                    "your location",
                    "426 Brannan",
                    "Southern Category",
                    "booze",
                    "some examples are",
                    "8 Sushi",
                    "damage or cost",
                    "Public Art Urban Light",
                )
            )
        except AssertionError:
            raise e
        return formatted_response

    return system_response.response
