#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import functools
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from pytod.command import CommandCollection, ServiceCommand, is_arg
from pytod.execution.auto_commands import Signal
from pytod.execution.policy_utils import RecommendedAction
from pytod.pytod_types.aliases import (
    IntentName,
    ServiceName,
    SlotName,
    SlotValueDict,
    StateDict,
    VariableName, DialogueID,
)
from pytod.sgd_metadata import NON_CUMULATIVE_SERVICES
from pytod.simulation.api_driver import APIDriver, dev_apis, test_apis
from pytod.simulation.command import Command
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.database import MongoDBCollection, dev_databases, test_databases
from pytod.simulation.entities import Entity
from pytod.simulation.search_command import SearchCommand
from pytod.utils import stringify_list

logger = logging.getLogger(__name__)

def _warn_state_tracking_invocation(dial_id: DialogueID, obj_name: str):
    logger.warning(
        f"{dial_id} State tracking invoked with {obj_name} type, expected Command subclass."
    )

CommandRecommendations = dict[
    ServiceName, dict[IntentName, Optional[list[RecommendedAction]]]
]

DatabaseCollection = dict[ServiceName, dict[IntentName, MongoDBCollection]]

APICollection = dict[ServiceName, dict[IntentName, APIDriver]]

NumItems = int
EntityInfo = tuple[NumItems, Entity]
IntentStates = list[dict[IntentName, SlotValueDict]]


def _recommendation_factory():
    return defaultdict(dict)


class AssignmentError(Exception):
    pass


class StateAccumulator:
    """Accumulate the state of APIs implementing
    the same service to obtain service level state."""

    def __init__(self, schema: CommandCollection):
        self._intent_states = defaultdict(list)
        self.service_state = {}
        self._schema = schema
        self._current_service: ServiceName = ""
        self._deletion_warned = defaultdict(list)

    def _maybe_warn_slot_deletion(self, deleted_slots: list[SlotName]):
        """Warn on possible slot deletion."""
        if (
            deleted_slots
            and sorted(deleted_slots)
            not in self._deletion_warned[self._current_service]
        ):
            logger.warning(
                f"Slots {stringify_list(deleted_slots)} appear to have been deleted. "
                "This is not expected"
            )
            self._deletion_warned[self._current_service].append(sorted(deleted_slots))

    def _maybe_delete_slots(
        self, current_state: StateDict, new_state: StateDict, new_schema: ServiceCommand
    ) -> list[SlotName]:
        """The agent is expected to call follow-up intents with keywords that reference
        entities or commands already generated from the first service. In practice, the agent
        may fail to generate some of the keywords. Therefore, we have to delete these slots
        from the cumulated state to avoid overestimating the performance."""
        to_delete = [
            arg
            for arg in current_state
            if is_arg(arg, new_schema) and arg not in new_state
        ]
        self._maybe_warn_slot_deletion(to_delete)
        for slot in to_delete:
            current_state.pop(slot)
        return to_delete

    def _merge_pair(
        self,
        first: dict[IntentName, SlotValueDict],
        second: dict[IntentName, SlotValueDict],
    ) -> dict[IntentName, SlotValueDict]:
        """Merge the state of two command from the same service.

        Parameters
        ----------
        first
            The state of the command the user invoked first.
        second
            The state of the second command the user invoked (possibly
            a follow-up command)
        """

        first_cmd, second_cmd = next(iter(first)), next(iter(second))
        # the state is cumulative, so we
        # always copy the state for the first cmd
        state, new_state = deepcopy(first[first_cmd]), deepcopy(second[second_cmd])
        self._maybe_delete_slots(
            state, new_state, self._schema.get(self._current_service, second_cmd)
        )
        # update the state with any new arguments relevant
        # to the new command or updates to previously
        # mentioned arguments
        for slot in new_state:
            state[slot] = new_state[slot]
        return {second_cmd: state}

    def _merge_drafts(self, drafts: IntentStates) -> SlotValueDict:
        """Merge the state of all commands from the same service."""
        if not drafts:
            return {}
        return next(iter(functools.reduce(self._merge_pair, drafts).values()))

    def _merge_intent_states(self, intent_states: IntentStates) -> SlotValueDict:
        """Accumulate the states of intents from the same service, accounting
        for missed slots during intent changes."""

        def _get_last_draft(
            key: IntentName, intent_states: IntentStates
        ) -> dict[IntentName, SlotValueDict]:
            """The model may generate the same command multiple times, but we
            only consider the state of the last generated instance for state
            tracking purposes."""
            for draft_state in reversed(intent_states):
                if key in draft_state:
                    return draft_state
            else:
                raise IndexError(f"Cannot find draft for {key}")

        if len(intent_states) == 1:
            return next(iter(intent_states[0].values()))

        # get a unique list of all the intents from the
        # current service, in the order in which they were
        # generated
        draft_cmd_names = []
        for draft_state in intent_states:
            cmd = next(iter(draft_state))
            if cmd not in draft_cmd_names:
                draft_cmd_names.append(cmd)
        # merge the state of the last draft for each intent in the service
        return self._merge_drafts(
            [_get_last_draft(intent, intent_states) for intent in draft_cmd_names]
        )

    def _non_cumulative_state(self) -> bool:
        return self._current_service in NON_CUMULATIVE_SERVICES

    def _update_state_intents(
        self,
        intent_states: list[dict[IntentName, dict[ServiceName, list[str]]]],
        key: IntentName,
        state: dict[SlotName, list[str]],
    ):
        """Track the state of all the commands the user invoked,
        keeping track of the invocation order."""

        # track the state of the first command the user invoked
        if not intent_states:
            intent_states.append({key: state})
            return

        # the commands accumulate the state as we go
        # along, we just store the latest update if the
        # command has been updated by the user
        if key in intent_states[-1]:
            intent_states[-1] = {key: state}
        else:
            if self._non_cumulative_state():
                self.service_state[self._current_service] = {}
                self._intent_states[self._current_service] = [{key: state}]
                return
            # commands should be generated once and
            # then updated, so we warn if this is not
            # the case and multiple instances of the same
            # cmd have been created
            for cmd_state in intent_states:
                if key in cmd_state:
                    logger.warning(
                        f"A new draft was created for command: {key}. "
                        f"Only the last draft will be used for state "
                        f"tracking."
                    )
            # create a container for tracking the state of follow-up
            # command the user invoked
            intent_states.append({key: state})

    def update_service_state(
        self, intent: IntentName, service: ServiceName, state: StateDict
    ):
        """Update the state of the current service"""
        self._current_service = service
        self._update_state_intents(
            self._intent_states[service], intent, state["slot_values"]
        )
        self.service_state[service] = self._merge_intent_states(
            self._intent_states[service]
        )


class PyTODAgentState:
    var_containers = (
        "assignments",
        "suggested_tasks",
        "suspended_tasks",
        "revealed_entities",
        "execution_errors",
    )

    def __init__(self, split: Literal["validation", "test"], schema: CommandCollection):
        match split:
            case "validation":
                split = "dev"
            case _:
                assert split in {"test", "dev"}
        self.current_recommendations: CommandRecommendations = defaultdict(dict)
        self.assignments: dict[
            VariableName, Command | Entity | Signal | NumItems | EntityInfo
        ] = {}
        # possible follow-up intents suggested by the agent to the user
        self.suggested_tasks: dict[VariableName, ConfirmedCommand] = {}
        # agent suggested follow-up tasks the user decided not to execute
        self.suspended_tasks: dict[VariableName, ConfirmedCommand] = {}
        # tasks the user suspended and resumed
        self.resumed_tasks: dict[VariableName, ConfirmedCommand] = {}
        # store the outputs of len | show | slice commands
        self.revealed_entities: dict[
            VariableName, tuple[NumItems, Entity] | str | NumItems
        ] = {}
        self.execution_errors: dict[VariableName, Signal] = {}
        self.confirmations: dict[VariableName, ConfirmedCommand] = {}
        self.databases: DatabaseCollection = (
            dev_databases if split == "dev" else test_databases
        )
        self.apis: APICollection = dev_apis if split == "dev" else test_apis
        self._state_accumulator = StateAccumulator(schema)
        self.dialogue_state: defaultdict[ServiceName, StateDict] = defaultdict(dict)
        self._current_task: SearchCommand | ConfirmedCommand | None = None

    def get_assignment(
        self, var_name: str, idx: Optional[int] = None
    ) -> Entity | Command | None:
        var_obj = self.assignments.get(var_name)
        if var_obj is None:
            raise ValueError(f"{var_name} not found")
        if not isinstance(var_obj, (Command, Entity, Signal)):
            logger.warning(
                f"{var_name} must be a Command, Entity or Signal (was {type(var_obj)})"
            )
            raise ValueError(
                f"{var_name} must be a Command, Entity or Signal (was {type(var_obj)})"
            )

        if isinstance(var_obj, SearchCommand) and idx is not None:
            entity = var_obj.get_entity_by_idx(idx)
            if entity is None:
                logger.warning(f"Entity not found at {idx}")
            return entity

        return var_obj

    @property
    def current_task(self) -> SearchCommand | ConfirmedCommand:
        return self._current_task

    def _set_current_task(self, command: Command):
        """Track the last task the user initiated."""
        if isinstance(command, Command):
            self._current_task = command
        else:
            _warn_state_tracking_invocation(
                self.current_task.uuid, command.__class__.__name__
            )

    @property
    def current_service(self) -> ServiceName:
        return self._current_task.service

    @property
    def current_intent(self) -> IntentName:
        return self._current_task.name

    @property
    def current_task_context(self) -> MongoDBCollection | APIDriver | None:
        registry = (
            self.databases
            if isinstance(self.current_task, SearchCommand)
            else self.apis
        )
        return registry.get(self.current_service, {}).get(self.current_intent, None)

    def track_state(self, command: Command):
        """Track the state of the service in the dialogue."""
        service, intent = command.service, command.name
        accumulator = self._state_accumulator
        self._set_current_task(command)
        accumulator.update_service_state(intent, service, command.state)
        self.dialogue_state[service]["slot_values"] = accumulator.service_state[service]
        self.dialogue_state[service]["active_intent"] = intent
        if "requested_slots" not in self.dialogue_state[service]:
            self.dialogue_state[service]["requested_slots"] = []

    def update_user_requests(self, requested_info: list[SlotName]):
        """Update the current state with the user requested info. Invoked during
        execution by the NLG tool."""
        service = self._current_task.service
        self.dialogue_state[service]["requested_slots"] += list(set(requested_info))


class TurnContext(Enum):
    MULTI_DOMAIN = "multi_domain"
    SINGLE_DOMAIN = "single_domain"


@dataclass
class ExecutionContext:
    turn_context: TurnContext
    conversation_end: bool
