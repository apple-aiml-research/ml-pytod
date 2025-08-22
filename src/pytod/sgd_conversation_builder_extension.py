#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This is a builder extension that provides additional annotation of slots
carried over across tasks (both in- and cross-domain)."""

import functools
import logging
import random
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Literal, Optional, Type, Union

from pydantic import BaseModel

from pytod.command import CommandCollection, ServiceName, is_arg
from pytod.pytod_types.aliases import ServiceIntent, SlotName, SlotValue
from pytod.pytod_types.sgd_conversation import (
    NO_ACTIVE_INTENT,
    APISlotCarryOverInfo,
    Author,
    Conversation,
    DialogueState,
    SlotCarryOverValues,
    SlotsCarriedOver,
    SystemDialogueAct,
    Turn,
    UserAction,
    UserDialogueAct,
    ValuesInfo,
    get_active_service,
    get_prev_mentioned_slots,
)
from pytod.sgd_conversation_builder import add_actions_dict, build_conversation
from pytod.sgd_utils import get_dialogue_services
from pytod.utils import dispatch_on_value, swap_keys_and_values

logger = logging.getLogger(__name__)

SlotCarryOverInfo = dict[ServiceName, Union[str, int, list[str]]]
TargetServiceName = str  # the service to which a slot is carried over
TargetIntentName = str  # the intent to which a slot is carried over
TargetServiceIntentName = str
SourceServiceIntentName = str
TargetSlotName = str  # the name of the slot whose value is carried over
SourceServiceName = (
    str  # the name of the service where the slot was mentioned (or copied from)
)
SourceIntentName = (
    str  # the name of the intent where the slot was mentioned (or copied from)
)
SourceSlotName = str  # the name of the slot which provides the value for this slot
SlotRelationsMapping = dict[
    TargetServiceName,
    dict[
        TargetIntentName,
        dict[
            TargetSlotName,
            list[
                dict[
                    [Literal["service", "intent", "slot"]],
                    Union[SourceSlotName, SourceIntentName, SourceServiceName],
                ]
            ],
        ],
    ],
]


BACKTRACK_TO_RESULTS: dict[
    TargetServiceIntentName, dict[SourceServiceIntentName, list[SourceSlotName]]
] = {
    "RideSharing_1.GetRide": {
        "Travel_1.FindAttractions": ["attraction_name"],
        "Homes_1.ScheduleVisit": ["address"],
        "Movies_1.GetTimesForMovie": ["street_address"],
        "Restaurants_1.ReserveRestaurant": ["street_address"],
        "Restaurants_2.ReserveRestaurant": ["address"],
        "Services_2.BookAppointment": ["address"],
        "Services_4.BookAppointment": ["address"],
        "Events_1.BuyEventTickets": ["event_location"],
    },
    "RideSharing_2.GetRide": {
        "Homes_1.ScheduleVisit": ["address"],
        "Homes_2.ScheduleVisit": ["address"],
        "Restaurants_1.ReserveRestaurant": ["street_address"],
        "Restaurants_2.ReserveRestaurant": ["address"],
        "Services_1.BookAppointment": ["street_address"],
        "Services_3.FindProvider": ["street_address"],
        "Services_3.BookAppointment": ["street_address"],
        "Services_4.BookAppointment": ["address"],
        "Travel_1.FindAttractions": ["attraction_name"],
        "Events_3.BuyEventTickets": ["venue"],
    },
    "Calendar_1.AddEvent": {
        "Services_2.BookAppointment": ["address"],
        "Services_3.BookAppointment": ["street_address"],
        "Services_1.BookAppointment": ["street_address"],
        "Homes_1.ScheduleVisit": ["address"],
        "Movies_1.GetTimesForMovie": ["street_address"],
        "Restaurants_1.ReserveRestaurant": ["street_address"],
    },
    "Messaging_1.ShareLocation": {
        "Homes_2.ScheduleVisit": ["address"],
        "Events_3.FindEvents": ["venue"],
        "Services_1.FindProvider": ["street_address"],
    },
}
"""Specifies target services and the source services where a carried-over
slot value can be looked up in the service results for the purposes of
discovering the source API."""


def _check_act_presence(
    turn: Turn, act: Union[UserDialogueAct, SystemDialogueAct]
) -> Optional[set[str]]:
    """Return the services whose corresponding frames contain an action annotated with
    dialogue act `act`."""
    services_present = set()
    if turn.author == Author.USER:
        actions = getattr(turn, "user_actions_dict")
    else:
        actions = getattr(turn, "system_actions_dict")
    for service, service_actions in actions.items():
        if act in service_actions:
            services_present.add(service)
    return services_present or None


class TaskChangeInfo(BaseModel):
    discuss_old_service_again: bool
    intent_changed_in_service: bool
    service_changed: bool
    user_retries_task: bool
    current_turn_idx: int
    active_service: str
    previous_services: dict[ServiceName, int]
    active_service_intent: str
    service_intent_hist: dict[str, list[int]]
    history: Conversation
    task_change_fields: tuple[
        Literal["discuss_old_service_again"],
        Literal["intent_changed_in_service"],
        Literal["service_changed"],
        Literal["user_retries_task"],
    ] = (
        "discuss_old_service_again",
        "intent_changed_in_service",
        "service_changed",
        "user_retries_task",
    )


def parse_as_carry_over_type(
    carry_over_info: Union[dict[str, list[str]], dict[str, list[dict, Any]]]
) -> Union[dict[str, SlotCarryOverValues], dict[str, list[APISlotCarryOverInfo]]]:
    """Parses raw slot backtracking algorithm into a structured
    object (either `APISlotCarryOverInfo` or `SlotCarryOverValues`).
    These objects build the SGD `Conversation` object carry-over annotations.
    """

    def parse_as_api_info(
        carry_over_info: dict[str, list[dict, Any]]
    ) -> dict[str, list[APISlotCarryOverInfo]]:
        api_info = {}
        for slot, possible_api in carry_over_info.items():
            api_info[slot] = [
                APISlotCarryOverInfo.model_validate(api) for api in possible_api
            ]
        return api_info

    def parse_as_values(
        carry_over_info: dict[str, list[str]]
    ) -> dict[str, SlotCarryOverValues]:
        return {
            slot: SlotCarryOverValues.model_validate({"values": slot_vals})
            for slot, slot_vals in carry_over_info.items()
        }

    def get_nested_value_type(
        carry_over_dict: Union[dict[str, list[str]], dict[str, list[dict, Any]]]
    ) -> Union[Type[dict], Type[str]]:
        values = list(chain(carry_over_dict.values()))
        types = list(set(map(type, values)))
        assert len(types) == 1
        assert types[0] is list
        nested_type = type(
            carry_over_dict[random.choice(list(carry_over_dict.keys()))][0]
        )
        assert all(
            isinstance(val, nested_type) for val in chain(*carry_over_dict.values())
        )
        return nested_type

    nested_val_type = get_nested_value_type(carry_over_info)
    if nested_val_type is dict:
        return parse_as_api_info(carry_over_info)
    else:
        assert nested_val_type is str
    return parse_as_values(carry_over_info)


def gather_task_change_info(
    dialogue: dict[str, Any],
    conversation: Conversation,
    current_service: str,
    current_user_turn_idx: int,
    previous_services: dict[ServiceName, int],
    active_service_intent: ServiceIntent,
    service_intent_hist: dict[ServiceIntent, list[int]],
) -> TaskChangeInfo:
    """Tracks changes in service and intent at the current dialogue turn.

    Parameters
    ----------
    dialogue
        The raw dialogue, in SGD format.
    conversation
        Structured object representing the dialogue.
    current_service
        The service the assistant should invoke to help the user with the current task.
    current_user_turn_idx
        The index of the current user turn in the conversation.
    previous_services
        Maps service names to the index of the turn where they last occurred in the dialog.
        These are expected to be indices of user turns.
    active_service_intent
        User's currently active intent and the service it belongs to.
        Represented as {service_name}.{intent_name}
    service_intent_hist
        Maps the service.intent name to the turn indices where the
        respective service and intent were active. These are user turn indices.
    """

    def sanity_check_annotation():
        """Check annotation follows expected pattern when the user retries tasks."""

        assert current_service in conversation[prev_user_turn_idx].dialogue_state
        prev_state = conversation[prev_user_turn_idx].dialogue_state[current_service]
        assert f"{current_service}.{prev_state.active_intent}" == active_service_intent

    def get_dialogue_history(
        conversation: Conversation, current_user_turn_idx: int
    ) -> Conversation:
        """For a given user turn index `current_user_turn_idx`
        return the conversation history up to, and including the previous user turn.
        The last system turn is not included."""

        assert current_user_turn_idx % 2 == 0
        if current_user_turn_idx == 0:
            return conversation
        turns = deepcopy(conversation[: current_user_turn_idx - 1])
        services = get_dialogue_services(
            {"turns": dialogue["turns"][: current_user_turn_idx - 1]}
        )
        return Conversation(turns=turns, services=services, id=conversation.id)

    task_info = {
        "service_changed": False,
        "discuss_old_service_again": False,
        "intent_changed_in_service": False,
        "user_retries_task": False,
        "current_turn_idx": current_user_turn_idx,
        "active_service": current_service,
        "previous_services": previous_services,
        "active_service_intent": active_service_intent,
        "service_intent_hist": service_intent_hist,
        "task_change_fields": (
            "discuss_old_service_again",
            "intent_changed_in_service",
            "service_changed",
            "user_retries_task",
        ),
        "history": get_dialogue_history(conversation, current_user_turn_idx),
    }
    prev_user_turn_idx = current_user_turn_idx - 2
    if previous_services:
        last_service_occurrence = previous_services.get(current_service, None)
        service_changed = (current_service not in previous_services) or (
            last_service_occurrence is not None
            and last_service_occurrence < prev_user_turn_idx
        )
        discuss_old_service_again = (
            current_service in previous_services
            and previous_services[current_service] < prev_user_turn_idx
        )
        intent_changed_in_service = False
        # check if the intent changed
        if not service_changed:
            try:
                assert prev_user_turn_idx >= 0
                intent_changed_in_service = (
                    prev_user_turn_idx not in service_intent_hist[active_service_intent]
                )
            except KeyError:
                assert not intent_changed_in_service
                intent_changed_in_service = True
        user_retries_task = False
        if all(
            not change_flag
            for change_flag in (
                service_changed,
                intent_changed_in_service,
                discuss_old_service_again,
            )
        ):
            user_actions = conversation[current_user_turn_idx].user_actions_dict[
                current_service
            ]
            user_retries_task = UserDialogueAct.INFORM_INTENT in user_actions
            if user_retries_task:
                sanity_check_annotation()

        task_info["discuss_old_service_again"] = discuss_old_service_again
        task_info["intent_changed_in_service"] = intent_changed_in_service
        task_info["service_changed"] = service_changed
        task_info["user_retries_task"] = user_retries_task

    return TaskChangeInfo.model_validate(task_info)


def get_last_service_invocation(
    active_service: str, usr_turn_idx: int, task_change_history: list[TaskChangeInfo]
) -> Optional[int]:
    """Return the index of the last turn the active service was invoked in the current conversation.
    Return `None` if the service was never invoked."""
    prev_user_turn_idx = usr_turn_idx - 2
    assert prev_user_turn_idx >= 0
    # tracks the last occurrence of this service
    # service_history: dict[ServiceName, int] = task_change_history[-1].previous_services
    service_occurrences = set()
    for change_info in reversed(task_change_history):
        service_history: dict[ServiceName, int] = change_info.previous_services
        if active_service not in service_history:
            continue
        service_occurrences.add(service_history[active_service])
    # a previous invocation of a service is one that
    # did not occur in the current turn or the turn before
    previous_invocations = sorted(
        list(service_occurrences.difference({usr_turn_idx, usr_turn_idx - 2}))
    )
    if not previous_invocations:
        return
    return previous_invocations[-1]


def get_task_name(state: DialogueState) -> str:
    return state.active_intent


def mentioned_in_results(
    conversation: Conversation,
    c_slot: SlotName,
    carry_over_values: list[SlotValue],
    usr_turn_idx: int,
    active_service_intent: str,
    source_service_intent: str,
) -> bool:
    """
    Parameters
    ----------
    conversation
    c_slot
        The slot whose value is carried over.
    carry_over_values
        Values of `c_slot`
    usr_turn_idx
        The index up to which we expect `c_slot` to be mentioned
    active_service_intent, source_service_intent
        Intent active and the intent that is currently assumed to
        contain the value for the slot
    """

    if (
        active_service_intent in BACKTRACK_TO_RESULTS
        and source_service_intent in BACKTRACK_TO_RESULTS[active_service_intent]
    ):
        assert (
            c_slot in BACKTRACK_TO_RESULTS[active_service_intent][source_service_intent]
        )
        source_service, source_intent = source_service_intent.split(".")
        for turn in conversation[:usr_turn_idx]:
            if (
                turn.service_results is not None
                and turn.service_call.method == source_intent
            ):
                assert source_service in turn.system_actions_dict
                results = turn.service_results.service_results
                for result in results:
                    if c_slot in result:
                        result_value = [result[c_slot], result[c_slot].lower()]
                        carry_over_values += [v.lower() for v in carry_over_values]
                        if set(result_value).intersection(carry_over_values):
                            return True
    return False


def maybe_remove_intent_actions_from_failure_recovery_turns(
    conversation: Conversation,
    command_collection: CommandCollection | None,
    retry_instruction_format: Literal["call", "assignment"] | None,
    retry_query_instruction_format: Literal["call", "assignment"] | None,
):
    """Remove INFORM_INTENT actions from turns where the user retries the call to
    a failing task.

    Parameters
    ----------
    conversation, command_collection
    retry_instruction_format
        If set to `assignment` then the `INFORM_INTENT` dialogue act
        annotation is removed from frames where transactional (ie confirmed)
        intent are communicated if the agent notified the user of a task
        completion failure.
    retry_query_instruction_format
        If set to `assignment` then the `INFORM_INTENT` dialogue act
        annotation is removed from frames where transactional (ie confirmed)
        intent are communicated if the agent notified the user of a task
        completion failure.
    """

    if retry_instruction_format not in (expected_values := ["assignment", "call"]):
        raise ValueError(
            f"Expected one of {expected_values} for user_task_retry_instruction_format."
            f" Got: {retry_instruction_format}"
        )

    if command_collection is None:
        raise ValueError(
            "Removing user intent annotations from turns following transaction "
            "failures requires SGD command collection access."
        )
    failed_api_calls: dict[int, dict[Literal["service", "method"], str]] = {}
    for idx, turn in enumerate(conversation):
        if turn.author == Author.SYSTEM:
            failed_services = _check_act_presence(
                turn, SystemDialogueAct.NOTIFY_FAILURE
            )
            if failed_services is not None:
                assert len(failed_services) == 1
                [failed_service] = failed_services
                failed_api_calls[idx] = {
                    "service": failed_service,
                    "method": turn.service_call.method,
                }
        else:
            if idx - 1 in failed_api_calls:
                failed_api_info = failed_api_calls[idx - 1]
                service = failed_api_info["service"]
                intent = failed_api_info["method"]
                command = command_collection.get(service, intent)
                if (
                    retry_instruction_format == "assignment"
                    and command.is_transactional
                ) or (
                    retry_query_instruction_format == "assignment"
                    and not command.is_transactional
                ):
                    user_actions = turn.user_actions[service]
                    to_remove = UserAction.model_validate(
                        {
                            "slot": "intent",
                            "act": "INFORM_INTENT",
                            "values": [f"{intent}"],
                            "canonical_values": [f"{intent}"],
                            "metadata": None,
                        },
                    )
                    try:
                        user_actions.remove(to_remove)
                    except ValueError:
                        pass
                    try:
                        turn.user_actions_dict[service].pop(
                            UserDialogueAct.INFORM_INTENT
                        )
                    except KeyError:
                        pass


def build_conversation_with_carryover_annotations(
    dialogue: dict[str, Any], **kwargs
) -> Conversation:
    """Compared to `build_conversation`, this builder:

    1. Populates the `slots_carried_over` property of frames where the intent/service
    changes and arguments relevant to the current API have been mentioned in other
    services or same service, different intents.

    2. Adds a special `CARRY_OVER` action to mark such slots.

    Parameters
    ----------
    kwargs
        user_task_retry_instruction_format
            If `True`, `INFORM_INTENT` actions are removed from turns where the user
            asks the agent to re-try a transaction that failed. As a result, PyTOD
            dialogues will show the changes made by the user while retrying as
            assignments to a previous call.
    """

    # keys: (service|function|values|intent_distance)
    # where function := intent['name'] from SGD schema

    def annotate_slot_carry_over_in_state(
        conversation: Conversation,
        command_collection: CommandCollection,
        slot_relations: SlotRelationsMapping,
    ):
        """Populates the `slots_carried_over` field of the SGD dialogue state tracking
        annotations, to differentiate between those slots that are mentioned by the user
        when they start a new task to those already mentioned in the context of a different
        task in the same service or a task in a different service.

        Returns
        -------
        A mapping containing the indices of the user turns where slots are carried over
        as keys and a dictionary mapping slot names to slot values as values.
        """

        def get_active_service_intent(
            turn: Turn,
            turn_idx: int,
            active_service: ServiceName,
            turn_idx_to_intent: dict[int, str],
        ) -> str:
            this_turn_intent = turn.dialogue_state[active_service].active_intent
            if this_turn_intent == NO_ACTIVE_INTENT:
                # there should be no carry over
                return turn_idx_to_intent[turn_idx - 2]
            return f"{active_service}.{this_turn_intent}"

        def discover_source_apis(
            carryover_slots: dict[str, list[str]],
            usr_turn_idx: int,
            task_change_history: list[TaskChangeInfo],
        ) -> dict[SlotName, SlotCarryOverInfo]:
            """Track slots carried-over to possible source APIs."""

            SPECIAL_CARRYOVER_HANDLING = (
                # check balance after making a payment and check balance again in the same account
                "Banks_1.CheckBalance",
                "Banks_2.CheckBalance",
                "Banks_1.TransferMoney",
                "Banks_2.TransferMoney",
                "Payment_1.MakePayment",
                "Payment_1.RequestPayment",
            )

            def should_stop() -> bool:
                """Stop signal, reached when all carryover slots have been
                tracked to their source."""
                return all(s in source_api_info for s in carryover_slots)

            def get_previous_api_info(
                task_change_info: TaskChangeInfo,
            ) -> dict[str, Union[str, int, list[str]]]:
                """Find out the intent and service at the last user turn
                before a task change."""
                prev_usr_turn_idx = task_change_info.current_turn_idx - 2
                turn_to_service_intent: dict[int, str] = swap_keys_and_values(
                    task_change_info.service_intent_hist
                )
                previous_service_intent = turn_to_service_intent[prev_usr_turn_idx]
                service, function = previous_service_intent.split(".")
                prev_intent_turn = task_change_info.history[-1]
                assert prev_intent_turn.author == Author.USER
                return {
                    "service": service,
                    "function": function,
                    "intent_distance": 1,
                    "values": [],
                    "slot": "",
                }

            def maybe_find_source_slots(
                carryover_slots: dict[SlotName, list[str]],
                active_service_intent: str,
                task_change_info: TaskChangeInfo,
                usr_turn_idx: int,
                handler: Optional[str] = None,
            ) -> dict[SlotName, SlotCarryOverInfo]:
                """Check if the values of the slots carried over are
                 mentioned by either agent in the task encoded in
                 `task_change_info`.

                carryover_slots
                    Slot - value list mapping of the slots we need
                    to find the source slot for.
                active_service_intent
                    The service and intent to which the slots are
                    carried over.
                task_change_info
                    Object storing relevant data about the current
                    task change analysed.
                handler:
                    The name of the calling function. Used for identification
                    of the slot that provides the value for the current slot.
                    If the handler indicates that the update is for carry-over
                    across domains, then slot relations mapping is used to
                    detect which slot to backtrack to in the conversation hist.

                Returns
                -------
                A mapping from slot to dictionaries of the form::

                    {
                        "service": service,
                        "function": function,
                       "intent_distance": 1,
                       "values": []

                    }

                where only `service`, function` and `values` are updated if a slot
                with matching value was found. Carried-over slots are excluded if the
                values list do not match the source slots.

                Notes
                -----
                1. Function has side-effects (destroys) on `carryover_slots`.
                """

                @dispatch_on_value
                def map_carry_over_slot_names(
                    handler: str,
                    carryover_slot: SlotName,
                    active_service_intent: str,
                    prev_service_intent: str,
                    slot_relations: SlotRelationsMapping,
                ) -> Optional[str]:
                    assert handler in [
                        "handle_intent_changes_same_service",
                        None,
                        "handle_related_task_in_history",
                    ], f"Unknown handler {handler}"
                    if handler != "handle_related_task_in_history":
                        # we do not check this condition when
                        # handler = "handle_related_task_in_history"
                        # because the task_change_info points to the task
                        # that follows the related task and so the
                        # assertion would fail
                        active_service = active_service_intent.split(".")[0]
                        previous_service = prev_service_intent.split(".")[0]
                        assert active_service == previous_service
                    return carryover_slot

                @map_carry_over_slot_names.register("handle_cross_service_carry_over")
                def _(
                    handler: str,
                    carryover_slot: SlotName,
                    active_service_intent: str,
                    prev_service_intent: str,
                    slot_relations: SlotRelationsMapping,
                ) -> Optional[str]:
                    active_service, active_intent = active_service_intent.split(".")
                    try:
                        source_apis_info = slot_relations[active_service][
                            active_intent
                        ][carryover_slot]
                    except KeyError:
                        raise KeyError(
                            f"Dialogue: {dialogue['dialogue_id']}. "
                            f"Undefined relation for slot {carryover_slot}. "
                            f"Service: {active_service}. "
                            f"Intent: {active_service_intent} "
                        )
                    for source_api in source_apis_info:
                        maybe_source_service = source_api["service"]
                        maybe_source_intent = source_api["intent"]
                        prev_service, prev_intent = prev_service_intent.split(".")
                        if (
                            maybe_source_service == prev_service
                            and maybe_source_intent == prev_intent
                        ):
                            return source_api["slot"]
                    return

                def get_values_mentioned_before(
                    turn_idx: int, mentions: list[ValuesInfo]
                ) -> list[list[SlotValue]]:
                    """Return the values mentioned before or at `turn_idx`."""
                    return [
                        m.values for m in mentions if m.mentioned_in_turn <= turn_idx
                    ]

                def _intersect_values(
                    expected_values: list[SlotValue],
                    found_values: list[ValuesInfo],
                    usr_turn_idx: int,
                ) -> list[SlotValue]:
                    previous_mentions = get_values_mentioned_before(
                        usr_turn_idx, found_values
                    )[-3:]
                    result = []
                    expected_values = set(expected_values)
                    for m in previous_mentions:
                        result.extend(expected_values.intersection(m))
                    return result

                intersect_values = functools.partial(
                    _intersect_values, usr_turn_idx=usr_turn_idx
                )

                current_service = active_service_intent.split(".")[0]
                updates = {}
                for c_slot, carry_over_values in carryover_slots.items():
                    new_service_slot = c_slot
                    # find out what task was active before the change occurred
                    prev_api_info = get_previous_api_info(task_change_info)
                    prev_service_intent = (
                        f"{prev_api_info['service']}.{prev_api_info['function']}"
                    )
                    # the current slot was not in the state of the previous intent
                    # - the cross-service carry-over handler will deal with that slot
                    if (
                        handler == "handle_intent_changes_same_service"
                        and current_service != prev_api_info["service"]
                    ):
                        continue
                    c_slot = map_carry_over_slot_names(
                        handler,
                        c_slot,
                        active_service_intent,
                        prev_service_intent,
                        slot_relations,
                    )
                    # this happens in multi-domain case, if the slot annotation relations
                    # did not find the API active at the intent change encoded in `task_change_info`
                    # amongst the annotated slot relations list. In this case, we backtrack to the
                    # previous change in history and track the source of the current slot value
                    # there
                    if c_slot is None:
                        continue
                    # see what slots were mentioned by the user
                    try:
                        user_mentioned_slots = conversation.slot_mentions.user[
                            prev_service_intent
                        ]
                    except KeyError:
                        user_mentioned_slots = {}
                    # see what slots were mentioned by the system
                    try:
                        sys_mentioned_slots = conversation.slot_mentions.system[
                            prev_service_intent
                        ]
                    except KeyError:
                        sys_mentioned_slots = {}
                    # see if a slot was carried over to the previous intent
                    inherited_slots = {}
                    if conversation.slots_carried_over is not None:
                        try:
                            inherited_slots: dict[
                                SlotName, list[APISlotCarryOverInfo]
                            ] = conversation.slots_carried_over.carryover_slots[
                                prev_service_intent
                            ]
                        # the slot we are looking for was not inherited in the
                        # previous intent, so we continue backtracking
                        except KeyError:
                            pass
                    # see if the slot is actually an entity property mentioned by the agent
                    information_provided = {}
                    if conversation.information_provided is not None:
                        try:
                            information_provided: dict[
                                SlotName, list[ValuesInfo]
                            ] = conversation.information_provided.provided_slot_values[
                                prev_service_intent
                            ]
                        except KeyError:
                            pass
                    # check if the current slot is mentioned by the user or system
                    if c_slot in user_mentioned_slots:
                        # we only consider the last list of values mentioned for a given slot
                        # (hence -1 indexing)
                        vals = intersect_values(
                            carry_over_values, user_mentioned_slots[c_slot]
                        )
                        # it can happen that the user mentions wildcard ("dontcare")
                        # whereas the system later mentions the actual value - we don't
                        # want this to trigger the assertion below (see train/50_00087)
                        if not vals and c_slot in sys_mentioned_slots:
                            vals = intersect_values(
                                carry_over_values, sys_mentioned_slots[c_slot]
                            )
                    elif c_slot in sys_mentioned_slots:
                        vals = intersect_values(
                            carry_over_values, sys_mentioned_slots[c_slot]
                        )
                    # some carried over slots are themselves inherited (see train/101_00021)
                    elif c_slot in inherited_slots:
                        vals = list(
                            set(carry_over_values).intersection(
                                inherited_slots[c_slot][-1].values
                            )
                        )
                    elif c_slot in information_provided:
                        # some carry-over is from slots which were mentioned by system
                        # to answer user questions (eg train/59_00122)
                        vals = intersect_values(
                            carry_over_values, information_provided[c_slot]
                        )
                    elif mentioned_in_results(
                        conversation,
                        c_slot,
                        carry_over_values,
                        usr_turn_idx,
                        active_service_intent,
                        prev_service_intent,
                    ):
                        vals = deepcopy(carry_over_values)
                    else:
                        # we did not find a matching slot, have to backtrack further
                        # back in the dialogue history
                        continue
                    # store the previous task info along with slot values carried over
                    # the slot was found but the values do not match - we continue
                    # searching for the slot source. This can happen if there is a
                    # related task in history (ie we searched for a flight at some point)
                    # but the values for the carry-over slot do not match. This means
                    # the value was corrected in a more recent task (see train/56_00058 for
                    # examples)
                    if not vals:
                        continue
                    prev_api_info.update(
                        {
                            "values": vals,
                            "slot": c_slot,
                            "new_service_slot": new_service_slot,
                        }
                    )
                    updates[new_service_slot] = prev_api_info
                return updates

            def handle_user_task_retries(
                task_change_history: list[TaskChangeInfo],
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                usr_turn_idx: int,
            ):
                """Handle dialogues where the user retries the task following
                an API call failure

                Examples
                --------
                See dialogue train/14_00119.
                """

                current_task_change_info = task_change_history[-1]
                if current_task_change_info.user_retries_task:
                    updates = maybe_find_source_slots(
                        carryover_slots,
                        active_service_intent,
                        current_task_change_info,
                        usr_turn_idx,
                    )
                    for slot, api_info in updates.items():
                        source_api_info[slot].append(api_info)

            def handle_repeated_intent_services(
                task_change_history: list[TaskChangeInfo],
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                usr_turn_idx: int,
            ):
                """Special backtracking case for banks and payments where the most recent
                intent is assumed to be the slot source. We *do not* backtrack to the part
                of the conversation where the same task was active as this does not make
                sense for these services.

                Parameters
                ----------
                usr_turn_idx
                    The index of the user turn at which slots are carried over.
                """

                current_task_change_info = task_change_history[-1]
                service_intent_hist = current_task_change_info.service_intent_hist
                active_service_intent = current_task_change_info.active_service_intent

                def only_one_other() -> bool:
                    """Check of the current task is the second task in the
                    current dialogue."""
                    return (
                        len(service_intent_hist) == 2
                        and active_service_intent in service_intent_hist
                    )

                if (
                    only_one_other()
                    and active_service_intent in SPECIAL_CARRYOVER_HANDLING
                ):
                    updates = maybe_find_source_slots(
                        carryover_slots,
                        active_service_intent,
                        current_task_change_info,
                        usr_turn_idx,
                    )
                    for slot, api_info in updates.items():
                        source_api_info[slot].append(api_info)

            def handle_intent_changes_same_service(
                task_change_history: list[TaskChangeInfo],
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                usr_turn_idx: int,
            ):
                """Handle dialogues where the intent changes but the service
                does not.

                Parameters
                ----------
                usr_turn_idx
                    The index of the user turn at which slots are carried over.
                """
                active_service_intent = task_change_history[-1].active_service_intent
                if task_change_history[-1].intent_changed_in_service:
                    backtrack_to_source(
                        active_service_intent,
                        source_api_info,
                        task_change_history,
                        usr_turn_idx,
                        handler="handle_intent_changes_same_service",
                    )

            def backtrack_to_source(
                active_service_intent: str,
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                task_change_history: list[TaskChangeInfo],
                usr_turn_idx: int,
                handler: str,
            ):
                """Trace back the slots in `carryover_slots` to their source.
                The source can be a user/system mention or a slot in the
                previous intent that itself inherited its value from a slot
                explicitly mentioned by the agent/user in a previous intent."""

                change_index, intent_distance = -1, 1
                remaining_steps = len(task_change_history)
                while remaining_steps > 0 and not should_stop():
                    current_task_change_info = task_change_history[change_index]
                    update = maybe_find_source_slots(
                        carryover_slots,
                        active_service_intent,
                        current_task_change_info,
                        usr_turn_idx,
                        handler=handler,
                    )
                    for slot, api_info in update.items():
                        api_info["intent_distance"] = intent_distance
                        source_api_info[slot].append(api_info)
                        carryover_slots.pop(slot)
                    change_index -= 1
                    intent_distance += 1
                    remaining_steps -= 1

            def handle_related_task_in_history(
                task_change_history: list[TaskChangeInfo],
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                usr_turn_idx: int,
            ):
                """Handles dialogues where the history contains a related
                task from the same service that was completed before the previous
                task, which is from another service (eg the user first searches event
                tickets, then transport, then books the event tickets).

                Parameters
                ----------
                usr_turn_idx
                    The index of the user turn at which slots are carried over.
                """
                active_service = task_change_history[-1].active_service
                # find if the service was invoked before the previous user turn
                last_invocation_turn_idx = get_last_service_invocation(
                    active_service, usr_turn_idx, task_change_history
                )
                # the service has not been invoked before, so this
                # handler relinquishes control
                if last_invocation_turn_idx is None:
                    return
                assert (
                    last_invocation_turn_idx % 2 == 0
                ), "Last invocation should be a user turn"
                if (
                    task_change_history[-1].service_changed
                    and last_invocation_turn_idx is not None
                ):
                    multi_frame_state = conversation[
                        last_invocation_turn_idx
                    ].dialogue_state
                    assert active_service in multi_frame_state
                    related_task = get_task_name(  # noqa
                        multi_frame_state[active_service]
                    )
                    changes = [
                        t
                        for t in task_change_history
                        if t.current_turn_idx == last_invocation_turn_idx + 2
                    ]
                    assert len(changes) == 1
                    change_index = task_change_history.index(changes[0])
                    intent_distance = len(task_change_history) - change_index
                    update = maybe_find_source_slots(  # noqa
                        carryover_slots,
                        task_change_history[-1].active_service_intent,
                        changes[0],
                        usr_turn_idx,
                        handler="handle_related_task_in_history",
                    )
                    for slot, api_info in update.items():
                        api_info["intent_distance"] = intent_distance
                        source_api_info[slot].append(api_info)
                        carryover_slots.pop(slot)

            def handle_cross_service_carry_over(
                task_change_history: list[TaskChangeInfo],
                source_api_info: dict[SlotName, list[SlotCarryOverInfo]],
                usr_turn_idx: int,
            ):
                """Handle dialogues where the carry-over is from a task in a
                different domain.

                Parameters
                ----------
                usr_turn_idx
                    The index of the user turn at which slots are carried over.
                """

                def carryover_slots_not_in_related_intent_state() -> bool:
                    return all(
                        s
                        not in conversation[usr_turn_idx - 2]
                        .dialogue_state[active_service]
                        .slot_values
                        for s in carryover_slots
                    )

                # this is the final handler - note that it can also be called when
                # the service has not changed if the value was carried over from
                # a previous intent to the current intent and the intent that just
                # completed is from the same service but does not carry over the
                # slot we are looking for
                active_service = task_change_history[-1].active_service
                assert task_change_history[-1].service_changed or (
                    task_change_history[-1].intent_changed_in_service
                    and carryover_slots_not_in_related_intent_state()
                )
                active_service_intent = task_change_history[-1].active_service_intent
                backtrack_to_source(
                    active_service_intent,
                    source_api_info,
                    task_change_history,
                    usr_turn_idx,
                    handler="handle_cross_service_carry_over",
                )

            source_api_info = defaultdict(list)
            handlers = [
                handle_cross_service_carry_over,
                handle_related_task_in_history,
                handle_intent_changes_same_service,
                handle_repeated_intent_services,
                handle_user_task_retries,
            ]
            while not should_stop():
                next_handler = handlers.pop()
                next_handler(task_change_history, source_api_info, usr_turn_idx)
                if not handlers and not should_stop():
                    # this is a temporary patch for harvesting
                    # app invocations from multi-domain conversations
                    logger.info(
                        f"Carryover annotation in multi-domain conversation {conversation.id} "
                        f"with services {conversation.services} is incomplete"
                    )
                    raise AssertionError(
                        f"Could not backtrack in {conversation.id}, turn {usr_turn_idx}. "
                        f"Task: {active_service_intent}. "
                        f"User turn: {conversation[usr_turn_idx].text}. "
                        f"Outstanding slots: {carryover_slots}"
                    )
            try:
                assert should_stop()
            except AssertionError:
                logger.debug(
                    f"Dialogue ({conversation.id}). Failed to discover same-service value. "
                )
                raise AssertionError
            try:
                assert source_api_info
            except AssertionError:
                assert False, "Something went wrong during slot-value mining"
            return dict(source_api_info)

        def mark_for_carryover(task_change_info: TaskChangeInfo) -> bool:
            """Marks the current turn as a potential candidate for annotation of
            slot relations. This is done on the basis of three attributes (`task_change_fields`)
            stored in the `TaskChangeInfo` object:

                * "intent_changed_in_service": `True` when intent changes in the same domain
                * "service_changed": `True` when the user changes task, and the domain also changes
                * "discuss_old_service_again": `True` when the user changes task to a task from a
                 domain active before, but the task just completed is from a different domain.
                * "user_retries_task" : `True` when the user asks the agent to re-try a task
                following a failure notification (nb: user changes constraints in this case)
            """
            flags = (
                getattr(task_change_info, f)
                for f in task_change_info.task_change_fields
            )
            return any(flags)

        def maybe_filter_slots(
            active_service,
            possible_carry_over_slots: Optional[
                dict[SlotName, list[Union[dict[str, Any], str]]]
            ],
            task_change_info: TaskChangeInfo,
            command_collection: CommandCollection,
        ):
            """Ensure slots marked as carried over slots are args of current API.
            Filtering happens in-place but side effects are isolated because the
            objects generating `possible_carry_over_slots` take deep copies of
            eg value lists and so on."""
            if possible_carry_over_slots is None or not possible_carry_over_slots:
                return

            active_intent = task_change_info.active_service_intent.split(".")[1]
            current_command = command_collection.get(active_service, active_intent)
            to_remove = {
                slot
                for slot in possible_carry_over_slots
                if not is_arg(slot, current_command)
            }
            for not_arg_slot in to_remove:
                possible_carry_over_slots.pop(not_arg_slot)

        previous_services = {}
        turn_idx_to_intent, intent_to_turn_id = {}, defaultdict(list)
        assert (
            command_collection is not None
        ), "Need command collection to generate carryover annotations"
        assert (
            slot_relations is not None
        ), "Slot relations must be specified to generate cross-domain carryover annotations"
        task_change_history = []
        slots_carried_over = defaultdict(dict)
        for prefix in conversation.prefix_conversations(last_speaker=Author.USER):
            usr_turn_idx = len(prefix.turns) - 1
            assert usr_turn_idx % 2 == 0
            active_service = get_active_service(prefix)
            user_turn = prefix.turns[-1]
            active_service_intent = get_active_service_intent(
                user_turn, usr_turn_idx, active_service, turn_idx_to_intent
            )
            task_change_info = gather_task_change_info(
                dialogue,
                conversation,
                active_service,
                usr_turn_idx,
                previous_services,
                active_service_intent,
                dict(intent_to_turn_id),
            )
            if mark_for_carryover(task_change_info):
                task_change_history.append(task_change_info)
                # detect slots in the state that were not mentioned
                # in the current turn
                prev_mentioned_slots = get_prev_mentioned_slots(
                    user_turn, active_service
                )
                # filter out slots that are not relevant to the current API
                maybe_filter_slots(
                    active_service,
                    prev_mentioned_slots,
                    task_change_info,
                    command_collection,
                )
                if prev_mentioned_slots is not None and prev_mentioned_slots:
                    carried_over_slots = discover_source_apis(
                        deepcopy(prev_mentioned_slots),
                        usr_turn_idx,
                        task_change_history,
                    )
                    carry_over_data = parse_as_carry_over_type(carried_over_slots)
                    conversation[usr_turn_idx].dialogue_state[
                        active_service
                    ].slots_carried_over = carry_over_data
                    slots_carried_over[active_service_intent].update(carry_over_data)
                    conversation.slots_carried_over = SlotsCarriedOver.model_validate(
                        {"carryover_slots": dict(slots_carried_over)}
                    )

            turn_idx_to_intent[usr_turn_idx] = active_service_intent
            intent_to_turn_id[active_service_intent].append(usr_turn_idx)
            previous_services[active_service] = usr_turn_idx

    def annotate_slot_carry_over_as_action(conversation: Conversation):
        """Add special actions annotated with the `CARRY_OVER` dialogue act
        to turns where the intent/service changes if there are any slot values
        in the dialogue state that were mentioned in the dialogue history during
        different services or intents.
        """

        def carry_over_action_factory():
            return {
                "slot": "",
                "act": UserDialogueAct.CARRY_OVER,
                "values": [],
                "canonical_values": [],
                "metadata": None,
            }

        @functools.singledispatch
        def populate_action_dict(
            info: Union[list[APISlotCarryOverInfo], SlotCarryOverValues],
            action_draft: dict[str, Any],
        ) -> dict[str, Any]:
            raise TypeError(f"Unknown type for carry over object {type(info)}")

        @populate_action_dict.register(list)
        def _(info: list[APISlotCarryOverInfo], action_draft: dict[str, Any]):
            assert all(isinstance(e, APISlotCarryOverInfo) for e in info)
            vals = {tuple(el.values) for el in info}
            try:
                assert len(vals) == 1
                action_draft["values"] = info[0].values
            except (
                AssertionError
            ):  # this should be a semantically equivalent value, we take the most recent value
                action_draft["values"] = info[0].values
            action_draft["metadata"] = {}
            action_draft["metadata"]["carryover"] = info

        @populate_action_dict.register(SlotCarryOverValues)
        def _(info: SlotCarryOverValues, action_draft: dict[str, Any]):
            action_draft["values"] = info.values

        for turn in conversation:
            if turn.author == Author.USER:
                for service, service_state in turn.dialogue_state.items():
                    carry_over = service_state.slots_carried_over
                    if carry_over is not None:
                        carry_over_actions = []
                        for slot, carry_over_info in carry_over.items():
                            new_action = carry_over_action_factory()
                            new_action["slot"] = slot
                            populate_action_dict(carry_over_info, new_action)
                            carry_over_actions.append(
                                UserAction.model_validate(new_action)
                            )
                        turn.user_actions[service].extend(carry_over_actions)

    conversation = build_conversation(dialogue, **kwargs)
    logger.info(f"Annotating conversation, {conversation.id}")
    command_collection = kwargs.get("command_collection")
    maybe_remove_intent_actions_from_failure_recovery_turns(
        conversation,
        command_collection,
        kwargs.get("user_task_retry_instruction_format"),
        kwargs.get("user_query_retry_instruction_format"),
    )
    slot_relations = kwargs.get("slot_relations")

    assert isinstance(slot_relations, dict)
    annotate_slot_carry_over_in_state(conversation, command_collection, slot_relations)
    annotate_slot_carry_over_as_action(conversation)
    # mapping from dialogue act to action to facilitate PyTOD dialogue generation
    for turn in conversation:
        add_actions_dict(turn)

    return conversation
