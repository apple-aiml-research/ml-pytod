#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple, Optional, Union

from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    ServiceName,
    SystemAction,
    SystemDialogueAct,
    Turn,
    UserAction,
    UserDialogueAct,
)

logger = logging.getLogger(__name__)

_NO_ACTIVE_INTENT = "NONE"


@dataclass
class Behaviour:
    detected: bool
    # filled for both usr/sys turns
    service: Optional[str] = None
    # only filled for usr turns
    intent: Optional[str] = None

    def __bool__(self):
        return self.detected


class FailedCallTurn(NamedTuple):
    turn: Turn
    index: int


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


def _gather_behaviours(
    turn: Turn, act: Union[UserDialogueAct, SystemDialogueAct]
) -> list[Behaviour]:
    """Utility to check the presence of `act` in the semantic annotations of each frame of `turn`"""
    behaviours = []
    services = _check_act_presence(turn, act)
    if services is None:
        return [Behaviour(detected=False)]
    for service in services:
        intent = None
        if turn.author == Author.USER:
            intent = turn.dialogue_state[service].active_intent
        behaviours.append(Behaviour(detected=True, service=service, intent=intent))

    return behaviours


def count_constraints(dialogue: Conversation) -> Counter:
    """Counts the number of times each slot was provided in a single intent dialogue."""
    slots_mentioned = []
    for i, turn in enumerate(dialogue.turns):
        if turn.author != Author.USER:
            continue
        user_actions = list(*turn.user_actions.values())
        slots_mentioned += [
            action.slot
            for action in user_actions
            if (
                action.act == UserDialogueAct.INFORM
                or (action.act == UserDialogueAct.SELECT and action.slot)
            )
        ]
        # handle slot value confirmation
        # ("agent: Would you like to take the bus today? // user: yes!")
        # this applies for multi-domain dialogues only
        if any(a.act == UserDialogueAct.AFFIRM for a in user_actions):
            prev_sys_turn = dialogue[i - 1]
            sys_actions = list(*prev_sys_turn.system_actions.values())
            sys_proposed_vals_for = [
                a.slot
                for a in sys_actions
                if a.act == SystemDialogueAct.REQUEST and a.values
            ]
            for slot in sys_proposed_vals_for:
                if slot not in slots_mentioned:
                    slots_mentioned.append(slot)
    return Counter(slots_mentioned)


def no_goal_changes(dialogue: Conversation) -> bool:
    """Determines if the user changed goal in a single intent dialogue
    by counting constraints."""
    slot_counts = count_constraints(dialogue)
    return slot_counts.total() == len(slot_counts)


def count_goal_changes(dialogue: Conversation) -> int:
    """Counts how many times the user changed a slot value."""
    slot_counts = count_constraints(dialogue)
    return slot_counts.total() - len(slot_counts)


def _check_system_act_present(turn: Turn, act: SystemDialogueAct) -> bool:
    """Returns `True` if a given dialogue act is present in any frame of turn."""
    if turn.author == Author.SYSTEM:
        actions = chain(*turn.system_actions.values())
        return any(action.act == act for action in actions)
    return False


def count_questions_asked(actions: Union[list[UserAction], list[SystemAction]]) -> int:
    """Count how many questions are asked in a given turn by either the agent or the user."""
    if actions is None or len(actions) == 0:
        return 0
    request_act = (
        UserDialogueAct.REQUEST
        if isinstance(actions[0], UserAction)
        else SystemDialogueAct.REQUEST
    )
    n_questions = [1 if a.act == request_act else 0 for a in actions]
    return sum(n_questions)


def infer_call_occurred(turn: Turn) -> bool:
    if turn.author == Author.USER:
        logger.warning("Function infer_call_occurred called on user turn...")
        return False
    expected_acts = [
        SystemDialogueAct.NOTIFY_FAILURE,
        SystemDialogueAct.NOTIFY_SUCCESS,
        SystemDialogueAct.OFFER,
    ]
    this_turn_acts = set(a.act for a in chain(*turn.system_actions.values()))
    return this_turn_acts.intersection(expected_acts) or False


def call_returns_results(turn: Turn) -> bool:
    """Check if a service call in `turn` returned results."""
    return (
        turn.service_call is not None and len(turn.service_results.service_results) > 0
    )


def count_result_turns(conversation: Conversation, exclude_last: bool = False) -> int:
    """Counts the number of times a call returned results in a conversation.

    Parameters
    ----------
    exclude_last
        Whether to count the calls in the last agent turn.
    """

    if exclude_last:
        if conversation.turns[-1].author == Author.SYSTEM:
            dialogue = conversation.turns[:-1]
        else:
            dialogue = conversation.turns[:-2]
    else:
        dialogue = conversation.turns
    count = 0
    for turn in dialogue:
        if turn.author == Author.USER:
            continue
        if call_returns_results(turn):
            count += 1
    return count


def is_single_action(turn: Turn) -> bool:
    """Detects dialogue turns where either agent takes a single action."""
    actions = turn.user_actions if turn.author == Author.USER else turn.system_actions
    if len(actions) > 1:
        assert all(len(service_actions) > 1 for _, service_actions in actions.items())
        return False
    actions = list(chain(*actions.values()))
    return len(actions) == 1


def get_service(turn: Turn) -> ServiceName:
    """Get the name of the service annotating the current turn.

    Raises
    ------
    ValueError
        If the turn is annotated with multiple semantic frames.
    """
    if len(turn.system_actions) > 1:
        raise ValueError(
            "Multiple semantic frames annotate the turn, service is not unique"
        )
    return list(turn.system_actions.keys())[0]


def get_active_intents(
    conversation: Conversation, include_none: bool = True
) -> set[str]:
    """Returns all active intents from the conversation."""
    intents = set()
    for turn in conversation:
        if turn.author == Author.USER:
            state = turn.dialogue_state
            for service, service_state in state.items():
                intent = service_state.active_intent
                if intent == _NO_ACTIVE_INTENT and not include_none:
                    continue
                intents.add(intent)
    return intents
