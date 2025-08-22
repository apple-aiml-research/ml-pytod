#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""A collection of predicates checking what the user did in a turn.
Utilities that operate over user turns."""


from collections import defaultdict
from itertools import chain
from typing import Optional

from pytod.pytod_types.aliases import SlotName
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    Turn,
    UserAction,
    UserDialogueAct,
)
from pytod.sgd_analysis_utils import (
    Behaviour,
    _gather_behaviours,
    count_constraints,
    count_questions_asked,
    no_goal_changes,
)


def _gather_user_behaviours(turn: Turn, act: UserDialogueAct) -> list[Behaviour]:
    """Return a list of behaviour objects, which encode which semantic frames
    in `turn` contain at least an action annotated with the `act` dialogue act."""
    assert isinstance(act, UserDialogueAct)
    return _gather_behaviours(turn, act)


def is_req_alts(turn: Turn) -> Behaviour:
    """Check if the user requested alternatives in this turn. These turns are annotated with
    the `REQUEST_ALTS` dialogue act in SGD."""
    # req_alts is a user act
    if turn.user_actions is None:
        assert turn.author == Author.SYSTEM
        return Behaviour(detected=False)
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.REQUEST_ALTS)
    # the user only searches one thing at a time, so alternatives cannot be
    # requested in multiple domains
    assert len(behaviours) == 1
    return behaviours[0]


def is_req_alts_with_changes(turn: Turn) -> Behaviour:
    """Check if the user requested alternatives but changed constraints. This is annotated
    with `REQUEST_ALTS` and one or more `INFORM(slot=value)` actions."""
    if turn.user_actions is None:
        assert turn.author == Author.SYSTEM
        return Behaviour(detected=False)

    if is_req_alts(turn):
        user_actions = list(*turn.user_actions.values())
        return Behaviour(
            detected=len(user_actions) > 1, service=list(turn.user_actions.keys())[0]
        )
    return Behaviour(detected=False)


def user_req_alt(turn: Turn) -> Behaviour:
    """User just requested alternatives, without updating constraints."""

    req_alts = is_req_alts(turn)
    req_alts_changes = is_req_alts_with_changes(turn)
    assert req_alts.service == req_alts_changes.service
    return Behaviour(
        detected=req_alts and not req_alts_changes, service=req_alts.service
    )


def user_req_alternatives(conversation: Conversation) -> bool:
    """Detects single intent conversations where the user requests alternatives
    without changing constraints.

    Conversations where this behaviour happens but the user also changes constraints
    are ignored.
    """

    turn_type = "no_alternatives"
    for prefix in conversation.prefix_conversations(last_speaker=Author.USER):
        last_turn = prefix.turns[-1]
        if is_req_alts(last_turn):
            if is_req_alts_with_changes(last_turn):
                return False
            turn_type = "req_alts"
    return turn_type == "req_alts"


def user_informs_constraints(turn: Turn) -> Behaviour:
    """Detect if the user has provided slot values in the current turn."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.INFORM)
    # SGD does not have multi-intent utterances, so this act should
    # never occur in semantic frames annotating different services
    assert len(behaviours) == 1
    return behaviours[0]


def user_negates(turn: Turn) -> Behaviour:
    """Check if the user negates a system offer in the current turn."""

    behaviours = _gather_user_behaviours(turn, UserDialogueAct.NEGATE)
    # this act annotates situations where the user does not accept a slot
    # value proposed by the system. Slots offered only relate to the
    # single active intent. Hence, only one semantic frame should
    # have this annotation.
    assert len(behaviours) == 1
    return behaviours[0]


def user_affirms(turn: Turn) -> Behaviour:
    """Check if the user affirms (aka accepts) the system offer in the
    current turn."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.AFFIRM)
    # AFFIRM either means the user confirms a transaction (follows CONFIRM dialogue
    # acts issued by the system in the previous turn) or that the user confirms
    # a slot value proposed by system during slot filling.
    # The two behaviours are incompatible and so only one of them should occur
    # in a single frame.
    assert len(behaviours) == 1
    return behaviours[0]


def user_ends_conversation(turn: Turn) -> Behaviour:
    """Check if the user ends the conversation in the current turn."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.GOODBYE)
    # only one task is active when the user ends the conversation
    assert len(behaviours) == 1
    return behaviours[0]


def user_states_intention(turn: Turn) -> Behaviour:
    """Check if the user states their intent in the current turn."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.INFORM_INTENT)
    # there are no multi-intent turns in SGD
    assert len(behaviours) == 1
    return behaviours[0]


def user_accepts_suggestion(turn: Turn) -> Behaviour:
    """Check if the user accepts a task suggested by the agent."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.AFFIRM_INTENT)
    # agent only ever suggests on task in SGD
    assert len(behaviours) == 1
    return behaviours[0]


def user_declines_suggestion(turn: Turn) -> Behaviour:
    """Check if the user declines a task suggested by the agent."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.NEGATE_INTENT)
    # there should be only one task proposed
    assert len(behaviours) == 1
    return behaviours[0]


def user_selects_entity(turn: Turn) -> Behaviour:
    """Check if the user selected an entity in the current turn.
    This is annotated with a `SELECT` action that has no arguments."""
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.SELECT)
    # only one query intent can be active, so the user is
    # expected to select a single entity per turn
    assert len(behaviours) == 1
    if behaviours[0]:
        for service, service_actions in turn.user_actions_dict.items():
            if UserDialogueAct.SELECT in service_actions:
                [selection] = service_actions[UserDialogueAct.SELECT]
                intent = turn.dialogue_state[service].active_intent
                if selection.slot:
                    return Behaviour(detected=False, service=service, intent=intent)
                else:
                    return Behaviour(detected=True, service=service, intent=intent)
    return behaviours[0]


def user_selects_from_system_options(turn: Turn) -> Behaviour:
    """Detects if the user selects an option offered by the system
    in the previous user turn. This is annotated with a `SELECT` action
    parametrise by the slot name used for the selection and its value.

    Example
    -------
    system: You want to watch Dumbo or Hellboy.
    user: Hellboy.
    """
    for service, service_actions in turn.user_actions_dict.items():
        try:
            [select_action] = service_actions[UserDialogueAct.SELECT]
            assert isinstance(select_action, UserAction)
            if select_action.slot:
                return Behaviour(
                    detected=True,
                    service=service,
                    intent=turn.dialogue_state[service].active_intent,
                )
        except KeyError:
            continue
    return Behaviour(detected=False)


def user_does_not_specify_next_task(turn: Turn) -> Behaviour:
    """Check if the user does not explicitly end the conversation in the
    current turn but no intent is active."""
    behaviour = Behaviour(
        detected=_check_user_act_present(turn, UserDialogueAct.THANK_YOU)
        and len(list(chain(*turn.user_actions.values()))) == 1
    )
    if behaviour.detected:
        behaviour.service = next(iter(turn.user_actions))
    return behaviour


def user_asked_multiple_questions(turn: Turn) -> Behaviour:
    """Check if the user asked multiple questions in a single turn.

    Example:

        _Does this restaurant have a live band and is it expensive?_
    """
    actions = turn.user_actions
    if actions is None:
        return Behaviour(detected=False)

    behaviours = []
    for service, service_actions in actions.items():
        if count_questions_asked(service_actions) > 1:
            behaviours.append(Behaviour(detected=True, service=service))
    # the user only asks questions about the results of the current query
    #  or transaction.
    assert len(behaviours) <= 1
    if behaviours:
        return behaviours[0]
    return Behaviour(detected=False)


def user_requests_information(turn: Turn) -> Behaviour:
    """Check if the user asked at least one question in the current turn."""
    actions = turn.user_actions
    if actions is None:
        return Behaviour(detected=False)
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.REQUEST)
    # either no questions asked or questions asked about one entity
    assert len(behaviours) == 1
    return behaviours[0]


def turn_carries_over_arguments(turn: Turn) -> Behaviour:
    """Check if some of the slots in the current turn have values that are
    inherited from slots mentioned during different intents (same service) or
    different services.

    Notes
    -----
    This behaviour should only occur in turns where the intent changes.
    """
    assert turn.author == Author.USER
    behaviours = _gather_user_behaviours(turn, UserDialogueAct.CARRY_OVER)
    # this would mean that there are multiple services active at the current
    # turn and that in both of them slot values are carried over from previous
    # tasks - this should never happen in SGD
    assert len(behaviours) == 1
    return behaviours[0]


def user_changes_goal(conversation: Conversation) -> bool:
    """Detects single intent conversations where the user changes a constraint
    during search. The change may be annotated with a REQ_ALTS dialogue act
    (the user requests the alternative explicitly) or not (the user just changes
    the constraint when providing more info).
    """

    for prefix in conversation.prefix_conversations(last_speaker=Author.USER):
        last_turn = prefix.turns[-1]
        if is_req_alts_with_changes(last_turn):
            assert user_informs_constraints(last_turn)
            return True

    if no_goal_changes(conversation):
        return False
    assert not no_goal_changes(conversation)
    return True


def get_goal_change_turn_indices(
    conversation: Conversation,
) -> Optional[dict[int, list[SlotName]]]:
    """Turn-level annotation of goal changes.

    Returns
    -------
    A mapping from turn index to the slots the user corrected at that turn.
    """
    if len(conversation.services) > 1:
        raise NotImplementedError(
            "Goal change tracking is not yet supported for multi-domain conversations"
        )
    if not user_changes_goal(conversation):
        return
    slot_info_counts = {
        slot: count
        for slot, count in count_constraints(conversation).items()
        if count > 1
    }
    already_communicated = set()
    goal_changes: defaultdict[int, list[SlotName]] = defaultdict(list)
    for i, turn in enumerate(conversation):
        if user_informs_constraints(turn):
            for service, service_actions in turn.user_actions.items():
                # we exclude intent changes - should be handled separately
                slots_informed = [a.slot for a in service_actions if a.slot != "intent"]
                for s in slots_informed:
                    slot_info_counts[s] -= 1
                    if slot_info_counts[s] == 0:
                        slot_info_counts.pop(s)
                    if s in already_communicated:
                        goal_changes[i].append(s)
                    else:
                        already_communicated.add(s)
    return dict(goal_changes) or None


def get_turns_matching_service_intent(
    conversation: Conversation,
    service: str,
    intent: str,
    exclude_carryover_turns: bool = True,
    stop_at_first_turn: bool = True,
    exclude_service_boundary_turns: bool = True,
) -> list[Turn]:
    """Returns the turns in a conversation where `service`
    and `intent` are active.

    Parameters
    ----------
    conversation
    service, intent
        The service and intent to match
    exclude_carryover_turns
        Excludes turns where some arguments are carried over
        from other services or conversations.
    stop_at_first_turn
        Only return the first turn in the conversation where
        `service` and `intent` are active
    exclude_service_boundary_turns
        Do not match turns where two services are active. These
        are turns where the user initiates a new task - there
        are no multi-intent turns in SGD.
    """

    to_return = []
    for turn in conversation:
        skip_turn = any(
            (
                turn.author == Author.SYSTEM,
                turn.author == Author.USER
                and exclude_carryover_turns
                and turn_carries_over_arguments(turn),
                not user_states_intention(turn),
                # when the service changes, the intent just complete
                # and the new intent are annotated as active in each
                # service frame. This would erroneously lead to matching
                # a turn where the user initiates a new task
                exclude_service_boundary_turns
                and turn.author == Author.USER
                and len(turn.user_actions) > 1,
            )
        )
        if skip_turn:
            continue
        if (
            service in turn.dialogue_state
            and turn.dialogue_state[service].active_intent == intent
        ):
            if stop_at_first_turn:
                return [turn]
            to_return.append(turn)

    return to_return


def get_first_intent_change_turn_index(conversation: Conversation) -> Optional[int]:
    """Return the index of the first intent change in a conversation."""

    for i, turn in enumerate(conversation[1:], start=1):
        assert isinstance(turn, Turn)
        if user_states_intention(turn):
            return i


def _check_user_act_present(turn: Turn, act: UserDialogueAct) -> bool:
    """Check if a given dialogue act is present in any frame of turn."""
    if turn.author == Author.USER:
        actions = chain(*turn.user_actions.values())
        return any(action.act == act for action in actions)
    return False
