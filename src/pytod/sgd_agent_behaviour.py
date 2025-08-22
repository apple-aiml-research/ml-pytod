#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""A collection of predicates checking what the agent did in a turn.
Utilities that operate over agent turns."""

from collections import Counter
from typing import Optional

from pytod.pytod_types.aliases import ServiceName
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    SystemDialogueAct,
    Turn,
    get_active_service,
)
from pytod.sgd_analysis_utils import (
    Behaviour,
    FailedCallTurn,
    _gather_behaviours,
    count_questions_asked,
)
from pytod.sgd_policy_assertions import (
    assert_sys_answered_questions,
    assert_user_asked_questions,
)


def _gather_system_behaviours(turn: Turn, act: SystemDialogueAct) -> list[Behaviour]:
    """Return a list of behaviour objects, which encode which semantic frames
    in `turn` contain at least an action annotated with the `act` dialogue act."""
    assert isinstance(act, SystemDialogueAct)
    return _gather_behaviours(turn, act)


def agent_notifies_failure(turn: Turn) -> Behaviour:
    """Checks if the agent informs the user a transaction
    failed or no results were available."""

    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.NOTIFY_FAILURE)
    # only one task is active, so only one failure can happen
    assert len(behaviours) == 1
    if turn.author == Author.SYSTEM:
        return behaviours[0]
    return Behaviour(detected=False)


def agent_makes_offer(turn: Turn) -> Behaviour:
    """Checks if the agent presents information about an entity.
    This occurs when the agent calls the API but also when API calls
    do not succeed and the agent offers alternatives.
    """
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.OFFER)
    # only one task is active, so only one failure can happen
    assert len(behaviours) == 1
    return behaviours[0]


def agent_suggests_next_task_turn(turn: Turn) -> Behaviour:
    """Check if the agent took initiative in the current turn
    by offering the user an intent. Such turns are annotated with
    `OFFER_INTENT` dialogue act"""
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.OFFER_INTENT)
    return behaviours[0]


def agent_suggests_next_task(conversation: Conversation) -> Optional[set[ServiceName]]:
    """Returns the set of services where the agent took initiative
    in the conversation in any conversation turn.
    """

    initiative = set()
    for turn_idx, turn in enumerate(conversation):
        if agent_suggests_next_task_turn(turn):
            initiative.add(
                get_active_service(Conversation(turns=conversation[:turn_idx]))
            )
    return initiative or None


def agent_prompts_for_more_help(turn: Turn) -> Behaviour:
    """Check if the system prompts the user to state the next task.
    This only happens when the user active intent is `NONE` (eg, when the user
    just thanks the agent for help with completing the next task.


    Example
    -------

    system: I booked you at the Gardenia, enjoy your meal.
    user: Thanks.
    system: Is there anything else you'd like help with?
    ...
    """
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.REQ_MORE)
    assert len(behaviours) == 1
    return behaviours[0]


def agent_informs_task_status(turn: Turn) -> Behaviour:
    """Whether the agent informed the task status (success or failure) in this turn."""

    sys_actions = turn.system_actions
    if sys_actions is None:
        return Behaviour(detected=False)
    behaviours = []
    for service, actions in sys_actions.items():
        if any(
            (
                a.act == SystemDialogueAct.NOTIFY_FAILURE
                or a.act == SystemDialogueAct.NOTIFY_SUCCESS
            )
            for a in actions
        ):
            behaviours.append(Behaviour(detected=True, service=service))
    # only one API call is made per system turn
    assert len(behaviours) <= 1
    if behaviours:
        return behaviours[0]
    return Behaviour(detected=False)


def agent_informs_slot_values(turn: Turn) -> Behaviour:
    """Whether the agent informs at least one question in the current turn."""

    sys_actions = turn.system_actions
    if sys_actions is None:
        return Behaviour(detected=False)
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.INFORM)
    # user does not ask questions about multiple entities at once
    assert len(behaviours) == 1
    return behaviours[0]


def agent_informs_task_status_and_answers_questions(
    conversation: Conversation,
) -> Optional[set[ServiceName]]:
    """Returns the set of services where the agent informed the user of
    whether the task succeeded or not and answered questions in a single turn.

    Examples
    --------
    An example of such a turn is:

     _system: Your booking was made without errors, but unfortunately they do not have live music._

    """

    services = set()
    for turn_idx, turn in enumerate(conversation):
        if agent_informs_task_status(turn) and agent_informs_slot_values(turn):
            assert len(turn.system_actions) == 1  # single frame turn
            assert turn.service_call is not None
            service = turn.service_call.service
            assert_user_asked_questions(turn, conversation[turn_idx - 1], service)
            assert_sys_answered_questions(turn, conversation[turn_idx - 1], service)
            services.add(service)

    return services


def agent_requests_confirmation(turn: Turn) -> Behaviour:
    """Check if the agent asks the user to confirm at least one
    value in this turn."""
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.CONFIRM)
    # user only talks about one task in a given turn, so confirmation is requested
    # for a single task
    assert len(behaviours) == 1
    return behaviours[0]


def agent_informs_entity_count(turn: Turn) -> Behaviour:
    """Check if the agent informs the user on how many results
    their search returned."""
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.INFORM_COUNT)
    # a turn can only contain results from one query
    assert len(behaviours) == 1
    return behaviours[0]


def agent_answers_questions(
    conversation: Conversation,
) -> Optional[set[ServiceName]]:
    """Returns the set of services where the agent responded to multiple
    user questions at once."""

    def get_service_questions_asked(user_turn: Turn) -> ServiceName:
        user_actions = user_turn.user_actions
        for service, service_actions in user_actions.items():
            try:
                assert count_questions_asked(service_actions) > 1
                return service
            except AssertionError:
                assert count_questions_asked(service_actions) == 0

    services = set()
    for turn_idx, turn in enumerate(conversation):
        if agent_informs_slot_values(turn):
            turn_service = get_service_questions_asked(conversation[turn_idx - 1])
            services.add(turn_service)

    return services or None


def agent_makes_system_call(turn: Turn) -> bool:
    return turn.service_call is not None


def agent_ends_conversation(turn: Turn) -> Behaviour:
    behaviours = _gather_system_behaviours(turn, SystemDialogueAct.GOODBYE)
    assert len(behaviours) == 1
    return behaviours[0]


def count_agent_questions(turn: Turn) -> int:
    assert len(turn.system_actions) == 1  # checking there is only one sys frame
    assert turn.system_actions is not None
    q_counts = {}
    for service, service_actions in turn.system_actions.items():
        n_questions = count_questions_asked(service_actions)
        if n_questions > 0:
            q_counts[service] = n_questions
    assert len(q_counts) in range(2)
    return sum(Counter(q_counts).values())


def agent_asked_multiple_questions(turn: Turn) -> bool:
    n_questions = count_agent_questions(turn)
    return n_questions > 1


def has_failed_api_calls(conversation: Conversation) -> bool:
    """Returns `True` if the agent informs the user a transaction failed or
    no results were available. For search intents, this happens if the user
    requests more results than are available. For transactions, these simulate
    actual errors."""
    return any(agent_notifies_failure(turn) for turn in conversation.turns)


def get_next_failed_api_call(
    conversation: Conversation,
    start_index: Optional[int] = None,
    stop_index: Optional[int] = None,
) -> Optional[FailedCallTurn]:
    """Get the next turn where the agent notifies the user the API
    call was not successful (ie no results or transaction failed).

    Parameters
    ----------
    start_index, stop_index
        Conversation turn range where the failed API called turn should be
        retrieved from. Search is exclusive (ie `conversation[stop_index]`) is
        not checked.

    Returns
    -------
    A `FailedCallTurn` object, containing the index


    Notes
    -----
    Pass the entire conversation as opposed to conversation slices to ensure
    the returned indices are correct.
    """

    if start_index is None:
        start_index = 0
    if stop_index is None:
        stop_index = len(conversation)
    for index, turn in enumerate(
        conversation[start_index:stop_index], start=start_index
    ):
        if index == stop_index:
            break
        if agent_notifies_failure(turn):
            return FailedCallTurn(turn=turn, index=index)


def count_confirmation_turns(conversation: Conversation) -> int:
    """Counts the number of turns where the agent requests the user to confirm
    transactions."""
    confirmations = 0
    for turn in conversation:
        if agent_requests_confirmation(turn):
            confirmations += 1

    return confirmations


def get_first_turn_after_first_call(conversation: Conversation) -> int:
    for idx, turn in enumerate(conversation):
        if agent_makes_system_call(turn):
            return idx
