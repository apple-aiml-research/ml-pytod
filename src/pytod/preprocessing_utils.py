#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This module contains utilities that are applied to the conversations
before converting them to `python` programs. These include utilities for
truncating and filtering conversations."""

import logging
from itertools import chain
from typing import Callable, NamedTuple, Optional, Union

from pytod.pytod_types.aliases import DialogueID
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    SystemDialogueAct,
    Turn,
    UserDialogueAct,
    get_active_service,
)
from pytod.sgd_agent_behaviour import (
    agent_asked_multiple_questions,
    agent_prompts_for_more_help,
    agent_suggests_next_task_turn,
)
from pytod.sgd_analysis_utils import infer_call_occurred
from pytod.sgd_user_behaviour import (
    user_does_not_specify_next_task,
    user_selects_entity,
    user_selects_from_system_options,
)

logger = logging.getLogger(__name__)


class ConversationPrefix(NamedTuple):
    prefix: Conversation
    # if -1, the entire input conversation was stored in the input
    last_turn_idx: int


def get_first_user_turn_prefix(conversation: Conversation) -> ConversationPrefix:
    first_turn = conversation[0]
    services = list(first_turn.user_actions.keys())
    return ConversationPrefix(
        prefix=Conversation(turns=[first_turn], id=conversation.id, services=services),
        last_turn_idx=0,
    )


def _get_conversation_prefix(
    conversation: Conversation,
    condition: Callable[[Turn], bool],
    *,
    author: Author,
) -> ConversationPrefix:
    """Iterate through turns of a dialogue until `condition` is met at a given user or agent turn. A
    condition is a predicate that detects a given user/agent behaviour (e.g. agent offers an
    intent to the user or asks them if they want help with another task).

    Parameters
    ---------
    condition
        Determines the prefix length.
    author
        On which turns sh
    """

    prefix_services = []
    last_turn_idx = -1
    prefix_dialogue = conversation
    for turn_idx, turn in enumerate(conversation):
        if turn.author == Author.USER:
            service = get_active_service(conversation)
            if service not in prefix_services:
                prefix_services.append(service)
        if author == turn.author and condition(turn):
            prefix_turns = conversation[:turn_idx]
            prefix_dialogue = Conversation(
                turns=prefix_turns, services=prefix_services, id=conversation.id
            )
            last_turn_idx = turn_idx
            break

    return ConversationPrefix(prefix=prefix_dialogue, last_turn_idx=last_turn_idx)


def get_turns_before_multiple_slot_filling_turn(
    conversation: Conversation,
) -> ConversationPrefix:
    """Extracts the prefix of `conversation` where the agent only takes one action to fill
      the API required slots.

    Returns
    -------
    A SingleSlotFillingActionPrefix object containing:

        - the prefix extracted, stored in the ``prefix`` property
        - the turn index, in the original SGD conversation where the single slot filling
        condition was no longer met, stored in the ``last_turn_index`` property.

    Notes
    -----

    1. If ``last_turn_index`` = -1, then the conversation was not filtered.
    2. The conversation stored in ``prefix`` does not contain `conversation[last_turn_index]`.
    The last user turn before it is the last user turn
    """
    return _get_conversation_prefix(
        conversation, agent_asked_multiple_questions, author=Author.SYSTEM
    )


def get_turns_before_next_task_suggestion(
    conversation: Conversation,
) -> ConversationPrefix:
    """Extracts the prefix of `conversation` before the agent suggests the user
    completes.

    Returns
    -------
    A ConversationPre object containing:

        - the prefix extracted, stored in the ``prefix`` property
        - the turn index, in the original SGD conversation where the agent suggested
        that the user completes a new task (e.g. book a restaurant they find)

    Notes
    -----

    1. If ``last_turn_index`` = -1, then the conversation was not filtered.
    2. The conversation stored in ``prefix`` does not contain `conversation[last_turn_index]`.
    The last user turn before it is the last user turn
    """
    return _get_conversation_prefix(
        conversation, agent_suggests_next_task_turn, author=Author.SYSTEM
    )


def get_turns_before_next_task_prompting(
    conversation: Conversation,
) -> ConversationPrefix:
    """Extracts the prefix of `conversation` before the agent prompts the user
    to complete another tasks.

    Returns
    -------
    A ConversationPrefix object containing:

        - the prefix extracted, stored in the ``prefix`` property
        - the turn index, in the original SGD conversation where the agent prompted
        the user to complete another task (e.g. _system: Would you like help with anything else?_)

    Notes
    -----

    1. If ``last_turn_index`` = -1, then the conversation was not filtered.
    2. The conversation stored in ``prefix`` does not contain `conversation[last_turn_index]`.
    The last user turn before it is the last user turn.
    """
    return _get_conversation_prefix(
        conversation, agent_prompts_for_more_help, author=Author.SYSTEM
    )


def get_turns_before_first_call(conversation: Conversation) -> ConversationPrefix:
    return _get_conversation_prefix(
        conversation, infer_call_occurred, author=Author.SYSTEM
    )


def get_turns_before_entity_selection(conversation: Conversation) -> ConversationPrefix:
    """Returns the turns before the user simply selects an entity."""

    def user_only_makes_selection(turn: Turn):
        return (
            user_selects_entity(turn)
            and len(list(chain(*turn.user_actions.values()))) == 1
        )

    return _get_conversation_prefix(
        conversation, user_only_makes_selection, author=Author.USER
    )


def get_only_active_intent_turns(conversation: Conversation):
    """Returns all the turns in the conversation up to the point the user does not specify
    the next task but does not end the conversation either"""
    return _get_conversation_prefix(
        conversation, user_does_not_specify_next_task, author=Author.USER
    )


def get_turns_before_selection_from_options(conversation: Conversation):
    """Returns all the turns in the conversation up to the point where the user selects
    an option from several alternatives offered in the previous turn."""
    return _get_conversation_prefix(
        conversation, user_selects_from_system_options, author=Author.USER
    )


def get_skipped_acts(
    author: str, acts: list[str]
) -> Union[set[UserDialogueAct], set[SystemDialogueAct]]:
    """Return a list of acts that have not PyTOD statement equivalents."""
    author = getattr(Author, author)
    act_enum = SystemDialogueAct if author == Author.SYSTEM else UserDialogueAct
    return {getattr(act_enum, act) for act in acts}


def by_services(conversation: Conversation, services: Optional[list[str]]) -> bool:
    """Filter out a conversation if it contains any of specified services."""
    if services is None:
        return False
    return bool(set(conversation.services).intersection(services))


def by_id(conversation: Conversation, noisy_ids: list[DialogueID]):
    """Filter out noisy conversations.

    Notes
    -----
    Adding an annotations correction module could easily fix these
    conversations in a separate pipeline.
    """
    if conversation.id in noisy_ids:
        return True
    return False
