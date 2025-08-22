#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from typing import Literal

from omegaconf import DictConfig

from pytod.command import CommandCollection, is_arg
from pytod.preprocessing_utils import ConversationPrefix
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    Turn,
    UserAction,
    UserDialogueAct,
    get_active_service,
)
from pytod.sgd_agent_behaviour import (
    agent_asked_multiple_questions,
)

logger = logging.getLogger(__name__)


def assert_prefix_correct(conversation: Conversation, prefix: ConversationPrefix):
    if prefix.last_turn_idx == -1:
        assert len(conversation) == len(prefix.prefix)
    else:
        if prefix.last_turn_idx == 0:
            assert len(prefix.prefix) == 1
        else:
            assert len(conversation[: prefix.last_turn_idx]) == len(prefix.prefix)


def assert_turn_valid(turn: Turn, interpreter_cfg: DictConfig):
    if turn is None:
        return
    if turn.author == Author.SYSTEM:
        # assert not agent_suggests_next_task_turn(turn)
        if not interpreter_cfg.backend.hints.multiple_slot_filling:
            assert not agent_asked_multiple_questions(turn)


def assert_no_carry_over_overlap(actions: dict[UserDialogueAct, list[UserAction]]):
    assert isinstance(actions, dict)
    if (
        UserDialogueAct.CARRY_OVER not in actions
        or UserDialogueAct.INFORM not in actions
        or UserDialogueAct.SELECT not in actions
    ):
        return
    carry_over_slots = set(a.slot for a in actions[UserDialogueAct.CARRY_OVER])
    informed_slot = set(a.slot for a in actions[UserDialogueAct.INFORM])
    selected_slot = set(a.slot for a in actions[UserDialogueAct.SELECT])
    assert not carry_over_slots.intersection(informed_slot, selected_slot)


TargetServiceName = str
TargetIntentName = str
TargetSlotName = str
SourceAPIInfo = str
SlotRelationsMapping = dict[
    TargetServiceName,
    dict[
        TargetIntentName,
        dict[
            TargetSlotName,
            list[dict[[Literal["service", "intent", "slot"]], SourceAPIInfo]],
        ],
    ],
]


def assert_slot_relations_correct(
    slot_relations: SlotRelationsMapping, command_collection: CommandCollection
):
    """Check annotation of slot relations is correct."""
    for target_service, target_intent_map in slot_relations.items():
        for target_intent, target_slot_map in target_intent_map.items():
            try:
                target_command = command_collection.get(target_service, target_intent)
            except AttributeError:
                # slot relations are defined for the entire corpus whereas command collection
                # only contains the commands from the split processed
                continue
            for slot, api_sources in target_slot_map.items():
                assert isinstance(api_sources, list)
                for source_api_info in api_sources:
                    assert isinstance(source_api_info, dict)
                    assert len(source_api_info) == 3
                    assert set(source_api_info.keys()) == {"service", "intent", "slot"}
                    assert is_arg(slot, target_command)
                    source_service = source_api_info["service"]
                    try:
                        source_command = command_collection.get(
                            source_service, source_api_info["intent"]
                        )
                        source_slot = source_api_info["slot"]
                    # the cmd is not seen in the split we are processing so it is not
                    # part of the cmd collection
                    except AttributeError:
                        continue
                    try:
                        assert is_arg(source_slot, source_command)
                    except AssertionError:
                        # it can be a property of the returned entity, so we check command slots
                        try:
                            assert command_collection.in_service_schema(
                                source_service, source_slot
                            )
                        except AssertionError:
                            pass
                    # the cmd is not seen in the split we are processing so it is not
                    # part of the cmd collection
                    except AttributeError:
                        continue


def assert_on_multi_frame_turns_policy(conversation: Conversation):
    expected_acts = {
        UserDialogueAct.SELECT,
        UserDialogueAct.THANK_YOU,
    }

    for prefix in conversation.prefix_conversations(last_speaker=Author.USER):
        if len(user_actions := prefix.current_turn().user_actions_dict) > 1:
            active_service = get_active_service(prefix)
            ending_task_acts = set(user_actions[active_service].keys())
            try:
                assert expected_acts.issubset(expected_acts)
            except AssertionError:
                logger.error(
                    f"Ending task acts set {ending_task_acts} is not a subset "
                    f"of expected acts {expected_acts}"
                )
