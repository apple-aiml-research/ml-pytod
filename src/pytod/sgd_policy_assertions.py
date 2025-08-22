#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import Counter
from copy import deepcopy
from itertools import chain

from pytod.pytod_types.aliases import ServiceName
from pytod.pytod_types.sgd_conversation import (
    Author,
    SystemDialogueAct,
    Turn,
    UserAction,
    UserDialogueAct,
)

logger = logging.getLogger(__name__)


def assert_only_requests(actions: list[UserAction]):
    actions = deepcopy(actions)
    while actions:
        next_action = actions.pop()
        try:
            assert next_action.act == UserDialogueAct.REQUEST
        except AssertionError:
            assert next_action.act == UserDialogueAct.AFFIRM


def assert_user_asked_questions(sys_turn: Turn, user_turn: Turn, service: ServiceName):
    sys_actions = sys_turn.system_actions[service]
    user_actions = user_turn.user_actions[service]

    # check all questions asked are answered
    for u_action in user_actions:
        if u_action.act == UserDialogueAct.REQUEST:
            queried_property = u_action.slot
            assert any(
                sys_a.slot == queried_property
                for sys_a in sys_actions
                if sys_a.act == SystemDialogueAct.INFORM
            )


def assert_sys_answered_questions(
    sys_turn: Turn, user_turn: Turn, service: ServiceName
):
    sys_actions = sys_turn.system_actions[service]
    user_actions = user_turn.user_actions[service]
    # check all answered questions are asked
    for sys_action in sys_actions:
        if sys_action.act == SystemDialogueAct.INFORM:
            mentioned_slot = sys_action.slot
            assert any(
                usr_a.slot == mentioned_slot
                for usr_a in user_actions
                if usr_a.act == UserDialogueAct.REQUEST
            )


def assert_only_slot_filling(turn: Turn):
    assert turn.author == Author.SYSTEM, "Something went wrong..."
    assert all(
        a.act == SystemDialogueAct.REQUEST for a in chain(*turn.system_actions.values())
    )


def assert_system_information_provision(turn: Turn):
    assert turn.author == Author.SYSTEM, "Something went wrong..."
    dialog_acts = Counter([a.act for a in chain(*turn.system_actions.values())])
    dialog_acts.pop(SystemDialogueAct.INFORM)
    if dialog_acts:
        if SystemDialogueAct.NOTIFY_SUCCESS in dialog_acts:
            dialog_acts.pop(SystemDialogueAct.NOTIFY_SUCCESS)
        else:
            # KeyError should never be raised
            dialog_acts.pop(SystemDialogueAct.NOTIFY_FAILURE)
            if dialog_acts:
                dialog_acts.pop(SystemDialogueAct.OFFER)
        assert not dialog_acts


def assert_sys_offer_behaviour(turn: Turn):
    assert turn.author == Author.SYSTEM, "Something went wrong..."
    assert len(turn.system_actions) == 1
    dialog_acts = Counter([a.act for a in chain(*turn.system_actions.values())])
    dialog_acts.pop(SystemDialogueAct.OFFER)
    if dialog_acts:
        dialog_acts.pop(SystemDialogueAct.INFORM_COUNT)
    assert not dialog_acts


def assert_sys_confirmation_behaviour(turn: Turn):
    assert turn.author == Author.SYSTEM, "Something went wrong..."
    assert len(turn.system_actions) == 1
    dialog_acts = Counter([a.act for a in chain(*turn.system_actions.values())])
    dialog_acts.pop(SystemDialogueAct.CONFIRM)
    assert not dialog_acts


def assert_user_intent_information_behaviour(
    actions: dict[UserDialogueAct, list[UserAction]]
):
    try:
        assert len(actions) == 1
    except AssertionError:
        try:
            assert len(actions) == 2 and UserDialogueAct.SELECT in actions
        except AssertionError:
            assert any(
                (
                    UserDialogueAct.THANK_YOU in actions,
                    UserDialogueAct.CARRY_OVER in actions,
                )
            )


def assert_user_result_selection_by_indexing_behaviour(turn: Turn):
    # NB: the user can select a film and move on to do something else
    # in the same turn (see train/99_00124)
    selection_services = set()
    for service, service_actions_dict in turn.user_actions_dict.items():
        if UserDialogueAct.SELECT in service_actions_dict:
            if len(service_actions_dict) > 1:
                this_service_actions = deepcopy(service_actions_dict)
                expected_acts = {
                    UserDialogueAct.INFORM_INTENT,
                    UserDialogueAct.INFORM,
                    UserDialogueAct.CARRY_OVER,
                    UserDialogueAct.SELECT,
                    # UserDialogueAct.REQUEST
                }
                selection_services.add(service)
                observed_acts = set(this_service_actions.keys())
                try:
                    assert observed_acts.issubset(expected_acts)
                except AssertionError:
                    logger.warning(
                        f"User selected an entity and ended the conversation in service {service}."
                        "Please check that the natural language utterance reflects this."
                    )
                    # train/100_00027 shows a paraphrase error where the GOODBYE act does not
                    # appear in the utterance
                    assert observed_acts == {
                        UserDialogueAct.SELECT,
                        UserDialogueAct.GOODBYE,
                    }
            else:
                assert len(service_actions_dict) == 1
                selection_services.add(service)
                assert UserDialogueAct.SELECT in service_actions_dict
    # assert len(selection_services) == 1, "Only one query should be active at any time"


def assert_user_only_informed_or_selected_entity(turn: Turn):
    dialog_acts = Counter([a.act for a in chain(*turn.user_actions.values())])
    expected_acts = {UserDialogueAct.INFORM_INTENT, UserDialogueAct.INFORM}
    try:
        # only informed
        assert len(dialog_acts) == 2
    except AssertionError:
        expected_acts = [
            # declined a suggested task and did something else instead (eg train/65_00069)
            {
                UserDialogueAct.NEGATE_INTENT,
                UserDialogueAct.INFORM_INTENT,
                UserDialogueAct.INFORM,
            },
            # selected an entity at the same time
            {
                UserDialogueAct.INFORM_INTENT,
                UserDialogueAct.INFORM,
                UserDialogueAct.SELECT,
            },
            # the user thanks and moves on to the next task (eg, test/8_00041)
            {
                UserDialogueAct.INFORM_INTENT,
                UserDialogueAct.INFORM,
                UserDialogueAct.THANK_YOU,
            },
        ]
        assert any(
            not set(dialog_acts.keys()).difference(acts) for acts in expected_acts
        )


def assert_single_action_or_entity_selection_turn(turn: Turn):
    try:
        actions = (
            turn.user_actions if turn.author == Author.USER else turn.system_actions
        )
        if len(actions) > 1:
            assert all(
                len(service_actions) > 1 for _, service_actions in actions.items()
            )
            return False
        actions = list(chain(*actions.values()))
        assert len(actions) == 1
    except AssertionError:
        # user selected some entity
        dialog_acts = Counter([a.act for a in chain(*turn.user_actions.values())])
        expected_acts = {
            UserDialogueAct.INFORM_INTENT,
            UserDialogueAct.SELECT,
        }
        try:
            assert not set(dialog_acts.keys()).difference(expected_acts)
        except AssertionError:
            # user thanks and gets on with the next task (test/8_00031)
            expected_acts = [
                {
                    UserDialogueAct.INFORM_INTENT,
                    UserDialogueAct.THANK_YOU,
                },
                {UserDialogueAct.INFORM_INTENT, UserDialogueAct.NEGATE_INTENT},
            ]
            assert any(
                not set(dialog_acts.keys()).difference(acts) for acts in expected_acts
            )


def assert_no_affirmation(turn: Turn):
    if turn.author == Author.USER:
        for service, service_actions in turn.user_actions_dict.items():
            assert UserDialogueAct.AFFIRM not in service_actions


def assert_on_implicit_termination(
    user_turn: Turn, prev_system_turn: Turn, next_system_turn: Turn
):
    """Check that the conversation either ends with the user explicitly dismissing the agent
    or with the user implicitly dismissing by saying they don't want anymore help."""
    if next_system_turn is None:
        return
    for service, service_actions in next_system_turn.system_actions_dict.items():
        if SystemDialogueAct.GOODBYE in service_actions:
            user_actions = user_turn.user_actions_dict[service]
            if UserDialogueAct.GOODBYE not in user_actions:
                prev_sys_actions = prev_system_turn.system_actions_dict[service]
                assert SystemDialogueAct.REQ_MORE in prev_sys_actions
                assert UserDialogueAct.NEGATE in user_actions
