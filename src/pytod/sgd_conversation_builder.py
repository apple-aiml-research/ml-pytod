#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from collections import defaultdict, namedtuple
from typing import Any, Union

from pytod.command import ServiceCall, ServiceCallResult, ServiceName
from pytod.pytod_types.sgd_conversation import (
    _SGD_AGENT_NAME,
    _SGD_SERVICE_CALL_KEY,
    _SGD_USER_NAME,
    Author,
    Conversation,
    DialogueState,
    InformationProvided,
    SlotMentions,
    SystemAction,
    SystemDialogueAct,
    Turn,
    UserAction,
    UserDialogueAct,
)
from pytod.sgd_annotation import (
    annotate_additional_inform_count,
    annotate_information_provided,
    annotate_slot_mentions,
    annotate_slot_value_confirmation,
    annotate_slots_mentioned_as_alternatives,
    annotate_user_selected_value_index,
    get_multi_value_offer_metadata,
)
from pytod.sgd_data_mining_utils import (
    collect_request_action_parameters_data,
    collect_slot_value_paraphrases,
    collect_user_slot_confirmation_data,
)
from pytod.sgd_utils import dialogue_iterator, is_dialogue_formality


def no_op_conversation_builder(dialogue: dict, **kwargs) -> dict[str, Any]:
    """A no-op conversation builder that returns an SGD conversation in raw format."""
    return dialogue


def is_error_turn(
    call_dict: dict[str, dict[str, Any]],
    output: list[dict[str, str]],
    sys_actions: list[SystemAction],
    utterance: str,
) -> bool:
    """Heuristic to approximately correctly parse `is_error`.

    Notes
    -------
     Consider the excerpt from `1_00108` (train):

        ...
        [3] user: "Could you give another suggestion? I would love a place with live music."
            agent: find_restaurants(city=Oakland, cuisine=African, has_live_music=True)
            results: {"phone_number": "510-531-6970", "restaurant_name": "Miliki Restaurant",
            "street_address": "3725 MacArthur Boulevard", "_id": "2"}
            agent: "I have found 1 place called Miliki Restaurant in Oakland."
        [4] user: "Where are they located? And do they serve liquor?"
            agent: "They do serve alcohol, and are located at 3725 MacArthur Boulevard."
        [5] user: "That works for me. I'd like to make a reservation on the 12th."
            agent: "At what time would you like me to schedule the reservation?"
        [6] user: "Please schedule it at 11:15 in the morning."
            agent: "Please confirm the following: You want a table
            at Miliki Restaurant in Oakland at 11:15 am for 2 people on March 12th."
        [7] user: "That would be great."
            agent: reserve_restaurant(city=Oakland, date=2019-03-12,
            party_size=2, restaurant_name=Miliki Restaurant, time=11:15)
            results: {"cuisine": "African", "has_live_music": "True",
             "phone_number": "510-531-6970", "price_range": "moderate",
             "serves_alcohol": "True", "street_address": "3725 MacArthur Boulevard", "_id": "3"}
            agent: "I'm sorry, I couldn't make the reservation.
              Is there anything else I could help you with?"
        ...

    Notice that after turn `[7]`, the agent says:

    _"I'm sorry, I couldn't make the reservation.
    Is there anything else I could help you with?"_

    but the `results:` turn is

        ```
         results:
            {
                "cuisine": "African",
                "has_live_music": "True",
                "phone_number": "510-531-6970",
                "price_range": "moderate",
                "serves_alcohol": "True",
                "street_address": "3725 MacArthur Boulevard",
                "_id": "3"
            }
        ```
    so the annotation contains a result. Note that the time does not appear in the result -
    this is because our conversation_to_text logic removes common slots (value notwithstanding)
    if they are in the service call to shorten the response.

    Meanwhile, the ground truth is annotated with is annotated with a response where
    `{"time": "11:30"}`, which does not match the call `{"time": "11:15"}`. The same
    applies to other dialogues (eg `1_00002`). Meanwhile, in `1_00009` the system does
    offer a different booking time, corresponding to the time annotated in the
    service_result. and the calls are annotated in the same way as for `1_00108`.
    On the other hand, in `1_00015` there are no results and the system does not offer
    alternative arrangements.
    """

    ArgMismatch = namedtuple("ArgMismatch", ["call", "response"])

    def _get_response_mismatched_args(
        call_dict: dict[str, Any], call_results: list[dict[str, str]]
    ) -> dict[str, ArgMismatch]:
        call_args = call_dict["parameters"]
        assert len(call_results) <= 1
        if len(call_results) == 0:
            return {}
        [response] = call_results
        mismatches = {}
        for arg, value in call_args.items():
            if arg in response and value != response[arg]:
                mismatches[arg] = ArgMismatch(call=value, response=response[arg])
        return mismatches

    def _agent_notifies_failure(actions: list[SystemAction]):
        return any(action.act == SystemDialogueAct.NOTIFY_FAILURE for action in actions)

    if _agent_notifies_failure(sys_actions):
        mismatched_args = _get_response_mismatched_args(call_dict, output)
        # if there are any mismatched slots whose vals appear
        # in sys utterance, it's not an actual failure
        if mismatched_args:
            for arg, mismatch_info in mismatched_args.items():
                maybe_offered_value = mismatch_info.response
                if maybe_offered_value in utterance:
                    return False
        return True
    else:
        assert output
        return not output


def get_actions_dict(
    turn: Turn,
) -> dict[
    ServiceName,
    dict[
        Union[UserDialogueAct, SystemDialogueAct], list[Union[UserAction, SystemAction]]
    ],
]:
    """In-place remapping of action lists to dictionaries where the keys
    are dialogue acts and the values are lists of actions corresponding to
    the dialogue act."""

    if turn.author == Author.USER:
        actions = turn.user_actions
    else:
        actions = turn.system_actions
    assert actions is not None
    actions_by_service = {}
    for service, actions_list in actions.items():
        actions_dict: dict[
            Union[UserDialogueAct, SystemDialogueAct],
            list[Union[UserAction, SystemAction]],
        ] = defaultdict(list)
        for action in actions_list:
            actions_dict[action.act].append(action)
        actions_by_service[service] = dict(actions_dict)

    return actions_by_service


def add_actions_dict(turn: Turn):
    """In-place turn modifications to create a mapping from dialogue
    acts to actions."""
    assert isinstance(turn, Turn)
    actions_dict = get_actions_dict(turn)
    if turn.author == Author.USER:
        turn.user_actions_dict = actions_dict
    else:
        turn.system_actions_dict = actions_dict


def maybe_copy_suggested_task_annotation(
    service: ServiceName, user_frame: dict[str, Any], prev_system_turn: dict[str, Any]
):
    """The AFFIRM_INTENT and NEGATE_INTENT user actions do not have values annotation.
    For convenience, we copy them to the user turn."""

    for a in user_frame["actions"]:
        if a["act"] in {"NEGATE_INTENT", "AFFIRM_INTENT"}:
            [system_frame] = [
                f for f in prev_system_turn["frames"] if f["service"] == service
            ]
            [offered_intent_action] = [
                a for a in system_frame["actions"] if a["act"] == "OFFER_INTENT"
            ]
            a["values"] = offered_intent_action["values"]
            a["canonical_values"] = offered_intent_action["values"]
            a["slot"] = offered_intent_action["slot"]


def build_conversation(dialogue: dict, **kwargs) -> Conversation:
    """Build a `Conversation` from a json-format SGD dialogue."""

    def assert_selected_value_in_utterance(
        actions: list[dict[str, Any]], utterance: str
    ):
        for a in actions:
            if a["act"] == "SELECT" and len(a["values"]) >= 1:
                assert a["values"][0] in utterance

    turns = []
    multi_value_offer = None
    prev_sys_turn = None
    for turn_idx, turn in enumerate(dialogue_iterator(dialogue)):
        # a frame continues state & sematic or API call annotations for a single service
        multi_frame_turn = len(turn["frames"]) > 1
        this_turn_data = {}
        parsed_states, parsed_actions = {}, {}
        collect_request_action_parameters_data(turn, dialogue["dialogue_id"], **kwargs)
        for frame in turn["frames"]:
            service: ServiceName = frame["service"]
            this_frame_actions = frame["actions"]
            collect_slot_value_paraphrases(
                turn["speaker"],
                service,
                this_frame_actions,
                paraphrase_map_collector=kwargs.get("paraphrase_map_collector", None),
                canonical_values_collector=kwargs.get(
                    "canonical_values_collector", None
                ),
            )
            if turn["speaker"] == _SGD_USER_NAME:
                maybe_copy_suggested_task_annotation(
                    service, frame, dialogue["turns"][turn_idx - 1]
                )
                # skip dialogue formality frames to simplify processing
                if multi_frame_turn and is_dialogue_formality(frame, act="THANK_YOU"):
                    assert len(frame["actions"]) == 1
                    continue
                assert_selected_value_in_utterance(frame["actions"], turn["utterance"])
                parsed_states[service] = DialogueState.model_validate(frame["state"])
                annotate_user_selected_value_index(turn, multi_value_offer)
                collect_user_slot_confirmation_data(turn, prev_sys_turn, **kwargs)
                annotate_slot_value_confirmation(turn, prev_sys_turn)
                parsed_actions[service] = [
                    UserAction.model_validate(usr_action)
                    for usr_action in this_frame_actions
                ]
                this_turn_data.update(
                    {
                        "author": Author.USER,
                        "text": turn["utterance"],
                        "dialogue_state": parsed_states,
                        "user_actions": parsed_actions,
                    }
                )
            else:
                assert turn["speaker"] == _SGD_AGENT_NAME
                annotate_slots_mentioned_as_alternatives(turn)
                annotate_additional_inform_count(turn, **kwargs)
                multi_value_offer = get_multi_value_offer_metadata(turn)
                prev_sys_turn = turn
                parsed_actions[service] = [
                    SystemAction.model_validate(sys_action)
                    for sys_action in this_frame_actions
                ]
                this_turn_data.update(
                    {
                        "author": Author.SYSTEM,
                        "text": turn["utterance"],
                        "system_actions": parsed_actions,
                    }
                )
                service_call, service_results = None, None
                if _SGD_SERVICE_CALL_KEY in frame:
                    # parse service call
                    service_call_info = frame["service_call"]
                    service_call_info.update(service=frame["service"])
                    service_call = ServiceCall.model_validate(frame["service_call"])
                    # parse service call output
                    error_message = kwargs.get("error_messages", None)
                    error_turn = is_error_turn(
                        frame["service_call"],
                        frame["service_results"],
                        parsed_actions[service],
                        turn["utterance"],
                    )
                    if error_turn and error_message is not None:
                        # this should never be raised
                        error_message = error_message[service][service_call.method]
                    service_results = ServiceCallResult.model_validate(
                        {
                            "service_results": frame["service_results"],
                            "error_message": error_message if error_turn else None,
                            "is_error": error_turn,
                        },
                    )
                if service_call is not None:
                    this_turn_data.update(
                        {
                            "service_call": service_call,
                            "service_results": service_results,
                        }
                    )
        assert this_turn_data, "Empty turn data should not be possible!"
        this_turn = Turn.model_validate(this_turn_data)
        add_actions_dict(this_turn)
        turns.append(this_turn)
    slot_mentions = annotate_slot_mentions(dialogue)
    info_provided = annotate_information_provided(dialogue)
    return Conversation(
        turns=turns,
        id=dialogue["dialogue_id"],
        services=dialogue["services"],
        slot_mentions=SlotMentions.model_validate(slot_mentions),
        information_provided=InformationProvided.model_validate(info_provided),
    )


def mine_offered_slots(
    dialogue: dict, collector: dict[str, Any], **kwargs
) -> Conversation:
    return build_conversation(dialogue, collector=collector)


def mine_request_parametrisation(
    dialogue: dict, request_parametrisation_collector: dict[str, Any], **kwargs
) -> Conversation:
    return build_conversation(
        dialogue, request_parametrisation_collector=request_parametrisation_collector
    )


def mine_user_slot_confirmations(
    dialogue: dict, user_slot_confirmation_collector: dict[str, Any], **kwargs
) -> Conversation:
    return build_conversation(
        dialogue, user_slot_confirmation_collector=user_slot_confirmation_collector
    )


def mine_slot_value_paraphrases(
    dialogue: dict,
    paraphrase_map_collector: dict[str, Any],
    canonical_values_collector: dict[str, list[str]],
    **kwargs,
):
    return build_conversation(
        dialogue,
        paraphrase_map_collector=paraphrase_map_collector,
        canonical_values_collector=canonical_values_collector,
    )
