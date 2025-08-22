#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This module contains functions that are used for additional automatic annotation
of the SGD corpus, to ensure a consistent representation of the dialogues as
python programs. The annotations are applied at conversation building time."""
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

from _operator import itemgetter
from pydantic import BaseModel, parse_obj_as

from pytod.sgd_metadata import (
    entities,
    requested_slots,
    result_slots,
    untracked_optionals,
)
from pytod.sgd_utils import (
    dialogue_iterator,
    get_frame,
    get_slots_with_requested_confirmation,
    get_user_informed_slots,
    user_affirms,
)
from pytod.utils import default_to_regular, nested_defaultdict

logger = logging.getLogger(__name__)


def annotate_slot_mentions(dialogue: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Use the dialogue act annotations to distinguish between the slots
    communicated by user/system while a service is active and slots carried
    over from previous intents.

    Parameters
    ----------
    dialogue
        Dialogue in raw SGD format.
    """

    def get_user_mentioned_slots(actions: list[dict]) -> dict[str, list[str]]:
        slot_mentions = {}
        for action in actions:
            # For simplicity, we also include  the values that the user
            # accepts using an AFFIRM() dialogue act. The actual values are
            # mentioned in the previous system utterance and are annotated
            # via REQUEST(slot=[vals,...]) actions. The same annotation is
            # used to present different values to the user (eg different
            # values of cuisine), in which case the user turn will always have
            # a INFORM(slot=['value']) annotation
            counts_as_mention = any(
                (
                    action["act"] == "INFORM",
                    action["act"] == "SELECT" and action["slot"],
                    action["act"] == "AFFIRM" and action["slot"],
                )
            )
            if counts_as_mention:
                assert action["values"]
                assert action["canonical_values"]
                slot_mentions[action["slot"]] = action["values"]
        return slot_mentions

    def get_system_mentioned_slots(actions: list[dict]) -> dict[str, list[str]]:
        slot_mentions = {}
        for action in actions:
            if action["act"] in ("OFFER", "CONFIRM"):
                slot_mentions[action["slot"]] = action["values"]
        return slot_mentions

    def get_user_requested_slots(actions: list[dict]) -> dict[str, list[str]]:
        slot_mentions = {}
        for action in actions:
            if action["act"] == "REQUEST":
                slot_mentions[action["slot"]] = action["values"]
        return slot_mentions

    def maybe_warn_untracked_slots(
        final_state: dict[str, dict[str, list[str]]],
        maybe_warn_about: dict[str, set[str]],
    ):
        """Warn if a system-mentioned slot is not found in the state at the end of the dialogue."""
        for service_intent, slots_not_tracked in maybe_warn_about.items():
            service, intent = service_intent.split(".")
            for slot in slots_not_tracked:
                if (
                    slot in final_state[f"{service}.{intent}"]
                    or slot in entities[service][intent]
                ):
                    continue
                other_intents = [i for i in entities[service] if i != intent]
                for other in other_intents:
                    if slot in entities[service][other]:
                        assert slot in final_state[f"{service}.{other}"]
                        break
                else:
                    logger.debug(
                        f"{dialogue['dialogue_id']}||{service_intent}: "
                        f"Slots {slot} mentioned by the system but they were not tracked"
                    )

    communicated_by_user = nested_defaultdict(list, depth=2)
    system_inherited = nested_defaultdict(list, depth=2)
    maybe_warn_about = defaultdict(set)
    final_state = {}
    prev_service_intent = []
    for idx, turn in enumerate(dialogue_iterator(dialogue, user=True, system=False)):
        user_turn_idx = 2 * idx
        this_turn_service_intents = []
        for frame in turn["frames"]:
            service = frame["service"]
            intent = frame["state"]["active_intent"]
            if intent == "NONE":
                # state of the service does not change
                continue
            service_intent = f"{service}.{intent}"
            user_mentioned = get_user_mentioned_slots(frame["actions"])
            user_requested = get_user_requested_slots(frame["actions"])
            current_state = frame["state"]["slot_values"]
            sys_mentioned = {}
            if user_turn_idx > 0:
                sys_turn = dialogue["turns"][user_turn_idx - 1]
                prev_turn_sys_frame = get_frame(sys_turn, service)
                # if we get no actions, service changed
                if prev_turn_sys_frame is not None:
                    sys_mentioned.update(
                        get_system_mentioned_slots(prev_turn_sys_frame["actions"])
                    )
            # we know what the user mentioned and what the system did
            assert set(user_mentioned).issubset(current_state.keys())
            # update the slots that were actually said by the user
            for slot, values in user_mentioned.items():
                communicated_by_user[f"{service}.{intent}"][slot].append(
                    {"values": deepcopy(values), "mentioned_in_turn": idx}
                )
            for slot, values in sys_mentioned.items():
                assert prev_service_intent
                matching_intents = [
                    i for i in prev_service_intent[-1] if i.startswith(service)
                ]
                assert len(matching_intents) == 1
                # not all system slots are not in the dialogue state of the current service
                # however, note that some of the requested slot (eg address) can be
                # required arguments to downstream tasks (eg `Services_*` addresses
                # are carried over as values to `destination` in `RideSharing_*` services)
                is_requested = requested_slots.is_only_requested(slot, service) or (
                    requested_slots.can_request(slot, service)
                    and slot in user_requested
                )
                is_result = result_slots.is_result(slot, service)
                is_known_untracked_optional = untracked_optionals.is_untracked(
                    slot, service
                )

                if (
                    is_requested
                    or (is_result is not None and is_result)
                    or is_known_untracked_optional
                ):
                    continue
                if values not in system_inherited[service_intent][slot]:
                    system_inherited[matching_intents[0]][slot].append(
                        {"values": deepcopy(values), "mentioned_in_turn": idx}
                    )
                if slot not in current_state:
                    maybe_warn_about[matching_intents[0]].add(slot)
            this_turn_service_intents.append(service_intent)
            final_state[service_intent] = current_state
        prev_service_intent.append(this_turn_service_intents)
    maybe_warn_untracked_slots(final_state, maybe_warn_about)

    return {
        "user": default_to_regular(communicated_by_user),
        "system": default_to_regular(system_inherited),
    }


def annotate_additional_inform_count(turn: dict[str, Any], **kwargs):
    """In some domains, the INFORM_COUNT dialogue act is missing if the system
    response only references one element. This function detects these cases and
    adds an additional INFORM_COUNT(count=1) action to ensure consistency.


    Parameters
    ----------
    turn
        Dialogue turn, possibly comprising multiple semantic frames, in raw SGD
        format.
    """

    def action_factory():
        return {
            "slot": "count",
            "act": "INFORM_COUNT",
            "values": ["1"],
            "canonical_values": ["1"],
            "metadata": None,
        }

    metadata_collector = kwargs.get("collector", None)
    assert turn["speaker"] == "SYSTEM"
    for frame in turn["frames"]:
        actions = frame["actions"]
        informed_number_of_entities = any(a["act"] == "INFORM_COUNT" for a in actions)
        if informed_number_of_entities:
            continue
        made_offer = any(a["act"] == "OFFER" for a in actions)
        notified_failure = any(a["act"] == "NOTIFY_FAILURE" for a in actions)
        if made_offer and not notified_failure:
            draft_action = action_factory()
            for a in actions:
                if (
                    a["act"] == "OFFER"
                    and (n_entities := len(a["canonical_values"])) > 1
                ):
                    if metadata_collector is not None:
                        metadata_collector[frame["service"]].add(a["slot"])
                    draft_action["values"][0] = str(n_entities)
                    draft_action["canonical_values"][0] = str(n_entities)
            actions.append(draft_action)


def annotate_slot_value_confirmation(
    user_turn: dict[str, Any],
    prev_sys_turn: Optional[dict[str, Any]],
    **kwargs,
):
    """In multi-domain conversations, one mechanism through which the user provides
    a slot value is accepting a value proposed by the agent. For example:

        ...
        system: Would you like your hotel booking for your arrival date, 11th of March?
        user: Yes, please.

    where the state updates with `11th of March`. This is not annotated as
    `INFORM(check_in_date="11th of March")`, as one would expected. Instead, in the
    previous system turn there is a `REQUEST(check_in_date=["11th of March"]` and the
    user turn is annotated with the AFFIRM() dialogue act.


    Parameters
    ----------
    user_turn, prev_sys_turn
        User and previous system turn, in raw SGD format.

    Note
    ----
    1. The user can also NEGATE() the proposal or override the value proposed.
    2. The REQUEST(value=['11th March']) annotation is ambiguous to the situation where
    the system simply proposes values to the user they can select from (eg "system: What
    kind of event would you like? We have sports, music, everything).
    """

    def remove_affirm_action(actions: list[dict[str, Any]]) -> dict:
        [index] = [i for i in range(len(actions)) if actions[i]["act"] == "AFFIRM"]
        return actions.pop(index)

    for frame in user_turn["frames"]:
        service = frame["service"]
        if user_affirms(frame):
            slots_with_proposed_values = get_slots_with_requested_confirmation(
                get_frame(prev_sys_turn, service)
            )
            if slots_with_proposed_values is not None:
                # ignore slots which the user overrides
                if (user_mentions := get_user_informed_slots(frame)) is not None:
                    slots_with_proposed_values = {
                        s: v
                        for s, v in slots_with_proposed_values.items()
                        if s not in user_mentions
                    }
                if slots_with_proposed_values:
                    user_actions = frame["actions"]
                    action_template = remove_affirm_action(user_actions)
                    for slot, value_info in slots_with_proposed_values.items():
                        new_action = deepcopy(action_template)
                        new_action["slot"] = slot
                        new_action["values"] = value_info["values"]
                        new_action["canonical_values"] = value_info["canonical_values"]
                        user_actions.append(new_action)


def annotate_slots_mentioned_as_alternatives(turn: dict[str, Any]):
    """Use the dialogue acts and call annotations to infer which
    slots triggered the failure during a transactional intent."""
    assert turn["speaker"] == "SYSTEM"
    for frame in turn["frames"]:
        actions = frame["actions"]
        notified_failure = any(a["act"] == "NOTIFY_FAILURE" for a in actions)
        if notified_failure:
            call_params = frame["service_call"]["parameters"]
            for a in (a for a in actions if a["act"] == "OFFER"):
                slot = a["slot"]
                # not all slots mentioned when an offer is made
                # are API parameters (eg `Buses_1.fare`
                # is mentioned when sys offers alternative bus tickets
                # but obvs not a BuyBusTickets param )
                if slot not in call_params:
                    continue
                slot_vals = a["values"]
                canonical_slot_vals = a["canonical_values"]
                call_result = call_params[slot]
                if call_result not in canonical_slot_vals:
                    a["metadata"] = nested_defaultdict(list, depth=2)
                    a["metadata"]["alternative_values"][slot].extend(slot_vals)
                    a["metadata"] = default_to_regular(a["metadata"])


def annotate_information_provided(dialogue: dict[str, Any]) -> dict[str, Any]:
    """Extract the information that the agent provides in response to user requests.

    Returns
    -------
    A mapping from service-intent types to slot, where the values contain the list
    of values provided by the system and turn index information.
    """

    info = nested_defaultdict(list, depth=2)
    user_turn = None
    for idx, turn in enumerate(dialogue["turns"]):
        match turn["speaker"]:
            case "USER":
                user_turn = turn
            case "SYSTEM":
                for frame in turn["frames"]:
                    for action in frame["actions"]:
                        if action["act"] == "INFORM":
                            service = frame["service"]
                            state = get_frame(user_turn, service)["state"]
                            service_intent = f"{service}.{state['active_intent']}"
                            info[service_intent][action["slot"]].append(
                                {"mentioned_in_turn": idx, "values": action["values"]}
                            )
    return {"provided_slot_values": default_to_regular(info)}


class MultiValueSlotMetadata(BaseModel):
    slot: str
    offered_values: list[str]
    offered_canonical_values: list[str]
    utterance: str


def annotate_user_selected_value_index(
    turn: dict[str, Any], multi_value_offer: MultiValueSlotMetadata
):
    """Annotates the index of the slot value SELECTed by the user on the corresponding action."""
    # system did not offer multiple values, nothing to annotate
    if multi_value_offer is None:
        return
    for frame in turn["frames"]:
        actions = frame["actions"]
        for a in actions:
            if a["act"] == "SELECT":
                assert a["slot"] == multi_value_offer.slot
                assert len(a["values"]) == 1
                index = multi_value_offer.offered_canonical_values.index(
                    a["canonical_values"][0]
                )
                assert (
                    a["values"][0] in turn["utterance"]
                ), "User did not specify selected value"
                a["metadata"] = {}
                a["metadata"]["slot_selection"] = {
                    "index": index,
                    "mentioned_in_utterance": True,
                }


def get_multi_value_offer_metadata(
    turn: dict[str, Any], **kwargs
) -> Optional[MultiValueSlotMetadata]:
    """Returns information about multiple entities proposed by the system in a
    single turn, if it exists."""

    def multi_value_offer_meta_factory():
        return {
            "slot": None,
            "offered_values": None,
            "offered_canonical_values": None,
            "utterance": turn["utterance"],
        }

    def assert_action_order_matches_nlg(
        utterance: str, values: list[str], canonical_vals: list[str]
    ):
        starts = [(v, utterance.index(v)) for v in values]
        sorted_starts = sorted(starts, key=itemgetter(1))
        # assume the order of values list is the NLG order
        try:
            assert starts == sorted_starts
        except AssertionError:
            # reorder value list according to the NLG to ensure
            # index is correct with respect to the SYSTEM utterance
            sorting_idxs = sorted(range(len(values)), key=lambda i: starts[i][1])
            values_copy = deepcopy(values)
            canonical_values_copy = deepcopy(canonical_vals)
            values.clear()
            canonical_vals.clear()
            for idx in sorting_idxs:
                values.append(values_copy[idx])
                canonical_vals.append(canonical_values_copy[idx])

    for frame in turn["frames"]:
        actions = frame["actions"]
        for a in actions:
            multi_value_offer = a["act"] == "OFFER" and len(a["canonical_values"]) > 1
            if multi_value_offer:
                vals = a["values"]
                canonical_vals = a["canonical_values"]
                assert_action_order_matches_nlg(turn["utterance"], vals, canonical_vals)
                metadata = multi_value_offer_meta_factory()
                metadata["slot"] = a["slot"]
                metadata["offered_values"] = vals
                metadata["offered_canonical_values"] = canonical_vals
                return MultiValueSlotMetadata.model_validate(metadata)
    return
