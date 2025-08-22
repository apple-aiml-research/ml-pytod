#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Utilities for working with SGD data in raw format. These support
on-the-fly annotation and conversation building."""
import logging
from copy import deepcopy
from itertools import chain
from typing import Any, Literal, Optional

from pytod.pytod_types.aliases import ServiceName

logger = logging.getLogger(__name__)


def dialogue_iterator(
    dialogue: dict[str, Any], user: bool = True, system: bool = True
) -> dict:
    if (not user) and (not system):
        raise ValueError("At least a speaker needs to be specified!")

    drop_speaker = "USER" if not user else "SYSTEM" if not system else ""

    for turn in dialogue["turns"]:
        if drop_speaker and turn.get("speaker", "") == drop_speaker:
            continue
        else:
            yield turn


def get_service_by_turn(dialogue: dict[str, Any]) -> list[list[str]]:
    """Return the services in every dialogue turn. List returned has length
    of the number of dialogue turns.

    Parameters
    ----------
    dialogue
        Nested dictionary of SGD-format dialogue.

    Returns
    -------
    services
        Each sublist contains the services annotated at a given turn.
    """

    services = []
    for turn in dialogue_iterator(dialogue):
        this_turn_services = []
        for frame in turn["frames"]:
            service = frame["service"]
            this_turn_services.append(service)
        services.append(this_turn_services)
    return services


def get_dialogue_services(dialogue: dict[str, Any]) -> list[str]:
    """Returns the intents in a dialogue.

    Parameters
    ----------
    dialogue
        See get_service_by_turn

    Returns
    -------
    intents
        A list of services annotating the dialogue.
    """

    return list(set(chain(*get_service_by_turn(dialogue))))


def get_intent_by_turn(
    dialogue: dict[str, Any], exclude_none: bool = True
) -> list[list[str]]:
    """Return the active intents in every dialogue turn. List returned has length
    of the number of user turns.

    Parameters
    ----------
    dialogue
        Nested dictionary of SGD-format dialogue.
    exclude_none
        If True, the `NONE` intent is not included in the intents set.


    Returns
    -------
    intents
        Each sublist contains the intents expressed by the user at a given turn.
    """
    intents = []
    for turn in dialogue_iterator(dialogue, user=True, system=False):
        this_turn_intents = []
        prev_service_intent = ""
        for frame in turn["frames"]:
            intent = frame["state"]["active_intent"]
            service = frame["service"]
            service_intent = f"{service}.{intent}"
            if exclude_none:
                if intent == "NONE":
                    assert prev_service_intent and "NONE" not in prev_service_intent
                    this_turn_intents.append(prev_service_intent)
                else:
                    this_turn_intents.append(service_intent)
            else:
                this_turn_intents.append(intent)
            prev_service_intent = service_intent
        intents.append(this_turn_intents)

    return intents


def get_frame(turn: dict[str, Any], service: str) -> dict:
    for frame in turn["frames"]:
        if frame["service"] == service:
            return frame


def get_active_service(dialogue: dict[str, Any], turn_idx: int) -> str:
    all_turns_services = get_service_by_turn(dialogue)
    req_turn_services = all_turns_services[turn_idx]
    if len(req_turn_services) == 1:
        return req_turn_services[0]
    while True:
        turn_idx -= 1
        prev_turn_services = all_turns_services[turn_idx]
        new_services = list(set(req_turn_services).difference(prev_turn_services))
        if len(new_services) == 1:
            return new_services[0]
        else:
            raise AssertionError("Could not determine active service")


def user_affirms(frame: dict[str, Any]) -> bool:
    """
    Parameters
    ----------
    frame
        Semantic frame in raw SGD format.
    """
    return any(a["act"] == "AFFIRM" for a in frame["actions"])


def get_slots_with_requested_confirmation(
    frame: dict[str, Any]
) -> Optional[dict[str, dict[Literal["values", "canonical_values"], list[str]]]]:
    """
    Parameters
    ----------
    frame
        Semantic frame in raw SGD format.
    """
    assert isinstance(frame, dict)

    def is_possible_value_confirmation(action: dict[str, Any]) -> bool:
        return action["act"] == "REQUEST" and action["values"]

    slots = {
        a["slot"]: {
            "values": deepcopy(a["values"]),
            "canonical_values": deepcopy(a["canonical_values"]),
        }
        for a in frame["actions"]
        if is_possible_value_confirmation(a)
    }
    return slots or None


def get_user_informed_slots(frame: dict[str, Any]) -> Optional[set[str]]:
    """
    Parameters
    ----------
    frame
        Semantic frame in raw SGD format.
    """
    return {a["slot"] for a in frame["actions"] if a["act"] == "INFORM"} or None


def is_dialogue_formality(frame: dict[str, Any], act: str) -> bool:
    """Marks frames where the user simply thanks the agent. These
    can appear at the end of the conversation when all tasks are
    completed or between task switches. At the end of the conversation
    it may be co-ordinated with `GOODBYE`."""

    if len(frame["actions"]) > 1:
        return False
    return frame["actions"][0]["act"] == act


def get_turn_level_services(dial: dict[str, Any]) -> dict[int, list[ServiceName]]:
    """Get the turn level services"""
    services = {}
    for i, turn in enumerate(dial["turns"]):
        if turn["speaker"] == "USER":
            services[i] = [frame_["service"] for frame_ in turn["frames"]]

    return services
