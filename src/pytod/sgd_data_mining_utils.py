#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import Any, Optional

from pytod.pytod_types.aliases import (
    CanonicalValueCollector,
    ParaphraseValueCollector,
    ServiceName,
    SGDAuthor,
)
from pytod.sgd_utils import (
    get_slots_with_requested_confirmation,
    get_user_informed_slots,
    user_affirms,
)
from pytod.utils import store_data


def collect_request_action_parameters_data(
    turn: dict[str, Any], dialogue_id: str, **kwargs
):
    """Collects information about the slots which appear as arguments of `REQUEST`
    actions.


    Notes
    ------
    See SGD corpus files for `turn` content information.
    """

    def is_parametrised_request(action: dict[str, Any]) -> bool:
        return action["act"] == "REQUEST" and action["values"]

    metadata_collector = kwargs.get("request_parametrisation_collector", None)
    if metadata_collector is None:
        return
    for frame in turn["frames"]:
        param_request_actions = [
            a for a in frame["actions"] if is_parametrised_request(a)
        ]
        for confirmation_request in param_request_actions:
            store_key = [
                turn["speaker"].lower(),
                "multivalue"
                if len(confirmation_request["values"]) > 1
                else "singlevalue",
                frame["service"],
                confirmation_request["slot"],
            ]
            store_data(dialogue_id, metadata_collector, store_key)


def collect_user_slot_confirmation_data(
    user_turn: dict[str, Any], prev_sys_turn: Optional[dict[str, Any]], **kwargs
):
    """Collect information about the occurrence of the `AFFIRM` dialogue act.

    user_turn, prev_sys_turn


    Notes
    -----
    See SGD corpus files for `user_turn` and `prev_sys_turn` content information.
    """

    metadata_collector = kwargs.get("user_slot_confirmation_collector", None)
    if metadata_collector is None:
        return
    for frame in user_turn["frames"]:
        service = frame["service"]
        active_intent = frame["state"]["active_intent"]
        if user_affirms(frame):
            assert prev_sys_turn is not None
            [sys_frame] = [
                f for f in prev_sys_turn["frames"] if f["service"] == service
            ]
            maybe_slots = set(get_slots_with_requested_confirmation(sys_frame).keys())
            if maybe_slots is not None:
                user_mentions = get_user_informed_slots(frame)
                if user_mentions is not None:
                    maybe_slots = maybe_slots.difference(user_mentions)
                metadata_collector[service][active_intent].update(maybe_slots)


def collect_slot_value_paraphrases(
    author: SGDAuthor, service: ServiceName, actions: list[dict[str, Any]], **kwargs
):
    """Updates a map of slot canonical values to values represented in the natural language
    utterances. This is useful for non-categorical (aka open value) slots, for which many forms
    can exist."""

    def collect_paraphrases(
        collector: ParaphraseValueCollector,
        canonical_value_collector: Optional[CanonicalValueCollector],
        service: ServiceName,
        acts: set[str],
        actions: list[dict[str, Any]],
        lowercase_service: bool = False,
    ):
        service = service.lower() if lowercase_service else service
        for action in actions:
            if action["act"] in acts:
                match (values := action["values"]):
                    # no parameters: for slot filling without value proposal (REQUEST(slot))
                    #  and entity selection (SELECT())
                    case []:
                        continue
                    case _:
                        assert len(values) == len(
                            canonical_values := action["canonical_values"]
                        )
                        for value, canonical_value in zip(values, canonical_values):
                            collector[service][canonical_value][action["slot"]].add(
                                value
                            )
                            if canonical_value_collector is not None:
                                canonical_value_collector[canonical_value].add(value)

    value_paraphrase_collector = kwargs.get("paraphrase_map_collector", None)
    if value_paraphrase_collector is None:
        return
    match author:
        case "USER":
            acts = {"INFORM", "SELECT"}
        case "SYSTEM":
            acts = {"REQUEST", "OFFER", "CONFIRM"}
        case _:
            raise ValueError(f"Unknown author {author}")

    collect_paraphrases(
        value_paraphrase_collector,
        kwargs.get("canonical_values_collector", None),
        service,
        acts,
        actions,
        lowercase_service=kwargs.get("lowercase_service", False),
    )
