#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import re
from collections import defaultdict
from typing import Literal, Optional

from pytod.command import ServiceCommand
from pytod.pytod_types.aliases import DialogueID, IntentName, ServiceName, SlotName
from pytod.simulation.command import Command
from pytod.simulation.command_registry import command_registry

MULTIPLE_VALUE_SEP = "###"
SLOT_VALUE_SEP = "==="
ACT_SLOT_SEP = "<<<"


def actions_iterator(
    frame: dict, patterns: Optional[list[str] | dict[str, list[str]]] = None
) -> tuple[Optional[str], dict]:
    """
    Iterate through actions in a frame.

    Parameters
    ----------
    frame
        An SGD-format semantic frame.
    patterns
        If supplied, only actions whose ``act`` field is matched by at least one pattern are
        returned.
    """

    for act_dict in frame.get("actions", []):
        if patterns:
            for pattern in patterns:
                if re.search(pattern, act_dict["act"]):
                    yield pattern, act_dict
        else:
            yield None, act_dict


def get_turn_actions(
    turn: dict,
    act_patterns: Optional[list[str]] = None,
    service_patterns: Optional[list[str]] = None,
    use_lowercase: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    act_slot_sep: str = ACT_SLOT_SEP,
    multiple_val_sep: str = MULTIPLE_VALUE_SEP,
    key: Literal["values", "canonical_values"] = "values",
) -> dict[str, list[str]]:
    """
    Retrieve actions from a given dialogue turn. An action is a parametrised dialogue act
    (e.g., INFORM(price=cheap)).

    Parameters
    ----------
    turn
        Contains turn and annotations, with the structure::

            {
            'frames': [
                    {
                        'actions': dict,
                        'service': str,
                        'slots': list[dict], can be empty if no slots are mentioned
                        (e.g., "I want to eat.") , in SYS turns or if the USER requests
                        a slot (e.g., address). The latter is tracked in the``'state'`` dict.
                        'state': dict
                    },
                    ...
                ],
            'speaker': 'USER' or 'SYSTEM',
            'utterance': str,

            }

        The ``'actions'`` dictionary has structure::

            {
            'act': str (name of the act, e.g., INFORM_INTENT(intent=findRestaurant), REQUEST(slot))
            'canonical_values': [str] (name of the acts).
                It can be the same as value for non-categorical slots.
                Empty for some acts (e.g., GOODBYE)
            'slot': str, (name of the slot that parametrizes the action,
                e.g., 'city'. Can be "" (e.g., GOODBYE())
            'values': [str], (value of the slot, e.g "San Jose").
                Empty for some acts (e.g., GOODBYE()), or if the user
                makes a request (e.g., REQUEST('street_address'))
            }

        When the user has specified all the constraints (e.g., restaurant type and location),
         the next ``'SYSTEM'`` turn has the following _additional_ keys
         of the ``'actions'`` dictionary:

            {
            'service_call': {'method': str, same as the intent,
                'parameters': {slot:value} specified by user}
            'service_result': [dict[str, str], ...] where each dict maps
                properties of the entity retrieved to their
                vals. Structure depends on the entity retrieved.
            }

        The dicts of the ``'slots'`` list have structure:

            {
            'exclusive_end': int (char in ``turn['utterance']`` where the slot value ends)
            'slot': str, name of the slot
            'start': int (char in ``turn['utterance']`` where the slot value starts)
            }

        The ``'state'`` dictionary has the structure::

            {
            'active_intent': str, name of the intent active at the current turn,
            'requested_slots': [str], slots the user requested in the current turn
            'slot_values': dict['str', list[str]], mapping of slots to values
                specified by USER up to current turn
            }
    act_patterns,
        Optionally specify these patterns to return only specific actions.
        The patterns are matched against
        ``turn['frames'][frame_idx]['actions'][action_idx]['act']
        for all frames and actions using ``re.search``.


    Returns
    -------
    Actions in the current dialogue turn.

    """

    # TODO: UPDATE DOCS
    # TODO: TEST THIS FUNCTION, VERY IMPORTANT => can a string end in slot value sep?

    formatted_actions = defaultdict(list)

    for frame in turn.get("frames", []):
        service = frame.get("service", "")
        # return patterns only for certain services
        if service_patterns:
            if not any((re.search(pattern, service) for pattern in service_patterns)):
                continue
        for _, action_dict in actions_iterator(frame, patterns=act_patterns):
            # empty frame
            if action_dict is None:
                continue
            # acts without parameters (e.g., goodbye)
            slot = ""
            if "slot" in action_dict:
                slot = action_dict["slot"] if action_dict["slot"] else ""
            val = ""
            if slot:
                val = (
                    f"{multiple_val_sep}".join(action_dict[key])
                    if action_dict[key]
                    else ""
                )

            if slot and val:
                f_action = (
                    f"{action_dict['act']}{act_slot_sep}{slot}{slot_value_sep}{val}"
                )
            else:
                if slot:
                    f_action = f"{action_dict['act']}{act_slot_sep}{slot}"
                else:
                    f_action = f"{action_dict['act']}"
            f_action = f_action.lower() if use_lowercase else f_action
            formatted_actions[service].append(f_action)

    return formatted_actions


def get_params(
    actions: list[str],
    include_values: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    act_slot_sep: str = ACT_SLOT_SEP,
    use_lowercase: bool = True,
) -> list[str]:
    """Retrieve action parameters given a formatted action.

    # TODO: DESCRIBE ACTION FORMAT OR X-REF TO REL DOC
    """
    action_params = []
    for f_action in actions:
        this_action_params = "".join(f_action.split(act_slot_sep)[1:])
        if this_action_params:
            if use_lowercase:
                action_params.append(this_action_params.lower())
            else:
                action_params.append(this_action_params)
    if not include_values:
        action_params = [
            "".join(action_param.split(slot_value_sep)[0])
            for action_param in action_params
        ]
    return action_params


def get_turn_action_params(
    turn: dict,
    act_patterns: Optional[list[str]] = None,
    service_patterns: Optional[list[str]] = None,
    include_values: bool = True,
    use_lowercase: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    multiple_val_sep: str = MULTIPLE_VALUE_SEP,
    key: Literal["values", "canonical_values"] = "values",
) -> defaultdict[str, list[str]]:
    """Obtain the parameters for all the actions in the turn.
    # TODO: ADD DOCS
    """

    turn_actions_by_service = get_turn_actions(
        turn,
        act_patterns=act_patterns,
        service_patterns=service_patterns,
        use_lowercase=use_lowercase,
        slot_value_sep=slot_value_sep,
        multiple_val_sep=multiple_val_sep,
        key=key,
    )

    turn_action_params = defaultdict(list)
    for service, formatted_actions in turn_actions_by_service.items():
        action_params = get_params(
            formatted_actions,
            include_values=include_values,
            slot_value_sep=slot_value_sep,
            use_lowercase=use_lowercase,
        )
        turn_action_params[service] = action_params

    return turn_action_params


def instantiate_command(
    dial_id: DialogueID,
    cmd_name: IntentName,
    service: ServiceName,
    parameters: dict[SlotName, str],
    service_schema: ServiceCommand,
) -> Command:
    cmd = command_registry.get(name=cmd_name, service=service)
    cmd = cmd.build(dial_id, command_schema=service_schema)
    cmd.state = parameters
    return cmd
