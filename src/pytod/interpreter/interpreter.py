#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This module is the entry point for converting the SGD semantic annotations
to `python` expressions and/or templates and backed hints/notifications messages and/or
templates."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional, Type, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.interpreter.utils import GrammarError, PolicyError, Tag
from pytod.pytod_types.aliases import ServiceName
from pytod.pytod_types.sgd_conversation import (
    Author,
    SystemAction,
    SystemDialogueAct,
    Turn,
    UserAction,
    UserDialogueAct,
)
from pytod.pytod_types.transcript import PyTODInstruction, PyTODInstructionTemplate
from pytod.transcript import APIInfo
from pytod.utils import typed_partial

logger = logging.getLogger(__name__)


# not all SGD dialogue acts have a python expression / backend message
# equivalent
UninterpretedDialogActs = dict[
    Author, Union[set[UserDialogueAct], set[SystemDialogueAct]]
]


def assert_performs_are_correct(combined_tags: list[Tag], system_turn: Turn):
    """The `perform` function should only appear in the transcript when the
    system makes a service call to a transactional intents."""
    if any(t.tag == "assign_query_result" for t in combined_tags):
        assert system_turn.service_call is not None


def assert_tags_correct(
    combined_tags: list[Tag],
    user_turn_tags: list[Tag],
    next_system_turn: Optional[Turn],
):
    """We expect to assign one or more tags from each system turn except when
    we are at the end of the conversation.

    Notes
    -----
    Sometimes the end of the conversation is not the end of the SGD conversation,
    case in which `next_system_turn` will be `None`.
    """
    try:
        assert len(combined_tags) > len(user_turn_tags)
    except AssertionError:
        assert (
            any(t.tag == "end_conversation" for t in user_turn_tags)
            or next_system_turn is None
        )


class UninterpretedActions(NamedTuple):
    """Collects actions that have not been converted to program statements."""

    user: dict[ServiceName, dict[UserDialogueAct, list[UserAction]]]
    system: dict[ServiceName, dict[SystemDialogueAct, list[SystemAction]]]


@dataclass
class TemplatesCollection:
    # the collection also contains expressions that are not templates
    # (eg first turns in all conversations are just expressions)
    expressions_and_templates: list[PyTODInstruction | PyTODInstructionTemplate]
    # some transitions in the SGD policy graph are not modelled
    uninterpreted_actions: Optional[UninterpretedActions] = None


def get_uninterpreted_dialogue_acts(
    interpreter_cfg: DictConfig,
) -> UninterpretedDialogActs:
    """Read the acts without `python` statement equivalent from the interpreter configuration."""
    skip_acts = {
        Author.USER: instantiate(interpreter_cfg.user.out_of_scope_acts),
        Author.SYSTEM: None,
    }
    if (
        to_skip_system := interpreter_cfg.backend.actions.out_of_scope_acts
    ) is not None:
        skip_acts.update({Author.SYSTEM: instantiate(to_skip_system)})

    return skip_acts


def collect_uninterpreted_actions(
    interpreter_cfg: DictConfig,
    system_actions: dict[ServiceName, dict[SystemDialogueAct, list[SystemAction]]],
    user_actions: dict[ServiceName, dict[UserDialogueAct, list[UserAction]]],
) -> Optional[UninterpretedActions]:
    """Collect actions that have not been converted to program statements."""

    skip_acts = get_uninterpreted_dialogue_acts(interpreter_cfg)
    outstanding_actions = {}
    if usr_not_converted := user_actions:
        usr_outstanding = {}
        for service in usr_not_converted:
            service_outstanding = {
                a: usr_not_converted[service][a]
                for a in usr_not_converted[service]
                if a not in skip_acts[Author.USER]
            }
            if service_outstanding:
                usr_outstanding[service] = service_outstanding
        if usr_outstanding:
            outstanding_actions.update({Author.USER: usr_outstanding})
    if sys_not_converted := system_actions:
        sys_outstanding = {}
        for service in sys_outstanding:
            service_outstanding = {
                a: sys_not_converted[service][a]
                for a in sys_not_converted[service]
                if a not in skip_acts[Author.SYSTEM]
            }
            if service_outstanding:
                sys_outstanding[service] = service_outstanding
        if sys_outstanding:
            outstanding_actions.update({Author.SYSTEM: sys_outstanding})

    user_outstanding = (
        outstanding_actions[Author.USER] if Author.USER in outstanding_actions else None
    )
    sys_outstanding = (
        outstanding_actions[Author.SYSTEM]
        if Author.SYSTEM in outstanding_actions
        else None
    )
    if user_outstanding is None and sys_outstanding is None:
        return
    return UninterpretedActions(system=sys_outstanding, user=user_outstanding)


def check_for_interpreter_errors(
    prev_system_turn: Turn,
    system_turn: Turn,
    outstanding_actions: Optional[UninterpretedActions],
    interpreter_cfg: DictConfig,
):
    """Assert on specific conversation structures that lead to annotated actions
    without a PyTOD instruction equivalent."""

    usr_uninterpreted_actions = dict[
        ServiceName, dict[UserDialogueAct, list[UserAction]]
    ]

    def unfulfilled_information_requests(
        user_not_interpreted: usr_uninterpreted_actions,
        prev_system_turn: Turn,
        system_turn: Turn,
        interpreter_cfg: DictConfig,
    ) -> bool:
        """REQUEST actions are not interpreted if the agent does not actually answer the
        question."""
        try:
            for service, uninterpreted_actions in user_not_interpreted.items():
                assert (
                    len(uninterpreted_actions) == 1
                    and UserDialogueAct.REQUEST in uninterpreted_actions
                )
                assert (
                    SystemDialogueAct.NOTIFY_FAILURE
                    in system_turn.system_actions_dict[service]
                )
        except AssertionError:
            return False
        return True

    def unfulfilled_slot_carryovers(
        user_not_interpreted: usr_uninterpreted_actions,
        prev_system_turn: Turn,
        system_turn: Turn,
        interpreter_cfg: DictConfig,
    ) -> bool:
        """There are uninterpreted carry-over actions if a transactional API call fails and we
        format the corresponding instruction as an assignment instead of a call."""
        try:
            assert (
                interpreter_cfg.user.user_task_retry_instruction_format == "assignment"
            )
            for service, usr_uninterpreted_actions in user_not_interpreted.items():
                assert UserDialogueAct.CARRY_OVER in usr_uninterpreted_actions
                assert (
                    SystemDialogueAct.CONFIRM
                    in system_turn.system_actions_dict[service]
                )
                assert (
                    SystemDialogueAct.NOTIFY_FAILURE
                    in prev_system_turn.system_actions_dict[service]
                )
        except AssertionError:
            return False
        return True

    if outstanding_actions is None:
        return
    msg = "The following {agent} actions {outstanding_actions} have not been converted."
    agent = ""
    not_interpreted = {}
    if outstanding_actions.user:
        not_interpreted.update({Author.USER: outstanding_actions.user})
        agent = "user"
    if outstanding_actions.system:
        agent = f"{agent} & system" if agent else "system"
        not_interpreted.update({Author.SYSTEM: outstanding_actions.system})
    if not_interpreted:
        # it can happen that we do not convert the REQUEST([slot]) action annotated
        # in the user turn because the request for info is represented as a positional
        # argument to say() but if the API fails, the system does not answer the question
        # (eg, train/14_00119)
        if len(not_interpreted) == 1 and Author.USER in not_interpreted:
            user_not_interpreted = not_interpreted[Author.USER]
            expected_residual_actions = (
                h(user_not_interpreted, prev_system_turn, system_turn, interpreter_cfg)
                for h in (unfulfilled_information_requests, unfulfilled_slot_carryovers)
            )
            if not any(expected_residual_actions):
                logger.error(
                    {msg.format(outstanding_actions=outstanding_actions, agent=agent)}
                )
                raise PolicyError(
                    f"{msg.format(outstanding_actions=outstanding_actions, agent=agent)}"
                )


def interpret_actions(
    interpreter_cfg: DictConfig,
    user_turn: Turn,
    next_system_turn: Turn,
    prev_system_turn: Optional[Turn],
    api_info: APIInfo,
) -> TemplatesCollection:
    """Interpret user and system actions from a turn pair to a collection
    of transcript elements or templates.

    The collection contains:

        - `python` expressions: only when an intent is invoked for the first time,
            without arguments carried over from other intents
        - `python` expressions templates: expressions for which the variable is not yet assigned
        - hints and notifications from the backend or their templates: hints/notifications from the
        backend simulation.

    grammar:
        A string identifying the grammar that is used for converting the semantic annotation
        to program statements.
    api_info
        An object which contains information such as the currently active intent and service.
    """

    grammar = interpreter_cfg.grammar
    assert grammar in [
        "v2",
        "v3",
    ], f"Only grammars v2, v3 are known to the interpreter, got {grammar}."
    if grammar == "v2":
        from pytod.interpreter.ontology_v2 import (
            combine_with_system_tags,
            get_user_turn_tags,
            interpret_system_actions,
            interpret_user_actions,
        )
    else:
        assert grammar == "v3"
        from pytod.interpreter.ontology_v3 import (
            combine_with_system_tags,
            get_user_turn_tags,
        )
        from pytod.interpreter.system_routines_registry_v3 import (
            interpret_system_actions,
        )
        from pytod.interpreter.user_routines_registry_v3 import interpret_user_actions

    def add_service_and_intent_info(instructions: list, tag: Tag):
        for instr in instructions:
            instr.service = tag.service
            instr.intent = tag.intent

    def tag_turn_pair(
        user_turn: Turn,
        prev_system_turn: Optional[Turn],
        next_system_turn: Optional[Turn],
        api_info: APIInfo,
    ) -> list[Tag]:
        """Tag turn pair with tags which reflect the user/system behaviour
        at this point in the dialog.

        Parameters
        ----------
        user_turn, prev_system_turn, next_system_turn
        api_info
            Tuple containing information about the active intent.
        """

        # tag the turns with semantic labels of PyTOD instructions
        user_turn_tags = get_user_turn_tags(user_turn, prev_system_turn, api_info)
        combined_tags = combine_with_system_tags(
            user_turn_tags, next_system_turn, api_info
        )
        assert_tags_correct(combined_tags, user_turn_tags, next_system_turn)
        assert_performs_are_correct(combined_tags, next_system_turn)

        return combined_tags

    if grammar == "v2":
        raise GrammarError("Backwards compatibility to 'v2' is not fully implemented")

    interpret_user_actions_: Type[interpret_user_actions] = typed_partial(
        partial(interpret_user_actions, api_info=api_info)
    )
    interpret_system_actions_: Type[interpret_system_actions] = typed_partial(
        partial(interpret_system_actions, api_info=api_info)
    )
    # tag turn-pair with a semantic labels corresponding to PyTOD instructions
    tags = tag_turn_pair(user_turn, prev_system_turn, next_system_turn, api_info)
    user_actions: dict[
        ServiceName, dict[UserDialogueAct, list[UserAction]]
    ] = user_turn.user_actions_dict
    # interpreter routines have side effects
    user_actions = deepcopy(user_actions)
    system_actions = None
    if next_system_turn is not None:
        system_actions: Optional[
            dict[ServiceName, dict[SystemDialogueAct, list[SystemAction]]]
        ] = next_system_turn.system_actions_dict
        system_actions = deepcopy(system_actions)
    instructions = []
    for tag in tags:
        assert (
            tag.service is not None
        ), "Tags should always contain a reference to the service they relate to"
        if tag.source_author == Author.USER:
            new_instructions = interpret_user_actions_(
                tag.tag, user_actions[tag.service], interpreter_cfg.user
            )
            add_service_and_intent_info(new_instructions, tag)
            instructions.extend(new_instructions)
        else:
            if system_actions is not None:
                new_instructions = interpret_system_actions_(
                    tag.tag, system_actions[tag.service], interpreter_cfg.backend
                )
                add_service_and_intent_info(new_instructions, tag)
                instructions.extend(new_instructions)
            else:
                system_actions = {}
    uninterpreted_actions = collect_uninterpreted_actions(
        interpreter_cfg, system_actions, user_actions
    )
    check_for_interpreter_errors(
        prev_system_turn, next_system_turn, uninterpreted_actions, interpreter_cfg
    )
    return TemplatesCollection(
        expressions_and_templates=instructions,
        uninterpreted_actions=uninterpreted_actions,
    )
