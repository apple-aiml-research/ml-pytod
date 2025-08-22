#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This registry contains routines for converting system dialogue acts to PyTOD
instructions or instruction templates. The backend simulation is therefore implemented
 in this module. The routines output certain pytod_types (see `pytod.pytod_types.transcript`)
 which are abstract (ie do not contain variable references). These are passed back to the
 interpreter, whose output is passed to the pytod.assignments.Assignments class, where
 the variables are assigned"""

import random
from itertools import chain
from typing import Literal, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError

from pytod.command import ServiceCall
from pytod.interpreter.interpreter_routines_utils import update_method_name
from pytod.interpreter.metadata import SKIP_QUERY_BINDING
from pytod.pytod_types.sgd_conversation import SystemAction, SystemDialogueAct
from pytod.pytod_types.transcript import (
    BackendHint,
    BackendHintTemplate,
    BackendNotification,
    BackendNotificationTemplate,
    ProgramStatement,
    ProgramStatementTemplate,
    TemplateField,
)
from pytod.text_formatter import PythonFunctionServiceCallFormatter
from pytod.transcript import APIInfo
from pytod.utils import dispatch_on_value, snake_case, stringify_values

SystemTurnTag = Literal[
    "slot_filling_hint",
    "assign_query_result",
    "task_performed",
    "inform_entity_count_signal",
    "task_status_signal",
    "offer_alternative_signal",
    "require_confirmation_hint",
]


def _format_call_params(
    params: dict[str, str], skip_quotes: Optional[set[str]] = None
) -> str:
    formatter = PythonFunctionServiceCallFormatter(
        {"quote_values": True, "skip_quotes": skip_quotes}
    )
    dialog_template = formatter.call_to_text(
        (
            ServiceCall.model_validate(
                {
                    "service": "",
                    "method": "",
                    "parameters": params,
                }
            )
        )
    )
    return dialog_template


@dispatch_on_value
def interpret_system_actions(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[
    BackendHint | BackendNotification | ProgramStatement | ProgramStatementTemplate
]:
    raise ValueError(f"Unknown tag {tag}")


@interpret_system_actions.register("slot_filling_hint")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendHintTemplate]:
    """Tags turns where the agent asks the user to provide the
    value of a required slot.


    Examples
    --------
    agent: What city would you like me to search in?
    user: Search in Pleasanton.
    5 x1.city="Pleasanton"
    ---> 6 Hint('request value: cuisine (American, Indian or other)')
    7 say(x6)
    agent: What type of cuisine are you looking for,
        for example American, Indian, or anything else?
    """

    hint_template = instantiate(config.hints.slot_filling_template)
    assert hint_template.tag == tag
    system_requests = actions.pop(SystemDialogueAct.REQUEST)
    multiple_slot_filling = config.hints.multiple_slot_filling
    unfilled_slots = api_info.unfilled_slots
    show_values = config.show_values_samples_in_slot_filling_hints
    if multiple_slot_filling:
        ground_truth_hints = []
        for action_to_hint in system_requests:
            ground_truth_slot = action_to_hint.slot
            try:
                unfilled_slots.remove(ground_truth_slot)
            except ValueError:
                assert ground_truth_slot in api_info.wildcard_slots
            value_examples = (
                stringify_values(action_to_hint.values) if show_values else ""
            )
            dialog = hint_template.dialog_template.format(
                arg=action_to_hint.slot,
                value_examples=value_examples,
            ).strip()
            ground_truth_hints.append(
                BackendHintTemplate(
                    dialog_template=dialog,
                    fields=[],
                    tag=tag,
                    is_assignable=hint_template.is_assignable,
                    origin_fields=hint_template.origin_fields,
                    origin_template=hint_template.origin_template,
                )
            )
        other_hints = []
        for slot in unfilled_slots:
            dialog = hint_template.dialog_template.format(
                arg=slot, value_examples=""
            ).strip()
            other_hints.append(
                BackendHintTemplate(
                    dialog_template=dialog,
                    fields=[],
                    tag=tag,
                    is_assignable=hint_template.is_assignable,
                    origin_fields=None,
                    origin_template=None,
                    pass_to_nlg=False,
                )
            )
        to_hint = ground_truth_hints + other_hints
        if config.hints.randomise_hint_order:
            random.shuffle(to_hint)
        return to_hint
    assert len(system_requests) == 1, (
        f"Multiple slot filling disabled but encountered actions: {system_requests}"
        f"Ensure your filters contain a `get_turns_before_multiple_slot_filling_turn` step"
    )
    to_hint = system_requests[0]
    dialog = hint_template.dialog_template.format(
        arg=to_hint.slot,
        value_examples=stringify_values(to_hint.values),
    ).strip()
    return [
        BackendHintTemplate(
            dialog_template=dialog,
            fields=[],
            tag=tag,
            is_assignable=hint_template.is_assignable,
            origin_fields=hint_template.origin_fields,
            origin_template=hint_template.origin_template,
        ),
    ]


@interpret_system_actions.register("inform_entity_count_signal")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendNotificationTemplate] | list[ProgramStatementTemplate]:
    count_actions = actions.pop(SystemDialogueAct.INFORM_COUNT)
    assert len(count_actions) == 1
    count = count_actions[0].values[0]
    args = {"tag": tag}
    try:
        template_obj = instantiate(config.notifications.inform_entity_count_template)
        template = template_obj.dialog_template.format(
            count=count, current_call=f"{{current_call}}"  # noqa
        )
        return_type = BackendNotificationTemplate
        args.update({"dialog_template": template})
    # v0.3.2
    except ConfigAttributeError:
        template_obj = instantiate(config.actions.inform_entity_count_template)
        # override the default show({current_call}) with slice({current_call})
        # for certain services where multiple entities are mentioned in the
        # same utterance (Media_*, Movies_1).
        if hasattr(template_obj.metadata, "apply_custom_template"):
            if api_info.service in SKIP_QUERY_BINDING and api_info.function in chain(
                *SKIP_QUERY_BINDING.values()
            ):
                template_obj.expression_template = template_obj.metadata.custom_template
        template = template_obj.expression_template.format(
            count=count, current_call=f"{{current_call}}"  # noqa
        )
        return_type = ProgramStatementTemplate
        args.update({"expression_template": template})

    args.update({"fields": [f for f in template_obj.fields if f.field != "count"]})
    return [return_type(**args)]


@interpret_system_actions.register("require_confirmation_hint")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendHintTemplate]:
    """This routine is responsible for outputting confirmation hints template.

    Example
    -------
    29 Hint('ask the user to confirm:
        event_name=x25.event_name,
        number_of_tickets=x25.number_of_tickets,
        date=x25.date,
        city=x25.city'
    )
    30 say(x29)
    agent: Okay then, so you want 4 tickets for the DC United Vs
    Revolution soccer match that takes place in Washington D.C. tomorrow?
    """

    template_obj = instantiate(config.hints.confirmation_template)
    template = template_obj.dialog_template
    to_confirm_actions = actions.pop(SystemDialogueAct.CONFIRM)
    all_confirmed_slots = api_info.system_confirmed_slots
    if val_refs := config.include_slot_value_references_in_confirmation_hints:
        to_confirm_arguments = [
            {a.slot: f"{{current_call}}.{a.slot}" for a in to_confirm_actions}
        ]
    else:
        to_confirm_arguments = [{a.slot: "" for a in to_confirm_actions}]
    # slots that are not actually re-confirmed
    not_reconfirmed = set()
    if config.multiple_confirmation_hints:
        single_hint_template = to_confirm_arguments.pop()
        already_confirmed = list(single_hint_template.keys())
        to_confirm_arguments = [
            {slot: single_hint_template[slot]} for slot in single_hint_template
        ]
        for slot in all_confirmed_slots:
            if slot not in already_confirmed:
                not_reconfirmed.add(slot)
                val = f"{{current_call}}.{slot}" if val_refs else ""
                to_confirm_arguments.append({slot: val})

    templates = []
    for args in to_confirm_arguments:
        arg_string = _format_call_params(args, skip_quotes={a for a in args})
        arg_string = arg_string.replace('"', "")
        if not_reconfirmed.intersection(args.keys()):
            nlg_control_params = {
                "pass_to_nlg": False,
                "origin_template": None,
                "origin_fields": None,
            }
        else:
            nlg_control_params = {
                "pass_to_nlg": True,
                "origin_template": template_obj.origin_template,
                "origin_fields": template_obj.origin_fields,
            }
        templates.append(
            BackendHintTemplate(
                dialog_template=template.format(entity_properties=arg_string[1:-1]),
                fields=[TemplateField(is_variable=True, field="current_call")],
                tag=tag,
                **nlg_control_params,
            )
        )
    random.shuffle(templates)
    return templates


@interpret_system_actions.register("offer_alternative_signal")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendNotificationTemplate] | list[BackendHintTemplate]:
    """Tags turns where the agent makes an alternative offer to the user
    in the wake of a transaction failure.

    Example
    -------

    user: Yeah, that's correct. Also, what's their address, and do they serve liquor there?
    20 confirm(x14)
    21 Signal: There was an error while completing the task
    ---> 22 Hint(
        'alternative:
            restaurant_name=x14.restaurant_name,
            party_size=x14.party_size,
            date=x14.date,
            time="1:30 pm"'
        )
    23 say(x13.street_address, x13.serves_alcohol, x21, x22)
    agent: They're located at 500 Main Street, and they don't serve alcohol, sorry.
        I'm sorry but that time is all booked.
        Would you like a reservation for 2 at Baci Bistro And Bar for 1:30 pm today instead?
    """
    to_offer = actions.pop(SystemDialogueAct.OFFER)
    try:
        template = instantiate(config.notifications.results_alternative_template)
    # v0.3.2
    except ConfigAttributeError:
        template = instantiate(config.hints.results_alternative_template)
    to_offer_params = {}
    skip_quotes = set()
    for offer_action in to_offer:
        this_action_slot = offer_action.slot
        if config.include_slot_value_references_in_alternative_hints:
            action_meta = offer_action.metadata
            if action_meta is None:
                skip_quotes.add(this_action_slot)
                to_offer_params.update(
                    {this_action_slot: f"{{current_call}}.{this_action_slot}"}
                )
            else:
                proposed_value = action_meta["alternative_values"][this_action_slot][0]
                to_offer_params.update({this_action_slot: proposed_value})
        else:
            to_offer_params.update({this_action_slot: ""})

    formatter = PythonFunctionServiceCallFormatter(
        {"convert_camel_case": True, "quote_values": True, "skip_quote": skip_quotes}
    )
    offer_params = formatter.call_to_text(
        ServiceCall.model_validate(
            {
                "service": "",
                "method": "",
                "parameters": to_offer_params,
            },
        )
    )[1:-1]
    template.dialog_template = template.dialog_template.format(
        offer_params=offer_params
    )
    fields = [TemplateField(field="current_call", is_variable=True)]
    template.fields = fields
    return [template]


@interpret_system_actions.register("task_status_signal")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendNotificationTemplate]:
    """Tags turns where the agent has information regarding task completion
    from the execution environment. Only task failure is included in the
    PyTOD transcripts.

    Example
    -------
    user: I do not find that interesting. Is there anything else?
    21 next(x13)
    ---> 22 Signal: No results found
    """
    if SystemDialogueAct.NOTIFY_SUCCESS in actions:
        actions.pop(SystemDialogueAct.NOTIFY_SUCCESS)
        return []
    else:
        actions.pop(SystemDialogueAct.NOTIFY_FAILURE)
        template = "{failure}"
    return [
        BackendNotificationTemplate(
            dialog_template=template,
            fields=[TemplateField(is_variable=False, field=template[1:-1])],
            tag=tag,
            is_assignable=True,
            pass_to_nlg=True,
            origin_template="{current_call}",
            origin_fields=[TemplateField(is_variable=True, field="current_call")],
        )
    ]


@interpret_system_actions.register("assign_query_result")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """Tags turns where search intents return 0 or more results.

    Example
    -------
    user: What else can you find?
        Also, I just realized I actually need to book four tickets for the 6th.
    8 x1.travelers=4; x1.leaving_date="the 6th"
    9 len(x1) # -> 9
    ---> 10 next(x1)
    11 say(x9)
    agent: Ok, I found 9 options.
        There's one with 0 transfers for $21 leaving at 9:40 am. How about that one?
    """
    fields = [
        TemplateField(field="current_call", is_variable=True),
    ]
    return [
        ProgramStatementTemplate(
            expression_template="next({current_call})", fields=fields, tag=tag
        )
    ]


@interpret_system_actions.register("task_performed")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """Tags turns where a call to a transactional intent is performed. Perform
    meaning is grounded to task success - entities returned by these intents
    are not modified in SGD but their properties can be queried or carried over to
    other intents.

    Example
    -------
    agent: Before watching it, please confirm the following details:
        playing Green Book without subtitles.
    user: Sure.
    18 confirm(x15)
    ---> 19 perform(x15)
    ---> 20 say(x19)
    """
    return [
        ProgramStatementTemplate(
            expression_template="perform({current_call})",
            fields=[TemplateField(field="current_call", is_variable=True)],
            tag=tag,
        ),
        ProgramStatementTemplate(
            expression_template="{current_transaction_index}",
            fields=[TemplateField(field="current_transaction_index", is_variable=True)],
            tag=tag,
            is_assignable=False,
            pass_to_nlg=True,
            is_inlined=True,
        ),
    ]


@interpret_system_actions.register("prompt_more_help_required")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[BackendHintTemplate]:
    """Tags turns where the agent asks the user if more help is required.

    Example
    -------
    ...
    user: What else you can find?
    7 next(x1)
    8 Signal: No results found
    --->9 Hint('ask the user if they require further assistance')
    10 say(x7, x8, -->x9<---)
    agent: All my research are fialed because I can't maching your preferences.
        Can I help you with somethink else?


    Notes
    -----
    Could really do with running a spell checker over SGD :).
    """
    template = config.hints.prompt_more_help_required
    try:
        actions.pop(SystemDialogueAct.REQ_MORE)
        if config.ground_req_more:
            hint_template = instantiate(template)
            return [hint_template]
    except KeyError:
        # this means the action was not annotated in the corpus
        # to start with
        if config.ground_req_more:
            hint_template = instantiate(
                template, pass_to_nlg=False, origin_template=None, origin_fields=None
            )
            return [hint_template]
    return []


@interpret_system_actions.register("task_suggestion")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement | ProgramStatementTemplate]:
    """
    This routine is responsible for:

        - outputting `suggest(task=..)` instruction
        - outputting a program template to ensure the variable bound
        to said instruction is referenced in the subsequent `say` call

    Example
    -------
    agent: I have found 1 movie. Would you like to watch Say Anything?
    user: Say Anything sounds like a great movie.
    11 select(title="Say Anything", from_results=x1)
    ---> 12 suggest(task=media_1_play_movie)
    ---> 13 say(x12)
    agent: Would you like begin watching the movie?
    """

    annotated = True
    try:
        [suggested_task] = actions.pop(SystemDialogueAct.OFFER_INTENT)
        task_name = suggested_task.canonical_values[0]
    except KeyError:
        task_name = api_info.followup_command
        annotated = False
    service_call_dict = {
        "method": "",
        "parameters": {},
        "service": "",  # not relevant for call formatting
    }
    update_method_name(
        config.function_naming_convention,
        service_call_dict,
        task_name,
        api_info,
        config.actions.use_snake_case,
        config.actions.skip_service_variations,
        config.actions.anonymize_service,
    )
    task_name = service_call_dict.pop("method")
    suggestion_arg = config.actions.suggestion_tool_arg_name
    service_call_dict["parameters"].update(
        {
            suggestion_arg: snake_case(task_name)
            if config.actions.use_snake_case
            else task_name,
        }
    )
    service_call_dict["method"] = config.actions.task_suggestion_tool_name
    service_call = ServiceCall.model_validate(service_call_dict)
    formatter = PythonFunctionServiceCallFormatter(
        {
            "convert_camel_case": config.actions.use_snake_case,
            "quote_values": True,
            # "skip_quote": {suggestion_arg},
        }
    )
    expression = formatter.call_to_text(service_call)
    statements = [
        ProgramStatement(
            tag=tag,
            expression=expression,
            service=api_info.service,
            intent=api_info.function,
        )
    ]
    # annotated=False when this action is not annotated in the system actions
    #  -  we created the tag so that the model learns to select the action
    if annotated:
        statements.append(
            ProgramStatementTemplate(
                expression_template="{current_suggestion}",
                fields=[TemplateField(field="current_suggestion", is_variable=True)],
                tag=tag,
                is_assignable=False,
                pass_to_nlg=True,
                is_inlined=True,
                service=api_info.service,
                intent=api_info.function,
            )
        )
    return statements


@interpret_system_actions.register("end_conversation")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """Tags the turns at the end of conversation."""
    actions.pop(SystemDialogueAct.GOODBYE)
    return [
        ProgramStatement(expression="say()", tag=tag),
    ]


@interpret_system_actions.register("_remove_offers")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list:
    """Remove OFFER action.

    The algorithm will raise a PolicyError unless all actions have been interpreted."""
    actions.pop(SystemDialogueAct.OFFER)
    return []


@interpret_system_actions.register("_remove_information_provided")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list:
    """Remove INFORM action.

    The algorithm will raise a PolicyError unless all actions have been interpreted."""
    actions.pop(SystemDialogueAct.INFORM)
    return []


@interpret_system_actions.register("_remove_entity_counts")
def _(
    tag: SystemTurnTag,
    actions: dict[SystemDialogueAct, list[SystemAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list:
    """Remove INFORM_COUNT action.

    The algorithm will raise a PolicyError unless all actions have been interpreted."""
    actions.pop(SystemDialogueAct.INFORM_COUNT)
    return []
