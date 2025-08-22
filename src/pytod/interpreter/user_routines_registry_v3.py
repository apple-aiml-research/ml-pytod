#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from typing import Literal

from omegaconf import DictConfig

from pytod.command import ServiceCall
from pytod.interpreter.assertions import assert_no_carry_over_overlap
from pytod.interpreter.interpreter_routines_utils import (
    cast_to_statement,
    update_method_name,
    update_service_call_dict,
)
from pytod.pytod_types.sgd_conversation import UserAction, UserDialogueAct
from pytod.pytod_types.transcript import (
    AttributeAccessTemplate,
    ProgramStatement,
    ProgramStatementTemplate,
    TemplateField,
)
from pytod.sgd_policy_assertions import assert_user_intent_information_behaviour
from pytod.text_formatter import PythonFunctionServiceCallFormatter, maybe_quote_or_cast
from pytod.transcript import APIInfo
from pytod.utils import dispatch_on_value


class InterpreterError(Exception):
    pass


UserTurnTag = Literal[
    "entity_selection",
    "entity_selection_with_new_task",
    "call",
    "call_with_params",
    "call_with_selected_params",
    "call_with_carry_over",
    "result_selection_by_indexing",
    "result_selection_by_indexing_with_new_task",
    "assignment",
    "confirmation",
    "entity_query",
    "results_iterator",
    "assign_query_result",
    "end_conversation",
]

call_turn_tags = [
    "call",
    "call_with_params",
    "call_with_selected_params",
    "call_with_carry_over",
]


logger = logging.getLogger(__name__)


@dispatch_on_value
def interpret_user_actions(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement | ProgramStatementTemplate]:
    raise ValueError(f"Unknown tag {tag}")


@interpret_user_actions.register("call")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """This tag is assigned to turns where the user specifies a new
    task without providing any slot values.


    Example
    ------
    user: I want to eat sth yummy.
    ---> restaurants_1_find_restaurant()
    """
    service_call_dict = {
        "method": "",
        "parameters": {},
        "service": "",  # not relevant for call formatting
    }
    assert_user_intent_information_behaviour(actions)
    # this should be just the bare function call (eg book_restaurant())
    assert all(
        (
            UserDialogueAct.CARRY_OVER not in actions,
            UserDialogueAct.INFORM not in actions,
            UserDialogueAct.SELECT not in actions,
        )
    )
    try:
        [intent_action] = actions.pop(UserDialogueAct.INFORM_INTENT)
    except KeyError:
        [intent_action] = actions.pop(UserDialogueAct.AFFIRM_INTENT)
    convention = config.function_naming_convention
    update_method_name(
        convention,
        service_call_dict,
        intent_action.canonical_values[0],
        api_info,
        config.use_snake_case,
        config.skip_service_variations,
        config.anonymize_service,
    )
    service_call = ServiceCall.model_validate(service_call_dict)
    return [cast_to_statement(service_call, tag)]


@interpret_user_actions.register("call_with_params")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """This tag is assigned to user turns where the user specifies
    a new task and provides one or more slot values.

    Example
    -------
    user: I'm ravenous, would like food, preferably in Dublin.
    ---> restaurants_1_find_restaurant(location="Dublin")
    """
    service_call_dict = {
        "method": "",
        "parameters": {},
        "service": "",
    }
    # these annotations are compatible with user specifying a new
    # intention, but are handled in different routines
    assert UserDialogueAct.CARRY_OVER not in actions
    assert UserDialogueAct.SELECT not in actions
    try:
        [intent_action] = actions.pop(UserDialogueAct.INFORM_INTENT)
    except KeyError:
        [intent_action] = actions.pop(UserDialogueAct.AFFIRM_INTENT)
    convention = config.function_naming_convention
    update_method_name(
        convention,
        service_call_dict,
        intent_action.canonical_values[0],
        api_info,
        config.use_snake_case,
        config.skip_service_variations,
        config.anonymize_service,
    )
    parameter_actions = actions.pop(UserDialogueAct.INFORM)
    unhappy_path_label = (
        "single-assignment-in-initial-intent-mention"
        if len(parameter_actions) == 1
        else "multi-assignment-in-initial-intent-mention"
    )
    for a in parameter_actions:
        assert len(a.values) == 1
        service_call_dict["parameters"].update({a.slot: a.values[0]})
    service_call = ServiceCall.model_validate(service_call_dict)
    return [
        cast_to_statement(
            service_call, tag, info={"unhappy_path_labels": [unhappy_path_label]}
        )
    ]


@interpret_user_actions.register("call_with_carry_over")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement | ProgramStatementTemplate]:
    """This tag is assigned to user turns where the user states
    a new task but some slots have already been specified in the
    dialogue history and should not be requested again by the agent.

    Example
    -------

    user: I want to check the balance of my checking account please
    banks_2_check_balance(account_type="checking")
    ....
    agent: Do you need to make any transfers?
    user: Yes please, send some money to Svetlana
    --> banks_2_transfer_money(
        account_type={call_or_entity_reference_{i}}.account_type,
        recipient_name="Svetlana"
    )
    """
    assert_no_carry_over_overlap(actions)
    service_call_dict = {
        "method": "",
        "parameters": {},
        "service": "",
    }
    carry_over = False
    template_fields = []
    skip_quote = set()
    if UserDialogueAct.CARRY_OVER in actions:
        carry_over = True
        wildcard_call_ref = config.reference_command_for_wildcard_carryover
        parameter_actions = actions.pop(UserDialogueAct.CARRY_OVER)
        update_service_call_dict(
            config.value_carryover,
            service_call_dict,
            template_fields,
            skip_quote,
            parameter_actions,
            reference_command_for_wildcard_carryover=wildcard_call_ref,
            system_notified_failure=api_info.notified_failure,
        )
    if UserDialogueAct.INFORM in actions:
        parameter_actions = actions.pop(UserDialogueAct.INFORM)
        update_service_call_dict(
            "natural_language",
            service_call_dict,
            template_fields,
            skip_quote,
            parameter_actions,
        )
    try:
        [intent_action] = actions.pop(UserDialogueAct.INFORM_INTENT)
    except KeyError:
        [intent_action] = actions.pop(UserDialogueAct.AFFIRM_INTENT)
    convention = config.function_naming_convention
    update_method_name(
        convention,
        service_call_dict,
        intent_action.canonical_values[0],
        api_info,
        config.use_snake_case,
        config.skip_service_variations,
        config.anonymize_service,
    )
    service_call = ServiceCall.model_validate(service_call_dict)
    assert UserDialogueAct.CARRY_OVER not in actions
    if not carry_over:
        raise InterpreterError(
            "Could not interpret actions. Are you missing carry-over annotations?"
        )
    formatter = PythonFunctionServiceCallFormatter(
        {
            "convert_camel_case": config.use_snake_case,
            "quote_values": True,
            "skip_quote": skip_quote,
        }
    )
    expression_template_or_statement = formatter.call_to_text(service_call)
    if template_fields:
        return [
            ProgramStatementTemplate(
                expression_template=expression_template_or_statement,
                fields=template_fields,
                tag=tag,
            )
        ]
    return [cast_to_statement(service_call, tag)]


@interpret_user_actions.register("call_with_selected_params")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement | ProgramStatementTemplate]:
    """This tag is assigned to turns where the user select an entity
    by referring to a slot name (eg select a movie by title).


    Example
    -------
    ..
    agent: Sure. I found 1 movie, The Best of Enemies. How does that sound?
    user: The Best of Enemies sound great.
    7 select(movie_name="The Best of Enemies", from_results=x1)
    ...
    agent: Should I buy the tickets?
    user: No. Not right now. What are the show times?
    10 suspend(x8)  // only in v0.6.0
    ---> 11 movies_1_get_times_for_movie(
        location=x7.location,
        **movie_name=x7.movie_name**,
        show_type=x7.show_type
    )
    ...
    Notes
    -----
    Selected parameters are marked with **. Contextual slot carry-over (location, show type)
    and slots informed in the current utterance are also handled.
    """
    template_fields = []
    skip_quote = set()
    service_call_dict = {
        "method": "",
        "parameters": {},
        "service": "",
    }

    # handle the slots inherited from previous intents/services
    assert_no_carry_over_overlap(actions)
    wildcard_call_ref = config.reference_command_for_wildcard_carryover
    if UserDialogueAct.CARRY_OVER in actions:
        update_service_call_dict(
            config.value_carryover,
            service_call_dict,
            template_fields,
            skip_quote,
            actions.pop(UserDialogueAct.CARRY_OVER),
            reference_command_for_wildcard_carryover=wildcard_call_ref,
            system_notified_failure=api_info.notified_failure,
        )
    # handle the selected action
    [select_action] = actions.pop(UserDialogueAct.SELECT)
    assert isinstance(select_action, UserAction)
    resolve_selected_value_to = config.value_carryover
    if config.value_carryover == "variable_reference":
        resolve_selected_value_to = "selection_variable_reference"

    # user selected a value from multiple slots offered
    if config.resolve_entity_in_invocation:
        selection_metadata = select_action.metadata
        if selection_metadata is not None:
            if selection_metadata["slot_selection"].mentioned_in_utterance:
                # ensure we display the value mentioned in the utterance
                # regardless of the other call params value resolution rules
                resolve_selected_value_to = "natural_language"
        # we asserted that the user mentions any selected value inside
        # sgd_conversation_builder.build_conversation, so we resolve to natural
        # language as a result
        else:
            resolve_selected_value_to = "natural_language"
    update_service_call_dict(
        resolve_selected_value_to,
        service_call_dict,
        template_fields,
        skip_quote,
        [select_action],
        reference_command_for_wildcard_carryover=wildcard_call_ref,
        system_notified_failure=api_info.notified_failure,
    )
    # finally, handle any slots communicated in the current turn
    if UserDialogueAct.INFORM in actions:
        update_service_call_dict(
            "natural_language",
            service_call_dict,
            template_fields,
            skip_quote,
            actions.pop(UserDialogueAct.INFORM),
        )
    [intent_action] = actions.pop(UserDialogueAct.INFORM_INTENT)
    convention = config.function_naming_convention
    update_method_name(
        convention,
        service_call_dict,
        intent_action.canonical_values[0],
        api_info,
        config.use_snake_case,
        config.skip_service_variations,
        config.anonymize_service,
    )
    service_call = ServiceCall.model_validate(service_call_dict)
    formatter = PythonFunctionServiceCallFormatter(
        {
            "convert_camel_case": config.use_snake_case,
            "quote_values": True,
            "skip_quote": skip_quote,
        }
    )
    expression_template_or_statement = formatter.call_to_text(service_call)
    if template_fields:
        return [
            ProgramStatementTemplate(
                expression_template=expression_template_or_statement,
                fields=template_fields,
                tag=tag,
            )
        ]
    return [cast_to_statement(service_call, tag)]


@interpret_user_actions.register("assignment")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is assigned to turns where the user provides
    more slots in response to an agent request. Note that the
    user may provide more slots than asked.

    Example
    ------
    user: find me a movie to watch please. Any show will work
    1 movies_1_find_movies(show_type="dontcare")
    ...
    agent: What location should I search?
    user: please search Mill Valley for movies.
    ---> 4 x1.location="Mill Valley"

    """

    @dispatch_on_value
    def build_statement_templates(assignment_layout: str):
        raise ValueError(f"Unknown assignment layout {assignment_layout}")

    @build_statement_templates.register("inlined")
    def _(assignment_layout: str):
        assignments, unhappy_path_labels = [], []
        info = None
        for target, value in zip(assignment_targets, assignment_values):
            assignments.append(f"{target}={value}")
        if len(assignments) > 1:
            unhappy_path_labels.append("multi_assignment")
            info = {"unhappy_path_labels": unhappy_path_labels}
        statement_templates.append(
            ProgramStatementTemplate(
                expression_template="; ".join(assignments),
                fields=[TemplateField(is_variable=True, field="current_call")],
                tag=tag,
                info=info,
            )
        )

    @build_statement_templates.register("newline")
    def _(assignment_layout: str):
        for target, value in zip(assignment_targets, assignment_values):
            statement_templates.append(
                ProgramStatementTemplate(
                    expression_template=f"{target}={value}",
                    fields=[TemplateField(is_variable=True, field="current_call")],
                    tag=tag,
                )
            )

    try:
        to_assign = actions.pop(UserDialogueAct.INFORM)
    except KeyError:
        # in some turns, the values are proposed by the system
        # and accepted by the user (see train/55_00119 - 7th user turn)
        assert UserDialogueAct.AFFIRM in actions
        to_assign = []
    if UserDialogueAct.AFFIRM in actions:
        # check annotation is as expected - we should have
        # added slot/values for what the user confirms
        confirmed_slot_val_actions = actions[UserDialogueAct.AFFIRM]
        assert all(all((a.slot, a.values)) for a in confirmed_slot_val_actions)
        to_assign += actions.pop(UserDialogueAct.AFFIRM)
    assignment_targets = [f"{{current_call}}.{a.slot}" for a in to_assign]
    assignment_values = [maybe_quote_or_cast(a.values[0]) for a in to_assign]
    assert len(assignment_targets) == len(assignment_values)
    statement_templates = []
    build_statement_templates(config.assignments_layout)
    return statement_templates


@interpret_user_actions.register("entity_query")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[AttributeAccessTemplate]:
    """This tag marks a user turn where the user requests to know the value
    of an entity attribute (REQUEST(attribute)) or checks a certain attribute
    has a certain value (REQUEST(attribute=value)).


    According to the PyTOD grammar, we only return variable templates which
    are communicated to the NLG


    Example
    -------
    user: How about on the 1st.
    14 x11.show_date="the 1st"
    15 len(x11) # -> 1
    16 next(x11)
    17 say(x15)
    agent: I show a 7 pm showing at CineArts at Sequoia.
    user: Wait. How much are the tickets? Is this a Horror movie?
    ---> 18 say(x16.genre, x16.price)

    Notes
    -----
    Both attribute value confirmation request (user: Can you confirm it's a first class ticket?")
    and attribute request (user: What is the ticket class?) are handled in the same manner.
    """
    to_print = actions.pop(UserDialogueAct.REQUEST)
    variables = [f"{{current_entity_or_call}}" for _ in to_print]  # noqa
    properties = [a.slot for a in to_print]
    return [
        AttributeAccessTemplate(
            expression_template=f"{var}.{prop}",
            attribute=prop,
            variable=var,
            fields=[
                TemplateField(
                    is_variable=True,
                    field="current_entity_or_call",
                    requires_disambiguation=True,
                )
            ],
            tag=tag,
            pass_to_nlg=True,
            requires_disambiguation=True,
            disambiguate=["expression_template", "variable"],
            is_assignable=False,
            is_inlined=True,
        )
        for (var, prop) in zip(variables, properties)
    ]


@interpret_user_actions.register("entity_selection")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag gets assigned to turns where the user selects an entity.


    Examples
    --------
    agent: How about Aka University City is a nice 3 star hotel.
    user: Do they allow smoking? What is the number?
    45 say(x43.smoking_allowed, x43.phone_number)
    agent: The number is +1 215-372-9000 and they do allow smoking.
    user: Sounds like a good match.
    ---> 46 select(x43)
    """

    selection = actions.pop(UserDialogueAct.SELECT)
    assert len(selection) == 1
    service_call_dict = {
        "method": "select",
        "parameters": {
            "{last_entity}": "",
            "from_results": "{current_results_list}",
        },
        "service": "",
    }
    formatter = PythonFunctionServiceCallFormatter(
        {
            "convert_camel_case": config.use_snake_case,
            "quote_values": True,
            "skip_quote": {"from_results"},
        }
    )
    expression_template = formatter.call_to_text(
        ServiceCall.model_validate(service_call_dict)
    )
    return [
        ProgramStatementTemplate(
            expression_template=expression_template,
            fields=[
                TemplateField(is_variable=True, field="last_entity"),
                TemplateField(is_variable=True, field="current_results_list"),
            ],
            tag=tag,
        )
    ]


@interpret_user_actions.register("entity_selection_with_new_task")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is assigned to turns where the user selects and entity
    and switches task in one go.


    Examples
    -------
    1. Same domain

        user: I love that restaurant you just mention. Go ahead and book for 3.

    2. New domain

        user: Okay, this event sounds great. What's the weather like on that date?


    Notes
    ----
    Often, the selection is quite subtle, the user just says "Ok" and goes ahead with
    the next task.
    """
    templates = interpret_user_actions("entity_selection", actions, config, api_info)
    for t in templates:
        t.tag = tag
    return templates
    # select = actions.pop(UserDialogueAct.SELECT)
    # assert len(select) == 1
    # expression = "select({last_entity})"
    # return [
    #     ProgramStatementTemplate(
    #         expression_template=expression,
    #         fields=[TemplateField(is_variable=True, field="last_entity")],
    #         tag=tag,
    #     )
    # ]


@interpret_user_actions.register("confirmation")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is assigned to turns where user confirms transactions.

    Example
    -------
    agent: You want 1 ticket to Anthony Green in Philadelphia on March 10th, correct?
    user: Yes that is correct.
    ---> 56 confirm(x53)
    """
    affirm = actions.pop(UserDialogueAct.AFFIRM)
    assert len(affirm) == 1
    expression = "confirm({current_call})"
    return [
        ProgramStatementTemplate(
            expression_template=expression,
            fields=[TemplateField(is_variable=True, field="current_call")],
            tag=tag,
        )
    ]


@interpret_user_actions.register("assign_query_result")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is automatically assigned to catch query results."""
    req_alts = actions.pop(UserDialogueAct.REQUEST_ALTS)
    assert len(req_alts) == 1
    fields = [
        TemplateField(field="current_call", is_variable=True),
    ]
    return [
        ProgramStatementTemplate(
            expression_template="next({current_call})", fields=fields, tag=tag
        )
    ]


@interpret_user_actions.register("results_iterator")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is assigned to turns where the user asks for further
    search results without providing additional constraints.


    Examples
    --------

    agent: I've found 10. 1 Hotel Brooklyn Bridge is a nice 4 star hotel.
    user: Are there any others?
    -->25 next(x21)
    """
    req_alts = actions.pop(UserDialogueAct.REQUEST_ALTS)
    assert len(req_alts) == 1
    fields = [
        TemplateField(field="current_call", is_variable=True),
    ]
    return [
        ProgramStatementTemplate(
            expression_template="next({current_call})", fields=fields, tag=tag
        )
    ]


@interpret_user_actions.register("result_selection_by_indexing")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is used to index in the current results list when
    the user does not mention the new intent in the same turn.
    The current result list variable is the same as the variable
    assigned to the call which returned the results.

    Notes
    -----
    This only applies to Movies_1 and Media_* domains where the
    user selects an entity from multiple options mentioned in a
    single utterance by the agent.
    """
    # pop this act, we will display a variable
    selection = actions.pop(UserDialogueAct.SELECT)
    assert len(selection) == 1
    [selection] = selection
    if config.index_lists_in_media_services:
        logger.warning(
            "Results list indexing will not be correctly represented in "
            "dialogues where the user iterates through options across "
            "multiple turns.. See 106_00052 (train) for an example."
        )
        if selection.metadata is None:
            # one option offered and that was selected
            index = 0
        else:
            # we have annotated the actual index of selected value
            assert selection.metadata.mentioned_in_utterance
            index = selection.metadata["slot_selection"].index
        fields = [
            TemplateField(field="current_results_list", is_variable=True),
        ]
        return [
            ProgramStatementTemplate(
                expression_template=f"{{current_results_list}}[{index}]",
                fields=fields,
                tag=tag,
                pass_to_nlg=False,
            )
        ]
    assert all((selection.slot, selection.values))
    service_call_dict = {
        "method": "select",
        "parameters": {
            selection.slot: selection.values[0],
            "from_results": "{current_results_list}",
        },
        "service": "",
    }
    formatter = PythonFunctionServiceCallFormatter(
        {
            "convert_camel_case": config.use_snake_case,
            "quote_values": True,
            "skip_quote": {"from_results"},
        }
    )
    expression_template = formatter.call_to_text(
        ServiceCall.model_validate(service_call_dict)
    )
    return [
        ProgramStatementTemplate(
            expression_template=expression_template,
            fields=[TemplateField(field="current_results_list", is_variable=True)],
            tag=tag,
            pass_to_nlg=False,
        )
    ]


@interpret_user_actions.register("result_selection_by_indexing_with_new_task")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is applied to turns where the user selects from multiple
    options offered by the system in the same turn and starts a new task
    for which the entity is relevant.

    This is represented as either:

     - indexing into the current results list variable  (ie x[2], where "x" the
     variable assigned to the call which returned the results)

     - a call to a `select` routine (ie `select(title="Dumbo", from_results=x)`
     where "x" is the variable assigned to the current results list.
     This variable is then referenced when the user asks questions about the entity
     and in the call to the new task.
    """

    # don't pop here, this will happen in the `call_with_selected_param` routine
    selection = actions[UserDialogueAct.SELECT]
    assert len(selection) == 1
    [selection] = selection
    if config.index_lists_in_media_services:
        logger.warning(
            "Results list indexing will not be correctly represented in "
            "dialogues where the user iterates through options across "
            "multiple turns..."
        )
        if selection.metadata is None:
            # one option offered and that was selected
            index = 0
        else:
            # here the user selected the value from options w/o mentioning it:
            # we assign the entity first, reference in the call
            # more than one option offered
            selection_metadata = selection.metadata["slot_selection"]
            if selection_metadata.mentioned_in_utterance:
                # this appears as a resolved string and is handled by
                # the `call_with_selected_params` routine
                return []
            index = selection_metadata.index
        # current_results_list binds to current_call automatically
        # on INFORM_COUNT annotation. This variable is passed forward to the
        # new task on this tag in the assignments code
        # (see `maybe_copy_entity_reference` in assignments.py)
        # before the template returned in this routine is rendered. Hence,
        # when rendered, this template will show the variable corresponding
        # to the query intent which returned the results (aka the list)
        # This tag also binds the index assigned to this statement
        # to an entity and so `call_with_selected_params` should
        # reference `current_entity` in the template
        expression_template = f"{{current_results_list}}[{index}]"
        field = "current_results_list"
        fields = [
            TemplateField(field=field, is_variable=True),
        ]
        return [
            ProgramStatementTemplate(
                expression_template=expression_template, fields=fields, tag=tag
            )
        ]
    slot = selection.slot
    assert slot
    program_statements_templates: list[
        ProgramStatementTemplate
    ] = interpret_user_actions(
        "result_selection_by_indexing", actions, config, api_info
    )
    assert len(program_statements_templates) == 1
    program_statements_templates[-1].tag = tag
    # add back the action so that the "call_with_selected_params" routine returns
    # the correct instruction
    actions[UserDialogueAct.SELECT] = [selection]
    return program_statements_templates


@interpret_user_actions.register("declines_suggestion")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """Tags turns where the user does not follow an agent suggestion

    Example
    -------
    user: Sounds like a good match.
    46 select(x43)
    47 suggest(task=hotels_4_reserve_hotel)
    48 say(x47)
    agent: Would you like to make a reservation?
    user: Not at this time thanks.
    ---> 49 suspend(x47)
    50 Hint('ask the user if they require further assistance')
    51 say(x50)
    agent: Can I do anything else for you?
    """
    actions.pop(UserDialogueAct.NEGATE_INTENT)
    cmd_name = config.decline_agent_suggested_task_cmd_name
    return [
        ProgramStatementTemplate(
            tag=tag,
            expression_template=f"{cmd_name}({{current_suggestion}})",
            fields=[TemplateField(field="current_suggestion", is_variable=True)],
        )
    ]


@interpret_user_actions.register("resumes_task")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatementTemplate]:
    """This tag is applied to turns where the user resumes a task
    they previously put on hold.

    Example
    -------
    ...
    3 next(x1)
    agent: A fancy restaurant in London is Heston Blumenthal.
    user: I love Heston Blumenthal.
    4 select(x3)
    agent: I bet you do, shall we get you in?
    5 suggest(task=restaurants_1_reserve_restaurant)
    user: Hold fire, check if there's money in the savings bank.
    6 decline(x5)
    7 banks_1_check_balance(account="savings")
    agent: Lots, $44,564.
    user: Cool, book me and my darling in
    ---> 8 resume(x6)
    9 reserve_restaurant(restaurant_name="Heston Blumenthal", people=2)
    """
    # SGD ontology does not have a corresponding action
    cmd_name = config.resume_task_cmd_name
    return [
        ProgramStatementTemplate(
            tag=tag,
            expression_template=f"{cmd_name}({{last_suspended_task}})",
            fields=[TemplateField(field="last_suspended_task", is_variable=True)],
        )
    ]


@interpret_user_actions.register("end_conversation")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """This tag is assigned to turns where the user
    dismisses the agent explicitly."""
    goodbye = actions.pop(UserDialogueAct.GOODBYE)
    assert len(goodbye) == 1
    return [
        ProgramStatement(
            expression="say()",
            tag=tag,
        )
    ]


@interpret_user_actions.register("conversation_pause")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """This tag is assigned to turns where the user does
    not finish the conversation but there are no active tasks."""
    thanks = actions.pop(UserDialogueAct.THANK_YOU)
    assert len(thanks) == 1
    return [
        ProgramStatement(
            expression="conversation_pause()",
            tag=tag,
        )
    ]


@interpret_user_actions.register("declines_alternative")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list[ProgramStatement]:
    """This tag is assigned to turns where the user does
    not finish the conversation but there are no active tasks."""
    negation = actions.pop(UserDialogueAct.NEGATE)
    assert len(negation) == 1
    return [
        ProgramStatement(
            expression="decline_alternative()",
            tag=tag,
        )
    ]


@interpret_user_actions.register("_remove_results_iteration")
def _(
    tag: UserTurnTag,
    actions: dict[UserDialogueAct, list[UserAction]],
    config: DictConfig,
    api_info: APIInfo,
) -> list:
    """Remove REQUEST_ALTS action. This is used only during queries
    in Media_1, Media_2, Media_3 and Movies_1.

    The algorithm will raise a PolicyError unless all actions have been interpreted."""
    actions.pop(UserDialogueAct.REQUEST_ALTS)
    return []
