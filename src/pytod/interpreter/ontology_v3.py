#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""This module defines a set of semantic labels mapping the SGD actions ontology to the
PyTOD ontology. Note there is no 1:1 correspondence between the two: Together, the functions
`get_user_turn_tags` and `combine_with_system_tags` represent the transitions in the SGD policy
graph as a sequence of PyTOD semantic labels (referred to as tags) which are converted to program
statements or backend hints/notifications by the interpreter using the routines implemented in the
registry modules."""

import logging
import random
from copy import deepcopy
from itertools import chain
from typing import Optional

from pytod.interpreter.metadata import should_skip_binding_first_result
from pytod.interpreter.utils import PolicyError, Tag
from pytod.pytod_types.sgd_conversation import Author, Turn, UserDialogueAct
from pytod.sgd_agent_behaviour import (
    agent_ends_conversation,
    agent_informs_entity_count,
    agent_informs_slot_values,
    agent_informs_task_status,
    agent_makes_offer,
    agent_notifies_failure,
    agent_prompts_for_more_help,
    agent_requests_confirmation,
    agent_suggests_next_task_turn,
    count_agent_questions,
)
from pytod.sgd_analysis_utils import get_service
from pytod.sgd_policy_assertions import (
    assert_no_affirmation,
    assert_only_slot_filling,
    assert_single_action_or_entity_selection_turn,
    assert_sys_confirmation_behaviour,
    assert_user_only_informed_or_selected_entity,
    assert_user_result_selection_by_indexing_behaviour,
)
from pytod.sgd_user_behaviour import (
    is_req_alts,
    turn_carries_over_arguments,
    user_accepts_suggestion,
    user_affirms,
    user_declines_suggestion,
    user_does_not_specify_next_task,
    user_ends_conversation,
    user_informs_constraints,
    user_negates,
    user_req_alt,
    user_requests_information,
    user_selects_entity,
    user_selects_from_system_options,
    user_states_intention,
)
from pytod.transcript import APIInfo

logger = logging.getLogger(__name__)

NLG_TAGS = {"results_iterator", "inform_entity_count_signal", "entity_query"}


def get_user_turn_tags(
    turn: Turn, prev_sys_turn: Optional[Turn], api_info: APIInfo
) -> list[Tag]:
    """Tag a user turn to with semantic labels determining the PyTOD instructions
    derived from it.

    Parameters
    ----------
    api_info
        Tuple containing information about the active service and intent.
    """

    def assert_type_tags_consistency(type_tags: list[Tag]):
        type_tags = [t.tag for t in type_tags]
        try:
            assert type_tags
        except AssertionError:
            try:
                assert any(
                    a.act == UserDialogueAct.NEGATE
                    for a in chain(*turn.user_actions.values())
                )
            except AssertionError:
                raise PolicyError(
                    f"No PyTOD conversion type defined for turn: {turn}."
                    f"User actions: {turn.user_actions}"
                )
        if any("call" in tag for tag in type_tags):
            assert "assignment" not in type_tags
        if "call" in type_tags:
            assert "call_with_params" not in type_tags
            assert "call_with_carry_over" not in type_tags
        if "end_conversation" in type_tags:
            # the user selects an entity at the end of a
            # search conversation  (eg, train/60_00108;
            # agent: A16 is a good restaurant in San Francisco. //
            # user: Yes, that is great. That is all I need.)
            assert len(type_tags) == 1 or (
                len(type_tags) == 2
                and any(
                    (
                        "entity_selection" in type_tags,
                        "result_selection_by_indexing" in type_tags,
                        "declines_suggestion" in type_tags,
                        "declines_alternative" in type_tags,
                    )
                )
            )
        if "entity_selection" in type_tags:
            assert "result_selection_by_indexing" not in type_tags
        if "entity_selection_with_new_task" in type_tags:
            assert "result_selection_by_indexing_with_new_task" not in type_tags
        if "result_selection_by_indexing_with_new_task" in type_tags:
            assert "call_with_selected_params" in type_tags
        for t in type_tags:
            count_with = t.count("_with_")
            assert count_with in range(2)

    assert turn.user_actions is not None, "Not a user turn?"
    type_tags = []
    # selection happens often in the same turn with an intention
    # change, so we display the selected results before the new
    # call
    if selection := user_selects_from_system_options(turn):
        tag = "result_selection_by_indexing"
        if intention := user_states_intention(turn):
            # we ignore the new task making the assumption that
            # the selected result is not passed to the new intent.
            # Entities are selected in this way only in
            # Movies_1/Media_* services, and they are not passed on
            # to other intents in other services
            if selection.service == intention.service:
                tag += "_with_new_task"
        assert (
            selection.intent is not None
        ), "Intent should be known to assign entities correctly"
        type_tags.append(
            Tag(
                tag=tag,
                source_author=Author.USER,
                service=selection.service,
                intent=selection.intent,
            )
        )
        assert_user_result_selection_by_indexing_behaviour(turn)
    # user selects an entity offered by the agent
    if entity_selection := user_selects_entity(turn):
        # this is a disjoint condition compared with indexing
        # in the system options
        assert not type_tags
        tag = "entity_selection"
        if user_explicit_intention := user_states_intention(turn):
            if user_explicit_intention.service == entity_selection.service:
                tag += "_with_new_task"
        type_tags.append(
            Tag(
                tag=tag,
                source_author=Author.USER,
                service=entity_selection.service,
                intent=entity_selection.intent,
            )
        )
    # user does not go ahead with a task suggested by the agent
    if user_negation := user_declines_suggestion(turn):
        tag = "declines_suggestion"
        if type_tags:
            assert (
                t.tag
                in {
                    "result_selection_by_indexing",
                    "entity_selection",
                    "entity_selection_with_new_task",
                    "result_selection_by_indexing_with_new_task",
                }
                for t in type_tags
            )
        type_tags.append(
            Tag(
                tag=tag,
                source_author=Author.USER,
                service=user_negation.service,
                intent=user_negation.intent,
            )
        )
    user_explicit_intention = user_states_intention(turn)
    suggestion_accepted = user_accepts_suggestion(turn)
    # user either explicitly states the tasks or accepts
    # a proposal
    if user_explicit_intention or suggestion_accepted:
        tag = "call"
        assert_no_affirmation(turn)
        intention_service = (
            user_explicit_intention.service or suggestion_accepted.service
        )
        assert intention_service is not None
        if constraints := user_informs_constraints(turn):
            assert constraints.service == intention_service
            if carryover := turn_carries_over_arguments(turn):
                options_selection = user_selects_from_system_options(turn)
                if options_selection and options_selection.service == intention_service:
                    # handles carry-over & constraints communicated in this turn
                    tag += "_with_selected_params"
                else:
                    assert intention_service == carryover.service
                    # also handles constraints communicated in this turn
                    tag += "_with_carry_over"
            elif options_selection := user_selects_from_system_options(turn):
                assert not any((t.tag == "entity_selection" for t in type_tags))
                # also handles constraints communicated in this turn
                if options_selection.service == intention_service:
                    tag += "_with_selected_params"
                else:
                    tag += "_with_params"
            else:
                assert_user_only_informed_or_selected_entity(turn)
                tag += "_with_params"
        elif carryover := turn_carries_over_arguments(turn):
            assert (
                carryover.service == user_explicit_intention.service
                or carryover.service == suggestion_accepted.service
            )
            options_selection = user_selects_from_system_options(turn)
            if options_selection.service == carryover.service:
                # this handles carry-over & selection from system options
                tag += "_with_selected_params"
            else:
                # this does not handle selection from system options
                tag += "_with_carry_over"
        elif options_selection := user_selects_from_system_options(turn):
            if options_selection.service == intention_service:
                tag += "_with_selected_params"
        else:
            # the user just stated the task or selected an entity without value mention
            assert_single_action_or_entity_selection_turn(turn)

        if (
            user_explicit_intention.service,
            user_explicit_intention.intent,
        ) in api_info.suspended_tasks:
            type_tags.append(
                Tag(
                    tag="resumes_task",
                    source_author=Author.USER,
                    service=intention_service,
                )
            )
        type_tags.append(
            Tag(tag=tag, source_author=Author.USER, service=intention_service)
        )
    if constraints := user_informs_constraints(turn):
        if not type_tags:
            type_tags.append(
                Tag(
                    tag="assignment",
                    source_author=Author.USER,
                    service=constraints.service,
                )
            )
        else:
            expected_suffixes = {
                "_with_params",
                "_with_carry_over",
                "_with_selected_params",
            }
            assert any(
                type_tags[-1].tag.endswith(suffix) for suffix in expected_suffixes
            )
    if affirmation := user_affirms(turn):  # (if user does not confirm they assign)
        if count_agent_questions(prev_sys_turn) > 0:
            # it is possible that the user informs something and confirms a slot value
            # in this case the turn should already have an "assignment" tag
            if (
                Tag(
                    tag="assignment",
                    source_author=Author.USER,
                    service=affirmation.service,
                )
                not in type_tags
            ):
                # the user just confirmed a slot value proposed by the system -
                # value proposed was not overridden and no new slots were mentioned
                # eg: agent: Do you want to check in on the 10th? // user: Yes pls!
                # nb: agent proposed values are carried over from related tasks, not
                # randomly sampled
                type_tags.append(
                    Tag(
                        tag="assignment",
                        source_author=Author.USER,
                        service=affirmation.service,
                    )
                )
        else:
            # the user confirms an entity the system has offered
            type_tags.append(
                Tag(
                    tag="confirmation",
                    source_author=Author.USER,
                    service=affirmation.service,
                )
            )
    # information was requested about an entity
    if info_request := user_requests_information(turn):
        assert not user_informs_constraints(turn)
        type_tags.append(
            Tag(
                tag="entity_query",
                source_author=Author.USER,
                service=info_request.service,
            )
        )
    if alternatives := is_req_alts(turn):
        # user requested another result without changing constraints
        if user_req_alt(turn):
            type_tags.append(
                Tag(
                    tag="results_iterator",
                    source_author=Author.USER,
                    service=alternatives.service,
                )
            )
        else:
            # some constraints were changed
            assert len(turn.user_actions_dict) == 1
            [state] = list(turn.dialogue_state.values())
            if not should_skip_binding_first_result(
                alternatives.service, api_info.function
            ):
                type_tags.append(
                    Tag(
                        tag="assign_query_result",
                        source_author=Author.USER,
                        service=alternatives.service,
                        intent=state.active_intent,
                    )
                )
            else:
                type_tags.append(
                    Tag(
                        tag="_remove_results_iteration",
                        source_author=Author.USER,
                        service=alternatives.service,
                        intent=state.active_intent,
                    )
                )
    if negation := user_negates(turn):
        if agent_makes_offer(prev_sys_turn) and agent_notifies_failure(prev_sys_turn):
            type_tags.append(
                Tag(
                    tag="declines_alternative",
                    source_author=Author.USER,
                    service=negation.service,
                )
            )
    if goodbye := user_ends_conversation(turn):
        type_tags.append(
            Tag(
                tag="end_conversation",
                source_author=Author.USER,
                service=goodbye.service,
            )
        )
    if conversation_pause := user_does_not_specify_next_task(turn):
        type_tags.append(
            Tag(
                tag="conversation_pause",
                source_author=Author.USER,
                service=conversation_pause.service,
            )
        )
    assert_type_tags_consistency(type_tags)
    return type_tags


def combine_with_system_tags(
    user_turn_tags: list[Tag], system_turn: Optional[Turn], api_info: APIInfo
) -> list[Tag]:
    """Combine PyTOD instruction semantic labels derived from the user turn with
    backend information. Crucially, ensure the order of PyTOD instructions is correct.

    Parameters
    ----------
    user_turn_tags
        Tags applied to the previous user turn.
    system_turn
        The system turn that follows the user turn from which `user_turn_tags` are
        derived. It is `None` if only the turns before a given turn in the original conversation
        are processed.
    api_info
        Tuple containing information about the active service and intent.
    """

    def get_entity_query_index(tags: list[Tag], service: Optional[str] = None) -> int:
        return tags.index(
            Tag(tag="entity_query", source_author=Author.USER, service=service)
        )

    def assert_type_tags_consistency(tags: list[Tag]):
        type_tags = [t.tag for t in tags]
        try:
            assert type_tags
        except AssertionError:
            raise PolicyError(
                f"No PyTOD conversion type defined for turn: {system_turn}."
                f"System actions: {system_turn.system_actions}"
            )
        if "offer_hint" in type_tags:
            assert "offer_alternative_signal" not in type_tags
        if "task_suggestion" in type_tags:
            assert any(
                t in type_tags
                for t in {"entity_selection", "result_selection_by_indexing"}
            )
        assert len(type_tags) == len(set(type_tags))

    def user_selected_entity(user_turn_tags: list[Tag]) -> bool:
        return any(t.tag == "entity_selection" for t in user_turn_tags)

    def can_followup(api_info: APIInfo) -> bool:
        return api_info.followup_command is not None

    if system_turn is None:
        return user_turn_tags
    assert system_turn.author == Author.SYSTEM
    tags = deepcopy(user_turn_tags)
    if count_agent_questions(system_turn) > 0:
        assert_only_slot_filling(system_turn)
        call_or_assignment_tags = {
            "call_with_params",
            "call_with_carry_over",
            "call_with_selected_params",
            "call",
            "assignment",
        }
        entity_selection_tags = {"entity_selection", "entity_selection_with_new_task"}
        slot_value_selection_tags = {
            "result_selection_by_indexing",
            "result_selection_by_indexing_with_new_task",
        }
        task_management_tags = {"resumes_task", "declines_suggestion"}
        slot_confirmation_tag = "user_accepts_proposed_slot_values"
        try:
            assert len(tags) == 1 and (
                tags[-1].tag in call_or_assignment_tags
                or tags[-1].tag == slot_confirmation_tag
            )
        except AssertionError:
            try:
                assert len(tags) == 2
            except AssertionError:
                assert len(tags) == 3
                assert any(t.tag in task_management_tags for t in tags)
            assert (
                any(t.tag in entity_selection_tags for t in tags)
                or any(t.tag in slot_value_selection_tags for t in tags)
                or any(t.tag in task_management_tags for t in tags)
            )
            assert any(t.tag in call_or_assignment_tags for t in tags)
        tags.append(
            Tag(
                tag="slot_filling_hint",
                source_author=Author.SYSTEM,
                service=get_service(system_turn),
            )
        )
        return tags
    if question_answers := agent_informs_slot_values(system_turn):
        # answer is managed by passing back a list of variables to the assignment algorithm
        # via the `entity_query` conversion utility
        tags.append(
            Tag(
                tag="_remove_information_provided",
                source_author=Author.SYSTEM,
                service=question_answers.service,
            )
        )
    if entities_offered := agent_makes_offer(system_turn):
        if entity_count := agent_informs_entity_count(system_turn):
            assert entity_count.service == entities_offered.service
            assert not agent_notifies_failure(system_turn)
            inform_count_tag = Tag(
                tag="inform_entity_count_signal",
                source_author=Author.SYSTEM,
                service=entities_offered.service,
            )
            try:
                _ = tags.index(
                    Tag(
                        tag="results_iterator",
                        source_author=Author.USER,
                        service=entities_offered.service,
                    )
                )
                tags.append(
                    Tag(
                        tag="_remove_entity_counts",
                        source_author=Author.SYSTEM,
                        service=entities_offered.service,
                    )
                )
            except ValueError:
                try:
                    user_req_alt_index = tags.index(
                        Tag(
                            tag="assign_query_result",
                            source_author=Author.USER,
                            service=entities_offered.service,
                        )
                    )
                    tags[user_req_alt_index:user_req_alt_index] = [inform_count_tag]
                except ValueError:
                    tags.append(inform_count_tag)
        if failure_notification := agent_notifies_failure(
            system_turn
        ):  # means an alternative is proposed
            assert failure_notification.service == entities_offered.service
            # the failure happens and the sys makes another offer -
            #   this can only happen if the user went ahead
            #   with a transaction, so we expect to have
            #   tagged the user turn accordingly
            confirmation_idx = tags.index(
                Tag(
                    tag="confirmation",
                    source_author=Author.USER,
                    service=entities_offered.service,
                )
            )
            tags[confirmation_idx + 1 : confirmation_idx + 1] = [
                Tag(
                    tag="task_status_signal",
                    source_author=Author.SYSTEM,
                    service=entities_offered.service,
                ),
                # NB: I feel the semantics is different from next so this is not assigned here
                Tag(
                    tag="offer_alternative_signal",
                    source_author=Author.SYSTEM,
                    service=entities_offered.service,
                ),
            ]
            # if there were any queries to be answered, find them and
            # respond at the end; the ref should be to the entity that
            # was accepted by the user which we should have assigned previously
            if agent_informs_slot_values(system_turn):
                relocate = [
                    tags.pop(
                        get_entity_query_index(
                            tags, service=failure_notification.service
                        )
                    )
                ]
                tags += relocate
        else:  # this means we are actually telling the user about an entity for the first time
            # we don't expect the user to ask a question before the entity was returned by the API
            assert not any(t.tag == "entity_query" for t in tags)
            # we have assigned already and the call has not failed we don't reassign
            service = get_service(system_turn)
            if not any(
                t in tags
                for t in (
                    Tag(
                        tag="assign_query_result",
                        source_author=Author.USER,
                        service=service,
                    ),
                    Tag(
                        tag="results_iterator",
                        source_author=Author.USER,
                        service=service,
                    ),
                )
            ):
                if not should_skip_binding_first_result(service, api_info.function):
                    tags.append(
                        Tag(
                            tag="assign_query_result",
                            source_author=Author.SYSTEM,
                            service=service,
                        )
                    )  # next({last_call_index})
            # special tag used to remove acts that we handled
            tags.append(
                Tag(tag="_remove_offers", source_author=Author.SYSTEM, service=service)
            )
    # this is either a successful call or error notification
    if task_status := agent_informs_task_status(system_turn):
        if not agent_makes_offer(
            system_turn
        ):  # the case where the agent made an offer and notified a failure is handled above
            if agent_notifies_failure(system_turn):
                try:  # requested an alternative and there was nothing to show
                    for t in (
                        Tag(
                            tag="assign_query_result",
                            source_author=Author.USER,
                            service=task_status.service,
                        ),
                        Tag(
                            tag="results_iterator",
                            source_author=Author.USER,
                            service=task_status.service,
                        ),
                    ):
                        if t in tags:
                            user_iteration_tag = t
                            break
                    else:
                        # if this is raised, it means we are not dealing with failed queries
                        raise ValueError
                    entity_assign_index = tags.index(user_iteration_tag)
                    assert not any(t.tag == "confirmation" for t in tags)
                    # the user shouldn't ask a question because they just req.
                    # an alternative so entity is not known
                    assert not any(t.tag == "entity_query" for t in tags)
                    # user iteration is followed by a notification from the backend
                    # that the task was not successful
                    tags[entity_assign_index + 1 : entity_assign_index + 1] = [
                        Tag(
                            tag="task_status_signal",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        ),
                    ]
                except ValueError:
                    # we did not fail because there were no more results to show
                    #   => hence, a transaction failed, so we look for confirmation from user
                    confirmation_idx = tags.index(
                        Tag(
                            tag="confirmation",
                            source_author=Author.USER,
                            service=task_status.service,
                        )
                    )
                    tags.insert(
                        confirmation_idx + 1,
                        Tag(
                            tag="task_status_signal",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        ),
                    )
                    assert not agent_informs_slot_values(system_turn)
                    # remove any questions the agent is not answering
                    # this only happens when calls to transactional APIs
                    # do not go through
                    try:
                        entity_index = get_entity_query_index(tags)
                        tags.pop(entity_index)
                    except ValueError:
                        pass
            else:
                if question_answers := agent_informs_slot_values(system_turn):
                    confirmation_idx = tags.index(
                        Tag(
                            tag="confirmation",
                            source_author=Author.USER,
                            service=task_status.service,
                        )
                    )
                    assert question_answers.service == task_status.service
                    # here we make sure the backend reports the task status first, assign the query
                    # and then go ahead and print the result
                    assert not any(t.tag == "assign_query_result" for t in tags)
                    tags[confirmation_idx + 1 : confirmation_idx + 1] = [
                        Tag(
                            tag="task_status_signal",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        ),
                        Tag(
                            tag="task_performed",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        ),
                    ]
                else:
                    tags.append(
                        Tag(
                            tag="task_status_signal",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        )
                    )
                    tags.append(
                        Tag(
                            tag="task_performed",
                            source_author=Author.SYSTEM,
                            service=task_status.service,
                        )
                    )
    if confirmation_request := agent_requests_confirmation(system_turn):
        tags.append(
            Tag(
                tag="require_confirmation_hint",
                source_author=Author.SYSTEM,
                service=confirmation_request.service,
            )
        )
        assert_sys_confirmation_behaviour(system_turn)
    if help_prompt := agent_prompts_for_more_help(system_turn):
        if can_followup(api_info) and user_selected_entity(user_turn_tags):
            assert not agent_suggests_next_task_turn(system_turn)
            new_tags = [
                Tag(
                    tag="prompt_more_help_required",
                    source_author=Author.SYSTEM,
                    service=help_prompt.service,
                ),
                Tag(
                    tag="task_suggestion",
                    source_author=Author.SYSTEM,
                    service=help_prompt.service,
                    # make it clear that this is not annotated in the
                    # system actions
                    annotated=False,
                ),
            ]
            random.shuffle(new_tags)
            tags.extend(new_tags)
        else:
            tags.append(
                Tag(
                    tag="prompt_more_help_required",
                    source_author=Author.SYSTEM,
                    service=help_prompt.service,
                ),
            )
    if task_suggestion := agent_suggests_next_task_turn(system_turn):
        # agent always suggests a transaction following a search
        # (eg book a restaurant you found) not sth random!
        new_tags = [
            Tag(
                tag="task_suggestion",
                source_author=Author.SYSTEM,
                service=task_suggestion.service,
            ),
            Tag(
                tag="prompt_more_help_required",
                source_author=Author.SYSTEM,
                service=task_suggestion.service,
                # make it clear that this is not annotated in the
                # system actions
                annotated=False,
            ),
        ]
        random.shuffle(new_tags)
        tags.extend(new_tags)
    if conversation_end := agent_ends_conversation(system_turn):
        # assume the user has explicitly dismissed the agent
        try:
            assert any(t.tag == "end_conversation" for t in tags)
        except AssertionError:
            # if the assertion fails, it means that the user has
            # declined an offer for help and implicitly dismissed the
            # agent (see train/34_00116)
            tags.append(
                Tag(
                    tag="end_conversation",
                    source_author=Author.SYSTEM,
                    service=conversation_end.service,
                )
            )

    assert_type_tags_consistency(tags)
    return tags
