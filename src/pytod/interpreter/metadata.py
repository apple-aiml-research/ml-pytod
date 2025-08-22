#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional

from pytod.pytod_types.aliases import IntentName, ServiceName

WILDCARD_VALUE = "dontcare"
SYSTEM_ARG_MARKER = "SYS_ARG"
PUBLIC_PROPERTY_MARKER = "PUBLIC_PROPERTY"
RESOLUTION_ERR = "RESOLUTION_ERR"
UNPARSEABLE_SLOTS = {
    "from": ["find_trains", "get_train_tickets"],
    "class": ["find_trains", "get_train_tickets"],
}
UNPARSEABLE_APIS = {
    "find_trains": "Trains_1",
    "get_train_tickets": "Trains_1",
}
SKIP_QUERY_BINDING = {
    "Media_1": ["FindMovies"],
    "Media_2": ["FindMovies"],
    "Media_3": ["FindMovies"],
    "Movies_1": ["FindMovies"],
}
"""For these intents, the next({current_call_variable}) statement
does not appear after len() because the user selects amongst options
offered by the system so the entity is bound when the user selects."""
ENTITY_SELECTION_BY_SLOT_MENTION = set(SKIP_QUERY_BINDING.keys())
"""For example, a movie is selected from options by specifying the title."""
ENTITY_REF_COMMAND = {"Banks_2": ["transfer_time"]}
"""For these slots, the variable referenced when the user asks questions during transactions
is the command and not the entity that was passed to the transaction."""


def get_unparseable_slot_map(service: ServiceName) -> Optional[dict[str, str]]:
    """Maps slots which are `python` special keywords so that calls containing
    these values can be AST-parsed."""
    if service == "Trains_1":
        return {
            "from": "journey_starts_from",
            "class": "ticket_fare_class",
        }


def should_skip_binding_first_result(service: ServiceName, intent: IntentName) -> bool:
    """When a query intent is called, the first result of the query is
    automatically extracted with a `next` call and bound to a variable.

    For certain services, we deviate from the above logic because the
    system mentions multiple entities in the utterance, and subsequent
    questions refer to a selected entity.

    Examples
    --------
        user: "Okay please find a movie to watch online."
        agent: "What type would you like to see?"
        user: "Drama would be nice."
        agent: find_movies(genre=Drama)
        agent: "Would you enjoy Dogman, Hackers, Or High Life?"
        user: "High Life sounds good. Please play it."
        agent: "You want to see High Life without subtitles, correct?"
        user: "Yes that is correct. Who directed it?"
        agent: play_movie(subtitles=False, title=High Life)

    Notes
    -----
    This is not active if there is a single result offered.
    """
    return service in SKIP_QUERY_BINDING and intent in SKIP_QUERY_BINDING[service]


SELECT_ALLOWED_KWARGS = ["movie_name", "title"]

NLG_CALL_TOOL = "say"
RESTART_TASK_TOOL = "resume"
ITERATION_TOOL = "next"
INTENT_UPDATE_TOOL = "assign"
FOLLOWUP_INTENT_TOOL = "suggest"
FOLLOWUP_INTENT_TOOL_KWARG = "task"
FOLLOWUP_INTENT_DECLINE_TOOL = "suspend"
QUERY_RESULTS_REF_KWARG = "from_results"
ENTITY_SELECTION_TOOL = "select"
COMMUNICATE_NUM_RESULTS_TOOL = "len"
COMMUNICATE_ENTITY_INFO_TOOL = "show"
SLICE_TOOL = "slice"
TRANSACTION_CONFIRMATION_TOOL = "confirm"
TRANSACTION_SUCCESS_TOOL = "perform"
PAUSE_TOOL = "conversation_pause"
DECLINE_ALTERNATIVE_TOOL = "decline_alternative"

QUERY_POS_ARG_TOOLS = {
    COMMUNICATE_NUM_RESULTS_TOOL,
    COMMUNICATE_ENTITY_INFO_TOOL,
    SLICE_TOOL,
}
NO_KWARG_TOOLS = {
    NLG_CALL_TOOL,
    *QUERY_POS_ARG_TOOLS,
    TRANSACTION_CONFIRMATION_TOOL,
    TRANSACTION_SUCCESS_TOOL,
    ITERATION_TOOL,
    RESTART_TASK_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
    PAUSE_TOOL,
    DECLINE_ALTERNATIVE_TOOL,
}
SINGLE_ARG_TOOLS = {
    *QUERY_POS_ARG_TOOLS,
    TRANSACTION_CONFIRMATION_TOOL,
    TRANSACTION_SUCCESS_TOOL,
    ITERATION_TOOL,
    RESTART_TASK_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
}
POSITIONAL_TOOLS = {
    ENTITY_SELECTION_TOOL,
    ITERATION_TOOL,
    *QUERY_POS_ARG_TOOLS,
    TRANSACTION_CONFIRMATION_TOOL,
    NLG_CALL_TOOL,
    RESTART_TASK_TOOL,
    TRANSACTION_SUCCESS_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
}
ZERO_ARG_TOOLS = {PAUSE_TOOL, DECLINE_ALTERNATIVE_TOOL}
PARSE_ERROR_TOOL = "parse_error"
PARSE_ERROR = f"{PARSE_ERROR_TOOL}()"
SPECIAL_TOOLS = {
    TRANSACTION_CONFIRMATION_TOOL,
    INTENT_UPDATE_TOOL,
    ITERATION_TOOL,
    ENTITY_SELECTION_TOOL,
    *QUERY_POS_ARG_TOOLS,
    TRANSACTION_SUCCESS_TOOL,
    PAUSE_TOOL,
    DECLINE_ALTERNATIVE_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
    FOLLOWUP_INTENT_TOOL,
    NLG_CALL_TOOL,
    RESTART_TASK_TOOL,
    PARSE_ERROR_TOOL,
}
SPECIAL_KWARGS = {FOLLOWUP_INTENT_TOOL_KWARG, QUERY_RESULTS_REF_KWARG}
VARIABLE_RESOLUTION_ERROR = "'var_resolution_error'"
ASSIGNMENT_START = r"x[0-9]+\."
SELECT_RESULTS_KWARG = "from_results"
TOOL_RESOLUTION_ERROR = "error"
