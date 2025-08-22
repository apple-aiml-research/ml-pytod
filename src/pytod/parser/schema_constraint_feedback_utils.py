#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging

from pytod.command import ServiceCommand, get_all_args, is_api_return_property, is_arg
from pytod.interpreter.metadata import INTENT_UPDATE_TOOL
from pytod.parser.expresion_validation_context import (
    SchemaConstraintContext,
    SchemaValidationContext,
)
from pytod.parser.expresion_validation_utils import ParserFeedback
from pytod.pytod_types.aliases import IntentName, ServiceName
from pytod.utils import snake_case, stringify_list

logger = logging.getLogger(__name__)


def _collect_service_argument_feedback(
    kwarg_name: str,
    kwarg_value: str,
    predicted_intent_schema: ServiceCommand,
    context: SchemaConstraintContext,
) -> ParserFeedback:
    """Provide feedback about whether `kwarg_name` is defined in the `predicted_intent_schema`."""
    cmd_collection = context.schema
    service = predicted_intent_schema.service
    intent = predicted_intent_schema.name
    tool_name = cmd_collection.get(service, intent).tool_name
    if is_arg(kwarg_name, cmd_collection.get(service, intent)):
        return ParserFeedback()
    else:
        apis, return_apis, feedback = [], [], []
        for cmd in cmd_collection.get_service_commands(service):
            if cmd.name != intent:
                if is_arg(kwarg_name, cmd):
                    apis.append(snake_case(cmd.name))
            if is_api_return_property(kwarg_name, cmd):
                return_apis.append(snake_case(cmd.name))
        ignore_argument = True
        if apis:
            apis_str = stringify_list(apis)
            msg = (
                f"Argument `{kwarg_name}` is not an argument of `{tool_name}`. "
                f"However, it is an argument for `{apis_str}` "
                "from the same service. Maybe the wrong API was chosen "
                f"or the value `{kwarg_value}` was wrongly attributed "
                f"to `{kwarg_name}`? Alternatively, maybe a call to "
                f"{apis_str} was missed and an assignment instruction "
                "issued instead?"
            )
            # don't ignore argument if from different intent as
            # intents are ambiguous and there can be data errors
            ignore_argument = False
        else:
            assert return_apis, f"{kwarg_name} should be in the schema"
            return_apis_str = stringify_list(return_apis)
            msg = (
                f"Argument `{kwarg_name}` is a property of the the "
                f"objects returned by the apis `{return_apis_str}`, "
                f"not an input argument."
            )
        feedback.append(msg)
        logger.warning(f"{context.dial_id}: {msg}")
        return ParserFeedback(feedback=feedback, ignore_argument=ignore_argument)


def _schema_validation_failure_msg(
    kwarg_name: str,
    cmd: ServiceCommand | None,
) -> str:
    if cmd is None:
        return ""
    arg_names = stringify_list(list(get_all_args(cmd)))
    tool_name = cmd.tool_name
    assert tool_name is not None
    if arg_names is None:
        arg_names_info = f"`{tool_name}` takes no arguments."
    else:
        arg_names_info = f" Arguments should be one of: {arg_names.lower()}"
    msg = (
        f"Slot `{kwarg_name}` is not part of `{tool_name}` schema. " f"{arg_names_info}"
    )
    return msg


def _collect_memorisation_feedback(
    kwarg_name: str,
    service: ServiceName,
    intent: IntentName,
    tool_name: str,
    context: SchemaValidationContext,
) -> list[str]:
    """Check whether `kwarg_name` is part of the schema of a different API."""
    cmd_collection = context.schema
    feedback = []
    for s in cmd_collection.services.difference({service}):
        if cmd_collection.in_service_schema(s, kwarg_name):
            api = ""
            for cmd in cmd_collection.get_service_commands(s):
                if is_arg(kwarg_name, cmd):
                    api = cmd.tool_name
                    break
            if api:
                tool_disp = (
                    snake_case(intent) if tool_name == INTENT_UPDATE_TOOL else tool_name
                )
                msg = f"'{kwarg_name}' is an argument of {api}, not {tool_disp}."
                if tool_name == INTENT_UPDATE_TOOL:
                    msg += (
                        " The variable referenced in property assignment is incorrect."
                    )
            else:
                msg = f"'{kwarg_name}' is part of service '{s.lower()}' schema, but is not mutable."
            logger.warning(f"{context.dial_id}: {msg}")
            feedback.append(msg)
    return feedback
