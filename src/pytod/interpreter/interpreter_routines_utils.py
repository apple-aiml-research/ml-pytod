#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
from typing import Any, Optional, Union

from pytod.command import ServiceCall
from pytod.interpreter.metadata import WILDCARD_VALUE
from pytod.pytod_types.sgd_conversation import UserAction
from pytod.pytod_types.transcript import ProgramStatement, TemplateField
from pytod.text_formatter import PythonFunctionServiceCallFormatter
from pytod.transcript import APIInfo
from pytod.utils import dispatch_on_value

logger = logging.getLogger(__name__)


@dispatch_on_value
def update_method_name(
    convention: str,
    service_call_dict: dict[str, Any],
    canonical_value: str,
    api_info: APIInfo,
    use_snake_case: bool,
    skip_service_variations: bool,
    anonymize_service: bool,
):
    assert convention in [
        "sgd",
        "toolbox",
    ], f"Unknown convention {convention} for function call formatting"


@update_method_name.register("sgd")
def _(
    convention: str,
    service_call_dict: dict[str, Any],
    canonical_value: str,
    api_info: APIInfo,
    use_snake_case: bool,
    skip_service_variations: bool,
    anonymize_service: bool,
):
    service_call_dict["method"] = canonical_value


@update_method_name.register("toolbox")
def _(
    convention: str,
    service_call_dict: dict[str, Any],
    canonical_value: str,
    api_info: APIInfo,
    use_snake_case: bool,
    skip_service_variations: bool,
    anonymize_service: bool,
):
    # assert api_info.function == canonical_value
    service = api_info.service
    if skip_service_variations or anonymize_service:
        service = service.split("_")[0]

    if use_snake_case:
        service_call_dict["method"] = f"{service}{canonical_value}"
    else:
        raise ValueError("Camel case is not supported")
        # service_call_dict["method"] = f"{service}.{canonical_value}"


def cast_to_statement(
    service_call: ServiceCall,
    tag,
    info: Optional[dict[str, Any]] = None,
    snake_case: bool = True,
) -> ProgramStatement:
    """Cast an SGD `ServiceCall` object to a PyTOD ProgramStatement."""
    formatter = PythonFunctionServiceCallFormatter(
        {"convert_camel_case": snake_case, "quote_values": True}
    )
    expression = formatter.call_to_text(service_call)
    statement = ProgramStatement(expression=expression, tag=tag, info=info)
    # sanity check that the signature is a valid python function call
    try:
        _ = ast.parse(statement.expression)
    except SyntaxError as e:
        logger.error(f"Could not parse {expression}...")
        raise e
    return statement


@dispatch_on_value
def update_service_call_dict(
    value_type: str,
    service_call_dict: dict[str, Union[str, dict[str, str]]],
    template_fields: list[TemplateField],
    skip_quote: set[str],
    actions: list[UserAction],
    reference_command_for_wildcard_carryover: bool = False,
    system_notified_failure: bool = False,
):
    raise ValueError(
        f"Unknown value {value_type}. Expected: 'natural_language' or 'variable_reference'."
    )


@update_service_call_dict.register("natural_language")
def _(
    value_type: str,
    service_call_dict: dict[str, Union[str, dict[str, str]]],
    template_fields: list[TemplateField],
    skip_quote: set[str],
    actions: list[UserAction],
    reference_command_for_wildcard_carryover: bool = False,
    system_notified_failure: bool = False,
):
    for a in actions:
        service_call_dict["parameters"].update({a.slot: a.values[0]})


@update_service_call_dict.register("selection_variable_reference")
def _(
    value_type: str,
    service_call_dict: dict[str, Union[str, dict[str, str]]],
    template_fields: list[TemplateField],
    skip_quote: set[str],
    actions: list[UserAction],
    reference_command_for_wildcard_carryover: bool = False,
    system_notified_failure: bool = False,
):
    for action in actions:
        if system_notified_failure:
            field = "last_call"
            value = f"{{last_call}}.{action.slot}"
        else:
            field = "current_entity"
            value = f"{{current_entity}}.{action.slot}"
        # note the mechanism of binding the current entity is a
        # result_selection_by_indexing_with_new_task tag, not a copy
        template_fields.append(TemplateField(is_variable=True, field=field))
        service_call_dict["parameters"].update({action.slot: value})
        skip_quote.add(action.slot)


@update_service_call_dict.register("variable_reference")
def _(
    value_type: str,
    service_call_dict: dict[str, Union[str, dict[str, str]]],
    template_fields: list[TemplateField],
    skip_quote: set[str],
    actions: list[UserAction],
    reference_command_for_wildcard_carryover: bool = False,
    system_notified_failure: bool = False,
):
    for i, action in enumerate(actions):
        if reference_command_for_wildcard_carryover and WILDCARD_VALUE in action.values:
            value = f"{{last_call}}.{action.metadata['carryover'][0].slot}"
            field = "last_call"
            resolve_with_metadata = False
        else:
            resolve_with_metadata = True
            value = f"{{call_or_entity_reference_{i}}}.{action.metadata['carryover'][0].slot}"
            field = f"call_or_entity_reference_{i}"
        if system_notified_failure:
            resolve_with_metadata = False
            value = f"{{last_call}}.{action.metadata['carryover'][0].slot}"
            field = "last_call"

        this_template_args = {
            "field": field,
            "metadata": action.metadata["carryover"][0],
            "resolve_with_metadata": resolve_with_metadata,
            "is_variable": True,
        }
        template_fields.append(TemplateField(**this_template_args))
        service_call_dict["parameters"].update({action.slot: value})
        skip_quote.add(action.slot)
