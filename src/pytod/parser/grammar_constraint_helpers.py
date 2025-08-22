#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import re
from typing import Any, Literal

from pytod.interpreter.metadata import (
    ASSIGNMENT_START,
    INTENT_UPDATE_TOOL,
    ITERATION_TOOL,
    NO_KWARG_TOOLS,
    PARSE_ERROR_TOOL,
    QUERY_RESULTS_REF_KWARG,
    SINGLE_ARG_TOOLS,
    SPECIAL_TOOLS,
    VARIABLE_RESOLUTION_ERROR,
    ZERO_ARG_TOOLS,
)
from pytod.parser.expresion_validation_context import (
    AssistantConstraintContext,
    SchemaConstraintContext,
    SchemaValidationContext,
)
from pytod.parser.expresion_validation_utils import contains_object_reference
from pytod.utils import camel_to_snake_case, snake_case

kwargs_context_keys = Literal["lineno", "allow_lineno_refs", "last_entity"]

logger = logging.getLogger(__name__)

FieldName = str  # any field of the Expression class


def var_index(var: str) -> int:
    return int(var[1:])


def validate_tool_name(field_name: str) -> list[str] | None:
    feedback = []
    try:
        assert snake_case(field_name) == field_name
    except AssertionError:
        feedback.append(
            f"Error validating function name: expecting {field_name} to be snake cased"
        )
        match ast.parse(field_name).body[0]:
            case ast.Expr(value=ast.Attribute(value, attr)):  # noqa
                feedback.append(
                    f"Attribute access is illegal in function names: {field_name}"
                )
    return feedback or None


def ensure_no_illegal_kwargs(
    data: dict[str, Any],
) -> tuple[FieldName, list[str]] | None:
    """Ensure `NO_KWARG_TOOLS` are not called with the keyword-value syntax."""
    feedback = []
    field_name = "keyword_args"
    if (tool := data["tool"]) in NO_KWARG_TOOLS:
        if kwargs := data[field_name]:
            feedback.append(
                f"Function {tool} does not take keyword arguments. "
                f"Ignoring arguments: {kwargs}."
            )
            data[field_name] = []

        if kwarg_values := data["kwarg_values"]:
            feedback.append(
                f"Function {tool} does not take keyword arguments. "
                f"Ignoring values: {kwarg_values}"
            )
            data["kwarg_values"] = []
    if feedback:
        return field_name, feedback
    return


def validate_assignment_syntax(
    data: dict[str, Any]
) -> tuple[FieldName, list[str]] | None:
    """Ensure assignments:

    - have exactly one slot and one value
    - that left hand side contains a reference to a variable property
    - that right hand side contains a slot value
    - that the value of a slot is modified only once
    """

    if "tool" in data and data["tool"] == INTENT_UPDATE_TOOL:
        feedback = []
        field_name = "keyword_args"
        field_values = data[field_name]
        try:
            assert (n_pieces := len(field_values)) == len(data["kwarg_values"]) == 1
        except AssertionError:
            feedback.append(
                f"Assignment must have exactly one slot name/value, got {n_pieces}"
            )
            data[field_name] = []
            data["kwarg_values"] = []
            return field_name, feedback
        # LHS correct
        try:
            assert re.match(ASSIGNMENT_START, field_values[0]) is not None
        except AssertionError:
            feedback.append(
                "Left hand side of assignment must be variable attribute access. "
                "Variable names match the x[0-9]+ regular expression pattern."
            )
            data[field_name] = []
            data["kwarg_values"] = []
            return field_name, feedback
        # RHS correct
        try:
            assigned_value = data["kwarg_values"][0]
            assert re.match(ASSIGNMENT_START, assigned_value) is None
        except AssertionError:
            feedback.append(
                "Right hand side of assignment must be a value, not a variable reference"
            )
            data[field_name] = []
            data["kwarg_values"] = []
            return field_name, feedback
        # proper quoting
        if assigned_value not in {"true", "false"}:
            try:
                _ = ast.literal_eval(assigned_value)
            except ValueError:
                # nb: should never be raised
                try:
                    assert not re.match(ASSIGNMENT_START, assigned_value)
                except AssertionError:
                    logger.error(
                        "Variable assignments on assignment RHS are disallowed"
                    )
                    raise AssertionError
                feedback.append("Incorrect quoting of assigned value in assignment")
                assigned_value = assigned_value.strip(""""'""").replace('"', "'")
                data["kwarg_values"][0] = f'"{assigned_value}"'
    return


def is_valid_variable_index(
    kwarg: str,
    value: str,
    context: dict[kwargs_context_keys, Any] | SchemaValidationContext,
) -> bool:
    """Ensure values containing variable references cannot refer to the
    current line unless the statement is part of a multi-line statement."""

    def _is_valid(value: str, context: dict[str, Any]) -> bool:
        """Ensure the variable index predicted is smaller than the
        current line number, unless multiple instructions have been
        predicted."""
        lineno_ = context["lineno"]
        pred_var_idx = var_index(value.split(".")[0])
        is_greater = pred_var_idx > lineno_
        is_equal = pred_var_idx == lineno_
        if is_greater:
            return False
        if is_equal:
            return context["allow_lineno_refs"]
        return True

    if context is None or isinstance(
        context, (SchemaConstraintContext, AssistantConstraintContext)
    ):
        return True
    if kwarg == QUERY_RESULTS_REF_KWARG:
        if re.match(r"x[0-9]{1,2}", value) is None:
            return False
        return _is_valid(value, context)

    if re.match(ASSIGNMENT_START, value) is not None:
        return _is_valid(value, context)

    return re.match(r"x[0-9]{1,2}", value) is None


def maybe_constrain_value_reference(
    data: dict[str, Any],
    value: str,
    context: dict[kwargs_context_keys, Any] | SchemaValidationContext | None,
    feedback: list[str],
) -> str:
    """Post-hoc constrain the value reference to refer to the last entity if variable
    references in a call refer to the current line and the model did not
    predict multiple instructions.

    Parameters
    ----------
    data, value
    context
        Contains information necessary for constraining value references
        in the current value.
    feedback
        Collects information about validation process.
    """

    if context is None or isinstance(
        context, (SchemaConstraintContext, AssistantConstraintContext)
    ):
        return value
    tool_name = data["tool"]
    if tool_name in SPECIAL_TOOLS:
        logger.error(f"Variable referenced current line for tool: {tool_name}")
        return value
    if (last_entity := context["last_entity"]) is None:
        msg = (
            f"It appears that no entity from the `{tool_name}` "
            f"has been bound to a variable. "
            f"Perhaps a query argument was missed?"
        )
        # ensure that the hallucinated variables are at most the current
        # line number
        if re.match(ASSIGNMENT_START, value):
            pred_var_idx = var_index(value.split(".")[0])
            if pred_var_idx > (lineno := context["lineno"]):
                value = re.sub(r"x[0-9]+", f"x{lineno}", value)
        data["action"] = {"insert": ITERATION_TOOL}
        if msg not in feedback:
            feedback.append(msg)
        return value
    feedback.append(
        "A `select` instruction may have been missed or else "
        f"the last entity {last_entity} should be referenced"
    )
    new_value = re.sub(r"x[0-9]+", last_entity, value)
    logger.info(f"Replaced {value} reference with {new_value}")
    return new_value


def is_valid_kwarg(kwarg: str) -> bool:
    return not re.match(r"x\d{1,2}[A-Za-z_]", kwarg)


def ensure_kwargs_correct(
    data: dict[str, Any],
    context: dict[kwargs_context_keys, Any] | SchemaConstraintContext | None,
) -> tuple[FieldName, list[str]] | None:
    """Ensures that:

    - the number of keyword arguments equals the number of keyword argument values
    - that the slot names are snake-cased
    - kwargs are not duplicated
    - the value references are correct
    """

    feedback = []
    kwargs = data["keyword_args"]
    kwarg_vals = data["kwarg_values"]
    n_kwargs, n_kwarg_vals = len(kwargs), len(kwarg_vals)
    try:
        n_kwargs == n_kwarg_vals
    except AssertionError:
        feedback.append(
            "The number of arguments names and values should be equal. "
            f"Got {n_kwargs} arguments and {n_kwarg_vals} values"
        )
        if any((n_kwargs == 0, n_kwarg_vals == 0)):
            data["keyword_args"] = []
            data["kwarg_values"] = []
            return "keyword_args", feedback
        min_args = min(n_kwargs, n_kwarg_vals)
        data["keyword_args"] = data["keyword_args"][:min_args]
        data["kwarg_values"] = data["kwarg_values"][:min_args]
    valid_kwargs, valid_kwarg_values = [], []
    for kwarg, kwarg_val in zip(data["keyword_args"], data["kwarg_values"]):
        if not is_valid_kwarg(kwarg):
            msg = (
                f"Invalid keyword argument: {kwarg}. "
                f"Keywords cannot start with variable names!"
            )
            logger.warning(msg)
            feedback.append(msg)
            continue
        if camel_to_snake_case(kwarg) == kwarg:
            if kwarg in valid_kwargs:
                feedback.append(
                    f"Duplicated keyword argument: {kwarg}. Value: {kwarg_val}"
                )
                continue
            if is_valid_variable_index(kwarg, kwarg_val, context):
                valid_kwargs.append(kwarg)
                valid_kwarg_values.append(kwarg_val)
            else:
                # was wrong but didn't match assignment start
                # means it starts with x{digit}
                valid_kwargs.append(kwarg)
                if re.match(ASSIGNMENT_START, kwarg_val) is None:
                    msg = f"Hallucinated value in expression: {kwarg} = {kwarg_val}"
                    feedback.append(msg)
                    logger.warning(msg)
                    # sometimes the model misses the variable name & outputs the
                    # subscript
                    if kwarg == QUERY_RESULTS_REF_KWARG:
                        try:
                            _ = int(kwarg_val)
                            value = f"x{kwarg_val}"
                        except ValueError:
                            value = VARIABLE_RESOLUTION_ERROR
                    else:
                        value = VARIABLE_RESOLUTION_ERROR
                    valid_kwarg_values.append(value)
                else:
                    feedback.append(
                        f"Value references cannot refer to the current line "
                        f" unless a multiple statements separated by <new> are generated. "
                        f"Got:  {kwarg} = {kwarg_val}"
                    )
                    valid_kwarg_values.append(
                        maybe_constrain_value_reference(
                            data, kwarg_val, context, feedback
                        )
                    )
        else:
            feedback.append(f"Expecting {kwarg_val} to be snake cased")
    data["keyword_args"] = valid_kwargs
    data["kwarg_values"] = valid_kwarg_values
    if feedback:
        return "keyword_args", feedback
    return


def ensure_obj_ref_syntax_correct(
    data: dict[str, Any],
    context: dict[kwargs_context_keys, Any] | SchemaConstraintContext | None,
) -> tuple[FieldName, list[str]] | None:
    """Ensures that:

    - the syntax for object references is in the format {variable}.{attribute}."""
    feedback = []
    valid_kwargs, valid_kwarg_values = [], []
    for kwarg, kwarg_val in zip(data["keyword_args"], data["kwarg_values"]):
        if not contains_object_reference(kwarg_val):
            valid_kwargs.append(kwarg)
            valid_kwarg_values.append(kwarg_val)
        else:
            parsed = ast.parse(kwarg_val)
            match parsed.body[0].value:
                case ast.Attribute(value=ast.Name(), attr=attr):  # noqa
                    valid_kwargs.append(kwarg)
                    valid_kwarg_values.append(kwarg_val)
                case _:
                    msg = f"Invalid object reference in keyword: {kwarg}={kwarg_val}"
                    logger.error(msg)
                    feedback.append(msg)
    data["keyword_args"] = valid_kwargs
    data["kwarg_values"] = valid_kwarg_values
    if feedback:
        return "keyword_args", feedback
    return


def ensure_no_args(data: dict[str, Any]) -> tuple[FieldName, list[str]] | None:
    """Ensure that `conversation_pause` and `decline_alternative`
    tools cannot be called with any args or kwargs
    """
    if (tool := data["tool"]) in ZERO_ARG_TOOLS:
        feedback = []
        field_name = "keyword_args"
        if pos := data["positional_args"]:
            field_name = "positional_args"
            feedback.append(
                f"Function {tool} takes no arguments. Got positional arguments: {pos}"
            )
        if kwargs := data["keyword_args"]:
            field_name = "keyword_args"
            feedback.append(
                f"Function {tool} takes no arguments. Got keyword arguments: {kwargs}"
            )
        data["positional_args"] = []
        data["keyword_args"] = []
        data["kwarg_values"] = []
        if feedback:
            return field_name, feedback
    return


def ensure_single_arg(data: dict[str, Any]) -> tuple[FieldName, list[str]] | None:
    """Ensure that tools that take a single argument are called accordingly"""
    if (tool := data["tool"]) in SINGLE_ARG_TOOLS:
        feedback = []
        field_name = "keyword_args"
        if not (pos_args := data["positional_args"]):
            field_name = "positional_args"
            feedback.append(
                f"Function {tool} takes a single positional argument. Got none."
            )
        if kwargs := data["keyword_args"]:
            field_name = "keyword_args"
            feedback.append(
                f"Function {tool} takes no keyword arguments. Got : {kwargs}"
            )
        if not pos_args:
            data["tool"] = PARSE_ERROR_TOOL
        data["keyword_args"] = []
        data["kwarg_values"] = []
        if feedback:
            return field_name, feedback
    return


ensure_no_illegal_kwargs.needs_context = False
validate_assignment_syntax.needs_context = False
ensure_kwargs_correct.needs_context = True
ensure_no_args.needs_context = False
ensure_single_arg.needs_context = False
ensure_obj_ref_syntax_correct.needs_context = True
