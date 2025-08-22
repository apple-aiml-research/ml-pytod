#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""A module containing utilities for working with Expression and ExpressionList parsers."""
import logging
from collections import Counter
from copy import deepcopy

from pytod.interpreter.metadata import INTENT_UPDATE_TOOL, NLG_CALL_TOOL, SPECIAL_TOOLS
from pytod.parser.expressions import Expression, ExpressionList
from pytod.pytod_types.aliases import SlotName, SlotValue, VariableName

logger = logging.getLogger(__name__)


def has_kwargs(expr: Expression) -> bool:
    return bool(expr.keyword_args)


def update_feedback(expression: ExpressionList, info: list[str]):
    """Add feedback to the ExpressionList object."""
    if not expression.feedback:
        expression.feedback.append({"expressions": info})
    else:
        for el in expression.feedback:
            if "expressions" in el:
                el["expressions"].extend(info)
                break
        else:
            expression.feedback.append({"expressions": info})


def get_assignment_vars(expression: ExpressionList) -> list[VariableName] | None:
    """Get the (unique) target variables of an assignment instruction."""
    if expression.get_tool_name() != INTENT_UPDATE_TOOL:
        return
    vars = set()
    for expr in expression.expressions:
        for kwarg in expr.keyword_args:
            try:
                var, _ = kwarg.split(".")
                vars.add(var)
            except ValueError:
                continue
    return list(vars) or None


def get_assignment_properties(
    expression: ExpressionList,
) -> dict[SlotName, SlotValue] | None:
    """Get the properties assigned to in an assignment instruction."""
    if expression.get_tool_name() != INTENT_UPDATE_TOOL:
        return
    args = {}
    for expr in expression.expressions:
        if expr.tool != INTENT_UPDATE_TOOL:
            continue
        for kwarg, kwval in zip(expr.keyword_args, expr.kwarg_values):
            try:
                _, arg = kwarg.split(".")
                args[arg] = kwval
            except ValueError:
                args[kwarg] = kwval
                continue
    return args or None


def replace_property(expression: ExpressionList, src: SlotName, tgt: SlotName):
    """Replace the `src` property with `tgt` property in `expression`."""
    if expression.get_tool_name() != INTENT_UPDATE_TOOL:
        return
    for expr in expression.expressions:
        if expr.tool != INTENT_UPDATE_TOOL:
            continue
        for i, (kwarg, val) in enumerate(zip(expr.keyword_args, expr.kwarg_values)):
            _, arg = kwarg.split(".")
            if src == arg:
                new_kwarg = kwarg.replace(src, tgt)
                expr.keyword_args[i] = new_kwarg


def extend_assignment(
    expression: ExpressionList,
    slot_name: SlotName,
    assigned_value: SlotValue,
    variable: str | None = None,
):
    """Extend an assignment expression with a slot-value pair."""
    if expression.get_tool_name() != INTENT_UPDATE_TOOL:
        return
    assert expression.expressions[0].tool == INTENT_UPDATE_TOOL
    try:
        variable = expression.expressions[0].keyword_args[0].split(".")[0]
        assignment = f"{variable}.{slot_name}"
        expression.expressions.append(
            Expression(
                tool=INTENT_UPDATE_TOOL,
                keyword_args=[assignment],
                kwarg_values=[assigned_value],
            )
        )
    except IndexError:
        assignment = f"{variable}.{slot_name}"
        expression.expressions[0].keyword_args = [assignment]
        expression.expressions[0].kwarg_values = [assigned_value]


def map_to_call(
    function_name: str, expression: ExpressionList
) -> ExpressionList | None:
    """Maps an assignment instruction to a function call."""
    if expression.get_tool_name() != INTENT_UPDATE_TOOL:
        return
    kwargs, kwarg_values = [], []
    for expr in expression.expressions:
        if expr.tool != INTENT_UPDATE_TOOL:
            continue
        for kwarg, value in zip(expr.keyword_args, expr.kwarg_values):
            kwarg_values.append(value)
            try:
                _, arg = kwarg.split(".")
                kwargs.append(arg)
            except ValueError:
                kwargs.append(kwarg)

    return ExpressionList(
        expressions=[
            Expression(
                tool=function_name,
                positional_args=[],
                keyword_args=kwargs,
                kwarg_values=kwarg_values,
            )
        ]
    )


def map_to_assignment(
    expression: ExpressionList, variable: VariableName
) -> ExpressionList | None:
    kwargs, kwarg_vals = [], []
    for e in expression.expressions:
        assert e.tool != INTENT_UPDATE_TOOL
        for i, kwarg in enumerate(e.keyword_args):
            kwargs.append(f"{variable}.{kwarg}")
            kwarg_vals.append(e.kwarg_values[i])
    if not kwargs:
        logger.warning(f"Expression {expression} could not be mapped to assignment")
        return
    return ExpressionList(
        expressions=[
            Expression(
                tool=INTENT_UPDATE_TOOL,
                positional_args=[],
                keyword_args=[kwargs[i]],
                kwarg_values=[kwarg_vals[i]],
            )
            for i in range(len(kwargs))
        ]
    )


def maks_conversation_end(expression: ExpressionList) -> bool:
    if len(expression.expressions) > 1:
        return False
    match expression.get_tool_name():
        case var if var == NLG_CALL_TOOL:
            return not bool(expression.expressions[0].positional_args)
        case _:
            return False


def merge_calls(expressions: list[ExpressionList]) -> list[ExpressionList]:
    def _merge(duplicates: list[ExpressionList]) -> ExpressionList:
        first_expression = deepcopy(duplicates[0].expressions[0])
        for d in duplicates[1:]:
            expr = d.expressions[0]
            for kwarg, value in zip(expr.keyword_args, expr.kwarg_values):
                if kwarg not in first_expression.keyword_args:
                    first_expression.keyword_args.append(kwarg)
                    first_expression.kwarg_values.append(value)
        return ExpressionList(expressions=[first_expression])

    tools = [expr.get_tool_name() for expr in expressions]
    tool_counts = Counter(tools)
    merged_expressions = []
    merged_tools = set()
    for i, t in enumerate(tools):
        if tool_counts[t] > 1:
            if t not in SPECIAL_TOOLS:
                if t not in merged_tools:
                    logger.error(f"Duplicated call: {t}")
                    merged_tools.add(t)
                    merged_expressions.append(
                        _merge([e for e in expressions if e.get_tool_name() == t])
                    )
                    merged_expressions[-1].feedback.append(
                        {"expressions": "Expected a single function call"}
                    )
            else:
                if t not in merged_tools:
                    logger.error(f"Duplicated call to tool: {t}")
                    merged_tools.add(t)
                    merged_expressions.append(expressions[i])
                    merged_expressions[-1].feedback.append(
                        {"expressions": f"Expected a single call to {t}"}
                    )
        else:
            merged_expressions.append(expressions[i])
    if len(merged_expressions) != len(expressions):
        logger.warning("Tool calls were  duplicated")
    return merged_expressions


def get_kwargs(expressions: ExpressionList) -> list[str] | None:
    """Return all the keyword arguments of all the expressions in
    the expression list."""

    if expressions.get_tool_name() == INTENT_UPDATE_TOOL:
        return list(get_assignment_properties(expressions).keys())
    try:
        assert len(expressions.expressions) == 1
    except AssertionError:
        logger.error(
            f"Multiple statements in a single expression: {str(expressions)}. "
            "This is not expected."
        )
    kwargs = []
    for expr in expressions.expressions:
        kwargs += expr.keyword_args
    return kwargs or None


def constrain_assignment_variables(
    expressions: ExpressionList,
    to_constrain: VariableName,
    constrained_value: VariableName,
):
    """In-place update assignment variables.

    Parameters
    ----------
    to_constrain
        The variable one or more assignments refer to.
    constrained_value
        The variable the assignment should refer to.
    """
    for e in expressions.expressions:
        logger.info(
            f"Constraining variable {to_constrain} to {constrained_value} in {str(expressions)}"
        )
        e.keyword_args[0] = e.keyword_args[0].replace(to_constrain, constrained_value)
