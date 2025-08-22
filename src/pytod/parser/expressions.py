#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from pytod.interpreter.metadata import (
    INTENT_UPDATE_TOOL,
    PARSE_ERROR_TOOL,
    POSITIONAL_TOOLS,
    SINGLE_ARG_TOOLS,
    SPECIAL_TOOLS,
)
from pytod.parser.expresion_validation_context import SchemaValidationContext
from pytod.parser.expresion_validation_utils import parse_error_dict
from pytod.parser.grammar_constraint_helpers import (
    ensure_kwargs_correct,
    ensure_no_args,
    ensure_no_illegal_kwargs,
    ensure_obj_ref_syntax_correct,
    ensure_single_arg,
    validate_assignment_syntax,
    validate_tool_name,
    var_index,
)
from pytod.parser.schema_constraint_helpers import (
    constrain_arguments,
    constrain_tool_name,
    constrain_value,
    constrain_value_object_reference,
)

logger = logging.getLogger(__name__)

FieldName = str  # any field of the Expression class

_init_context_var = ContextVar("_init_context_var", default=None)


@contextmanager
def expression_init_context(value: dict[str, Any] | Any) -> Iterator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


class Expression(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:  # noqa
        __pydantic_self__.__pydantic_validator__.validate_python(
            data,
            self_instance=__pydantic_self__,
            context=_init_context_var.get(),
        )

    feedback: defaultdict[
        Literal["tool", "positional_args", "keyword_args", "kwarg_values"], list[str]
    ] = defaultdict(list)
    tool: str
    positional_args: list[str] = []
    keyword_args: list[str] = []
    kwarg_values: list[str] = []
    # information for execution. this can be {"select" : 'auto'} or {"next": 'auto'},
    # directives which are inserted while constraining variable names
    # src/pytod/inference/dialogue_session_robust_utils.py::_escape_selection_mapping_errors and
    # src/pytod/inference/dialogue_session_robust_utils.py::_escape_function_call_mapping_errors
    exec_info: dict[str, str] = {}
    # set during parsing by maybe_constrain_value_reference only
    # for intents (inferred by checking membership of the tool
    # name in interpreter/metadata.py::SPECIAL_TOOLS
    action: dict[str, str] = {}

    def __eq__(self, other: Self) -> bool:
        return (
            self.tool == other.tool
            and sorted(self.keyword_args) == sorted(other.keyword_args)
            and sorted(self.kwarg_values) == sorted(other.kwarg_values)
            and sorted(self.positional_args) == other.positional_args
        )

    @model_validator(mode="before")
    @classmethod
    def schema_validation(cls, data: dict[str, Any], info: ValidationInfo):
        """Ensure that the predictions follow schema constraints."""

        context = info.context
        if not isinstance(context, SchemaValidationContext):
            return data
        validation_feedback = defaultdict(list)
        for validator in (
            constrain_tool_name,
            constrain_arguments,
            constrain_value,
            constrain_value_object_reference,
        ):
            if validator.needs_context:
                feedback = validator(data, context)
            else:
                feedback = validator(data)
            if feedback is not None:
                key, content = feedback
                validation_feedback[key].extend(content)
            if data["tool"] == PARSE_ERROR_TOOL:
                break
        cls.update_feedback(data, validation_feedback)
        return data

    @model_validator(mode="before")
    @classmethod
    def ensure_expression_correct(cls, data: dict[str, Any], info: ValidationInfo):
        """Ensure the model generated data follows the PyTOD grammar."""

        validation_feedback = defaultdict(list)
        feedback = validate_tool_name(data["tool"])
        # validate tool name
        if feedback is not None:
            validation_feedback["tool"].extend(feedback)
            match ast.parse(data["tool"]).body[0]:
                case ast.Expr(value=ast.Attribute(value, attr)):  # noqa
                    logger.warning("Attribute access detected in tool name")
                    data["tool"] = attr
                case _:
                    modified_s, count = re.subn(r"^x\d{1,2}_", "", data["tool"])
                    if count > 0:
                        logger.warning("Function started with variable name")
                        feedback.append(
                            f"Functions should never start with variable names: {data['tool']}"
                        )
                        data["tool"] = modified_s
                    else:
                        data["tool"] = PARSE_ERROR_TOOL
                        data["keyword_args"] = []
                        data["kwarg_values"] = []
                        data["positional_args"] = []
                        data["feedback"] = dict(validation_feedback)
                    return data

        # context accessible by all validators
        context = info.context
        # validate keyword arguments and values
        for validator in (
            validate_assignment_syntax,
            ensure_no_illegal_kwargs,
            ensure_kwargs_correct,
            ensure_obj_ref_syntax_correct,
            ensure_no_args,
            ensure_single_arg,
        ):
            if validator.needs_context:
                feedback = validator(data, context)
            else:
                feedback = validator(data)
            if feedback is not None:
                key, content = feedback
                validation_feedback[key].extend(content)
        cls.update_feedback(data, validation_feedback)

        return data

    @classmethod
    def update_feedback(
        cls, data: dict[str, Any], validation_feedback: defaultdict[str, list[str]]
    ):
        if "feedback" not in data:
            data["feedback"] = defaultdict(list)
        if validation_feedback:
            for k in validation_feedback:
                data["feedback"][k].extend(validation_feedback[k])

    @field_validator("positional_args")
    @classmethod
    def check_ordered_args_on_assign(
        cls, field_value: list[str], info: ValidationInfo
    ) -> list[str]:
        if "tool" in info.data and (tool := info.data["tool"]) not in POSITIONAL_TOOLS:
            try:
                assert len(field_value) == 0
                return field_value
            except AssertionError:
                info.data["feedback"][info.field_name].extend(
                    [f"Function {tool} cannot have positional args, got {field_value}"]
                )
                return []
        return field_value

    @field_validator("positional_args")
    @classmethod
    def check_positional_args_are_variables(
        cls, field_values: list[str], info: ValidationInfo
    ) -> list[str]:
        valid_args = []
        feedback = []
        for arg in field_values:
            try:
                assert re.match(r"x[0-9]+", arg) is not None
                valid_args.append(arg)
            except AssertionError:
                feedback.append(
                    f"Positional arguments must start with `x<digit>`. Got {arg}. "
                    f"Ignoring {arg}"
                )
        if feedback:
            info.data["feedback"][info.field_name].extend(feedback)
        return valid_args

    @field_validator("positional_args")
    @classmethod
    def ensure_single_arg(
        cls, field_values: list[str], info: ValidationInfo
    ) -> list[str] | None:
        """Ensure `SINGLE_ARG_TOOLS` can only be called with a single argument."""
        feedback = []
        all_args = field_values
        if (tool := info.data["tool"]) in SINGLE_ARG_TOOLS:
            if (n_positional := len(field_values)) > 1:
                all_args = copy(field_values)
                ignored = ", ".join(all_args[1:])
                feedback.append(
                    f"Function {tool} takes a single positional argument but"
                    f" {n_positional} were found. The following arguments "
                    f"will be ignored: {ignored}"
                )
                all_args = [all_args[0]]
        if feedback:
            info.data["feedback"][info.field_name].extend(feedback)
        return all_args

    @field_validator("kwarg_values")
    @classmethod
    def ensure_object_references_unquoted(
        cls, field_values: list[str], info: ValidationInfo
    ) -> list[str] | None:
        """In some cases, the model can quote object references,
        which breaks downstream validation."""

        feedback = []
        for i, val in enumerate(field_values):
            if re.match(r"'x\d{1,2}\.[a-z_]+'", val):
                field_values[i] = val[1:-1]
                if not feedback:
                    feedback.append(
                        "One or multiple values containing object "
                        "references were quoted. This is not expected."
                    )
        if feedback:
            info.data["feedback"][info.field_name].extend(feedback)
        return field_values

    def __str__(self) -> str:
        if self.tool == INTENT_UPDATE_TOOL:
            return f"{self.keyword_args[0]} = {self.kwarg_values[0]}"
        else:
            arg_str = ", ".join(
                self.positional_args
                + [f"{s} = {t}" for s, t in zip(self.keyword_args, self.kwarg_values)]
            )
            return f"{self.tool}({arg_str})"

    @classmethod
    def from_ast(cls, expr: ast.AST, expression_str: str | None = None) -> "Expression":
        match expr:
            case ast.Expr(value=ast.Call(func, args, keywords)):
                slot_names = []
                slot_values = []
                for k in keywords:
                    slot_name, slot_value = k.arg, ast.unparse(k.value)
                    assert slot_name is not None
                    assert slot_value is not None
                    slot_names.append(slot_name)
                    slot_values.append(slot_value)
                return Expression(
                    tool=ast.unparse(func),
                    positional_args=[ast.unparse(a) for a in args],
                    keyword_args=slot_names,
                    kwarg_values=slot_values,
                )
            case ast.Assign([target], value):
                return Expression(
                    tool=INTENT_UPDATE_TOOL,
                    positional_args=[],
                    keyword_args=[ast.unparse(target)],
                    kwarg_values=[ast.unparse(value)],
                )
            case _:
                logger.warning(
                    f"Ignoring AST tree {ast.dump(expr)} (expression: {expression_str}) "
                    "as it matched neither assignment nor call syntax."
                )
                return Expression(
                    tool=PARSE_ERROR_TOOL,
                    positional_args=[],
                    keyword_args=[],
                    kwarg_values=[],
                )


class ExpressionList(BaseModel):
    feedback: list[dict] = []
    expressions: list[Expression]
    comment: str | None = None

    @classmethod
    def from_string(cls, expression_str: str) -> "ExpressionList":
        comment = None
        if (m := re.search(r"#\s+(?P<comment>.*)$", expression_str)) is not None:
            comment = m["comment"]

        try:
            parsed = ast.parse(expression_str.strip())
        except SyntaxError as e:
            # this code path should not be reached because we run ast.parse on the
            # raw strings before attempting to parse them into Expression objects.
            logger.error(
                f'Could not parse expression "{expression_str}. Error: {e.msg}"'
            )
            raise SyntaxError

        exprs = []
        feedback = []
        for ast_expr in parsed.body:
            expression = Expression.from_ast(ast_expr, expression_str=expression_str)
            if expression.feedback:
                feedback.append(expression.feedback)
                match expression.tool:
                    # filter out assignments that were not correct
                    case var if all(
                        (not expression.keyword_args, not expression.kwarg_values)
                    ) and var == INTENT_UPDATE_TOOL:
                        continue
                    case var if var == PARSE_ERROR_TOOL:
                        expression.positional_args = []
                        expression.kwarg_values = []
                        expression.keyword_args = []
            exprs.append(expression)
        if not exprs:
            logger.warning(
                f"Illegal expression: {expression_str}. " "Calling 'parse_error' tool."
            )
            exprs = [Expression.model_validate(parse_error_dict())]

        return cls(expressions=exprs, comment=comment, feedback=feedback)

    def __str__(self) -> str:
        expr = "; ".join([str(e) for e in self.expressions])
        return expr if self.comment is None else f"{expr} # {self.comment}"

    def __getitem__(self, item: int) -> Expression:
        return self.expressions[item]

    def __len__(self) -> int:
        return len(self.expressions)

    @field_validator("expressions")
    @classmethod
    def check_duplicated_expressions(
        cls, v: list[Expression], info: ValidationInfo
    ) -> list[Expression]:
        feedback = []
        unique_expr = []
        for expr in v:
            if expr not in unique_expr:
                unique_expr.append(expr)
        if len(unique_expr) < len(v):
            feedback.append(
                "Expression contained duplicated instructions, this is not expected"
            )
        if feedback:
            info.data["feedback"].append({"expressions": feedback})
        return unique_expr

    @field_validator("expressions")
    @classmethod
    def check_multi_expressions(
        cls, v: list[Expression], info: ValidationInfo
    ) -> list[Expression]:
        tools = [e.tool for e in v]
        all_assigns = all(tool == INTENT_UPDATE_TOOL for tool in tools)
        feedback = []
        try:
            assert len(v) == 1 or all_assigns
        except AssertionError:
            if set(tools).difference({INTENT_UPDATE_TOOL, PARSE_ERROR_TOOL}):
                feedback.append(
                    "Only multiple assignments can be combined in one line."
                )
                info.data["feedback"].append({"expressions": feedback})
            match tools:
                case [func, *vars_] if (
                    func not in SPECIAL_TOOLS
                    and all(var == INTENT_UPDATE_TOOL for var in vars_)
                ):
                    call, [*assignment] = v[0], v[1:]
                    assignment_disp = "; ".join([str(a) for a in assignment])
                    msg = (
                        f"Incorrect use of tools: {tools}. {assignment_disp} is "
                        "only used to update function calls when the user has "
                        "stated the task in a previous dialogue turn."
                    )
                    feedback.append(msg)
                    for e in assignment:
                        for keyword, value in zip(e.keyword_args, e.kwarg_values):
                            call.keyword_args.append(keyword.split(".")[1])
                            call.kwarg_values.append(value)
                    expressions = [call]
                case _ if all(t not in SPECIAL_TOOLS for t in tools):
                    msg = (
                        "Multiple calls to prompt APIs are not expected. A single"
                        "function call with appropriate arguments should be "
                        "generated instead"
                    )
                    feedback.append(msg)
                    call = v[0]
                    for v in v[1:]:
                        for keyword, value in zip(v.keyword_args, v.kwarg_values):
                            call.keyword_args.append(keyword)
                            call.kwarg_values.append(value)
                    expressions = [call]
                case _:
                    expressions = [expr for expr in v if expr.tool != PARSE_ERROR_TOOL]
            try:
                assert expressions, "Not expecting empty expression list"
            except AssertionError:
                feedback.append(
                    "All assignments referenced incorrect arguments. Please check"
                    "the API definitions carefully!"
                )
                info.data["feedback"].append({"expressions": feedback})
                return [parse_error_expression()]
            return expressions
        var_names = set()
        valid_expressions = v
        if all_assigns:
            valid_expressions = []
            expression_kwargs = []
            for e in v:
                assert (
                    len(e.keyword_args) == 1
                ), f"Assignment must have exactly 1 slot name, got {len(e.keyword_args)}"
                m = re.match(r"(x[0-9]+)\b.*", e.keyword_args[0])

                assert (
                    m is not None
                ), f'Variable must be x[0-9]+ or an attribute thereof, got "{e.keyword_args[0]}"'
                var_names.add(m[1])
                msg = "Assignment target must work on the same variable."
                try:
                    assert len(var_names) == 1
                except AssertionError:
                    feedback.append(msg)
            if feedback and len(var_names) > 1:
                first_var = f"x{min([var_index(v) for v in var_names])}"
                logger.info(
                    f"Ensuring multi-assignment consistency. Target var: {first_var}"
                )
                for e in v:
                    logger.info(f"Original expression: {str(e)}")
                    m = re.match(r"(x[0-9]+)\b.*", e.keyword_args[0])
                    e.keyword_args[0] = re.sub(m[1], first_var, e.keyword_args[0])
                    logger.info(f"Consistent expression: {str(e)}")
            # ignore multiple assignments to the same kwarg
            for e in v:
                if (target := e.keyword_args[0]) not in expression_kwargs:
                    expression_kwargs.append(e.keyword_args[0])
                    valid_expressions.append(e)
                else:
                    feedback.append(
                        f"Target should only be assigned to once. "
                        f"Attempted multiple assignments to: {target}."
                    )
        if feedback:
            info.data["feedback"].append({"expressions": feedback})
        return valid_expressions

    @field_validator("comment")
    @classmethod
    def check_comment_validity(cls, v: str | None, info: ValidationInfo) -> str | None:
        if "expressions" in info.data and info.data["expressions"]:
            if info.data["expressions"][0].tool == "len":
                return v
        return None

    def get_tool_name(self) -> str:
        return self.expressions[0].tool


def parse_error_expression(
    expression_list: bool = False,
) -> Expression | ExpressionList:
    parse_error = Expression(
        tool=PARSE_ERROR_TOOL,
        positional_args=[],
        keyword_args=[],
        kwarg_values=[],
    )
    if expression_list:
        return ExpressionList(expressions=[parse_error])
    return parse_error
