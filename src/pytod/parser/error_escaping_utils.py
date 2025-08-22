#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import re
from enum import Enum

from pytod.parser.expressions import Expression
from pytod.utils import snake_case

CALL_PATTERN = r"^[a-z_]\w+\(.*"

ASSIGNMENT_PATTERN = r"^x\d{1,2}\.\s?[a-z_]+(?=\s?=)"

logger = logging.getLogger(__name__)


class SyntaxErrorType(Enum):
    CLOSING_PARENTHESIS_MISSING = "closing_parenthesis_missing"
    CLOSING_PARENTHESIS_EXTRA = "closing_parenthesis_extra"
    DECIMAL_LITERAL = "invalid_decimal_literal"
    INVALID_SYNTAX = "invalid_syntax"
    NO_ASSIGNMENT_ALLOWED = "no_assignment_allowed"
    NO_EXPRESSION_ASSIGN = "no_expression_assign"
    UNEXPECTED_POSITIONAL = "unexpected_positional"
    UNTERMINATED_LITERAL = "unterminated_string_literal"
    UNEXPECTED_ASSIGNMENT = "unexpected_assignment"
    LEADING_ZEROS_IN_DECIMAL = "leading_zeros"
    NO_ASSIGN_EXPRESSION = "assignment_to_expression_not_allowed"
    NO_ASSIGN_FCN_CALL = "assignment_to_function_call_not_allowed"
    UNKNOWN = "UNKNOWN"


class ExpressionType(Enum):
    ASSIGNMENT = "assignment"
    CALL = "call"
    SYNTAX_ERR_CALL = "call_syntax_err"
    SYNTAX_ERR_ASSIGNMENT = "assignment_syntax_err"
    ASSIGNMENT_CALL = "assignment_call"


def classify_error(err_msg: str) -> SyntaxErrorType:
    """Standardise the `ast.parse` error messages."""
    if "unterminated string literal" in err_msg:
        return SyntaxErrorType.UNTERMINATED_LITERAL
    if "invalid decimal literal" in err_msg:
        return SyntaxErrorType.DECIMAL_LITERAL
    if "'(' was never closed" in err_msg:
        return SyntaxErrorType.CLOSING_PARENTHESIS_MISSING
    if "unmatched ')'" in err_msg:
        return SyntaxErrorType.CLOSING_PARENTHESIS_EXTRA
    if "invalid syntax" in err_msg:
        return SyntaxErrorType.INVALID_SYNTAX
    if "positional argument follows keyword argument" in err_msg:
        return SyntaxErrorType.UNEXPECTED_POSITIONAL
    if "cannot assign to attribute here" in err_msg:
        return SyntaxErrorType.UNEXPECTED_ASSIGNMENT
    if "cannot assign to expression here" in err_msg:
        return SyntaxErrorType.NO_EXPRESSION_ASSIGN
    if "expression cannot contain assignment" in err_msg:
        return SyntaxErrorType.NO_ASSIGNMENT_ALLOWED
    if "leading zeros in decimal integer literals are not permitted" in err_msg:
        return SyntaxErrorType.LEADING_ZEROS_IN_DECIMAL
    if "cannot assign to expression" in err_msg:
        return SyntaxErrorType.NO_ASSIGN_EXPRESSION
    if "cannot assign to function call" in err_msg:
        return SyntaxErrorType.NO_ASSIGN_FCN_CALL
    logger.error(f"Unclassified syntax error: {err_msg}")
    return SyntaxErrorType.UNKNOWN


def classify_turn_type(model_output: str) -> ExpressionType:
    """Classify a model generation as a `call` or `assignment`
    (or error thereof)."""

    open_bracket_cnt = model_output.count("(")
    closed_bracket_cnt = model_output.count(")")
    if re.match(CALL_PATTERN, model_output):
        if open_bracket_cnt == closed_bracket_cnt == 1:
            return ExpressionType.CALL
        return ExpressionType.SYNTAX_ERR_CALL
    if re.match(ASSIGNMENT_PATTERN, model_output):
        if open_bracket_cnt == closed_bracket_cnt == 0:
            eq_operators = model_output.count("=")
            n_separators = model_output.count(";")
            if eq_operators == n_separators + 1:
                return ExpressionType.ASSIGNMENT
            return ExpressionType.SYNTAX_ERR_ASSIGNMENT
        if open_bracket_cnt == closed_bracket_cnt > 1:
            return ExpressionType.ASSIGNMENT_CALL
        return ExpressionType.SYNTAX_ERR_ASSIGNMENT
    if open_bracket_cnt == closed_bracket_cnt and ";" not in model_output:
        if open_bracket_cnt == 0:
            return ExpressionType.SYNTAX_ERR_ASSIGNMENT
        return ExpressionType.SYNTAX_ERR_CALL
    if ";" in model_output and any((open_bracket_cnt > 0, closed_bracket_cnt > 0)):
        return ExpressionType.ASSIGNMENT_CALL
    return ExpressionType.SYNTAX_ERR_ASSIGNMENT


def get_keywords(model_output: str, turn_type: ExpressionType) -> tuple[str, list[str]]:
    """Get the assignment expressions from the model output."""
    match turn_type:
        case ExpressionType.CALL:
            keywords = re.split(
                r",\s",
                model_output[model_output.find("(") + 1 : model_output.find(")")],
            )
            tool = model_output[: model_output.find("(")]
        case ExpressionType.ASSIGNMENT:
            keywords = re.split(r";\s", model_output)
            tool = "assign"
        case _:
            raise ValueError(f"Unexpected expression type: {turn_type}")
    return tool, keywords


def format_as_call(tool_name: str, valid_expr: list[str]) -> str:
    """Return a properly formatted call given a list of
    valid expressions.

    Parameters
    ----------
    valid_expr
        List of "kw='value'" strings.
    """
    keyword_args, kwarg_values = [], []
    for e in valid_expr:
        try:
            kw, val = e.split("=")
            if is_valid_keyword(kw.strip()):
                _ = ast.parse(f"{tool_name}({kw.strip()}={val.strip()})")
                keyword_args.append(kw.strip())
                kwarg_values.append(val.strip())
            elif contains_variable_name(kw):
                try:
                    assert kw.count(".") == 1
                except AssertionError:
                    continue
                kwarg = kw.strip().split(".")[1].strip()
                _ = ast.parse(f"{tool_name}({kwarg}={val.strip()})")
                keyword_args.append(kwarg)
                kwarg_values.append(val.strip())
        except ValueError:
            continue
        except SyntaxError:
            continue

    return str(
        Expression(
            tool=tool_name,
            positional_args=[],
            keyword_args=keyword_args,
            kwarg_values=kwarg_values,
        )
    )


def is_valid_keyword(keyword: str, max_len: int = 25) -> bool:
    """A valid keyword in a function call must be snake cased and should not
    contain a variable name.

    Parameters
    ----------
    keyword
    max_len
        Max length of a slot in the corpus.
    """
    keyword = keyword[: max_len + 1]
    return (
        snake_case(keyword) == keyword
        and not contains_variable_name(keyword)
        and len(keyword) < max_len + 1
    )


def contains_variable_name(arg_or_value: str) -> bool:
    return re.match(r"x[0-9]+", arg_or_value) is not None


def maybe_truncate_on_repeating_pattern(slot_name: str, threshold: int = 1):
    """Truncate slot names if they contain repeated words separated by
    underscores."""

    parts = slot_name.split("_")
    prev_word = parts[0]
    count = 1
    for i in range(1, len(parts)):
        if parts[i] == prev_word:
            count += 1
        else:
            prev_word = parts[i]
            count = 1
        if count > threshold:
            return "_".join(parts[:i])
    return slot_name
