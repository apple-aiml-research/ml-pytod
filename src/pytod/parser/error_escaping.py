#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Utilities for sanitising model outputs and enforcing syntax correctness."""
import ast
import logging
import re
from typing import Optional

from pytod.interpreter.metadata import NLG_CALL_TOOL, PARSE_ERROR, PARSE_ERROR_TOOL
from pytod.parser.error_escaping_utils import (
    ExpressionType,
    SyntaxErrorType,
    classify_error,
    classify_turn_type,
    contains_variable_name,
    format_as_call,
    get_keywords,
    is_valid_keyword,
    maybe_truncate_on_repeating_pattern,
)
from pytod.parser.expressions import Expression, ExpressionList
from pytod.utils import dispatch_on_value

logger = logging.getLogger(__name__)

MISMATCHED_QUOTES = {("'", '"'), ('"', "'")}
QUOTES = {'"', "'"}


patterns = {
    "included_line_number": r"^x\d{1,2}\s\w+\([^)]*\)$",
    "assigned_to_line_var": r"^x\d{1,2} = ",
    "space_in_slot_name": r"_\s(?=[a-z])",
    "space_in_slot_name_nlg": r"(x\d+)\.(\w+)\s",
}


def ensure_parseable(model_output: str) -> str:
    """Basic preprocessing to extract expressions from model output."""

    def get_arg_string(s: str) -> str:
        """Get the substring between the leftmost "("
        and rightmost ")"."""
        first_opening_bracket_index = s.find("(")
        last_opening_bracket_index = s.rfind(")")
        keywords_str = s[first_opening_bracket_index + 1 : last_opening_bracket_index]
        return keywords_str

    input = model_output
    for pname, pattern in patterns.items():
        match_result = re.match(pattern, model_output)
        if match_result:
            match pname:
                case "included_line_number":
                    model_output = re.sub(r"^x\d{1,2}\s", "", model_output)
                    break
                case "assigned_to_line_var":
                    model_output = re.sub(pattern, "", model_output)
                    break
    # sometimes the "_" is missed in slot names
    model_output = re.sub(patterns["space_in_slot_name"], "_", model_output)
    if classify_turn_type(model_output) == ExpressionType.ASSIGNMENT:
        pattern = r""", (?=x\d{1,2}\.[a-z_]+\s*=\s*)"""
        replacement = "; "
        model_output = re.sub(pattern, replacement, model_output)
    if classify_turn_type(model_output) == ExpressionType.SYNTAX_ERR_CALL:
        keywords_str = get_arg_string(model_output)
        if keywords_str.count(")") == 1:
            new_keywords_str = keywords_str.replace(")", "")
        else:
            new_keywords_str = keywords_str
        model_output = model_output.replace(keywords_str, new_keywords_str)
    if model_output.startswith(NLG_CALL_TOOL):
        args_string = get_arg_string(model_output)
        args_string = args_string.replace(";", ",")
        pieces = []
        for piece in args_string.split(","):
            piece = re.sub(patterns["space_in_slot_name_nlg"], r"\1.\2_", piece)
            piece = piece.replace(" - ", "_")
            pieces.append(piece)
        new_args_string = ", ".join(pieces)
        model_output = f"{NLG_CALL_TOOL}({new_args_string})"
    if model_output != input:
        logger.debug(f"Input: {input}")
        logger.debug(f"Extracted expression: {model_output}")
    return model_output


def is_valid_arg(s: str, max_arg_len: int = 30) -> bool:
    """Positional args should always start with a variable name and have at most
    one dot.

    Parameters
    ----------
    max_arg_len
        The maximum argument length for an argument (SGD)
    """
    pattern = r"\bx\d+"
    return (
        re.match(pattern, s) is not None
        and s[-1] != "."
        and s.count(".") <= 1
        and len(s) < max_arg_len + 1
    )


def is_valid_assigment_target(s: str, max_target_len: int = 105) -> bool:
    """A string is a valid assignment target if it does not contain a variable name
    and does not exceed the maximum len of a slot value in the corpus."""
    return not contains_variable_name(s) and len(s) < max_target_len + 1


def sanitise_value(value: str, max_value_len: int = 20) -> str:
    """Truncate and ensure that valid values are properly quoted."""
    if value in {"true", "false"}:
        return value
    if contains_variable_name(value):
        if value.count(".") > 1:
            return ".".join(value.split(".")[:2])
        return maybe_truncate_on_repeating_pattern(value)
    value = truncate_value(value, max_value_len)

    match value[0]:
        case "'" as start:
            # close quotes
            if value[-1] != start:
                # use double quotes if there are
                # apostrophes
                if start in value[1:]:
                    return f'"{value[1:]}"'
                return f"{value}'"
            return value
        case '"' as start:
            # close quotes
            if value[-1] != start:
                return f'{value}"'
            return value
        case _:
            match value[-1]:
                case "'" | '"' as end_quote:
                    return f"{end_quote}{value}"
            for t in (float, int):
                try:
                    _ = t(value)
                    return value
                except ValueError:
                    continue
            else:
                if "'" in value:
                    return f'"{value}"'
                return f"'{value}'"
    return value


def truncate_value(value: str, max_value_len: int, max_rep_count: int = 2) -> str:
    """Truncate values using the following strategies:

    1. to `max_value_len` if the value is a single word
    2. to first `k` unique words if there are multiple words
    """
    value_pieces = value.split(" ")
    if len(value_pieces) == 1:
        value = value[:max_value_len]
    else:
        rep_count = 0
        filtered_pieces = []
        for v in value_pieces:
            if v not in filtered_pieces:
                v = v[:max_value_len]
                filtered_pieces.append(v)
            else:
                rep_count += 1
            if rep_count > max_rep_count:
                break
        value = " ".join(filtered_pieces).strip()
    return value


def sanitise_call(model_output: str) -> str:
    """Sanitise a call output by the model which was not terminated with <eos>."""

    # we never expect this pattern, so we just ignore this string
    if model_output.count("(") > 1:
        return f"{model_output[:model_output.find('(')]}()"

    function_name = get_function_name(model_output)

    # Extract arguments string
    pattern = r"\((.*?)\)"
    match = re.search(pattern, model_output)
    if match is None:
        model_output = f"{model_output})"
        match = re.search(pattern, model_output)
    arguments_str = match.group(1)
    keyword_arguments, kwarg_values = [], []
    positional_args = set()
    for piece in arguments_str.split(","):
        if "=" in piece:
            try:
                keyword, value = piece.split("=")
            except ValueError:
                # incorrect assignment syntax slot = value = value = value ...
                continue
            if is_valid_keyword(keyword):
                keyword, value = keyword.strip(), value.strip()
                if not value:
                    continue
                if keyword not in keyword_arguments:  # ignore kw repetitions
                    sanitised_val = sanitise_value(value)
                    if not sanitised_val:
                        continue
                    keyword_arguments.append(keyword.strip())
                    kwarg_values.append(sanitised_val.strip())
        else:
            if keyword_arguments:
                logger.warning(
                    f"Ignoring, positional argument after keyword argument: {piece}"
                )
                continue
            if piece and is_valid_arg(piece.strip()):
                positional_args.add(piece.strip())
    # Split the arguments string by commas and filter out the keyword arguments
    expr = Expression(
        tool=function_name.strip(),
        positional_args=list(positional_args),
        keyword_args=keyword_arguments,
        kwarg_values=kwarg_values,
    )
    match expr.tool:
        case "resume" | "suspend" | "confirm":
            if expr.positional_args and len(expr.positional_args) > 1:
                expr.positional_args = [sorted(expr.positional_args)[0]]

    expr = str(expr).strip()

    try:
        _ = ast.parse(expr)
    except SyntaxError:
        logger.error(f"Sanitised call output {expr} cannot be parsed.")
    return expr


def get_function_name(model_output: str) -> Optional[str]:
    function_name = re.findall(r"^(.*)\(", model_output)[0]
    return function_name


def sanitise_assignment(model_output: str, max_value_len: int = 30) -> str:
    """Sanitise assignment statements by removing repetitions and discarding
    unterminated statements."""
    logger.info(f"Sanitising: {model_output}")
    keyword_args = []
    kwarg_values = []
    for term in model_output.split(";"):
        try:
            assignee, target = term.split("=")
            assignee = assignee.strip()
            target = target.strip()
            if is_valid_arg(assignee):
                if not is_valid_assigment_target(target):
                    if contains_variable_name(target):
                        continue
                    target = sanitise_value(target, max_value_len)
                if assignee not in keyword_args:
                    keyword_args.append(assignee)
                    try:
                        parsed_val = ast.parse(target).body[0].value.value
                    except AttributeError:
                        logger.error(f"Could not parse target: {target}")
                        continue
                    except SyntaxError:
                        parsed_val = correct_value_quotes(target)
                        # checking that we actually fixed the error
                        _ = ast.parse(parsed_val)
                        kwarg_values.append(parsed_val)
                        continue
                    if isinstance(parsed_val, str) and parsed_val:
                        if "'" in parsed_val:
                            parsed_val = f'"{parsed_val}"'
                        else:
                            parsed_val = f"'{parsed_val}'"
                    else:
                        parsed_val = str(parsed_val)
                    kwarg_values.append(parsed_val)
        except ValueError:
            continue
    # we could not parse this, so we ignore this
    # output in the history
    if not keyword_args:
        return "parse_error()"
    expr_list = []
    for keyword, value in zip(keyword_args, kwarg_values):
        expr_list.append(
            Expression(
                tool="assign",
                positional_args=[],
                keyword_args=[keyword],
                kwarg_values=[value],
            )
        )
    expr = str(ExpressionList(expressions=expr_list))
    try:
        _ = ast.parse(str(expr))
    except SyntaxError:
        logger.error(f"Sanitised assignment {expr} cannot be parsed.")
    return expr


def escape_syntax_errors(model_output: str, err_msg: str) -> str:
    return sanitise_syntax_errors(classify_error(err_msg), model_output)


@dispatch_on_value
def sanitise_syntax_errors(error_type: SyntaxErrorType, model_output: str) -> str:
    logger.warning(f"Unhandled syntax error for output: {model_output}.")
    return model_output


def handle_apostrophes(string: str) -> str:
    """Ensure double quotes are applied to values to avoid
    parsing errors due to single quoted values that contain apostrophes."""
    # Replace single quotes with double quotes, except for apostrophes within words
    sanitized_string = re.sub(r"('(?!\w))", '"', string)
    sanitized_string = re.sub(r"(?<!\w)'", '"', sanitized_string)

    # Replace single quotes with double quotes for apostrophes within words
    sanitized_string = re.sub(r"'s", "'s", sanitized_string)
    return sanitized_string


def get_valid_expressions(model_output: str, split_by: str = ";") -> list[str]:
    """Collect any valid program statements in the model output."""
    expressions = []
    pattern = rf"{split_by}\s"
    seen_expressions = set()
    for expr in re.split(pattern, model_output):
        expr = expr.strip()
        try:
            _ = ast.parse(expr)
            if expr not in seen_expressions:
                expressions.append(expr)
                seen_expressions.add(expr)
        except SyntaxError:
            try:
                k, v = expr.split("=")
                try:
                    _ = ast.parse(v.strip())
                except SyntaxError:
                    expr = escape_quote_errors("", [expr.strip()]).strip()
                    try:
                        _ = ast.parse(expr)
                        expressions.append(expr)
                    except SyntaxError:
                        continue
            except ValueError:
                match = re.match(r'(\w+)\s(["\'].*["\'])', expr)
                if match:
                    kw = match.group(1)
                    val = sanitise_value(match.group(2)).strip()
                    try:
                        _ = ast.parse(kw)
                        _ = ast.parse(val)
                    except SyntaxError:
                        continue
                    expressions.append(f"{kw} = {val}")
                continue

            continue
    return expressions


def correct_value_quotes(misquoted: str) -> str:
    """Helper function that corrects the quotes of a given value."""
    double_quote = False
    start, end = misquoted[0], misquoted[-1]
    if "'" in misquoted[1:-1]:
        double_quote = True
        if '"' in (middle := misquoted[1:-1]):
            middle = middle.replace('"', "'")
            misquoted = f"{start}{middle}{end}"
    # if the misquoted string had both " and ' inside
    # the above would be replaced and we can return
    # early
    try:
        _ = ast.parse(misquoted)
        return misquoted
    except SyntaxError:
        pass
    match (start, end) in MISMATCHED_QUOTES:
        case True:
            val = misquoted.replace('"', "'")
            val = f'"{val}"'
            _ = ast.parse(val)
            return val
        case False:
            # value not quoted at all
            if all((start not in QUOTES, end not in QUOTES)):
                if double_quote:
                    return f'"{misquoted}"'
                return f"'{misquoted}'"
            # wrong quote use, swap for the other type
            if all((start in QUOTES, end in QUOTES, start == end)):
                other_quote = next(iter({q for q in QUOTES if q != start}))
                return f"{other_quote}{misquoted[1:-1]}{other_quote}"
            # missing start/end quote
            match start:
                case "'" | '"':
                    assert end not in QUOTES
                    if double_quote:
                        return f'"{misquoted[1:]}"'
                    if start in misquoted[1:]:
                        other_quote = next(iter({q for q in QUOTES if q != start}))
                        misquoted = misquoted[1:].replace(start, other_quote)
                        return f"{start}{misquoted}{start}"
                    return f"{misquoted}{start}"
                case _:
                    assert end in QUOTES
                    if double_quote:
                        return f'"{misquoted[:-1]}"'
                    return f"{end}{misquoted}"


def escape_quote_errors(tool_name: str, keywords: list[str]) -> str:
    positional_args, keyword_args, kwarg_values = [], [], []
    for kw in keywords:
        kw = kw.strip()
        try:
            kw, maybe_unparseable_val = kw.split("=")
            keyword_args.append(kw.strip())
        except ValueError:
            if tool_name == "select" and re.match(r"^x\d{1,2}$", kw):
                positional_args.append(kw)
            continue
        maybe_unparseable_val = maybe_unparseable_val.strip()
        try:
            _ = ast.parse(maybe_unparseable_val)
            kwarg_values.append(maybe_unparseable_val)
        except SyntaxError:
            val = correct_value_quotes(maybe_unparseable_val)
            # this should pass - if it does not, we missed
            # a corner case
            _ = ast.parse(val)
            kwarg_values.append(val)
    match tool_name:
        case "assign":
            if len(kwarg_values) >= 1:
                expressions = []
                for kw, val in zip(keyword_args, kwarg_values):
                    expressions.append(
                        Expression(
                            tool=tool_name,
                            positional_args=[],
                            keyword_args=[kw],
                            kwarg_values=[val],
                        )
                    )
                return str(ExpressionList(expressions=expressions))
            return str(
                Expression(
                    tool=PARSE_ERROR_TOOL,
                    positional_args=[],
                    keyword_args=[],
                    kwarg_values=[],
                )
            )
        case "select" if positional_args:
            positional_args = sorted(positional_args)[0]
        case "":
            assert len(keywords) == 1
            return f"{keyword_args[0]}={kwarg_values[0]}"
    return str(
        Expression(
            tool=tool_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            kwarg_values=kwarg_values,
        )
    )


def sanitise_quotes(turn_type: ExpressionType, model_output: str) -> str:
    """Ensure the values are properly quoted."""
    tool_name, keywords = get_keywords(model_output, turn_type)
    return escape_quote_errors(tool_name, keywords)


@sanitise_syntax_errors.register(SyntaxErrorType.UNTERMINATED_LITERAL)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.CALL | ExpressionType.ASSIGNMENT:
            if ExpressionType.ASSIGNMENT:
                model_output = re.sub(r",(?=\s*x\d{1,2}\.[a-z_]+)", ";", model_output)
            orig_model_output = model_output
            model_output = handle_apostrophes(model_output)
            try:
                _ = ast.parse(model_output)
            except SyntaxError:
                split_by = ";" if turn_type == ExpressionType.ASSIGNMENT else ","
                if turn_type == ExpressionType.ASSIGNMENT:
                    valid_expressions = get_valid_expressions(
                        orig_model_output, split_by
                    )
                    return "; ".join(valid_expressions)
                if turn_type == ExpressionType.CALL:
                    tool = model_output[: model_output.find("(")]
                    valid_expressions = get_valid_expressions(
                        model_output[model_output.find(tool) + len(tool) + 1 : -1],
                        split_by,
                    )
                    return format_as_call(tool, valid_expressions)
                else:
                    logger.error(
                        "Failed to appropriately handle unterminated literal error \n"
                        f"Original expression: {orig_model_output} \n"
                        f"Corrected expression: {model_output} \n"
                        f"Turn type: {turn_type}"
                    )
                    return PARSE_ERROR
        case ExpressionType.ASSIGNMENT_CALL | ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            valid_expressions = get_valid_expressions(model_output)
            if valid_expressions:
                return "; ".join(valid_expressions)
            return PARSE_ERROR
        case ExpressionType.SYNTAX_ERR_CALL:
            model_output = model_output.replace(";", ",")
            tool_name = model_output[: model_output.find("(")]
            variables = sorted(re.findall(r"x\d{1,2}", model_output))
            positional_args, keyword_args, kwarg_values = [], [], []
            match (tool_name, variables):
                case ("select", variables):
                    if variables:
                        positional_args.append(variables[-1])
                    if len(variables) == 2:
                        keyword_args.append("from_results")
                        kwarg_values.append(variables[0])
                    return str(
                        Expression(
                            tool=tool_name,
                            positional_args=positional_args,
                            keyword_args=keyword_args,
                            kwarg_values=kwarg_values,
                        )
                    )
                case _:
                    end = model_output.rfind(")")
                    try:
                        start_idx = model_output.rindex("(")
                        arg_string = model_output[start_idx + 1 : end]
                        valid_expr = [
                            e
                            for e in get_valid_expressions(arg_string, split_by=",")
                            if e
                        ]
                        if valid_expr:
                            try:
                                _ = ast.parse(tool_name)
                                return format_as_call(tool_name, valid_expr)
                            except SyntaxError:
                                logger.warning(
                                    f"Unparseable tool name: {tool_name}. "
                                    f"Model output was: {model_output}"
                                )
                                return PARSE_ERROR
                        raise ValueError
                    except ValueError:
                        return PARSE_ERROR
    return model_output


@sanitise_syntax_errors.register(SyntaxErrorType.DECIMAL_LITERAL)
def _(error_type: str, model_output: str) -> str:
    match (expr_type := classify_turn_type(model_output)):
        case ExpressionType.CALL | ExpressionType.ASSIGNMENT:
            return sanitise_quotes(expr_type, model_output)
        case ExpressionType.SYNTAX_ERR_CALL:
            return sanitise_syntax_errors(
                SyntaxErrorType.UNTERMINATED_LITERAL, model_output
            )
        case ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            valid_expr = get_valid_expressions(model_output, ";")
            if valid_expr:
                return "; ".join(valid_expr)
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.CLOSING_PARENTHESIS_EXTRA)
def _(error_type: str, model_output: str) -> str:
    idx = model_output.rfind(")")
    prefix = f"{model_output[:idx]}"
    try:
        _ = ast.parse(prefix)
        assert prefix.count("(") == prefix.count(")")
        return prefix
    except SyntaxError as e:
        return sanitise_syntax_errors(classify_error(e.msg), prefix)
    # return the prediction unchanged if the error wasn't simply
    # an extra parenthesis
    except AssertionError:
        return model_output


@sanitise_syntax_errors.register(SyntaxErrorType.CLOSING_PARENTHESIS_MISSING)
def _(error_type: str, model_output: str) -> str:
    try:
        new_model_output = f"{model_output})"
        _ = ast.parse(new_model_output)
        match turn_type := classify_turn_type(new_model_output):
            case ExpressionType.CALL:
                return f"{model_output})"
            case ExpressionType.SYNTAX_ERR_CALL:
                tool_name = model_output[: model_output.find("(")]
                args = model_output[model_output.rindex("(") :]
                return f"{tool_name}{args}"
            case ExpressionType.SYNTAX_ERR_ASSIGNMENT | ExpressionType.ASSIGNMENT_CALL:
                valid_expressions = get_valid_expressions(model_output, ";")
                if valid_expressions:
                    return "; ".join(valid_expressions)
                return PARSE_ERROR
            case _:
                logger.error(f"{turn_type} not handled for error {error_type}")
                return PARSE_ERROR
    except SyntaxError as e:
        return sanitise_syntax_errors(classify_error(e.msg), f"{model_output})")


@sanitise_syntax_errors.register(SyntaxErrorType.UNEXPECTED_ASSIGNMENT)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.ASSIGNMENT | ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            split_by = ";" if ";" in model_output else ","
            valid_expressions = get_valid_expressions(model_output, split_by)
            if valid_expressions:
                return "; ".join(valid_expressions)
            return PARSE_ERROR
        case _:
            logger.error(f"{turn_type} not handled while escaping {error_type}")
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.UNEXPECTED_POSITIONAL)
def _(error_type: str, model_output: str) -> str:
    tool_name, keywords_args = get_keywords(model_output, ExpressionType.CALL)
    keywords, values = [], []
    for k in keywords_args:
        try:
            kw, val = k.split("=")
            kw = kw.strip()
            val = sanitise_value(val.strip()).strip()
            try:
                _ = ast.parse(kw)
                _ = ast.parse(val)
            except SyntaxError:
                continue
            if not contains_variable_name(kw):
                keywords.append(kw)
                values.append(val)
        except ValueError:
            match = re.match(r'(\w+)\s(["\'].*["\'])', k.strip())
            if match:
                kw = match.group(1)
                val = sanitise_value(match.group(2)).strip()
                try:
                    _ = ast.parse(kw)
                    _ = ast.parse(val)
                except SyntaxError:
                    continue
                if not contains_variable_name(kw):
                    keywords.append(kw)
                    values.append(val)
            continue
    return str(Expression(tool=tool_name, keyword_args=keywords, kwarg_values=values))


@sanitise_syntax_errors.register(SyntaxErrorType.INVALID_SYNTAX)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.ASSIGNMENT_CALL:
            try:
                call_str = model_output[: model_output.find(";")]
                _ = ast.parse(f"{call_str})")
            except SyntaxError:
                tool_name = model_output[: model_output.find("(")]
                try:
                    _ = ast.parse(tool_name)
                except SyntaxError:
                    return PARSE_ERROR
                return f"{tool_name}()"
            return f"{call_str})"
        case ExpressionType.CALL:
            try:
                pattern = r"\(([^)]*);([^)]*)\)"
                normalised_sep_call = re.sub(
                    pattern,
                    lambda match: f"({match.group(1)}, {match.group(2)})",
                    model_output,
                )
                if normalised_sep_call != model_output:
                    _ = ast.parse(normalised_sep_call)
                    return normalised_sep_call
            except SyntaxError:
                pass
            tool_name, args = get_keywords(model_output, ExpressionType.CALL)
            try:
                _ = ast.parse(tool_name)
                new_tool_name = tool_name
            except SyntaxError:
                new_tool_name = tool_name.replace(" ", "_")
            match new_tool_name:
                case "say":
                    positional_args = []
                    for a in args:
                        try:
                            _ = ast.parse(a.strip())
                            positional_args.append(a.strip())
                        except SyntaxError:
                            continue
                    return str(
                        Expression(
                            tool=new_tool_name,
                            positional_args=positional_args,
                            keyword_args=[],
                            kwarg_values=[],
                        )
                    )
                case _:
                    arg_string = model_output[
                        model_output.find(new_tool_name) + len(new_tool_name) + 1 : -1
                    ]
                    valid_expressions = get_valid_expressions(arg_string, ",")
                    return format_as_call(new_tool_name, valid_expressions)
        case ExpressionType.ASSIGNMENT:
            try:
                model_output = sanitise_syntax_errors(
                    SyntaxErrorType.UNTERMINATED_LITERAL, model_output
                )
                _ = ast.parse(model_output)
                return model_output
            except SyntaxError:
                logger.warning(
                    f"Could not escape errors in assignment for prediction {model_output}"
                )
                return PARSE_ERROR
        case ExpressionType.SYNTAX_ERR_CALL | ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            if turn_type == ExpressionType.SYNTAX_ERR_CALL:
                tool_name, args = get_keywords(model_output, ExpressionType.CALL)
                try:
                    _ = ast.parse(tool_name)
                except SyntaxError:
                    # "x0.return(x0) ..."
                    if len(tool_name.split(".")) > 1 and "return" in tool_name:
                        logger.warning(
                            f"Could not escape errors in assignment for prediction {model_output}"
                        )
                        return PARSE_ERROR
                    new_tool_name = tool_name.replace(" ", "_")
                    model_output = model_output.replace(tool_name, new_tool_name)
            return sanitise_syntax_errors(
                SyntaxErrorType.UNTERMINATED_LITERAL, model_output
            )
            # end = model_output.rfind(")")
            # tool_name = model_output[: model_output.find("(")]
            # if tool_name == model_output:
            #     return PARSE_ERROR
            # try:
            #     start_idx = model_output.rindex("(")
            #     arg_string = model_output[start_idx + 1: end]
            #     valid_expr = get_valid_expressions(arg_string, split_by=",")
            #     return format_as_call(tool_name, valid_expr)
            # except ValueError:
            #     return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.NO_ASSIGNMENT_ALLOWED)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.CALL | ExpressionType.SYNTAX_ERR_CALL:
            tool = model_output[: model_output.find("(")]
            if turn_type == ExpressionType.CALL:
                valid_expressions = get_valid_expressions(
                    model_output[model_output.find(tool) + len(tool) + 1 : -1],
                    ",",
                )
            else:
                try:
                    arg_start = model_output.rindex("(")
                    arg_end = model_output.find(")")
                    valid_expressions = get_valid_expressions(
                        model_output[arg_start + 1 : arg_end], split_by=","
                    )
                except IndexError:
                    return f"{tool}()"
            try:
                call = format_as_call(tool, valid_expressions)
                _ = ast.parse(call)
                return call
            except SyntaxError:
                return f"{tool}()"
        case ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            valid_expr = get_valid_expressions(model_output, split_by=";")
            if valid_expr:
                return "; ".join(valid_expr)
            return PARSE_ERROR
        case _:
            logger.error(f"{turn_type} not handled while escaping {error_type}")
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.LEADING_ZEROS_IN_DECIMAL)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.ASSIGNMENT:
            tool, keywords = get_keywords(model_output, ExpressionType.ASSIGNMENT)
            return escape_quote_errors(tool, keywords)
        case _:
            logger.error(f"{turn_type} not handled while escaping {error_type}")
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.NO_EXPRESSION_ASSIGN)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.SYNTAX_ERR_ASSIGNMENT | ExpressionType.ASSIGNMENT:
            valid_expr = get_valid_expressions(model_output, split_by=";")
            if valid_expr:
                return "; ".join(valid_expr)
            return PARSE_ERROR
        case _:
            logger.error(f"{turn_type} not handled while escaping {error_type}")
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.NO_ASSIGN_EXPRESSION)
def _(error_type: str, model_output: str) -> str:
    match turn_type := classify_turn_type(model_output):
        case ExpressionType.SYNTAX_ERR_ASSIGNMENT:
            valid_expr = get_valid_expressions(model_output, split_by=";")
            if valid_expr:
                return "; ".join(valid_expr)
            return PARSE_ERROR
        case _:
            logger.error(f"{turn_type} not handled while escaping {error_type}")
            return PARSE_ERROR


@sanitise_syntax_errors.register(SyntaxErrorType.NO_ASSIGN_FCN_CALL)
def _(error_type: str, model_output: str) -> str:
    valid_expr = []
    for expr in re.split(r";\s", model_output):
        try:
            _ = ast.parse(expr)
            valid_expr.append(expr)
        except SyntaxError:
            expr, _ = expr.rsplit("=", 1)
            expr = expr.strip()
            try:
                _ = ast.parse(expr)
                valid_expr.append(expr)
            except SyntaxError:
                continue
    return "; ".join(valid_expr)


def handle_error_escape_failure(model_output: str, escaped_output: str) -> str:
    """Simple post-processing for the case when the parser failed
    to escape the syntanx error."""
    match classify_turn_type(escaped_output):
        case ExpressionType.SYNTAX_ERR_CALL:
            open_bracket_idx = escaped_output.find("(")
            tool = escaped_output[:open_bracket_idx]
            args = escaped_output[open_bracket_idx + 1 : escaped_output.find(")")]
            args = ", ".join(get_valid_expressions(args, split_by=","))
            call = f"{tool}({args})"
            try:
                _ = ast.parse(call)
                return call
            except SyntaxError:
                try:
                    _ = ast.parse(tool)
                    return f"{tool}()"
                except SyntaxError:
                    return PARSE_ERROR
        case ExpressionType.ASSIGNMENT:
            pattern = (
                r'\bx\d{1,2}\.[a-z_]+=\s*(?:"(?:\\"|[^"])*"|\'(?:\\\'|[^\'])*\'|\S*)'
            )
            pairs = re.findall(pattern, escaped_output)
            if pairs:
                return "; ".join(pairs)
            return PARSE_ERROR
        case ExpressionType.CALL:
            return f"{escaped_output[:escaped_output.find('(')]}()"
        case _:
            logger.error(
                f"Failed to escape parser error for {escaped_output}.\n"
                f"Original model output: {model_output}"
            )
            return PARSE_ERROR
