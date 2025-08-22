#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import random
import re
from typing import Optional, Union

import rich
from omegaconf import DictConfig
from pydantic import BaseModel
from rich.columns import Columns
from rich.syntax import Syntax
from rich.text import Text

from pytod.parser.expressions import ExpressionList
from pytod.pytod_types.pytod import (
    AnyTurn,
    HintTurn,
    PyTODConversation,
    ResponseTurn,
    SignalTurn,
    SystemTurn,
    UserTurn,
)


def apply_mapping(expr: str | ExpressionList, mapping: dict[str, str]) -> str:
    result = str(expr)
    values = []
    for orig, new in mapping.items():
        values.append(new)
        result = re.sub(rf"\b{orig}\b", f"@values_{len(values)-1}@", result)
    for i, value in enumerate(values):
        result = re.sub(rf"@values_{i}@", value, result)
    return result


def apply_to_dialog_text(mapping: dict[str, str], dialog_str: str) -> str:
    """Reindex the text inside a dialog according to the mapping."""

    def apply(original: str) -> str:
        if (m := re.match(r"(x[0-9]+)(.*)", original)) is not None:
            if m[1] not in mapping:
                raise RuntimeError(
                    "Attempt at mapping a variable for which there is no mapping ({slot_name})"
                )
            return f"{mapping[m[1]]}{m[2]}"
        return original

    for match in re.findall(r"x[0-9]+", dialog_str):
        dialog_str = dialog_str.replace(match, apply(match))
    return dialog_str


class FormatterOutput(BaseModel):
    # map from turn index to
    # variable index
    ndx_mapping: dict[str, str]
    result: list[str]
    # split indices from expression for display with `rich.print`
    result_for_rich_display: list[tuple[Union[int, str], str]]


class PyTODDisplayFormatter:
    def __call__(
        self,
        history: list[AnyTurn],
        user_name: str = "user",
        agent_name: str = "agent",
        randomise: bool = False,
    ) -> FormatterOutput:
        result: list[str] = []
        result_for_display: list[tuple[Union[int, str], str]] = []
        ndx = 1
        if randomise:
            ndx = random.randint(0, 9)
        ndx_mapping = {}
        for i, turn in enumerate(history):
            match turn.author:
                case "User":
                    assert isinstance(turn, UserTurn)
                    query = turn.query
                    text = f"{user_name}: {query}"
                    result.append(text)
                    text_for_display = text.replace(f"{user_name}:", "")
                    result_for_display.append((user_name, text_for_display))
                case "System":
                    assert isinstance(turn, SystemTurn)
                    ndx_mapping[f"x{turn.index}"] = f"x{ndx}"
                    expression = apply_mapping(turn.expression, ndx_mapping)
                    result.append(f"{ndx} {expression}")
                    result_for_display.append((ndx, expression))
                    ndx += 1
                case "Response":
                    assert isinstance(turn, ResponseTurn)
                    msg = turn.text.replace('"', "'")
                    text = f"{agent_name}: {msg}"
                    result.append(text)
                    text_for_display = text.replace(f"{agent_name}:", "")
                    result_for_display.append((agent_name, text_for_display))
                case "Signal":
                    assert isinstance(turn, SignalTurn)
                    msg = turn.dialog
                    assert f"x{turn.index}" not in ndx_mapping
                    ndx_mapping[f"x{turn.index}"] = f"x{ndx}"
                    signal_body = f"Signal: {msg}"
                    result.append(f"{ndx} {signal_body}")
                    result_for_display.append((ndx, signal_body))
                    ndx += 1
                case "Hint":
                    assert isinstance(turn, HintTurn)
                    dialog = apply_to_dialog_text(ndx_mapping, turn.dialog)
                    assert f"x{turn.index}" not in ndx_mapping
                    ndx_mapping[f"x{turn.index}"] = f"x{ndx}"
                    hint_body = f"Hint('{dialog}')"
                    result.append(f"{ndx} {hint_body}")
                    result_for_display.append((ndx, hint_body))
                    ndx += 1
                case _:
                    raise ValueError(f'Unknown author "{turn.author}"')
        result.append(f"{ndx}")
        return FormatterOutput.model_validate(
            {
                "result": result,
                "ndx_mapping": ndx_mapping,
                "result_for_rich_display": result_for_display,
            },
        )


class PyTODTextFormatter:
    """Display PyTOD conversations in human-readable format."""

    def __init__(
        self,
        system_name: str = "agent",
        user_name: str = "user",
        rich_config: Optional[DictConfig] = None,
        show_programs: bool = True,
    ):
        self.system_name = system_name
        self.user_name = user_name
        self._formatter = PyTODDisplayFormatter()
        self._rich_config = rich_config
        self._syntax_highlights = rich_config.syntax_highlight
        self._show_programs = show_programs

    def conversation_to_text(self, conversation: PyTODConversation) -> str:
        formatter_out: FormatterOutput = self._formatter(
            conversation.turns, user_name=self.user_name, agent_name=self.system_name
        )
        return "\n".join(formatter_out.result)

    def rich_display(self, conversation: PyTODConversation):
        formatter_out: FormatterOutput = self._formatter(
            conversation.turns,
            user_name=self.user_name,
            agent_name=self.system_name,
            randomise=False,
        )
        for maybe_index, expr in formatter_out.result_for_rich_display:
            if (
                maybe_index not in [self.user_name, self.system_name]
                and self._show_programs
            ):
                maybe_index = Text(str(maybe_index))
                maybe_index.stylize(
                    style=self._syntax_highlights.get("varindex", "bold"),
                    start=0,
                    end=len(maybe_index),
                )
                syntax = (
                    Syntax(
                        expr,
                        "python",
                        code_width=self._rich_config.get("code_with", 80),
                        word_wrap=True,
                    ),
                )
                if "Hint" in expr:
                    syntax[0].word_wrap = self._rich_config.wrap.hints
                rich.print(Columns([maybe_index, syntax[0]]))
            else:
                author = Text(f"{maybe_index}:")
                if maybe_index == "user":
                    author.stylize(style=self._rich_config.get("user_style", "bold"))
                else:
                    author.stylize(
                        style=self._rich_config.get("agent_style", "bold"),
                        end=len(maybe_index),
                    )
                obj = author + expr
                rich.print(
                    Columns(
                        [obj], width=self._rich_config.get("width", 79), align="left"
                    ),
                )
