#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import random
import re

from pytod.parser.expressions import Expression, ExpressionList


class VariableMapper:
    def __init__(self, current_ndx: int | None = None) -> None:
        self._current_ndx = random.randint(0, 9) if current_ndx is None else current_ndx
        self._start_ndx = self._current_ndx
        self._mapping: dict[str, str] = {}
        self._reverse_mapping: dict[str, str] = {}

    def add_mapping(self, src: str):
        tgt = f"x{self._current_ndx}"
        if re.match(r"x([0-9])+", src) is None:
            raise ValueError(f"Source of mapping must of form `x<digit>`, got {src}")
        if src in self._mapping:
            raise ValueError(
                f"Attempted to add the same source variable ({src}) more than "
                "once. This is not allowed"
            )

        self._mapping[src] = tgt
        self._reverse_mapping[tgt] = src
        self._current_ndx += 1
        return

    def remove_mapping(self, src: str):
        reverse_key = self._mapping.pop(src)
        if reverse_key in self._reverse_mapping:
            self._reverse_mapping.pop(reverse_key)
        self._current_ndx -= 1

    @property
    def current_ndx(self) -> int:
        return self._current_ndx

    @property
    def start_ndx(self) -> int:
        return self._start_ndx

    @property
    def mapping(self) -> dict[str, str]:
        return {k[:]: v[:] for k, v in self._mapping.items()}

    @property
    def reverse_mapping(self) -> dict[str, str]:
        return {k[:]: v[:] for k, v in self._reverse_mapping.items()}

    def _apply(self, original: str, reverse: bool = False) -> str:
        mapping = self._reverse_mapping if reverse else self._mapping
        if (m := re.match(r"(x[0-9]+)(.*)", original)) is not None:
            if m[1] not in mapping:
                raise RuntimeError(
                    "Attempt at mapping a variable for which there is no mapping ({slot_name})"
                )
            return f"{mapping[m[1]]}{m[2]}"
        return original

    def apply(
        self, expression: ExpressionList, reverse: bool = False
    ) -> ExpressionList:
        mapped_expressions: list[Expression] = []
        for exp in expression.expressions:
            mapped_expressions.append(
                Expression(
                    tool=exp.tool,
                    positional_args=[
                        self._apply(a, reverse=reverse) for a in exp.positional_args
                    ],
                    keyword_args=[
                        self._apply(s, reverse=reverse) for s in exp.keyword_args
                    ],
                    kwarg_values=[
                        self._apply(v, reverse=reverse) for v in exp.kwarg_values
                    ],
                    exec_info=exp.exec_info,
                )
            )
        return ExpressionList(expressions=mapped_expressions)

    def apply_to_dialog_text(self, dialog_str: str) -> str:
        """Reindex the text inside a dialog according to the mapping."""
        for match in re.findall(r"x[0-9]+", dialog_str):
            dialog_str = dialog_str.replace(match, self._apply(match))
        return dialog_str
