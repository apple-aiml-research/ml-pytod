#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from omegaconf import DictConfig

from pytod.prompting.pytod_text2text_formatters_utils import VariableMapper
from pytod.pytod_types.pytod import SystemTurn

FormattedCompletion = str


class TargetFormatter:
    def __init__(self, config: DictConfig | None = None):
        self._config = config
        self.expression_list_sep = ""
        if config is not None:
            self.expression_list_sep = config.program_newline_sep

    def __call__(
        self, target_turns: list[SystemTurn], var_mapping: VariableMapper
    ) -> str:
        assert target_turns, "Something went wrong, no target!"
        if len(target_turns) > 1 and not self.expression_list_sep:
            raise ValueError("Expected line separator for target outputs")
        target_expressions = []
        for turn in target_turns:
            target_str = str(var_mapping.apply(turn.expression))
            var_mapping.add_mapping(f"x{turn.index}")
            target_expressions.append(target_str)
        return f" {self.expression_list_sep} ".join(target_expressions).lower().strip()
