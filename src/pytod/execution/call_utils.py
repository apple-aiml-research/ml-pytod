#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Utilities for PyTOD program execution."""
import logging
from typing import Optional

from pytod.execution.policy_utils import ActionResult, Inform, RecommendedAction
from pytod.execution.pytod_agent_state import PyTODAgentState
from pytod.pytod_types.aliases import VariableName
from pytod.simulation.command import Command

logger = logging.getLogger(__name__)


def _index(var_name: str) -> int:
    """Return the index of a variable."""
    return int(var_name[1:])


def _get_assigned_vars(state: PyTODAgentState) -> list[VariableName]:
    """Get a sorted list of the assigned variables in the program."""
    assignments = sorted(list(state.assignments.keys()), key=lambda x: _index(x))
    return assignments


def _get_last_command(state: PyTODAgentState) -> Optional[Command]:
    """Get the last assigned command in the program."""

    assignments = _get_assigned_vars(state)
    for v in reversed(assignments):
        try:
            if isinstance(obj := state.get_assignment(v), Command):
                return obj
        except ValueError:
            logger.warning(f"Variable {v} is not assigned.")
            for key in state.var_containers:
                container = getattr(state, key)
                if v in container:
                    logger.info(
                        f"Variable found in container {key}. Value: {container[v]}"
                    )
            continue
    return


def _recommend_prompting_user(
    index: int, state: PyTODAgentState, issuer: str
) -> ActionResult:
    """Helper function for recommending that the agent prompts the user for
    next task (annotated with REQ_MORE dialogue act in original annotation)."""
    command = _get_last_command(state)
    user_hint = command.hint_generator.prompt_user_hint()
    action = RecommendedAction(dialog=f"{user_hint}")
    return ActionResult(
        index=index,
        result=Inform(dialog="Success"),
        recommended_action=[action],
        issuing_cmd=issuer,
    )
