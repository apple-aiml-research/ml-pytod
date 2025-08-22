#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Runtime executors for expressions which are not function calls."""
import logging

from pytod.execution.calls import Call
from pytod.execution.policy_utils import ActionResult, Inform
from pytod.execution.pytod_agent_state import PyTODAgentState
from pytod.pytod_types.aliases import SlotName, VariableName
from pytod.simulation.command import Command

logger = logging.getLogger(__name__)


class AssignmentCall(Call):
    def __init__(
        self, var_name: VariableName, attribute_name: list[SlotName], value: list[str]
    ):
        super().__init__(var_name)
        # var_name inherited form `Call`
        # is the name of the variable
        # to which the assignment is made, not
        # the line number of the assignment
        # instruction in the generated program

        # the slot name ...
        self.attribute_name = attribute_name
        # ... to which values are assigned
        self.value = value

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Set command properties that the user updated
        and recommend actions the agent could take next."""
        command = None
        for att, val in zip(self.attribute_name, self.value):
            try:
                command = state.get_assignment(self.var_name)
            except ValueError:
                self._handle_failure(state, att, val)
                return ActionResult(
                    index=self.index, issuing_cmd="assign", result=Inform("Failed")
                )
            if isinstance(command, Command):
                setattr(command, att, val)
            else:
                self._handle_failure(state, att, val)
        assert command is not None
        if isinstance(command, Command):
            state.track_state(command)
            # nb: we do not auto-insert len() and slice()
            # instructions as these are handled inside the
            # dialogue session
            return self._perform_or_recommend(state, command)
        return ActionResult(
            index=self.index, issuing_cmd="assign", result=Inform("Failed")
        )

    def _handle_failure(self, state: PyTODAgentState, attr: str, value: str):
        """Set attributes to the last command if an assignment on immutable
        object."""
        try:
            command = state.get_assignment(self.var_name)
            obj = command.__class__.__name__
            logger.warning(
                f"Cannot set attribute {attr} to {obj}, {obj} is immutable. "
                f"Maybe a function call should have been generated?"
            )
        except ValueError:
            logger.warning("Attempted to set an attribute to an unassigned variable. ")
        setattr(state.current_task, attr, value)
        state.track_state(state.current_task)
