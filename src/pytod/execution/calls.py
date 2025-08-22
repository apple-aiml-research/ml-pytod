#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Base class for all expression executors."""
import logging
from typing import Any

from pytod.execution.call_utils import _index
from pytod.execution.policy_utils import ActionResult
from pytod.execution.pytod_agent_state import PyTODAgentState
from pytod.pytod_types.aliases import VariableName
from pytod.simulation.api_driver import TransactionResult
from pytod.simulation.command import Command
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.search_command import SearchCommand

logger = logging.getLogger(__name__)


class Call:
    def __init__(self, var_name: VariableName):
        # see subclass docs for semantics
        self.var_name = var_name

    def _run(self, state: PyTODAgentState) -> ActionResult:
        raise NotImplementedError

    @property
    def index(self) -> int:
        return _index(self.var_name)

    def run(self, state: PyTODAgentState) -> ActionResult:
        return self._run(state)

    def _perform_or_recommend(
        self, state: PyTODAgentState, command: Command
    ) -> ActionResult:
        """Make recommendations to the agent to request `command`
        positional arguments if these have not been specified already.
        Otherwise, call the databases or API and communicate the
        results the user or make further appropriate recommendations."""
        assert command is not None
        recommended_actions = command.recommend_action()
        if recommended_actions is not None:
            state.current_recommendations[command.service][
                command.name
            ] = recommended_actions
            return ActionResult(
                recommended_action=recommended_actions, index=self.index
            )
        else:
            if isinstance(command, SearchCommand):
                state.current_recommendations[command.service][command.name] = None
                database = state.databases.get(command.service, {}).get(
                    command.name, None
                )
                result = ActionResult(
                    result=command.perform(database), index=self.index
                )
            else:
                assert isinstance(command, ConfirmedCommand)
                endpoint = state.apis.get(command.service, {}).get(command.name, None)
                perform_result = command.perform(endpoint)
                # if the transaction succeeds, a `perform` statement will be inserted
                # into the transcript to indicate successful execution
                # if the transaction fails, then the agent will request the user what
                # to do next
                if isinstance(perform_result, TransactionResult):
                    result = ActionResult(result=perform_result, index=self.index)
                # the application recommends some actions the user could take if
                # the transaction cannot be executed
                else:
                    assert isinstance(perform_result, list)
                    result = ActionResult(
                        recommended_action=perform_result, index=self.index
                    )
            return result


class Value(Call):
    def __init__(self, var_name: str, value: Any):
        super().__init__(var_name)
        self.value = value

    def _run(self, state: PyTODAgentState) -> ActionResult:
        return ActionResult(index=self.index, result=self.value)
