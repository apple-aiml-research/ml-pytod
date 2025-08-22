#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from copy import deepcopy
from typing import Literal, NamedTuple

from pytod.execution.policy_utils import ActionResult
from pytod.execution.pytod_agent import AgentArgs, PyTODAgent
from pytod.execution.pytod_agent_state import ExecutionContext, TurnContext
from pytod.interpreter.metadata import (
    ENTITY_SELECTION_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
    FOLLOWUP_INTENT_TOOL,
    NLG_CALL_TOOL,
    RESTART_TASK_TOOL,
    SPECIAL_TOOLS,
)
from pytod.parser.expressions_utils import maks_conversation_end
from pytod.pytod_types.aliases import DialogueID, ServiceName
from pytod.pytod_types.pytod import AnyTurn, PyTODConversation, SystemTurn
from pytod.utils import default_to_regular, nested_defaultdict

UserTurnIdx = int
StateDict = dict[
    ServiceName,
    dict[Literal["slot_values", "active_intent", "requested_slots"], str | list[str]],
]
DialogueState = dict[Literal["utterance", "state"], str | StateDict]



class TurnEvalInfo(NamedTuple):
    query: str
    query_idx: int
    turns: list[SystemTurn]
    context: ExecutionContext


class OfflineSessionExecutor:
    def __init__(
        self,
        transcript: list[AnyTurn],
        agent: PyTODAgent,
        exclude_signals: bool = True,
        service_info: dict[UserTurnIdx, list[ServiceName]] | None = None,
    ):
        """
        Parameters
        ----------
        transcript
            Transcript of the conversation to execute.
        agent:
        exclude_signals
            If `True`, signal turns are skipped during execution.
        service_info
            If provided, it is used to inform the agent whether the
            turn is multi-domain or not, to allow the agent to take
            correct action.
        """
        self._transcript = transcript
        self._agent = agent
        self._exclude_signals = exclude_signals
        self.dialogue_id = agent.dialogue_id
        self._state = nested_defaultdict(dict, depth=2)
        self._service_info = service_info

    @property
    def session_state(self) -> dict[DialogueID, dict[UserTurnIdx, DialogueState]]:
        return default_to_regular(self._state)

    def _update_state(self, expr_info: TurnEvalInfo):
        """Read the state update from the PyTOD agent."""
        dial_id = self.dialogue_id
        user_turn_idx = expr_info.query_idx
        turn_dial_state = {
            "utterance": expr_info.query,
            "state": dict(self._agent.state.dialogue_state),
        }
        self._state[dial_id][user_turn_idx] = deepcopy(turn_dial_state)

    @staticmethod
    def _infer_execution_context(turns: list[AnyTurn]) -> ExecutionContext:
        """Infer whether we are predicting at the service boundary based on
        whether any statements follow a `select` call or not."""

        context = TurnContext.SINGLE_DOMAIN
        conversation_end = any(maks_conversation_end(t.expression) for t in turns)
        exec_context = ExecutionContext(
            conversation_end=conversation_end, turn_context=context
        )
        for i, t in enumerate(turns):
            if t.get_tool_name() == ENTITY_SELECTION_TOOL:
                if i == len(turns) - 1:
                    return exec_context
                else:
                    next_tool = turns[i + 1].get_tool_name()
                    try:
                        assert next_tool not in SPECIAL_TOOLS
                    except AssertionError:
                        try:
                            assert next_tool == NLG_CALL_TOOL
                            assert conversation_end
                        except AssertionError:
                            if next_tool == RESTART_TASK_TOOL:
                                exec_context.turn_context = TurnContext.MULTI_DOMAIN
                                return exec_context
                            assert next_tool == FOLLOWUP_INTENT_TOOL
                        return exec_context
                    exec_context.turn_context = TurnContext.MULTI_DOMAIN
                    return exec_context

        match [t.get_tool_name() for t in turns]:
            case [tool, other_tool] if (tool, other_tool) == (
                FOLLOWUP_INTENT_DECLINE_TOOL,
                NLG_CALL_TOOL,
            ):
                assert conversation_end
                return exec_context
            case [tool] if tool == FOLLOWUP_INTENT_DECLINE_TOOL:
                return exec_context
            case [
                tool,
                other_tool,
                *_,
            ] if tool == FOLLOWUP_INTENT_DECLINE_TOOL and other_tool not in SPECIAL_TOOLS:
                return exec_context
            case [
                tool,
                other_tool,
                *_,
            ] if tool == FOLLOWUP_INTENT_DECLINE_TOOL and other_tool == RESTART_TASK_TOOL:
                exec_context.turn_context = TurnContext.MULTI_DOMAIN
                return exec_context

    def _program_iterator(self, exclude_signals: bool = True) -> TurnEvalInfo:
        """Iterate through the statements generated by the action parser from a user turn."""
        user_query, user_query_idx = None, -2
        turns = []
        for i, t in enumerate(self._transcript):
            match t.author:
                case "User":
                    user_query_idx += 2
                    user_query = t.query
                case "System":
                    turns.append(t)
                case "Signal":
                    if exclude_signals:
                        continue
                    turns.append(t)
                case "Response":
                    if self._service_info is not None:
                        conversation_end = any(
                            maks_conversation_end(t.expression) for t in turns
                        )
                        turn_context = (
                            TurnContext.SINGLE_DOMAIN
                            if len(self._service_info[user_query_idx]) == 1
                            else TurnContext.MULTI_DOMAIN
                        )
                        context = ExecutionContext(
                            conversation_end=conversation_end, turn_context=turn_context
                        )
                    else:
                        context = self._infer_execution_context(turns)
                    yield TurnEvalInfo(
                        turns=turns,
                        query=user_query,
                        query_idx=user_query_idx,
                        context=context,
                    )
                    turns = []

    def _set_execution_context(self, context: ExecutionContext):
        """Inform the execution whether a multiple-intent
        turn is currently executed. This is necessary for
        generating the specific actions that may happen upon
        domain change."""
        self._agent.context = context

    def execute_program(self) -> list[list[ActionResult]]:
        """Execute a complete PyTOD program, offline."""

        actions = []
        for expr_info in self._program_iterator(exclude_signals=self._exclude_signals):
            self._set_execution_context(expr_info.context)
            actions.append(self._agent.execute_instructions(expr_info.turns))
            self._update_state(expr_info)
        return actions


def execute_and_evaluate(
    dialogue: PyTODConversation,
    agent_args: AgentArgs,
    service_info: dict[int, list[ServiceName]],
):
    dial_id = dialogue.id.replace(f"{agent_args.split}_", "")

    agent = PyTODAgent(
        agent_args.split,
        dial_id,
        agent_args.schema,
        lenient=agent_args.lenient,
        debug=agent_args.debug,
    )
    executor = OfflineSessionExecutor(dialogue.turns, agent, service_info=service_info)
    executor.execute_program()
    return executor.session_state
