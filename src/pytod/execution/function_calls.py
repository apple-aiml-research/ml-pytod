#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Runtime executors for all function calls."""
import logging
from typing import Literal, Optional

from pytod.command import is_arg
from pytod.execution.auto_commands import Signal
from pytod.execution.call_utils import (
    _get_assigned_vars,
    _get_last_command,
    _recommend_prompting_user,
)
from pytod.execution.calls import Call
from pytod.execution.policy_utils import ActionResult, Inform, RecommendedAction
from pytod.execution.pytod_agent_state import (
    ExecutionContext,
    PyTODAgentState,
    TurnContext,
)
from pytod.pytod_types.aliases import VariableName
from pytod.simulation._command_utils import APICallStatus
from pytod.simulation.api_driver import TransactionResult
from pytod.simulation.command import Command
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.entities import Entity
from pytod.simulation.search_command import SearchCommand

logger = logging.getLogger(__name__)


class FunctionCallError(Exception):
    pass


class InvalidVariableReferenceError(Exception):
    """Raised when a tool is invoked with a variable
    that exists in the transcript but is not assigned."""

    pass


class CommandCall(Call):
    """The PyTOD agent executes a command call to receive recommendations about the next
    actions to take or interact with external knowledge (eg databases or API endpoints).
    """

    def __init__(
        self, var_name: VariableName, command: Command, assigned: bool = False
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command
        self.assigned = assigned

    def _run(self, state: PyTODAgentState) -> ActionResult:
        if not self.assigned:
            state.assignments[self.var_name] = self.command
        state.track_state(self.command)
        return self._perform_or_recommend(state, self.command)


class PerformCall(Call):
    """A perform call indicates the successful execution of a transaction.
    Running this call binds a message indicating task success to the variable
    assigned to the perform statement."""

    def __init__(self, var_name: VariableName, command: ConfirmedCommand):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Bind a message indicating task success to the variable assigned
        to the perform call."""
        state.assignments[self.var_name] = Signal(
            dialog=f"Call to {self.command.get_full_command_name(snake_cased=True)} succeeded."
        )
        return ActionResult(
            index=self.index, issuing_cmd="perform", result=Inform("Success")
        )


class ConfirmCall(Call):
    def __init__(self, var_name: VariableName, command: ConfirmedCommand):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _get_action_result(self, state: PyTODAgentState) -> ActionResult:
        """Perform the given command, optionally returning recommendations."""
        action_result = self._perform_or_recommend(state, self.command)
        match action_result.result:
            # the application recommended alternative transaction params
            case None:
                assert action_result.recommended_action is not None
                action_result.issuing_cmd = "confirm"
                action_result.result = Inform("Failed")
                return action_result
            case TransactionResult(status=status):
                match status:
                    case APICallStatus.SUCCESS:
                        # nb we do not return a perform() statement here,
                        # this is already done in the dialogue session
                        return ActionResult(
                            index=self.index,
                            result=Inform("Success"),
                            issuing_cmd="confirm",
                        )
                    # transaction did not succeed & there is no alternative,
                    # ask the user if they need more help
                    case APICallStatus.FAILURE:
                        user_hint = self.command.hint_generator.prompt_user_hint()
                        action = RecommendedAction(dialog=f"{user_hint}")
                        return ActionResult(
                            index=self.index,
                            result=Inform("Failed"),
                            recommended_action=[action],
                            issuing_cmd="confirm",
                        )
                    case _:
                        raise RuntimeError(
                            f"Unexpected value for transaction result status: {status}"
                        )
            case _:
                raise RuntimeError(
                    f"Unexpected value for transaction result: {action_result.result}"
                )

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Mark a command as confirmed and recommend specific
        actions depending on transaction status."""

        self.command.__confirm__()
        state.confirmations[self.var_name] = self.command
        state.track_state(self.command)
        return self._get_action_result(state)


class SelectCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: SearchCommand,
        execution_context: ExecutionContext,
        entity: Optional[Entity] = None,
        select_kwargs: Optional[dict[Literal["key", "value"], str]] = None,
        assigned: bool = False,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command: SearchCommand = command
        self.select_kwargs = select_kwargs
        self.assigned = assigned
        self.entity = entity
        self._context = execution_context

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Call the `__select__` special method on the command
        and track the state of the dialogue. An action is
        recommended only if the user does not initiate the next
        task."""

        if not self.assigned:
            state.assignments[self.var_name] = self.entity
        if self.select_kwargs is not None:
            key = next(iter(self.select_kwargs))
            value = self.select_kwargs[key]
            self.command.__select__(key=key, value=value)
            state.assignments[self.var_name] = self.command.selected_entity
        else:
            self.command.__select__(entity=self.entity)
        action, result = None, Inform(dialog="Success")
        # following selection the agent either states a follow-up intent
        # or asks the user what to do next if the user doesn't immediately
        # state the next task or ends the conversation
        should_recommend = (
            self._context.turn_context == TurnContext.SINGLE_DOMAIN
            and not self._context.conversation_end
        )
        state.track_state(self.command)
        if should_recommend:
            # NB: suggest() instructions are auto-inserted
            # during the call to session.update() for now, for simplicity.
            # we only recommend to prompt for the next task
            return _recommend_prompting_user(self.index, state, "select")
        return ActionResult(
            index=self.index,
            result=result,
            recommended_action=action,
            issuing_cmd="select",
        )


class NextCall(Call):
    def __init__(self, var_name: VariableName, command: SearchCommand):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _maybe_update_assignments(self, state: PyTODAgentState):
        """`show`, `slice` and `len` commands are generated before
        we iterate the command. We update the assignment so that
        the NLG can receive the correct input"""
        if (prev_var := f"x{self.index - 1}") in state.revealed_entities:
            match assigned := state.revealed_entities[prev_var]:
                case (_, obj) if isinstance(obj, (type(None), Entity)):
                    state.revealed_entities[prev_var] = (
                        assigned[0],
                        state.assignments[self.var_name],
                    )
                case _:
                    raise NotImplementedError(
                        f"Reveled entities assignment not implement for {assigned}"
                    )

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Call the `next` on the command and recommend a follow-up
        action if the iteration raises `StopIteration` error."""
        action, result = None, Inform(dialog="Success")
        try:
            # technically the state does not change when we iterate
            state.track_state(self.command)
            entity = next(self.command)
            state.assignments[self.var_name] = entity
            self._maybe_update_assignments(state)
        except StopIteration:
            # NB: signal turns are auto-inserted
            # in the session for now, for simplicity
            result = Inform(dialog="Failed")
            if self.command.executed:
                user_hint = self.command.hint_generator.prompt_user_hint()
                action = [RecommendedAction(dialog=f"{user_hint}")]
                return ActionResult(
                    index=self.index,
                    result=result,
                    recommended_action=action,
                    issuing_cmd="next",
                )
            raise StopIteration
        return ActionResult(
            index=self.index,
            result=result,
            recommended_action=action,
            issuing_cmd="next",
        )


class SuggestCall(Call):
    def __init__(self, var_name: VariableName, command: ConfirmedCommand):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _find_source_task(self, state: PyTODAgentState) -> Optional[SearchCommand]:
        """Find a reference to a query the user completed in the execution history,
        if it exists."""
        assignments = _get_assigned_vars(state)
        source_task = None
        # we look for the most recent call to a command for which the
        # suggest intent can be a follow-up
        for var in reversed(assignments):
            maybe_source_task = state.assignments[var]
            if (
                isinstance(maybe_source_task, SearchCommand)
                and maybe_source_task.followup_command == self.command.name
            ):
                source_task = maybe_source_task
                break
        else:
            logger.warning(
                f"Could not find compatible source task for command: {self.command.name}"
            )
        return source_task

    def _update_suggested_command(self, state: PyTODAgentState):
        """Find tasks for which the suggested command is a follow-up
        command and carry-over properties to the suggested task."""
        source_task = self._find_source_task(state)
        if source_task is not None:
            # set required and optional slots that were already
            # mentioned for the new command
            args_mentioned = source_task.get_mentioned_args()
            for arg, value in args_mentioned.items():
                if is_arg(arg, self.command.schema):
                    setattr(self.command, arg, value)
            # result slots that are tracked when the query is
            # complete
            tracked_slots = source_task.system_tracked_slots
            for slot in tracked_slots:
                if is_arg(slot, self.command.schema):
                    value = getattr(source_task.selected_entity, slot, None)
                    if value is not None:
                        setattr(self.command, slot, value)

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Create an instance of the followup command,
        carrying-over relevant state from the source command and
        recommend an appropriate action."""
        self._update_suggested_command(state)
        state.suggested_tasks[self.var_name] = self.command
        return _recommend_prompting_user(self.index, state, "suggest")


class SuspendCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: ConfirmedCommand,
        execution_context: ExecutionContext,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command
        self._context = execution_context

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Mark a task as suspended. Recommend that the agent
        prompts the user for next task if the user has not
        already taken an action."""
        state.suspended_tasks[self.var_name] = self.command
        action, result = None, Inform(dialog="Success")
        should_recommend = (
            self._context.turn_context == TurnContext.SINGLE_DOMAIN
            and not self._context.conversation_end
        )
        # nb: technically this is only necessary if should_recommend=True
        state.track_state(self.command)
        if should_recommend:
            return _recommend_prompting_user(self.index, state, issuer="suspend")
        return ActionResult(
            index=self.index,
            result=result,
            recommended_action=action,
            issuing_cmd="suspend",
        )


class ResumeCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: ConfirmedCommand,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Resume a previously suspended task."""
        state.assignments[self.var_name] = self.command
        state.resumed_tasks[self.var_name] = self.command
        state.track_state(self.command)
        return ActionResult(
            index=self.index, result=Inform(dialog="Success"), issuing_cmd="resume"
        )


class SliceCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: SearchCommand,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Placeholder for reading the entities that the agent
        should be talking about from the command results."""
        # nb: we store a placeholder, this should call __slice__ protocol
        #  which we do not implement
        self._assign_entity(state)
        return ActionResult(
            index=self.index, result=Inform(dialog="Success"), issuing_cmd="slice"
        )

    def _assign_entity(self, state: PyTODAgentState):
        if self.command.name.startswith("Find"):
            state.revealed_entities[self.var_name] = "[entity_slice]"
        # hack to prevent ground truth execution failure on <v0.8.0, slice() should
        # not follow GetTimesForMovie
        else:
            state.revealed_entities[self.var_name] = (
                len(self.command),
                self.command.current_entity,
            )


class LenCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: SearchCommand,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Store the number of entities the user query returned."""
        state.revealed_entities[self.var_name] = len(self.command)
        return ActionResult(index=self.index, result=Inform(dialog="Success"))


class ShowCall(Call):
    def __init__(
        self,
        var_name: VariableName,
        command: SearchCommand,
    ):
        super().__init__(var_name)
        try:
            assert command is not None
        except AssertionError:
            raise InvalidVariableReferenceError
        self.command = command

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Store the number of entities the user query returned
        along with the top result."""
        state.revealed_entities[self.var_name] = (
            len(self.command),
            self.command.current_entity,
        )
        return ActionResult(index=self.index, result=Inform(dialog="Success"))


class ConversationPauseCall(Call):
    def __init__(
        self,
        var_name: VariableName,
    ):
        super().__init__(var_name)

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Recommend that the agent prompts the user for
        next task."""
        index = self.index
        command = _get_last_command(state)
        state.track_state(command)
        return _recommend_prompting_user(index, state, "conversation_pause")


class DeclineAlternativeCall(Call):
    def __init__(
        self,
        var_name: VariableName,
    ):
        super().__init__(var_name)

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Recommend that the agent prompts the user for
        next task."""
        index = self.index
        command = _get_last_command(state)
        state.track_state(command)
        return _recommend_prompting_user(index, state, "decline_alternative")


class ParseErrorCall(Call):
    def __init__(
        self,
        var_name: VariableName,
    ):
        super().__init__(var_name)

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Recommend actions following a parse error failure."""
        return ActionResult(index=self.index, result=Inform("Failed"))


class SayCall(Call):
    def __init__(
        self, var_name: VariableName, properties_requested: list[str] | None = None
    ):
        super().__init__(var_name)
        self.properties_requested = properties_requested

    def _run(self, state: PyTODAgentState) -> ActionResult:
        """Placeholder for NLG implementation. Currently only
        used to track the user requests."""
        if self.properties_requested is not None:
            state.update_user_requests(self.properties_requested)
        return ActionResult(index=self.index, result=Inform("Success"))
