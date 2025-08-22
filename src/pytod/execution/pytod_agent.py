#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Parse the language model generated statements into python-executable code."""

import ast
import logging
from copy import deepcopy
from typing import Any, Literal, NamedTuple, cast

from pytod.command import CommandCollection, ServiceCommand
from pytod.execution.calls import Call
from pytod.execution.expression_calls import AssignmentCall
from pytod.execution.function_calls import (
    CommandCall,
    ConfirmCall,
    ConversationPauseCall,
    DeclineAlternativeCall,
    LenCall,
    ParseErrorCall,
    PerformCall,
    ResumeCall,
    SayCall,
    ShowCall,
    SliceCall,
    SuspendCall, SelectCall, NextCall, SuggestCall,
)
from pytod.execution.policy_utils import ActionResult
from pytod.execution.pytod_agent_state import ExecutionContext, PyTODAgentState
from pytod.inference.utils import map_tool_name_to_service_intent
from pytod.interpreter.metadata import (
    COMMUNICATE_ENTITY_INFO_TOOL,
    COMMUNICATE_NUM_RESULTS_TOOL,
    DECLINE_ALTERNATIVE_TOOL,
    ENTITY_SELECTION_TOOL,
    FOLLOWUP_INTENT_DECLINE_TOOL,
    FOLLOWUP_INTENT_TOOL,
    FOLLOWUP_INTENT_TOOL_KWARG,
    ITERATION_TOOL,
    NLG_CALL_TOOL,
    PARSE_ERROR_TOOL,
    PAUSE_TOOL,
    QUERY_POS_ARG_TOOLS,
    QUERY_RESULTS_REF_KWARG,
    RESTART_TASK_TOOL,
    SELECT_ALLOWED_KWARGS,
    SLICE_TOOL,
    SPECIAL_TOOLS,
    TRANSACTION_CONFIRMATION_TOOL,
    TRANSACTION_SUCCESS_TOOL,
    VARIABLE_RESOLUTION_ERROR,
)
from pytod.pytod_types.aliases import DialogueID, ToolName, VariableName
from pytod.pytod_types.pytod import HintTurn, SignalTurn, SystemTurn
from pytod.simulation.api_driver import APIDriver
from pytod.simulation.command import Command
from pytod.simulation.command_registry import command_registry
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.entities import Entity
from pytod.simulation.search_command import SearchCommand

logger = logging.getLogger(__name__)


def _var_name(index: int) -> str:
    return f"x{index}"


def _normalise_args(
    args: list[ast.expr], keywords: list[ast.keyword], command: Command
) -> dict[str, ast.expr]:
    keyword_args = {}
    if len(args):
        pos_arg_names = command.positional_args
        if len(args) > len(pos_arg_names):
            raise ValueError("Too many positional arguments passed")
        for arg, arg_name in zip(args, pos_arg_names):
            keyword_args[arg_name] = arg

    for keyword in keywords:
        assert keyword.arg is not None
        keyword_args[keyword.arg] = keyword.value

    return keyword_args


def _parse_program_turn(turn: SystemTurn | SignalTurn | HintTurn) -> ast.Module:
    """Convert turns to ast modules."""
    match turn:
        case SystemTurn():
            program = str(turn.expression)
        case _:
            program = turn.dialog
    # assert (m := re.search(pattern, program)) is not None
    return ast.parse(program)


def _parse_fcn_name(expr: ast.Call) -> str:
    """Extract the function name from an ast
    expression."""
    match expr:
        case ast.Call(ast.Name() as func):
            return func.id
        case _:
            raise SyntaxError(f"Cannot extract fcn name from: {ast.unparse(expr)}")


class PyTODAgent:
    def __init__(
        self,
        split: Literal["validation", "test"],
        dialogue_id: DialogueID,
        schema: CommandCollection,
        lenient: bool = False,
        debug: bool = False,
        entity_properties: bool = False,
    ):
        """
        Parameters
        ----------
        split, dialogue_id
        lenient
            Whether the execution is lenient or not. If `True` then:

                - If `resume()` is correctly invoked, then arguments
                that were not correctly predicted in the subsequent
                API call are not overwritten
        entity_properties
            If `True`, then the attribute setting allows setting
            `Entity` subclasses as attributes.
        debug
            If True, turns are predicted as executed.
        """
        self.state: PyTODAgentState = PyTODAgentState(split, schema)
        self.context: ExecutionContext | None = None
        self._lenient = lenient
        self.dialogue_id = dialogue_id
        self.schema = schema
        self._debug = debug
        self._entity_properties = entity_properties

    def _get_assignment_calls(
        self, attributes: list[ast.Attribute], values: list[ast.Constant]
    ) -> list[AssignmentCall]:
        """
        Represent (possibly multi-) assignment statements as a call.

        Parameters
        ----------
        attributes
            Each attribute contains the target variable for assignment (eg)
            as a `value` and the command argument which is assigned a value
            to as `attr`.
        values
            The values assigned to each command argument.
        """
        grouped: dict[str, list[tuple[str, str]]] = {}
        for att, val in zip(attributes, values):
            match att.value:
                case ast.Name():
                    # the variable which is the assignment target
                    # (ie variable bound to command representing
                    # user intent for SGD; can be bound to entities,
                    # more generally)
                    var_name = att.value.id
                    attr_name = att.attr
                case _:
                    raise ValueError("Invalid assignment instruction.")
            # val is an ast.Constant expression
            executed_value = self._execute_expr(val)
            grouped.setdefault(var_name, []).append((attr_name, executed_value))
        return [
            AssignmentCall(
                var_name=call_bound_variable,
                attribute_name=[g[0] for g in properties_and_values],
                value=[g[1] for g in properties_and_values],
            )
            for call_bound_variable, properties_and_values in grouped.items()
        ]

    def forget(self):
        """Clear the agent state."""
        self.state = PyTODAgentState(self.schema.split, self.schema)

    def _get_select_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> SelectCall | ParseErrorCall:
        """Parse entity selection into a call that can be executed."""
        selected_kwargs, command, entity = None, None, None
        match expr:
            case ast.Call(ast.Name() as func, args, keywords):
                match args:
                    case [ast.Name() as entity]:
                        entity = self._execute_expr(entity)
                    case [ast.Attribute(value=ast.Name() as entity)]:
                        entity = self._execute_expr(entity)
                    case []:
                        # this means something went wrong
                        # and we called select()
                        if not keywords:
                            command = self.state.current_task
                            if isinstance(
                                context := self.state.current_task_context, APIDriver
                            ):
                                entity = command.get_entity_obj()
                            else:
                                entity = command.get_entity_obj(
                                    context, command.get_self_args()
                                )
                            return SelectCall(
                                var_name=var_name,
                                command=command,
                                entity=entity,
                                execution_context=self.context,
                                select_kwargs=selected_kwargs,
                            )
                        entity = None
                    case _:
                        logger.warning(
                            f"{self.dialogue_id} Incorrect positional arguments for `{func.id}`"
                            f" tool. Expression: {ast.unparse(expr)}"
                        )
                        return ParseErrorCall(var_name)
                match keywords:
                    case [
                        ast.keyword(
                            arg=arg,
                            value=ast.Name() as command,
                        )
                    ] if arg == QUERY_RESULTS_REF_KWARG:
                        command = self._execute_expr(command)
                    case [
                        ast.keyword() as kwarg,
                        ast.keyword(
                            arg=arg,
                            value=ast.Name() as command,
                        ),
                    ] if arg == QUERY_RESULTS_REF_KWARG:
                        command = self._execute_expr(command)
                        match kwarg:
                            case ast.keyword(
                                arg=key, value=value
                            ) if key in SELECT_ALLOWED_KWARGS:
                                selected_kwargs = {key: self._execute_expr(value)}
                            case ast.keyword(arg=key, value=value):
                                logger.error(
                                    f"Incorrect keyword in `{func.id}` syntax: {key}={value}"
                                )
                                selected_kwargs = {key: self._execute_expr(value)}
                    case _:
                        if entity is not None:
                            if isinstance(entity, Command):
                                command = entity
                                entity = None
                        else:
                            command = self.state.current_task
            case _:
                raise SyntaxError("Unexpected error during parsing select.")

        return SelectCall(
            var_name=var_name,
            command=command,
            entity=entity,
            execution_context=self.context,
            select_kwargs=selected_kwargs,
        )

    def _execute_single_positional_arg(self, call_ast: ast.Call) -> Command:
        """Get a `python` object representing the variable in
        a single-argument positional function call."""
        match call_ast:
            case ast.Call(ast.Name() as func, args):
                match args:
                    case [ast.Name() as command]:
                        command = self._execute_expr(command)
                    case _:
                        raise SyntaxError(
                            f"Invalid syntax for {func} statement: {ast.unparse(args)}"
                        )
            case _:
                raise SyntaxError(f"Invalid syntax for: {ast.unparse(call_ast)}")
        return command

    def _get_iteration_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> NextCall:
        """Parse iteration calls into executable code."""

        try:
            command = self._execute_single_positional_arg(expr)
        except SyntaxError:
            logger.warning(
                f"{self.dialogue_id}: "
                f"Invalid positional argument {ast.unparse(expr)} in iteration call"
            )
            return NextCall(var_name=var_name, command=self.state.current_task)
        return NextCall(var_name=var_name, command=cast(SearchCommand, command))

    def _get_followup_intent_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> SuggestCall:
        """Parse calls to follow-up intent tools into executable code."""

        match expr:
            case ast.Call(ast.Name() as func, args, keywords):  # noqa
                match keywords:
                    case [
                        ast.keyword(
                            arg=arg,
                            value=ast.Constant(value=value as tool_name),  # noqa
                        )
                    ] if arg == FOLLOWUP_INTENT_TOOL_KWARG:  # noqa
                        suggested_command = self._execute_expr(
                            ast.parse(f"{tool_name}()").body[0].value
                        )
                        return SuggestCall(
                            var_name=var_name, command=suggested_command
                        )
                    case _:
                        raise SyntaxError(
                            f"Invalid keywords to {func.id}: {ast.unparse(keywords)}"
                        )
            case _:
                raise SyntaxError(
                    "Unexpected error while parsing `suggest` instruction."
                )

    def _get_followup_intent_declined_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> SuspendCall:
        """Parse declined follow-up intent calls into executable code."""

        try:
            command = self._execute_single_positional_arg(expr)
        except SyntaxError as e:
            match expr:
                case ast.Call(ast.Name(), args):
                    match args:
                        case [ast.Attribute(value=ast.Name(id=id_))]:
                            command = self.state.assignments.get(id_)
                        case _:
                            raise e
                case _:
                    raise e
        return SuspendCall(
            var_name=var_name,
            command=cast(ConfirmedCommand, command),
            execution_context=self.context,
        )

    def _get_task_resumption_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> ResumeCall:
        """Parse user requests to resume previously suspended tasks."""
        try:
            command = self._execute_single_positional_arg(expr)
        except SyntaxError as e:
            raise e
        return ResumeCall(
            var_name=var_name,
            command=cast(ConfirmedCommand, command),
        )

    def _get_query_result_tool_call(
        self, var_name: VariableName, expr: ast.Call
    ) -> LenCall | SliceCall | ShowCall:
        """Parse calls to tools that operate on intents which return database results."""

        try:
            command = cast(SearchCommand, self._execute_single_positional_arg(expr))
        except SyntaxError as e:
            raise e

        match fcn := _parse_fcn_name(expr):
            case tool if tool == COMMUNICATE_NUM_RESULTS_TOOL:
                return LenCall(var_name=var_name, command=command)
            case tool if tool == SLICE_TOOL:
                return SliceCall(var_name=var_name, command=command)
            case _:
                assert fcn == COMMUNICATE_ENTITY_INFO_TOOL
                return ShowCall(var_name=var_name, command=command)

    def _get_transaction_confirmation_calls(
        self, var_name: VariableName, expr: ast.Call
    ) -> ConfirmCall | PerformCall:
        """Parse user transaction confirmation and successful transaction statements."""

        try:
            command = cast(ConfirmedCommand, self._execute_single_positional_arg(expr))
        except SyntaxError as e:
            raise e

        match fcn := _parse_fcn_name(expr):
            case tool if tool == TRANSACTION_SUCCESS_TOOL:
                return PerformCall(var_name, command)
            case _:
                assert fcn == TRANSACTION_CONFIRMATION_TOOL
                return ConfirmCall(var_name, command)

    @staticmethod
    def _get_conversation_pause_calls(
        var_name: VariableName, expr: ast.Call
    ) -> DeclineAlternativeCall | ConversationPauseCall:
        """Parse conversation pauses."""

        match fcn := _parse_fcn_name(expr):
            case tool if tool == PAUSE_TOOL:
                return ConversationPauseCall(var_name)
            case _:
                assert fcn == DECLINE_ALTERNATIVE_TOOL
                return DeclineAlternativeCall(var_name)

    @staticmethod
    def _get_nlg_call(var_name: VariableName, expr: ast.Call) -> SayCall:
        """Parse arguments to the NLG call. Since we do not implement
        NLG, this simply extracts the requested slots at this time."""

        properties_requested = []
        match expr:
            case ast.Call(ast.Name(), args):
                for arg in args:
                    match arg:
                        # a variable
                        case ast.Name():
                            continue
                        case ast.Attribute(value=ast.Name(), attr=attr):
                            properties_requested.append(attr)
                        case _:
                            logger.warning(
                                f"Invalid positional argument: {ast.unparse(arg)}"
                            )
            case _:
                raise SyntaxError(
                    f"Unexpected error while parsing nlg call: {ast.unparse(expr)}."
                )
        return SayCall(
            var_name=var_name, properties_requested=properties_requested or None
        )

    def _get_function_call(
        self, command: Command | str, var_name: VariableName, expr: ast.Call
    ) -> Call:
        """Parse the functions generated by the language model into python-executable
        code."""
        match command:
            case Command():
                return CommandCall(var_name=var_name, command=command, assigned=False)
            case command if isinstance(command, str):  # noqa
                assert command in SPECIAL_TOOLS
                match command:
                    case func if func == ENTITY_SELECTION_TOOL:
                        return self._get_select_call(var_name, expr)
                    case func if func == ITERATION_TOOL:
                        return self._get_iteration_call(var_name, expr)
                    case func if func == FOLLOWUP_INTENT_TOOL:
                        return self._get_followup_intent_call(var_name, expr)
                    case func if func == FOLLOWUP_INTENT_DECLINE_TOOL:
                        return self._get_followup_intent_declined_call(var_name, expr)
                    case func if func == RESTART_TASK_TOOL:
                        return self._get_task_resumption_call(var_name, expr)
                    case func if func in QUERY_POS_ARG_TOOLS:
                        return self._get_query_result_tool_call(var_name, expr)
                    case func if func in {
                        TRANSACTION_CONFIRMATION_TOOL,
                        TRANSACTION_SUCCESS_TOOL,
                    }:
                        return self._get_transaction_confirmation_calls(var_name, expr)
                    case func if func in {PAUSE_TOOL, DECLINE_ALTERNATIVE_TOOL}:
                        return self._get_conversation_pause_calls(var_name, expr)
                    case func if func == PARSE_ERROR_TOOL:
                        return ParseErrorCall(var_name)
                    case _:
                        assert func == NLG_CALL_TOOL
                        return self._get_nlg_call(var_name, expr)
            case _:
                raise RuntimeError(f"Unexpected type for command: {type(command)}")

    def _get_calls(self, index: int, ast_module: ast.Module) -> list[Call]:  # type: ignore[return]
        """Parse expressions generated by the language model into python-executable
        calls."""
        match ast_module.body:
            # matches function calls e.g. find_provider(city='Berkeley')
            case [ast.Expr(value=ast.Call() as expr)]:
                var_name = _var_name(index)
                value = self._execute_expr(expr)
                return [self._get_function_call(value, var_name, expr)]
            # matches a single assignment
            case [ast.Assign(targets=[ast.Attribute() as attribute], value=value)]:
                # e.g. x0.is_unisex = 'True'
                # but also x0[0].time_booked = '10:00 AM'
                return self._get_assignment_calls(
                    [attribute], [cast(ast.Constant, value)]
                )
            # matches multiple assignments
            case assigns if all(ast.Assign(assign, ast.Assign) for assign in assigns):
                calls, attributes, values = [], [], []
                for assign in assigns:
                    try:
                        attributes.append(assign.targets[0])
                        values.append(assign.value)
                    except AttributeError:
                        calls.extend(self._get_calls(index, ast.Module(body=[assign])))
                assignments = self._get_assignment_calls(attributes, values)
                # It is possible for multiple tasks to be updated in a multi-assign statement
                # In this case we will just return the task with the highest number of updated
                # values. However, this does not happen in SGD, so we assert on a single task
                # update
                try:
                    assert len(assignments) == 1
                except AssertionError:
                    assert not assignments
                    assert calls
                    logger.warning(
                        f"{self.dialogue_id} Multiple calls generated. This is not expected"
                    )
                return calls + assignments
            case []:
                raise ValueError("No expressions found")
            case _:
                raise ValueError("Exactly one expression or assignment allowed")

    def _build_command(self, tool_name: ToolName) -> Command:
        """Create an instance of a command representing a user intent."""
        service, intent = map_tool_name_to_service_intent(tool_name)
        service_schema: list[ServiceCommand] = self.schema.get_service_commands(service)
        command_schema = self.schema.get(service, intent)
        command_type = command_registry.get(name=intent, service=service)
        if command_type is None:
            raise ValueError(f"{tool_name} is not registered")
        command = command_type.build(
            self.dialogue_id,
            command_schema,
            [s for s in service_schema if s.name != intent] or None,
        )
        return command

    def _execute_expr(self, ast_expr: ast.expr) -> Any:
        """Executes a PyTOD instruction (or part of an instruction) represented as an abstract
        syntax tree (AST).

         Execution means that the expression tree is recursively
         traversed and variables are resolved to their values. A
         `python` object required to execute the current action is returned. The execution:

         1. Converts instructions to instances of corresponding `python` objects
         2. Generated positional/keyword args are set as command attributes. If an arg
            is a variable previously assigned to, the executor recursively resolves
            it to its value (which can be a python object or simple type (str, int, None but
            also immutable container types such af frozenset/tuple)

        Args
            ast_expr: the abstract syntax tree of the instruction currently executed (or
             a subtree of said instruction)
        """
        match ast_expr:
            # matches a fcn call e.g., find_restaurant(location="Cambridge")
            case ast.Call(ast.Name() as func, args, keywords):
                command_name = func.id  # "find_restaurant"
                match command_name:
                    case tool if tool not in SPECIAL_TOOLS:
                        command = self._build_command(command_name)
                        self._set_command_properties(command, args, keywords)
                        return command
                    case _:
                        return command_name
            # Constant(value='Cambridge') -> 'Cambridge'
            case ast.Constant(value):
                return value
            # expression is a variable name (eg x3) -> return what is assigned to x3
            case ast.Name(id_) if id_ in {"true", "false"}:
                return id_
            case ast.Name(id_):
                var = None
                # we must have assigned the variable to resolve it.
                # we assign the command calls, suggested and suspended
                # tasks separately
                for container in self.state.var_containers:
                    container_dict = getattr(self.state, container)
                    var = container_dict.get(id_)
                    if var is not None:
                        return var
                return var
            # expression is attribute access to a variable (eg x1.restaurant_name):
            # return the value of the attribute
            case ast.Attribute(value=ast.Name(id=id_), attr=attr):
                var = self.state.assignments.get(id_)
                if var is None:
                    # this issue arises in the ground truth data when queries
                    # fail because the variables refer to the last variable
                    # bound to next() which raises StopIteration error
                    # more generally, they can arise if a variable reference
                    # in a value is incorrect
                    try:
                        val = getattr(self.state.current_task, attr)
                    except AttributeError as e:
                        logger.warning(f"{self.dialogue_id}: {e.args[0]}")
                        return
                    if val is not None:
                        return val
                    logger.warning(
                        f"Invalid variable reference {id_} in expression"
                        f" {ast.unparse(ast_expr)}"
                    )
                    return
                try:
                    return getattr(var, attr)
                except AttributeError as e:
                    logger.warning(f"{self.dialogue_id}: {e.args[0]}")
                    return
            # this matches values which access a properties of entities in a
            # list assigned to a variable ("x1[0].name"), returning the value
            # of that property
            case ast.Attribute(
                value=ast.Subscript(
                    value=ast.Name(id=id_), slice=ast.Constant(value=idx)
                ),
                attr=attr,
            ):
                var = self.state.assignments.get(id_)
                if var is None:
                    raise ValueError(f"Variable {id_} not found")
                var = var[idx]
                return getattr(var, attr)
            case _:
                dial_id = self.dialogue_id
                logger.warning(
                    f" {dial_id} Cannot execute unsupported expression: "
                    f"{ast.unparse(ast_expr)}"
                )
                return

    def _execute_and_set(self, command: Command, keywords: dict[str, ast.expr]):
        """Resolve the keywords to strings and set the command properties."""
        # e.g., set 'location' to
        # self._execute_expr(value=Constant(value='Cambridge'))
        # which recursively resolves to "Cambridge"
        for arg, value in keywords.items():
            value = self._execute_expr(value)
            if value is None:
                logger.warning(
                    f"{self.dialogue_id}: "
                    f"Incorrect reference detected while resolving '{arg}' value"
                )
            elif isinstance(value, Entity) and not self._entity_properties:
                logger.warning(
                    f"{self.dialogue_id}"
                    f"{command}.{arg} does not support Entity members"
                )
            else:
                setattr(command, arg, value)

    def _get_draft_command(self, command: Command) -> Command | None:
        """Check if the current command has been previously suspended
        and therefore a draft exists."""

        for _, draft_cmd in self.state.resumed_tasks.items():
            try:
                if (
                    draft_cmd.name == command.name
                    and draft_cmd.service == command.service
                ):
                    return draft_cmd
            except AttributeError:
                logger.warning(f"{self.dialogue_id}: Failed to resume session")
                return
        return

    def _set_command_properties(
        self, command: Command, args: list[ast.expr], keywords: list[ast.keyword]
    ):
        """In-place setting of `command` attributes."""
        # {'location': Constant(value='New York')}
        keyword_args = _normalise_args(args, keywords, command)
        # in lenient evaluation, we take into account any
        # drafts created when the agent suggests the task
        if self._lenient:
            draft_command = self._get_draft_command(command)
            if draft_command is None:
                self._execute_and_set(command, keyword_args)
            else:
                # set any other attributes that could be resolved
                for arg, value in keyword_args.items():
                    value = self._execute_expr(value)
                    try:
                        assert value is not None
                    except AssertionError:
                        orig_value = ast.unparse(keyword_args[arg])
                        logger.info(
                            f"{command.uuid}: Could not execute value: {orig_value}. "
                            f"Maybe a quote was missing?"
                        )
                        continue
                    # do not override draft properties with variable
                    # resolution errors. If a property was not set,
                    # then it is not overridden with a resolution error
                    if value == VARIABLE_RESOLUTION_ERROR:
                        draft_value = getattr(command, arg, None)
                        if draft_value is None:
                            logger.warning(
                                f"{self.dialogue_id}: "
                                f"Attempted to set {arg} to {VARIABLE_RESOLUTION_ERROR}."
                                "Previous value was `None`."
                            )
                        if draft_value == VARIABLE_RESOLUTION_ERROR:
                            setattr(command, arg, VARIABLE_RESOLUTION_ERROR)
                        continue
                    setattr(command, arg, value)
                # copy draft properties in case they have been missed
                for arg, value in draft_command.get_mentioned_args().items():
                    setattr(command, arg, value)
        else:
            self._execute_and_set(command, keyword_args)

    def execute_instructions(
        self, turns: list[SystemTurn | SignalTurn | HintTurn]
    ) -> list[ActionResult]:
        """Execute a PyTOD programme."""

        actions = []
        for turn in turns:
            if self._debug:
                print(str(turn.expression))
            match turn.author:
                case "System":
                    turn_ast = _parse_program_turn(turn)
                    for call in self._get_calls(turn.index, turn_ast):
                        actions.append(call.run(self.state))
                # these are not tracked at the minute, they will
                # be taken into account when executing after
                # policy prediction
                case "Hint" | "Signal":
                    continue
        return actions


class AgentArgs(NamedTuple):
    schema: CommandCollection
    split: Literal["dev", "test"]
    lenient: bool
    debug: bool
