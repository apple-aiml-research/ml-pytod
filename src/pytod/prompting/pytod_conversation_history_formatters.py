#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import random
import re
from dataclasses import dataclass, field

from omegaconf import DictConfig

from pytod.command import ArgumentDefinition, ServiceCommand
from pytod.inference.utils import VARIABLE_PATTERN, map_tool_name_to_service_intent
from pytod.interpreter.metadata import (
    ENTITY_REF_COMMAND,
    ENTITY_SELECTION_TOOL,
    INTENT_UPDATE_TOOL,
    ITERATION_TOOL,
    PUBLIC_PROPERTY_MARKER,
    RESOLUTION_ERR,
    SPECIAL_TOOLS,
    SYSTEM_ARG_MARKER,
    TRANSACTION_CONFIRMATION_TOOL,
    WILDCARD_VALUE,
)
from pytod.prompting.pytod_text2text_formatters_utils import VariableMapper
from pytod.prompting.pytod_text2text_header_formatter_utils import (
    DisplayedProperties,
    EntityInfo,
    ObjectReferenceInfo,
    RequestableInfo,
    ValueType,
)
from pytod.prompting.template_factories import (
    developer_turn_confirmed_argument_instructions_resolved_carryover_arg_values,
)
from pytod.prompting.templates import (
    ConfirmationPropertyListingDeveloperTurnTemplate,
    ConfirmedArgumentInstructionsDeveloperTurnTemplate,
    IterationDeveloperTurnTemplate,
    SelectedEntityDeveloperTurnTemplate,
)
from pytod.pytod_types.aliases import ServiceName, SlotName, ToolName, VariableName
from pytod.pytod_types.pytod import (
    AnyTurn,
    HintTurn,
    ResponseTurn,
    SignalTurn,
    SystemTurn,
    UserTurn,
)
from pytod.utils import snake_case

logger = logging.getLogger(__name__)
FormattedConversationHistory = list[str]


LOWERCASE_BOOLEANS = {"true", "false"}


@dataclass
class ConversationHistoryFormatterState:
    """Track state relevant to formatting the conversation history.

    Properties
    ----------
    entity_or_cmd_service
        This map is used to resolve symbolic value references in API calls.
        It provides the active service at each entity or API call. This can
        be used to look up actual values of slots the user said when resolving references.
    state
        This tracks the values the user said along with two special values:
            - a value to identify system-tracked slots (which are either tracked entity properties
            or confirmed slots) (pytod.interpreter.metadata.SYSTEM_ARG_MARKER)
            - a value to indentify entity properties
                (pytod.interpreter.metadata.PUBLIC_PROPERTY_MARKER).
    current_cmd_var
        The variable bound to the last API call in the dialogue history.
    """

    var_mapping: VariableMapper | None = None
    current_turn_pointer: int = 0
    state: dict[ServiceName, dict[SlotName, str]] = field(default_factory=dict)
    entity_or_cmd_service: dict[VariableName, ServiceName] = field(default_factory=dict)
    history: list[AnyTurn] = field(default_factory=list)
    formatted_history: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    current_task_schema: ServiceCommand | None = None
    current_cmd_var: VariableName | None = None

    def clear(self):
        """Clears the state of the formatter so that it can be used
        to process another example."""
        self.var_mapping = None
        self.current_turn_pointer = 0
        self.history = []
        self.formatted_history = []
        # instructions which may be inserted after system turns
        self.instructions = []
        self.current_task_schema = None
        self.state = {}
        self.entity_or_cmd_service = {}
        self.current_cmd_var = None


class ConversationHistoryFormatter:
    def __init__(self, config: DictConfig | None = None):
        self._config = config
        self._line_start = ""
        self._start_index = None
        self._record_state = False
        self._schema = None
        if config is not None:
            if hasattr(config, "line_start"):
                self._line_start = "x" if config.line_start == "variable" else ""
            if hasattr(config, "start_index"):
                self._start_index = config.start_index
            if hasattr(config, "record_state"):
                self._record_state = config.record_state
                self._schema = config.schema
        self._state = ConversationHistoryFormatterState()

    @staticmethod
    def should_suppress_hint(history: list[AnyTurn], turn_idx: int) -> bool:
        if turn_idx == len(history) - 1:
            return False

        # if a user/system turn follows, this is a hint
        # from a previous turn
        for idx, turn in enumerate(history[turn_idx + 1 :]):
            match turn.author:
                case "User" | "System":
                    # 'suggest' can follow a hint and the agent should
                    # choose between them, we take care not to suppress
                    # the hint in this case
                    if turn.get_tool_name() == "suggest":
                        continue
                    return True
        return False

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, index: int):
        self._start_index = index

    @property
    def current_task_schema(self) -> ServiceCommand:
        return self._state.current_task_schema

    @current_task_schema.setter
    def current_task_schema(self, value: ServiceCommand):
        self._state.current_task_schema = value

    @property
    def state(self) -> ConversationHistoryFormatterState:
        return self._state

    def reset(self):
        self._state.clear()

    @staticmethod
    def should_suppress_suggestion(history: list[AnyTurn], turn_idx: int) -> bool:
        """Suppress `suggest` statements in the dialogue history if they do not
        ground the next agent utterance."""

        # the agent may choose this action at the current turn, shouldn't suppress
        if turn_idx in {len(history) - 1, len(history) - 2}:
            return False

        adjacent_turns = [history[turn_idx - 1], history[turn_idx + 1]]
        try:
            [adjacent_hint] = [t for t in adjacent_turns if isinstance(t, HintTurn)]
            # if the hint origin is None, then the agent actually suggested the task
            # so the negation means the agent did not suggest the task
            return not (adjacent_hint.origin is None)
        except ValueError:
            # in DST inference, we predict states without running the app first, so
            # we won't have an adjacent Hint turn
            return False

    def _display_expression(self, turn: SystemTurn):
        """Re-index the variables of an expression to account for statements
        not displayed in the history and display the expression."""
        displ_expr = str(self._state.var_mapping.apply(turn.expression))
        self._state.var_mapping.add_mapping(f"x{turn.index}")
        return displ_expr

    def _display_selected_entity(self, turn: SystemTurn) -> str:
        """Default selected entity representation is a call to the
        `select` with appropriate args and/or kwargs."""
        return self._display_expression(turn)

    def _display_transaction_result(self, turn: SystemTurn) -> str:
        """Default representation of a successful transaction result
        is a call to `perform` with appropriate args."""
        return self._display_expression(turn)

    def _display_entity(self, turn: SystemTurn):
        """Default representation of an entity that has been
        offered to the user by the agent or the user's results
        iteration is a call to `next` with appropriate args."""
        return self._display_expression(turn)

    def _display_property_reference_instruction(self, turn: SystemTurn):
        """Default representation of the user confirmation. This can be
        overridden to display the confirmation in a custom format, or
        can be used to add instructions after confirmation statements."""
        return self._display_expression(turn)

    def add_instructions(self, instructions: list[str]):
        """Remove elements of `instructions` and add them
        to the formatted conversation history."""
        while instructions:
            next_instr = instructions.pop()
            self._state.formatted_history.append(next_instr)

    def __call__(self, history: list[AnyTurn]) -> tuple[list[str], VariableMapper]:
        self._state.var_mapping = VariableMapper(self._start_index)
        var_mapping = self._state.var_mapping
        self._state.history = history
        result: list[str] = []
        self._state.formatted_history = result
        for i, turn in enumerate(history):
            self._state.current_turn_pointer = i
            match turn.author:
                case "User":
                    assert isinstance(turn, UserTurn)
                    query = turn.query
                    result.append(f"user: {query}")
                case "System":
                    assert isinstance(turn, SystemTurn)
                    tool_name = turn.get_tool_name()
                    if self._record_state:
                        self._maybe_update_state(tool_name, turn)
                    match tool_name:
                        # say messages can be targets, but they should never be in the source
                        case "say":
                            continue
                        # suggest messages are also ignored if they don't ground the agent utterance
                        case "suggest" if self.should_suppress_suggestion(history, i):
                            continue
                        # in inference, the model may sometimes produce unparseable outputs, which
                        # should be ignored when formatting the conversation
                        case "parse_error":
                            continue
                        case "next":
                            result.append(
                                f"{self._line_start}{var_mapping.current_ndx} "
                                f"{self._display_entity(turn)}"
                            )
                            # model may be prompted with specific instructions after
                            # each system turn
                            self.add_instructions(self._state.instructions)
                            continue
                        # we treat entity selection and transaction results separately
                        # because we may experiment with different entity representations
                        case "select":
                            result.append(
                                f"{self._line_start}{var_mapping.current_ndx} "
                                f"{self._display_selected_entity(turn)}"
                            )
                            self.add_instructions(self._state.instructions)
                            continue
                        case "perform":
                            result.append(
                                f"{self._line_start}{var_mapping.current_ndx} "
                                f"{self._display_transaction_result(turn)}"
                            )
                            self.add_instructions(self._state.instructions)
                            continue
                        case "confirm":
                            result.append(
                                f"{self._line_start}{var_mapping.current_ndx} "
                                f"{self._display_property_reference_instruction(turn)}"
                            )
                            self.add_instructions(self._state.instructions)
                            continue
                    result.append(
                        f"{self._line_start}{var_mapping.current_ndx} {var_mapping.apply(turn.expression)}"  # noqa
                    )
                    var_mapping.add_mapping(f"x{turn.index}")
                    self.add_instructions(self._state.instructions)
                case "Response":
                    assert isinstance(turn, ResponseTurn)
                    msg = turn.text.replace('"', "'")
                    result.append(f"agent: {msg}")
                case "Signal":
                    assert isinstance(turn, SignalTurn)
                    msg = turn.dialog
                    result.append(
                        f"{self._line_start}{var_mapping.current_ndx} Signal: {msg}"
                    )
                    var_mapping.add_mapping(f"x{turn.index}")
                case "Hint":
                    assert isinstance(turn, HintTurn)
                    if not self.should_suppress_hint(history, i):
                        dialog = var_mapping.apply_to_dialog_text(turn.dialog)
                        result.append(
                            f'{self._line_start}{var_mapping.current_ndx} Hint("{dialog}")'
                        )
                        var_mapping.add_mapping(f"x{turn.index}")
                case _:
                    raise ValueError(f'Unknown author "{turn.author}"')
        result.append(f"{self._line_start}{var_mapping.current_ndx}")
        return result, var_mapping

    def _update_current_task_schema(self, tool_name: ToolName):
        service, tool = map_tool_name_to_service_intent(tool_name)
        assert service is not None
        self._state.current_task_schema = self._schema.get(service, tool)

    def _maybe_update_state(self, tool_name: ToolName, turn: SystemTurn):
        """Record last assignment to a slot either via assignment or
        through function calling. This may be used to get the dialogue
        state for state-dependent transcript formatting."""
        match tool_name:
            case var if var in {
                ENTITY_SELECTION_TOOL,
                TRANSACTION_CONFIRMATION_TOOL,
                ITERATION_TOOL,
            }:
                if (
                    tool_name == ITERATION_TOOL
                    and not turn.expression.expressions[0].exec_info
                ):
                    return
                self._entity_or_confirmation_state_update(turn)
                return
            case var if var in SPECIAL_TOOLS and var != INTENT_UPDATE_TOOL:
                return

        if tool_name == INTENT_UPDATE_TOOL:
            service = self.current_task_schema.service
            service_state = self._state.state.setdefault(service, {})
            for expr in turn.expression.expressions:
                kwarg_name = expr.keyword_args[0].split(".")[1]
                val = expr.kwarg_values[0]
                if val in LOWERCASE_BOOLEANS:
                    service_state[kwarg_name] = val
                else:
                    service_state[kwarg_name] = str(ast.literal_eval(val))
        else:
            self._update_current_task_schema(tool_name)
            service = self.current_task_schema.service
            service_state = self._state.state.setdefault(service, {})
            var = f"x{turn.index}"
            self._state.entity_or_cmd_service[var] = service
            self._state.current_cmd_var = var
            for expr in turn.expression.expressions:
                for kwarg, val in zip(expr.keyword_args, expr.kwarg_values):
                    if val in LOWERCASE_BOOLEANS:
                        service_state[kwarg] = val
                        continue
                    try:
                        service_state[kwarg] = str(ast.literal_eval(val))
                    except ValueError:
                        try:
                            assert re.match(VARIABLE_PATTERN, val) is not None
                        except AssertionError:
                            logger.warning(
                                f"Expression '{kwarg}={val}' triggered an ast.literal_eval  "
                                f"exception."
                            )
                            service_state[kwarg] = RESOLUTION_ERR
                            continue
                        obj_ref, arg_name = val.split(".")
                        try:
                            service = self._state.entity_or_cmd_service[obj_ref]
                        except KeyError:
                            service_state[kwarg] = RESOLUTION_ERR
                            logger.warning(
                                f"Could not resolve kwarg {kwarg} ({tool_name}), "
                                f"definition will be rendered instead"
                            )
                            continue
                        try:
                            service_state[kwarg] = self._state.state[service][arg_name]
                        # happens if we reference some argument that is not actually in the
                        # state (eg the model hallucinated a property that not in the schema
                        #  of the intent referenced)
                        except KeyError:
                            service_state[kwarg] = RESOLUTION_ERR
                            logger.warning(
                                f"Could not resolve property {arg_name} for object {obj_ref}. "
                                f"This is likely caused by a hallucinated slot name."
                            )

    def _entity_or_confirmation_state_update(self, turn: SystemTurn):
        """Keep a mapping between variables and services, so that values
        represented via object references can read appropriate value from
        the service state."""
        var_name = f"x{turn.index}"
        schema = self._state.current_task_schema
        service = schema.service
        tool = turn.get_tool_name()
        sys_args = (
            schema.system_tracked_slots
            if tool == ENTITY_SELECTION_TOOL
            else schema.system_confirmed_slots
        )
        if sys_args is not None:
            for s in sys_args:
                if s not in self._state.state[service]:
                    self._state.state[service][s] = SYSTEM_ARG_MARKER
        if (public_members := schema.followup_metadata.public_members) is not None:
            for s in public_members:
                if s not in self._state.state[service]:
                    logger.debug(
                        f"Argument {s}, a public member of {schema.tool_name}",
                        "was not tracked",
                    )
                    self._state.state[service][s] = PUBLIC_PROPERTY_MARKER
                else:
                    logger.debug(
                        f"Argument {s}, a public member of {schema.tool_name}",
                        "was already in state",
                    )
        # nb: if the variables in select are hallucinated or there are self-references
        #  in the intent keyword values, a dummy next() instruction is inserted. This
        #  allows the execution to carry over whatever slots were correctly predicted
        #  to follow-up tasks
        if tool in {ENTITY_SELECTION_TOOL, ITERATION_TOOL}:
            self._state.entity_or_cmd_service[var_name] = service


def get_displayed_properties(
    current_task: ServiceCommand,
    conversation_state: dict[ServiceName, dict[SlotName, str]],
) -> DisplayedProperties:
    """Return a list of properties to display for a selected entity,
    differentiating between optional and mandatory arguments."""
    optional = current_task.followup_metadata.optional or []
    permanent = [
        a for a in current_task.followup_metadata.get_args() if a not in optional
    ]
    state = conversation_state[current_task.service]
    optional_displayed = [a for a in optional if a in state]
    return DisplayedProperties(optional_displayed, permanent)


@dataclass
class RenderedEntitiesConversationHistoryFormatterState(
    ConversationHistoryFormatterState
):
    active_service: str | None = None
    entity_info: list[EntityInfo] = field(default_factory=list)
    last_entity_idx: int = 0

    def clear(self):
        super().clear()
        self.active_service = None
        self.entity_info = []
        self.last_entity_idx = 0


class RenderedEntitiesConversationHistoryFormatter(ConversationHistoryFormatter):
    def __init__(self, config: DictConfig | None = None):
        super().__init__(config)
        # whether carried-over values are represented symbolically
        self.value_object_references = config.value_object_references
        # shows developer turn with instructions to refer entity after select instruction
        self._instruct_obj_reference = config.instruct_obj_reference
        # shows developer turn with requested arguments def after next instruction
        self._show_api_return_defs = config.show_api_return_defs
        # shows developer turn with arguments that can be requested by the user after confirmation
        self._display_properties_after_confirmation = (
            config.display_properties_after_confirmation
        )
        # instructs the model to reference confirmed optional slots in API call
        # retries
        self._instruct_confirmed_optional_reference = (
            config.instruct_confirmed_optional_reference
        )
        self._schema = config.schema
        self._entity_selection_developer_turn_template = (
            SelectedEntityDeveloperTurnTemplate(config.schema)
        )
        self._iteration_developer_turn_template = IterationDeveloperTurnTemplate(
            config.schema
        )
        self._confirmation_developer_turn_template = (
            ConfirmationPropertyListingDeveloperTurnTemplate(config.schema)
        )
        if self.value_object_references:
            self._confirmed_arguments_instructions_developer_turn_template = (
                ConfirmedArgumentInstructionsDeveloperTurnTemplate(config.schema)
            )
        else:
            template_obj = ConfirmedArgumentInstructionsDeveloperTurnTemplate
            factory = developer_turn_confirmed_argument_instructions_resolved_carryover_arg_values
            self._confirmed_arguments_instructions_developer_turn_template = (
                template_obj(config.schema, template_factory=factory)
            )
        self._randomise_prompt_elements = config.randomise_prompt_elements
        self._state = RenderedEntitiesConversationHistoryFormatterState()

    @property
    def active_service(self) -> ServiceName:
        """Returns the current service"""
        return self._state.active_service

    @active_service.setter
    def active_service(self, value: ServiceName):
        self._state.active_service = value

    @property
    def conversation_state(self) -> dict[ServiceName, dict[SlotName, str]]:
        return self._state.state

    @property
    def entities(self) -> list[EntityInfo]:
        """Return the entities in the conversation, grouped by service."""
        return self._state.entity_info

    @property
    def state(self) -> RenderedEntitiesConversationHistoryFormatterState:
        return self._state

    def reset(self):
        self._state.clear()

    def _get_arg_definition(
        self,
        arg: SlotName,
    ) -> ArgumentDefinition | None:
        """Get the argument schema."""
        arg_def = self._schema.get_arg_schema(self.current_task_schema.service, arg)
        if arg_def is None:
            logger.warning(
                f"ConversationHistoryFormatter could not retrieve schema for argument: {arg}"
            )
        return arg_def

    def _get_value_type(self, arg: SlotName, value: str) -> ValueType:
        """Classify the value as categorical, non-categorical or wildcard
        for the purposes of generating instructions for describing references
        to the model."""

        if value == WILDCARD_VALUE:
            return ValueType.WILDCARD
        elif value == SYSTEM_ARG_MARKER:
            return ValueType.SYSTEM_ARGUMENT
        elif value == PUBLIC_PROPERTY_MARKER:
            return ValueType.PUBLIC_VARIABLE
        elif value == RESOLUTION_ERR:
            return ValueType.RESOLUTION_ERROR
        else:
            arg_def = self._get_arg_definition(arg)
            assert arg_def is not None
            return (
                ValueType.CATEGORICAL
                if arg_def.is_categorical
                else ValueType.NON_CATEGORICAL
            )

    def _maybe_add_system_tracked_slots_instructions(
        self,
        turn: SystemTurn,
        object_references: list[ObjectReferenceInfo],
        system_tracked_slots: list[str],
    ):
        """If we missed a slot and the entity selection was not predicted (but a variable
        error in function call was made) or the entity selection had incorrect variable
        references, we still display the system tracked slots definitions in the prompt
        for consistency with training setup to avoid future errors. This does not impact
        the accuracy since during execution the database won't be called and any references
        in future APIs will be void."""
        if not system_tracked_slots:
            return

        def contain(
            object_references: list[ObjectReferenceInfo], arg_name: str
        ) -> bool:
            if any(
                ref.argument_definition.name == arg_name for ref in object_references
            ):
                return True
            return False

        tool_name = turn.get_tool_name()
        expression = turn.expression.expressions[0]
        if (
            tool_name in {ITERATION_TOOL, ENTITY_SELECTION_TOOL}
            and expression.exec_info
        ):
            for arg in system_tracked_slots:
                if contain(object_references, arg):
                    continue
                logger.info("Adding system tracked slots definitions in the prompt")
                object_references.append(
                    ObjectReferenceInfo(
                        **{
                            "reference": f"{self._state.var_mapping.mapping[f'x{turn.index}']}.{arg}",  # noqa
                            "value": RESOLUTION_ERR,
                            "value_type": ValueType.RESOLUTION_ERROR,
                            "argument_definition": self._get_arg_definition(arg),
                            "entity": self.current_task_schema.entity_name,
                            "active_intent": self.current_task_schema.tool_name,
                            "current_task_reference": self._state.var_mapping.mapping[
                                self._state.current_cmd_var
                            ],
                            "system_arg": True,
                        }
                    )
                )

    def _maybe_add_reference_carryover_instruction(
        self, turn: SystemTurn, entity_info: EntityInfo
    ) -> object:
        """If object reference instructions are enabled, this function adds a developer
        turn to instruct the model to use object references instead of values
        in follow-up API calls."""
        if not self._instruct_obj_reference or not entity_info.properties_displayed:
            return
        service = entity_info.task_schema.service
        arg_values: dict[SlotName, str] = self.conversation_state[service]
        object_references = []
        sys_tracked_slots = self._state.current_task_schema.system_tracked_slots or []
        for arg, val in arg_values.items():
            if arg in entity_info.properties_displayed:
                ref = ObjectReferenceInfo.model_validate(
                    {
                        "reference": f"{self._state.var_mapping.mapping[f'x{turn.index}']}.{arg}",
                        "value": val,
                        "value_type": self._get_value_type(arg, val),
                        "argument_definition": self._get_arg_definition(arg),
                        "entity": self.current_task_schema.entity_name,
                        "active_intent": self.current_task_schema.tool_name,
                        "current_task_reference": self._state.var_mapping.mapping[
                            self._state.current_cmd_var
                        ],
                        "system_arg": arg in sys_tracked_slots,
                    }
                )
                object_references.append(ref)
        self._maybe_add_system_tracked_slots_instructions(
            turn, object_references, sys_tracked_slots
        )
        dev_turn: str = self._entity_selection_developer_turn_template.get_prompt(
            object_references
        ).prompt
        dev_turn = dev_turn.strip()
        assert dev_turn
        if dev_turn:
            self._state.instructions.append(dev_turn)

    def _maybe_add_return_type_descriptions(self, turn: SystemTurn):
        """If `show_api_return_defs` is enabled, this flag appends a developer
        turn containing descriptions of entity properties that the user can
        request along with `say` usage instructions.
        """
        properties = self.current_task_schema.api_returns
        if not self._show_api_return_defs or not properties:
            return
        entity_name = self.current_task_schema.entity_name
        req = RequestableInfo.model_validate(
            {
                "current_cmd_variable": self._state.var_mapping.mapping[
                    self._state.current_cmd_var
                ],
                "properties": list(properties),
                "randomise": self._randomise_prompt_elements,
                "entity": snake_case(entity_name),
                "entity_var": self._state.var_mapping.mapping[f"x{turn.index}"],
            }
        )
        dev_turn = self._iteration_developer_turn_template.get_prompt(req)
        assert dev_turn
        self._state.instructions.append(dev_turn)

    def _display_selected_entity(self, turn: SystemTurn) -> str:
        """Display an entity using the entity name
        and properties other commands can reference."""

        if turn.get_tool_name() == ITERATION_TOOL:
            logger.warning("Rendering entity for auto-inserted call ")
        else:
            self._state.var_mapping.add_mapping(f"x{turn.index}")
        optional_properties, properties = get_displayed_properties(
            self.current_task_schema, self.conversation_state
        )
        entity_info = EntityInfo.model_validate(
            {
                "turn_or_api_call_idx": turn.index,
                "task_schema": self.current_task_schema,
                "command_collection": self._schema,
                "properties_displayed": properties + optional_properties,
            }
        )
        self._state.entity_info.append(entity_info)
        self._maybe_add_reference_carryover_instruction(turn, entity_info)
        name = snake_case(self.current_task_schema.entity_name)
        if self._randomise_prompt_elements:
            random.shuffle(properties)
        properties_str = ", ".join(properties)
        if optional_properties:
            if self._randomise_prompt_elements:
                random.shuffle(optional_properties)
            return f"{name}({properties_str}, optional={optional_properties})"
        return f"{name}({properties_str})"

    def _is_last_entity(self) -> bool:
        """Return true if a `next` instruction occurs
        before the next service change."""
        for turn in self._state.history[self._state.current_turn_pointer + 1 :]:
            if turn.author == "System":
                expr = turn.expression.expressions[0]
                match expr.tool:
                    case tool if tool == ITERATION_TOOL:
                        return bool(expr.exec_info)
                    case tool if tool not in SPECIAL_TOOLS:
                        return True
                    case _:
                        continue
        return True

    def _is_last_confirmation(self) -> bool:
        """Return `true` if a `confirm` statement
        follows before the next service change."""
        for turn in self._state.history[self._state.current_turn_pointer + 1 :]:
            if turn.author == "System":
                expr = turn.expression.expressions[0]
                match expr.tool:
                    case tool if tool == TRANSACTION_CONFIRMATION_TOOL:
                        return False
                    case tool if tool not in SPECIAL_TOOLS and tool != self.current_task_schema.tool_name:  # noqa
                        return True
                    case _:
                        continue
        return True

    def _display_entity(self, turn: SystemTurn) -> str:
        """Display the result of an iteration (ie call to `next` function). This works as follows:

        1. If the active service returned the entity display depends on
        the value of _show_api_return_defs. If `true`, a type annotation follows
        next and a developer turn containing properties descriptions and other instructions is
        inserted into the history:

            user: i need weather info for sebastopol on the 11th of this month.
            0 weather_1_get_weather(city = 'sebastopol', date = '11th of this month')
            1 show(x0)
            2 next(x0)  # type: forecast
            developer: to answer questions about 'humidity' (percentage humidity), 'wind'
             (wind speed in miles per hour) and 'temperature' (temperature in fahrenheit)
             pass the relevant property to `say` (eg say(x2.temperature))

        Otherwise, the instruction is annotated with relevant properties as shown below.

        2. If the service has changed, we simply display the relevant
        object properties alongside the statement.

            5 next(x1) // properties: humidity, temperature
            agent: ...
            user: ... // assume intent change
        """
        displ_expr = super()._display_expression(turn)
        if ITERATION_TOOL in turn.expression.expressions[0].exec_info:
            return self._display_selected_entity(turn)
        if self.current_task_schema.service == self.active_service:
            last_entity = self._is_last_entity()
            entity_name = self.current_task_schema.entity_name
            if last_entity:
                self._maybe_add_return_type_descriptions(turn)
            if self._show_api_return_defs and last_entity:
                return f"{displ_expr} # type: {snake_case(entity_name)}"
            return self.show_entity_properties(displ_expr, self.current_task_schema)
        else:
            return self.show_entity_properties(displ_expr, self.current_task_schema)

    @staticmethod
    def show_entity_properties(displ_expr: str, schema: ServiceCommand) -> str:
        """Show the properties that the user may request alongside the iteration
        instruction.

        Parameters
        ----------
        displ_expr
            The iteration instruction (eg next(x5))
        schema
            The schema of the active user intent.
        """
        assert schema.entity_name
        names = (
            [p.name for p in schema.api_returns]
            if schema.api_returns is not None
            else []
        )
        if names:
            return f"{displ_expr} // properties: {', '.join(names)} "
        return displ_expr

    def _display_transaction_result(self, turn: SystemTurn) -> str:
        """We do not render transaction results but we log the
        entities so that we can display task completion information
        in the prompt."""
        properties = self.current_task_schema.followup_metadata.get_args()
        api_call_idx = int(turn.expression[0].positional_args[0][1:])
        self._state.entity_info.append(
            EntityInfo.model_validate(
                {
                    "turn_or_api_call_idx": api_call_idx,
                    "task_schema": self.current_task_schema,
                    "properties_displayed": properties,
                    "command_collection": self._schema,
                    "transaction": True,
                }
            )
        )
        return super()._display_transaction_result(turn)

    def _display_property_reference_instruction(self, turn: SystemTurn) -> str:
        """We do not render the confirmation but may optionally display instructions
        containing requested slots definitions after the confirmation."""
        displ_expr = super()._display_expression(turn)
        if (
            self.current_task_schema.service == self.active_service
            and self._is_last_confirmation()
        ):
            self._maybe_add_property_reference_instructions(turn)
            self._maybe_add_confirmed_optional_carryover_instructions(turn)
        return displ_expr

    def _maybe_add_property_reference_instructions(self, turn: SystemTurn):
        """If the `_display_properties_after_confirmation` flag is enabled, the
        requestable properties definitions along with instructions regarding object
        references are displayed after the confirmation.


        Example
        -------
        user: i'll need to make a transfer please.
        3 account(account_type)
        developer: use the following object references, if relevant,
         in subsequent api function calls:
        - `x3.account_type` to refer to the user's account type
        4 banks_2_transfer_money(account_type = x3.account_type)
        agent: how much and to whom?
        user: i want to send $30 to jasbir.
        5 x4.transfer_amount = '$30'; x4.recipient_name = 'jasbir'
        agent: confirming a $30 transfer from savings to the checking account of jasbir.
        user: thanks. how long will that take?
        6 confirm(x4)
        developer: unless a signal indicates banks_2_transfer_money calling error with no
         alternative arrangements, the properties
        - transfer_time: number of days for the transfer to go through
        may be communicated to the user upon their request by referencing x4 while calling `say`
         (eg, say(x4.transfer_time)).
        7 perform(x4)
        8


        """
        properties = self.current_task_schema.api_returns
        if not self._display_properties_after_confirmation or not properties:
            return
        entity_name = self.current_task_schema.entity_name
        relevant_entities = [
            entity
            for entity in self._state.entity_info
            if entity.task_schema.service == self.active_service
            and not entity.transaction
        ]
        if relevant_entities:
            if self.active_service in ENTITY_REF_COMMAND:
                variable = self._state.current_cmd_var
            else:
                last_entity = relevant_entities[-1]
                variable = f"x{last_entity.turn_or_api_call_idx}"
        else:
            # can happen either due to error during runtime or if questions are
            # asked about the entity specified for a transactional intent (eg
            #  ask the name of a song you just played -> see 83_00050)
            variable = self._state.current_cmd_var
        req = RequestableInfo.model_validate(
            {
                "entity_var": f"{self._state.var_mapping.mapping[variable]}",
                "current_cmd_variable": self._state.var_mapping.mapping[
                    self._state.current_cmd_var
                ],
                "properties": list(properties),
                "randomise": self._randomise_prompt_elements,
                "entity": snake_case(entity_name),
                "active_tool": self.current_task_schema.tool_name,
            }
        )
        dev_turn = self._confirmation_developer_turn_template.get_prompt(req)
        assert dev_turn
        self._state.instructions.append(dev_turn)

    def _maybe_add_confirmed_optional_carryover_instructions(self, turn: SystemTurn):
        """If the `instruct_confirmed_optional_reference` flag is enabled, an instruction
         to carry-over arguments confirmed (but not mentioned) by the user to
         future calls to the same service is displayed.

        Example
        -------
        agent: if you're serious about it, confirm me your request:
            you want a table for 2 at 71 saint peter in san jose.
            you'll be there on march 11th at 2 pm
        user: i confirm what you just said
        15 confirm(x14)
        developer: in the event of a `restaurants_2_reserve_restaurant` failure,
            pass the following keywords (confirmed by the user) to
            subsequent `restaurants_2_reserve_restaurant` calls:
        - number_of_seats = x14.number_of_seats
        developer: unless a signal indicates a `restaurants_2_reserve_restaurant`
            calling error with no alternative arrangements, the properties
        - has_seating_outdoors: whether the restaurant has outdoor seating available
        - has_vegetarian_options: whether the restaurant has adequate vegetarian options
        - phone_number: phone number to contact restaurant
        - rating: average user rating for restaurant on a scale of 5
        - address: address of restaurant
        - price_range: price range for the restaurant
        - category: the category of food offered by the restaurant
        may be communicated to the user upon their request by referencing `x13`
            while calling `say` (eg, say(x13.has_seating_outdoors, x13.has_vegetarian_options)).
        16 signal: there was an error while completing the task
        agent: the reservation was not possible, ask me any other thing
        user: i really want to eat there so try to see if they have
            a free table at 10:30 in the morning
        17 restaurants_2_reserve_restaurant(
            date = x14.date, location = x14.location, number_of_seats = x14.number_of_seats,
            restaurant_name = x14.restaurant_name, time = '10:30 in the morning'
        )
        agent: confirm me this second attempt. you want a table for 2 at 71 saint peter restaurant
         located in san jose. you want to eat there on march 11th at 10:30 am
        user: yup, proceed now
        """

        task_schema = self._state.current_task_schema
        ongoing_transaction = task_schema.is_transactional
        if not self._instruct_confirmed_optional_reference or not ongoing_transaction:
            return
        sys_confirmed = task_schema.system_confirmed_slots or []
        confirmed_optionals = {
            arg.name for arg in task_schema.optional_slots if arg.name in sys_confirmed
        }
        state = self._state.state[task_schema.service]
        # instruct the model only if the user did not already mention this argument
        to_instruct = []
        for arg in confirmed_optionals:
            if arg not in state or (
                arg in state and state[arg] == ValueType.SYSTEM_ARGUMENT.value
            ):
                to_instruct.append(arg)
        if bool(to_instruct):
            refs = []
            for arg in to_instruct:
                kw = f"{arg} = {self._state.var_mapping.mapping[self._state.current_cmd_var]}.{arg}"
                refs.append(
                    ObjectReferenceInfo.model_validate(
                        {
                            "reference": kw,
                            "value": "",
                            "value_type": None,
                            # nb: this is likely a hack (?)
                            "entity": task_schema.tool_name,
                            "system_arg": False,
                            "current_task_reference": self._state.var_mapping.mapping[
                                self._state.current_cmd_var
                            ],
                            "active_intent": task_schema.tool_name,
                        }
                    )
                )
            dev_turn = self._confirmed_arguments_instructions_developer_turn_template.get_prompt(
                refs
            ).prompt
            assert dev_turn
            self._state.instructions.append(dev_turn)


ConversationHistoryFormatterType = (
    ConversationHistoryFormatter | RenderedEntitiesConversationHistoryFormatter
)
