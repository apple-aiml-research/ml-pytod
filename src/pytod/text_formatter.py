#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pytod.command import (
    CommandCollection,
    ServiceCall,
    ServiceCallResult,
    ServiceCommand,
)
from pytod.pytod_types.aliases import ArgumentValue
from pytod.pytod_types.sgd_conversation import Author, Conversation, Turn
from pytod.sgd_agent_behaviour import agent_notifies_failure
from pytod.sgd_user_behaviour import is_req_alts, is_req_alts_with_changes
from pytod.utils import snake_case

logger = logging.getLogger(__name__)

_MISSING_SPAN_MARKER = "#missing_span"

_RETURN_ATTRIBUTES = ["call", "response"]
ArgMismatch = namedtuple("ArgMismatch", _RETURN_ATTRIBUTES)


class ServiceCallFormatter(ABC):
    @abstractmethod
    def call_to_text(self, call: ServiceCall, **kwargs: Any) -> str:
        """Convert a `ServiceCall` object to text."""

    @abstractmethod
    def command_to_syntax(self, command: ServiceCall) -> str:
        """Syntax documentation for the given command call, showing arguments and types."""


class ServiceCallResultFormatter(ABC):
    @abstractmethod
    def command_output_to_text(
        self, command_output: ServiceCallResult, **kwargs: Any
    ) -> str:
        """Convert a `ServiceCallResult` object to text."""


class SGDServiceCallResultFormatter(ServiceCallResultFormatter):
    """A simple formatter that formats as a line with `result: ` or `error: `.

    Note this does not look at the action_result coming from the iOS backend.
    """

    default_filtering_opts = OmegaConf.create(
        {
            "call_params": False,
            "opt_slots": False,
        }
    )

    def __init__(self, format_config: dict):
        self.prefix = format_config.get("prefix", "results: ")
        self.error_prefix = format_config.get("error_prefix", "error: ")
        self.failure_message = format_config.get("failure_message", "")
        # set options for filtering results
        filtering_defaults = OmegaConf.create(
            SGDServiceCallResultFormatter.default_filtering_opts
        )
        filtering_opts = format_config.get("filtering_opts", {})
        self.filtering_opts = OmegaConf.merge(filtering_defaults, filtering_opts)
        self.index_results = format_config.get("index_results", False)
        self.max_results = format_config.get("max_results", 1)

    def command_output_to_text(
        self, command_output: ServiceCallResult, **kwargs: Any
    ) -> str:
        command_output = deepcopy(command_output)
        n_results_shown = kwargs.get("n_results_shown", 0)
        # used to display subsequent results when the user requests alternatives
        # without updating constraints
        result_index = kwargs.get("result_index", 0)
        if command_output.is_error:
            return f"{self.error_prefix} {command_output.error_message}"

        # filter records to prevent summary hallucination
        self._filter_record(command_output, self.filtering_opts, **kwargs)
        text_results: list[str] = []
        for i, result in enumerate(
            command_output.service_results[
                result_index : result_index + self.max_results
            ]
        ):
            if self.index_results:
                entity_id_offset = kwargs.get("entity_id_offset", 0)
                result.update({"_id": str(i + 1 + n_results_shown + entity_id_offset)})
            text_results.append(
                f"{self.prefix}{json.dumps(result, indent=None, ensure_ascii=False)}"
            )
        assert text_results
        if len(text_results) > 1:
            return "\n".join(text_results)
        else:
            return text_results[0]

    @staticmethod
    def _filter_record(
        command_output: ServiceCallResult, to_filter: DictConfig, **kwargs: Any
    ):
        """In-place filtering of raw SGD call result. The raw annotations always
        contain the call parameters as well as optional slots with their default
        values if they have not been mentioned by the suer. This can affect the
        summarisation, as the model will mention these entities as relating to the
        user intention.

        Args:
            to_filter: what information should be removed from the SGD call result.

        """

        if not to_filter:
            return
        call: ServiceCall = kwargs.get("service_call")

        if to_filter.call_params:
            for arg_name in call.parameters:
                for result in command_output.service_results:
                    if arg_name in result:
                        result.pop(arg_name)

        if to_filter.opt_slots:
            command_collection: CommandCollection = kwargs.get("command_collection")
            assert command_collection is not None, (
                "Cannot filter optional slots from service call results w/o access to "
                "agent command collection!"
            )
            opt_arg_spec = command_collection.get(
                call.service, call.method
            ).optional_slots
            for spec in opt_arg_spec:
                for result in command_output.service_results:
                    if spec.name in result:
                        result.pop(spec.name)


def _escape_newlines(text: str) -> str:
    return re.sub(r"(?<!\\)(\n)", r"\\n", text)


class JsonServiceCallFormat(ServiceCallFormatter):
    """Uses a JSON format where params are flattened into the object.

    Args:
        json_schema_syntax: whether to use JSON schema in the command syntax.
    """

    def __init__(self, json_schema_syntax: bool = False):
        self._use_json_schema_syntax = json_schema_syntax

    def call_to_text(self, call: ServiceCall, **kwargs: Any) -> str:
        json_obj: dict[str, ArgumentValue] = {"method": call.method}
        json_obj.update(
            {arg: value for arg, value in call.parameters.items() if value is not None}
        )
        return json.dumps(json_obj, indent=None, ensure_ascii=False)

    def command_to_syntax(self, command: ServiceCall) -> str:
        raise NotImplementedError


def command_to_text(command: ServiceCommand) -> str:
    """Converts a command to a function signature."""
    req_args = [arg.name for arg in command.required_slots]
    req_str = ", ".join(req_args) if req_args else ""
    opt_args = [f"{arg.name}{': Optional'}" for arg in command.optional_slots]
    opt_str = ", ".join(opt_args) if opt_args else ""
    if not req_str and not opt_str:
        arg_str = ""
    else:
        arg_str = f"{req_str}, {opt_str}".strip(
            ","
        ).strip()  # if req_str is '', remove leading ","
        arg_str = arg_str.strip().strip(
            ","
        )  # if there are no opt args, then remove trailing ", "

    return f"{command.name}({arg_str}) # {command.description.lower()}"


def maybe_quote_or_cast(value: str) -> int | str | bool:
    """Quote values so that the generated calls are AST-parseable strings."""
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        value_int = int(value)
        return value_int
    except ValueError:
        try:
            assert '"' not in value
        except AssertionError:
            logger.warning(f"Value _{value}_ contained quotes...")
            value = value.replace('"', "'")
            logger.info(f"Reformatted value to _{value}_")
        value = f'"{value}"'
    return value


class PythonFunctionServiceCallFormatter(ServiceCallFormatter):
    def __init__(self, format_config: dict):
        self.convert_camel_case = format_config.get("convert_camel_case", False)
        self.apply_quotes = format_config.get("quote_values", False)
        self.skip_quote = format_config.get("skip_quote", set())

    def transform_method_name(self, name: str) -> str:
        return snake_case(name)

    def call_to_text(self, call: ServiceCall, **kwargs: Any) -> str:
        call_params_str = ""
        for arg, value in call.parameters.items():
            if value:
                if self.apply_quotes and arg not in self.skip_quote:
                    if value:
                        call_params_str += f"{arg}={maybe_quote_or_cast(value)}, "
                else:
                    call_params_str += f"{arg}={value}, "
            else:
                call_params_str += f"{arg}, "
        if call_params_str:
            call_params_str = call_params_str[:-2]
        if self.convert_camel_case:
            method = f"{snake_case(call.method)}"
        else:
            method = f"{call.method}"
        call_str = f"{method}({call_params_str})"
        return call_str

    def command_to_syntax(self, command: ServiceCall) -> str:
        raise NotImplementedError


class TextFormatter:
    """Class for converting to/from text formats."""

    def __init__(
        self,
        system_name: str = "agent",
        user_name: str = "user",
        service_call_formatter: ServiceCallFormatter | None = None,
        service_results_formatter: SGDServiceCallResultFormatter | None = None,
        command_collection: CommandCollection | None = None,
        index_user_turns: bool = False,
        **kwargs: Any,
    ) -> None:
        self.system_name = system_name
        self.user_name = user_name
        self.index_user_turns = index_user_turns
        self.command_formatter = service_call_formatter
        if service_call_formatter is None:
            logger.warning(
                "No service call formatter specified. "
                "Service calls will not be included after relevant system turns"
            )

        self.command_output_formatter = service_results_formatter
        if service_results_formatter is None:
            logger.warning(
                "No service call results formatter specified. "
                "Results will not be included in relevant system turns "
            )
        self.command_collection = command_collection

    def _sys_turn_to_text_pieces(self, turn: Turn, **kwargs: DictConfig) -> list[str]:
        assert turn.author == Author.SYSTEM
        pieces = []
        n_results_shown = kwargs.get("n_results_shown", 0)

        def has_results(turn: Turn) -> bool:
            return turn.service_results is not None

        if self.command_formatter is not None:
            if turn.service_call is not None:
                pieces.append(
                    f"{self.system_name}: {self.command_formatter.call_to_text(turn.service_call)}"
                )
        if self.command_output_formatter is not None:
            display_next_result = kwargs.get("display_next_result", False)
            if has_results(turn):
                try:
                    assert not display_next_result
                except AssertionError:
                    # in this case, the user requested alternatives but there are no
                    # more results. For some reason, there is a call annotation in
                    # these turns, and so the failure message for no results will get
                    # displayed
                    assert agent_notifies_failure(turn)

                cmd_output_text = self.command_output_formatter.command_output_to_text(
                    turn.service_results,
                    service_call=turn.service_call,
                    n_results_shown=n_results_shown,
                    command_collection=self.command_collection,
                    entity_id_offset=kwargs.get("entity_id_offset", 0),
                )
                pieces.append(f"{cmd_output_text}")
            # frame annotations contain all API calls but we only disp. the top result
            # in conversation. When the user requests alternatives, we backtrack
            # to the last turn annotated with results and pick the next result
            elif display_next_result:
                if turn.service_call is not None:
                    cmd_output_text = (
                        self.command_output_formatter.command_output_to_text(
                            turn.service_results,
                            service_call=turn.service_call,
                            n_results_shown=n_results_shown,
                            command_collection=self.command_collection,
                            entity_id_offset=kwargs.get("entity_id_offset", 0),
                        )
                    )
                    pieces.append(f"{cmd_output_text}")
                else:
                    pieces = self._handle_request_alternative_turns(
                        turn,
                        n_results_shown,
                        conversation_history=kwargs.get("conversation_history", []),
                        next_result_index_lookup=kwargs.get(
                            "next_result_index_lookup", {}
                        ),
                        entity_id_offset=kwargs.get("entity_id_offset", 0),
                    )

        if turn.text:
            pieces.append(f'{self.system_name}: "{_escape_newlines(turn.text)}"')
        return pieces

    def _handle_request_alternative_turns(
        self, turn: Turn, n_results_shown: int, **kwargs
    ) -> list[str]:
        """Display API call and second, third, ... result if the user requests alternatives."""

        def get_prev_results_turn(current_turn: Turn, history: list[Turn]) -> Turn:
            """Backtrack to the previous turn where the API was called."""

            services = list(current_turn.system_actions.keys())
            assert all(
                (history, current_turn.service_results is None, len(services) == 1)
            )
            current_service = services[0]
            while history:
                turn = history.pop()
                if turn.author == Author.SYSTEM and turn.service_results is not None:
                    assert turn.service_call.service == current_service
                    return turn

        pieces = []
        prev_results_turn = get_prev_results_turn(
            turn,
            kwargs.get("conversation_history", []),
        )
        current_service = prev_results_turn.service_call.service
        assert self.command_formatter is not None
        pieces.append(
            f"{self.system_name}: "
            f"{self.command_formatter.call_to_text(prev_results_turn.service_call)}"
        )
        # KeyError should never be raised
        start_index = kwargs.get("next_result_index_lookup", {})[current_service]
        cmd_output_text = self.command_output_formatter.command_output_to_text(
            prev_results_turn.service_results,
            service_call=prev_results_turn.service_call,
            n_results_shown=n_results_shown,
            command_collection=self.command_collection,
            result_index=start_index,
            entity_id_offset=kwargs.get("entity_id_offset", 0),
        )
        pieces.append(f"{cmd_output_text}")
        return pieces

    def turn_to_text(self, turn: Turn, **kwargs: Any) -> str | list[str]:
        if turn.author == Author.USER:
            text = turn.text or ""
            return f'{self.user_name}: "{_escape_newlines(text)}"'

        assert turn.author == Author.SYSTEM
        return self._sys_turn_to_text_pieces(turn, **kwargs)

    @staticmethod
    def _update_start_index_lookup(turn: Turn, lookup: defaultdict[str, int]) -> bool:
        """SGD search API results can contain multiple results.

        When the user requests alternatives without updating results, we need to know
        which result to display next, possibly for multiple services."""

        services = list(turn.user_actions.keys())
        # this happens when the user wants to switch task -
        # at this point results retrieval is not necessary
        if len(services) > 1:
            return False
        current_service = services[0]

        # if the user requests alternatives w/o updating constraints,
        # increment the index to display the next result
        if is_req_alts(turn) and not is_req_alts_with_changes(turn):
            lookup[current_service] += 1
            return True
        return False

    @staticmethod
    def _maybe_reset_next_result_idx(turn: Turn, lookup: defaultdict[str, int]):
        """If a new API call is made, and we had previously displayed the next results, we need
        to reset the index back to zero to ensure correct results are displayed when the next call
        is made."""
        # reset the next result index because there was another API call
        if turn.author == Author.SYSTEM:
            if turn.service_call is not None and turn.service_call.service in lookup:
                lookup[turn.service_call.service] = 0
        else:
            services = list(turn.user_actions.keys())
            if len(services) > 1:
                return
            current_service = services[0]
            # if the user updates the values s.t. there will be a new API call
            # reset next result index to 0
            if current_service in lookup and is_req_alts_with_changes(turn):
                lookup[current_service] = 0

    @staticmethod
    def _missing_span(turn: Turn) -> bool:
        return _MISSING_SPAN_MARKER in turn.text

    def conversation_to_text(self, conversation: Conversation, **kwargs: Any) -> str:
        lines = []
        # tracks which result is to be displayed next for each service
        next_result_index = defaultdict(int)
        # how many results we have included in the conversation history so far
        results_shown_so_far = 0
        display_next_result, has_missing_span = False, False
        entity_id_offset = 0
        for current_turn_idx, turn in enumerate(conversation.turns, start=1):
            if not has_missing_span and self._missing_span(turn):
                has_missing_span = True
                entity_id_offset += kwargs.get("entity_id_offset", 0)
            current_turn_idx = (
                current_turn_idx + kwargs.get("index_offset", 0)
                if has_missing_span
                else current_turn_idx
            )
            index = f"[{current_turn_idx // 2 + 1}] " if self.index_user_turns else ""
            self._maybe_reset_next_result_idx(turn, next_result_index)
            if turn.author == Author.USER:
                lines.append(f"{index}{self.turn_to_text(turn, **kwargs)}")
                display_next_result = self._update_start_index_lookup(
                    turn, next_result_index
                )
                if display_next_result:
                    assert not current_turn_idx == 0
            else:
                if has_missing_span:
                    conversation_history = conversation.turns[
                        : current_turn_idx - kwargs.get("index_offset")
                    ]
                else:
                    conversation_history = conversation.turns[:current_turn_idx]
                for piece in self.turn_to_text(
                    turn,
                    n_results_shown=results_shown_so_far,
                    turn_idx=current_turn_idx,
                    entity_id_offset=entity_id_offset,
                    conversation_history=conversation_history,
                    next_result_index_lookup=dict(next_result_index),
                    display_next_result=display_next_result,
                ):
                    lines.append(f"{' ' * len(index)}{piece}")
                if display_next_result and not agent_notifies_failure(turn):
                    results_shown_so_far += 1
            if (
                self.command_output_formatter is not None
                and turn.service_results is not None
            ):
                results_shown_so_far += min(
                    len(turn.service_results.service_results),
                    self.command_output_formatter.max_results,
                )

        return "\n".join(filter(lambda x: x != "", lines))


def get_response_mismatched_args(
    call: ServiceCall, result: ServiceCallResult
) -> dict[str, ArgMismatch]:
    """Transactional calls annotations may contain responses even the agent
    notifies the user the call was unsuccessful.

    In this case, the agent _may_ inform the user of alternative
    bookings arrangements that can be made. The alternative is sometimes
    annotated by providing a slot from the API schema with a different value
    to the call.

    Notes
    -----
    It's not clear if the annotations for providing transactions alternatives are
    fully correct. In `1_00108` (train) the agent turn

    "I'm sorry, I couldn't make the reservation. Is there anything else I could help you with?"

    is annotated with a response where {"time": "11:30"} whereas the call has {"time": "11:15"}.
    The same applies to other dialogues (eg `1_00002`). Meanwhile, in `1_00009` the system does
    offer a different booking time and the calls are annotated in the same way as for `1_00108`.
    Meanwhile, in `1_00015` there are no results and the system does not offer alternative
    arrangements.

    To ensure correctness, one should detect turns where the agent offers alternatives when
    transactions are unsuccessful using both `NOTIFY_FAILURE` and `OFFER` dialogue acts, and not
    by looking at the `service_results` field alone.
    """

    assert len(result.service_results) == 1
    [response] = result.service_results
    mismatches = {}
    for arg, value in call.parameters.items():
        if arg in response and value != response[arg]:
            mismatches[arg] = ArgMismatch(call=value, response=response[arg])
    return mismatches
