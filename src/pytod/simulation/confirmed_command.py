#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from copy import deepcopy
from typing import Generic, NamedTuple, Optional, Type

from pytod.command import ServiceCommand
from pytod.execution.policy_utils import HintGenerator, RecommendedAction
from pytod.pytod_types.aliases import DefaultValue, DialogueID, SlotName
from pytod.simulation._command_utils import APICallStatus
from pytod.simulation.api_driver import APIDriver, TransactionResult
from pytod.simulation.command import (
    WILDCARD_VALUE,
    Command,
    CommandT,
    SlotState,
    SlotValue,
    State,
    _init_args,
)
from pytod.simulation.command_argument import CommandArgument
from pytod.simulation.entities import Entity, get_entity

logger = logging.getLogger(__name__)


class ConfirmedCommandState(State):
    pass


class ConfirmedCommandFeedback(NamedTuple):
    entity: Entity


class ConfirmedCommandArgument(CommandArgument, Generic[SlotValue]):
    """Descriptor for ConfirmedCommand arguments, which stores arguments that
    were not confirmed upon agent confirmation request."""

    def __set__(self, instance: CommandT, value: SlotValue):
        if self._name in instance.__dict__:
            dial_id = instance._dialogue_id
            cmd_name = instance.get_full_command_name(snake_cased=False)
            logger.debug(
                f"{dial_id} || {cmd_name} || "
                f"Logging slot. User updated parameter instead of confirming: {self._name}."
            )
            instance._log_arg_correction(self._name)
        else:
            # the agent requests confirmation for optional slots which
            # may not have been specified. We log this as a correction
            # even though it is the first time the user mentions it
            if (
                self._name in instance._args_for_confirmation
                and self._name in instance.keyword_args
                and instance._keywords_state[self._name] == SlotState.NOTSET
                and instance._requested_confirmation
            ):
                instance._log_arg_correction(self._name)
        super().__set__(instance, value)


class ConfirmedCommand(Command):
    state = ConfirmedCommandState()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)

        # the agent notified the user the command failed
        #  to execute and suggested values for some slots
        self._offered_alternative = False
        # the agent notified the user the command failed
        # to execute without proposing an alternative
        self._notified_failure = False
        # if the agent has requested the user to
        #   confirm the arguments at the end of slot filling
        self._requested_confirmation = False
        # if the user confirmed the command should be
        #  executed upon agent request
        self._user_confirmed = False
        # the response of the last API call made
        # by the command
        self._api_response = {}
        # the arguments which the agent will confirm with the
        # user before executing a transaction for the first time
        self._args_for_confirmation: tuple[SlotName] = ()
        # the arguments the user corrected
        self._corrected_args: list[SlotName] = []
        # each entry represents the correction for a given turn
        self._corrections_history: list[list[SlotName]] = []
        self._entity_confirmation_requests = 0
        # arguments the agent communicates to the user when
        # offering an alternative following an execution failure
        self._alternative_offer: tuple[SlotName] = ()
        # the slots and values the agent has proposed as an alternative
        # following an unsuccessful API call
        self._current_alternative_offer: dict[SlotName, SlotValue] = {}
        # set during build
        self._schema = None

    def __getattr__(self, item: str) -> Optional[SlotValue]:
        """Handle access to undefined attributes."""
        if item in self._api_response:
            return self._api_response[item]
        dial_id = self._dialogue_id
        cmd = self.get_full_command_name(snake_cased=False)
        logger.warning(
            f"{dial_id} || {cmd} || Attempted reading unknown attribute: {item}."
        )
        self._unknown_properties_accessed.append(item)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __confirm__(self):
        # set keyword args to defaults if they are
        # not provided before confirmation
        for arg in self.keyword_args:
            if self._keywords_state[arg] == SlotState.NOTSET:
                default_value = self.keyword_args[arg]
                if default_value != WILDCARD_VALUE:
                    setattr(self, arg, default_value)
                    self._keywords_state[arg] = SlotState.SET

        self._user_confirmed = True
        # automatically set the values to the values proposed by the aget
        # if the user confirms the command
        if self._offered_alternative:
            for arg, arg_value in self._current_alternative_offer.items():
                setattr(self, arg, arg_value[0])
                assert arg not in self.keyword_args

    def get_entity_obj(self):
        """Return an entity object containing the dialogue state.
        This is returned only in situations where an iteration is
        attempted on a confirmed command."""
        properties = {
            arg: values[-1] for arg, values in self.state["slot_values"].items()
        }
        for property_ in self.schema.result_slots:
            if property_ not in properties:
                properties[property_] = None
        entity = get_entity(
            self.entity_name,
            attributes=properties,
            cmp_attributes=None,
        )
        return ConfirmedCommandFeedback(entity=entity)

    def _log_arg_correction(self, arg: SlotName):
        """Log arguments that the user provides a different value
        for upon confirmation"""
        self._corrected_args.append(arg)

    def _maybe_recommend_confirmation(self) -> Optional[list[RecommendedAction]]:
        """Recommend slot or entity confirmation.

        Notes
        -----
        1. Entity confirmation when::

            - the user has not yet confirmed

            - the user retries execution following a failure notification

        2. Slot re-confirmation

            - when the user changes the slot instead of confirming its value. If a
            single slot is changed, an additional slot (randomly chosen) is confirmed.
            If two or more slots are changed, only slots changed are re-confirmed. Slot
            changes can follow confirmation requests made when the user re-tries execution
            following a failure notification.
        """
        # user has confirmed - command should be executed
        if self._user_confirmed:
            return
        dial_id = self._dialogue_id
        cmd_name = self.get_full_command_name(snake_cased=False)
        if not self._requested_confirmation and self._corrected_args:
            logger.warning(
                f"{dial_id} || {cmd_name} || "
                f"User corrected arguments before confirmation: {self._corrected_args}"
            )
            self._forget_corrections()
        match n_corrections := len(self._corrected_args):
            # slot filling complete, confirm all the relevant slots
            case 0:
                self._requested_confirmation = True
                self._entity_confirmation_requests += 1
                hints = [
                    self._hint_generator.get_confirmation_hint(arg)
                    for arg in self._args_for_confirmation
                ]
            # user changed one argument, policy is to reconfirm it alongside a
            # randomly chosen argument
            case 1:
                # if the command is not executed successfully and the user changes
                # some arguments as a result, all the arguments should be re-confirmed
                if self._notified_failure:
                    hints = [
                        self._hint_generator.get_confirmation_hint(arg)
                        for arg in self._args_for_confirmation
                    ]
                # otherwise, if the user corrected an argument instead of confirming it,
                # only a subset of arguments are reconfirmed
                else:
                    confirmation_choices = [
                        arg
                        for arg in self._args_for_confirmation
                        if arg not in self._corrected_args
                    ]
                    match len(confirmation_choices):
                        case 0:
                            # this should not be possible since at least one arg
                            # should be confirmed
                            raise AssertionError
                        # if there is only one other slot in the confirmed list,
                        # then both slots are confirmed
                        case 1:
                            hints = [
                                self._hint_generator.get_confirmation_hint(arg)
                                for arg in self._args_for_confirmation
                            ]
                        case _:
                            hints = [
                                self._hint_generator.get_confirmation_hint(
                                    self._corrected_args[0]
                                )
                            ]
            # user changed two arguments, policy is to confirm the new values for both
            case 2:
                # reconfirm the entire transaction if the use retries following a failure
                if self._notified_failure:
                    hints = [
                        self._hint_generator.get_confirmation_hint(arg)
                        for arg in self._args_for_confirmation
                    ]
                # reconfirm only changes if the user changed arguments
                # instead of confirming their value
                else:
                    hints = [
                        self._hint_generator.get_confirmation_hint(arg)
                        for arg in self._corrected_args
                    ]
            case _:
                # same policy as for case 2
                if self._notified_failure:
                    hints = [
                        self._hint_generator.get_confirmation_hint(arg)
                        for arg in self._args_for_confirmation
                    ]
                else:
                    dial_id = self._dialogue_id
                    cmd_name = self.get_full_command_name(snake_cased=False)
                    logger.warning(
                        f"{dial_id} || {cmd_name} || "
                        f"User corrected {n_corrections} arguments: {self._corrected_args}."
                    )
                    hints = [
                        self._hint_generator.get_confirmation_hint(arg)
                        for arg in self._corrected_args
                    ]
        self._forget_corrections()
        self._reset_failure_state()
        return [RecommendedAction(dialog=hint) for hint in hints]

    def _forget_corrections(self):
        """Forget corrections the user has made.
        This allows the agent to deal with situations where the user
        changes parameters upon confirmation request in multiple turns."""
        if self._corrected_args:
            self._corrections_history.append(deepcopy(self._corrected_args))
            self._corrected_args = []

    def _reset_failure_state(self):
        """Reset the failure state.
        This allows the agent to reconfirm only relevant slots if the
        user changes parameters when the agent confirms the entity
        during an attempt to re-execute a failed command.
        """
        self._notified_failure = False

    def recommend_action(self) -> Optional[list[RecommendedAction]]:
        """Recommend slot-filling or confirmation actions."""
        slot_filling_actions = super().recommend_action()
        # all slots have been filled
        if slot_filling_actions is None:
            if not self._offered_alternative:
                confirmation_actions = self._maybe_recommend_confirmation()
                return confirmation_actions
        return slot_filling_actions

    def recommend_alternative(
        self, availability_info: Optional[dict[SlotName, list[SlotValue]]]
    ) -> Optional[list[RecommendedAction]]:
        """Recommend an alternative if the transaction called with the args user provided
        failed to execute."""
        if availability_info is None:
            return
        hints = [
            self._hint_generator.get_alternative_hint(slot_name=arg)
            for arg in self._alternative_offer
        ]
        self._offered_alternative = True
        return [RecommendedAction(dialog=hint) for hint in hints]

    def perform(
        self, endpoint: APIDriver
    ) -> TransactionResult | list[RecommendedAction]:
        """Execute a command following user confirmation, optionally providing alternatives
        if the API call fails with the user specified arguments."""
        assert (
            endpoint is not None
        ), f"Something went wrong, performed {self.name} ({self.service}) without endpoint."
        response = self._execute_transaction(endpoint)
        self._api_response = response.response or {}
        self._executed = True
        match response.status:
            case APICallStatus.SUCCESS:
                return response
            case APICallStatus.FAILURE:
                self._notified_failure = True
                alternative = self.recommend_alternative(response.alternative)
                if alternative is None:
                    self._user_confirmed = False
                else:
                    self._current_alternative_offer = response.alternative
                    return alternative
                return response

    @staticmethod
    def _remove_wildcard_values(parameters: dict[SlotName, SlotValue | DefaultValue]):
        """The API simulation ignores wildcards, so we remove wildcard parameters before
        calling the API."""
        return {
            par: value for par, value in parameters.items() if value != WILDCARD_VALUE
        }

    def _execute_transaction(self, endpoint: APIDriver) -> TransactionResult:
        """Call an endpoint with the arguments provided by the user so far."""
        parameters = self._remove_wildcard_values(self.get_api_call_parameters())
        normalised_params = self._normalizer.normalise(self._service, parameters)
        self.log_normalisation(normalised_params)
        parameters = normalised_params.result
        parameters.update({"dialogue_id": self._dialogue_id})
        return endpoint(parameters)

    @classmethod
    def build(
        cls: Type[CommandT],
        dialogue_id: DialogueID,
        command_schema: Optional[ServiceCommand] = None,
        service_schemata: Optional[list[ServiceCommand]] = None,
        hint_generator: Optional[HintGenerator] = HintGenerator(),
    ) -> CommandT:
        """Build a Command instance for the current dialogue session
        using the schema and the entity name.

        Parameters
        ----------
        dialogue_id
            The ID of the current dialogue session.
        command_schema
            The command schema, parsed from the augmented SGD schema.
        service_schemata
            Schema of other commands from the same service.
        hint_generator
            An object the command uses to generate hints that guide the
            user-agent interaction.
        """

        command = cls(dialogue_id)
        # NB: normally this would not be exposed, but we
        #  do so to facilitate experimentation with
        #  different ways to control the model
        command._hint_generator = hint_generator
        if command_schema is not None:
            if (system_confirmed_args := command_schema.system_confirmed_slots) is None:
                command._args_for_confirmation = ()
            else:
                command._args_for_confirmation = tuple(system_confirmed_args)
            if (
                alternative_offer_args := command_schema.alternative_transaction
            ) is None:
                command._alternative_offer = ()
            else:
                command._alternative_offer = alternative_offer_args
            command._entity_name = command_schema.entity_name
            command._service = command_schema.service
            _init_args(command, command_schema, service_schemata)
        if command._positional_args:
            assert (
                command._hint_generator is not None
            ), "Expected a hint generator for commands with positional arguments"
        return command
