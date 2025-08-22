#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from typing import Generic, Type, cast

from pytod.evaluation.helpers import ValueProcessor
from pytod.pytod_types.aliases import SlotName
from pytod.sgd_metadata import UNANNOTATED_INT_SLOTS
from pytod.simulation.command import WILDCARD_VALUE, CommandT, SlotState, SlotValue

logger = logging.getLogger(__name__)


class CommandArgument(Generic[SlotValue]):
    """Descriptor base class for all command arguments."""

    def __set_name__(self, owner: Type[CommandT], name: SlotName):
        self._name = name

    def __get__(
        self, instance: CommandT | None, owner: Type[CommandT]
    ) -> SlotValue | None:
        logger.debug(f"Calling descriptor __get__: {self._name}")
        if instance is None:
            raise AttributeError(
                "Slot descriptors are defined for command instances only."
            )
        try:
            return cast(SlotName, instance.__dict__[self._name])
        except KeyError:
            dial_id = instance._dialogue_id
            cmd_name = instance.get_full_command_name(snake_cased=False)
            logger.debug(
                f"{dial_id} || {cmd_name} || Attribute {self._name} has not been set."
            )
            return

    def __set__(self, instance: CommandT, value: SlotValue | int):
        """Set the value of a command argument."""
        is_valid = self.value_is_valid(instance, value)
        if not is_valid:
            value = self.maybe_constrain(instance, value)
        if self.value_is_valid(instance, value):
            if self._name in instance.keyword_args:
                instance._keywords_state[self._name] = SlotState.SET
            # the evaluator is fuzzy so non-categorical slots
            # will be correctly evaluated even if it is lower cased.
            # However, the categorical values are cased and compared by
            # exact match, so we restore their case when storing the
            # argument value in the Command object. The API/database
            # simulators lowercase the parameters internally.
            cased_value = ValueProcessor.restore_case(value, instance.service)
            logger.debug(f"Setting {self._name} attribute")
            instance.__dict__[self._name] = cased_value
        else:
            dial_id = instance._dialogue_id
            logger.warning(
                f"{dial_id}: Invalid value `{value}` for argument `{self._name}`. "
                "Maybe the argument name is incorrect?"
            )

    def value_is_valid(self, instance: CommandT, value: SlotValue) -> bool:
        """Validate predicted categorical values against valid values."""
        if value == WILDCARD_VALUE:
            return True
        if value is None:
            return False
        arg_def = instance.schema.get_argument_definition(self._name)
        digit_map = ValueProcessor.DIGIT_NORMALISATION_MAP
        if arg_def.is_categorical:
            possible_vals = set(arg_def.possible_values).union(
                {v.lower() for v in arg_def.possible_values}
            )
            match arg_def.data_type:
                case "int" | "enum":
                    return str(value) in possible_vals
                case "bool":
                    return str(value) in {"true", "false", "True", "False"}
                case _:
                    raise NotImplementedError(
                        f"Validation is not implemented for type: {arg_def.data_type}"
                    )
        elif arg_def.name in UNANNOTATED_INT_SLOTS:
            try:
                _ = int(value)
                return True
            except ValueError:
                if str(value) in digit_map or str(value) in digit_map.values():
                    return True
                return False
        else:
            if arg_def.data_type == "str":
                if value in digit_map:
                    logger.warning(
                        "Invalid value for str argument. "
                        f"Got value: {arg_def.name}={value}"
                    )
                    return False
        return True

    def maybe_constrain(
        self, instance: CommandT, value: int | str | None
    ) -> str | None:
        """Constrain categorical values to be one of the correct categories.
        For now this is backed by a simple lookup table.
        """
        if value is None:
            return value
        arg_def = instance.schema.get_argument_definition(self._name)
        if (
            arg_def.is_categorical and arg_def.data_type == "int"
        ) or arg_def.name in UNANNOTATED_INT_SLOTS:
            if isinstance(value, int):
                assert arg_def.data_type == "int"
                return str(arg_def.default_value)
            elif isinstance(value, float):
                logger.warning(f"Incorrect float type for argument: {self._name}")
                return str(int(value))
            return ValueProcessor.map_to_digit(value)
        return value
