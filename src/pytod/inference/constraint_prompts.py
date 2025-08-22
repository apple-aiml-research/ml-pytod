#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from dataclasses import dataclass
from typing import Any, Callable

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel

from pytod.command import ArgumentDefinition, CommandCollection
from pytod.evaluation.helpers import is_date, is_int
from pytod.inference.constraint_prompts_factories import (
    bool_enum_arg_value_template_factory,
    hallucinated_arg_name_template_factory,
    hallucinated_arg_name_with_cat_value_template_factory,
    memorised_arg_name_template_factory,
)
from pytod.inference.constraint_prompts_utils import (
    MQAChoice,
    _collect_arg_defs,
    _get_slot_map,
    build_categorical_choice_map,
    int2alpha,
)
from pytod.inference.constraint_request import (
    SlotConstraintRequest,
    ValueConstraintRequest,
)
from pytod.parser.expresion_validation_utils import ConstrainedKeyword
from pytod.prompting.utils import TemplateMixin
from pytod.sgd_metadata import UNANNOTATED_INT_SLOTS

logger = logging.getLogger(__name__)


class PendingConstraintException(Exception):
    """Raised by the schema validation parser
    if a call was issued for the KeywordConstraintAssistant
    and the update should not continue."""

    pass


@dataclass
class SlotDefinitionFormatter:
    ENUM_SEPARATOR = ", "
    # number of values displayed instead of a
    # slot definition
    MAX_ENUM_DISPLAYED = 4
    # number of example values displayed along with
    # the definition for enum slots
    MAX_EXAMPLE_VALUES = 2
    DESCRIBE_INT_ARGS = True

    @staticmethod
    def arg_name_with_description_or_type(
        argument_definition: ArgumentDefinition,
    ) -> str:
        """Format an argument schema as a string with the format::

                `{argument_name}: {data_type} | {argument_definition}`

        {data_type} is rendered if the argument type is int | bool | enum.
        {argument_definition} is rendered if the argument type is flot | str.
        """

        arg_name = argument_definition.name
        match (arg_type := argument_definition.data_type):
            case _ if "date" in arg_name:
                description = argument_definition.description.lower()
                return f"{arg_name}: {description} (date)".strip()
            case "int":
                if SlotDefinitionFormatter.DESCRIBE_INT_ARGS:
                    description = argument_definition.description.lower()
                    return f"{arg_name}: {description} (int)".strip()
                return f"{arg_name}: int".strip()
            case "bool":
                return f"{arg_name}: true|false".strip()
            case "enum":
                max_vals_displayed = SlotDefinitionFormatter.MAX_ENUM_DISPLAYED
                if len(argument_definition.possible_values) > max_vals_displayed:
                    description = argument_definition.description.lower()
                    return f"{arg_name}: {description}".strip()
                options = [opt.lower() for opt in argument_definition.possible_values]
                sep = SlotDefinitionFormatter.ENUM_SEPARATOR
                return f"{arg_name}: {f'{sep}'.join(options[:-1])} or {options[-1]}".strip()
            case type_ if type_ in {"str", "float"}:
                if arg_name in UNANNOTATED_INT_SLOTS:
                    if SlotDefinitionFormatter.DESCRIBE_INT_ARGS:
                        description = argument_definition.description.lower()
                        return f"{arg_name}: {description} (int)".strip()
                    return f"{arg_name}: int".strip()
                description = argument_definition.description.lower()
                return f"{arg_name}: {description}".strip()
            case _:
                raise TypeError(f"Unknown argument type: {arg_type}")

    @staticmethod
    def arg_name_with_description(argument_definition: ArgumentDefinition) -> str:
        """Format an argument schema as a string with the format::

        `{argument_name}: {argument_definition}`
        """
        arg_name = argument_definition.name
        description = argument_definition.description.lower()
        return f"{arg_name}: {description}".strip()

    @staticmethod
    def arg_definition(argument_definition: ArgumentDefinition) -> str:
        """Format an argument schema using the definition alone."""
        description = argument_definition.description.lower()
        return description.strip()

    @staticmethod
    def arg_definition_with_enums(argument_definition: ArgumentDefinition) -> str:
        """Format an argument schema using the definition and enum values."""
        description = argument_definition.description.lower()
        if argument_definition.data_type == "enum":
            max_examples = SlotDefinitionFormatter.MAX_EXAMPLE_VALUES
            vals = argument_definition.possible_values
            enum_string = f"(eg, {', '.join(vals[:max_examples])})".lower()
            return f"{description.strip()} {enum_string}"
        return description.strip()


constraint_environment = Environment()
constraint_environment.filters["int2alpha"] = int2alpha
constraint_environment.filters[
    "slot_definition_formatter"
] = SlotDefinitionFormatter().arg_name_with_description_or_type


class Prompt(BaseModel):
    prompt: str


class HallucinatedSlotConstraintPrompt(Prompt):
    slot_mapping: dict[MQAChoice, ConstrainedKeyword]


class ConstraintTemplateMixin(TemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = lambda: "",
        optimise_prompt: bool = True,
    ):
        super().__init__(schema, template_factory=template_factory)
        self._optimise_prompt = optimise_prompt

    def get_prompt(
        self, request: SlotConstraintRequest
    ) -> HallucinatedSlotConstraintPrompt:
        template_vars = self._get_template_variables(request)
        prompt = self._template.render(
            **template_vars,
            undefined=StrictUndefined,
        )
        return HallucinatedSlotConstraintPrompt.model_validate(
            {
                "prompt": prompt,
                "slot_mapping": self._get_slot_map(template_vars["slot_schemas"]),
            }
        )

    def _get_slot_map(
        self, slot_schemas: list[ArgumentDefinition] | dict[ArgumentDefinition, Any]
    ) -> dict[MQAChoice, ConstrainedKeyword]:
        raise NotImplementedError("Subclasses must implement _get_slot_map.")


class HallucinatedArgNameConstraintTemplate(ConstraintTemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = hallucinated_arg_name_template_factory,
        optimise_prompt: bool = True,
    ):
        super().__init__(
            schema, template_factory=template_factory, optimise_prompt=optimise_prompt
        )
        environment = Environment()
        environment.filters["int2alpha"] = int2alpha
        environment.filters[
            "slot_definition_formatter"
        ] = SlotDefinitionFormatter().arg_name_with_description_or_type
        self._set_template_and_variables(environment)

    def _get_slot_schemas(
        self, request: SlotConstraintRequest
    ) -> dict[ArgumentDefinition, MQAChoice]:
        """Retrieve the schemas of the slots which were not mentioned in the
        conversation so far."""
        return _collect_arg_defs(self._schema, request, self._optimise_prompt)

    def _get_slot_map(
        self, arg_defs: dict[ArgumentDefinition, MQAChoice]
    ) -> dict[MQAChoice, ConstrainedKeyword]:
        return _get_slot_map(arg_defs)

    @staticmethod
    def _get_predicted_argument(request: SlotConstraintRequest) -> str:
        """The slot name template contains the name of the slot predicted
        and sometimes the value or the data type.

        Notes
        -----
        1. Value is included if the predicted value is a categorical value
         (or part of one).
        2. Data type is included if the predicted value can be cast to int
        """
        if (predicted_value := request.predicted_slot_value) is None:
            return f"'{request.predicted_slot}'"
        predicted_slot = request.predicted_slot
        if is_int(predicted_value) and all(
            slot not in predicted_slot
            for slot in ("time", "hotel", "address", "destination")
        ):
            logger.info(
                f"Cast predicted value '{predicted_value}' for hallucinated slot "
                f"'{predicted_slot}' to int."
            )
            return f"'{predicted_slot}:int'"
        elif is_date(predicted_value):
            return f"'{predicted_slot}:date'"
        else:
            return f"'{predicted_slot}'"

    @staticmethod
    def _get_predicted_argument_with_definition(request: SlotConstraintRequest) -> str:
        """In some cases, it is possible to include a definition alongside the argument
        name. For example, if we are constraining a value object reference and the
        property referenced is equal to the keyword name and the keyword name is a member
        of the active service schema, then we can add the description of the keyword
        to the prompt to allow the model to better identify the relations."""
        self_ = HallucinatedArgNameConstraintTemplate
        predicted_argument_with_type = self_._get_predicted_argument(request)
        if (descr := request.predicted_slot_description) is None:
            return predicted_argument_with_type
        if ":int" in predicted_argument_with_type:
            predicted_argument_with_type = predicted_argument_with_type.replace(
                ":int", ""
            )
            predicted_argument_with_type += " (int)"
        return f"{predicted_argument_with_type}, defined as '{descr}'"


class HallucinatedArgNameCatValueConstraintTemplate(ConstraintTemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[
            [], str
        ] = hallucinated_arg_name_with_cat_value_template_factory,
        optimise_prompt: bool = True,
    ):
        super().__init__(
            schema, template_factory=template_factory, optimise_prompt=optimise_prompt
        )
        environment = Environment()
        environment.filters["int2alpha"] = int2alpha
        environment.filters[
            "slot_definition_formatter"
        ] = SlotDefinitionFormatter().arg_name_with_description
        self._set_template_and_variables(environment)

    @staticmethod
    def _get_predicted_keyword(request: SlotConstraintRequest) -> str:
        assert request.predicted_slot_value is not None
        return f"'{request.predicted_slot} = {request.predicted_slot_value}'"

    def _get_slot_schemas(
        self, request: SlotConstraintRequest
    ) -> list[ArgumentDefinition]:
        """Retrieve the schemas of the boolean and enum slots in the schema. If
        self._optimise_prompt is `True` then slots in `request.known_slots` are
        excluded."""

        assert not request.is_value_object_reference
        all_args = [
            arg
            for arg in self._schema.get_service_arg_definitions(request.service)
            if arg.data_type in {"bool", "enum"}
        ]
        if self._optimise_prompt:
            slot_schemas = [
                arg for arg in all_args if arg.name not in request.known_slots
            ]
        else:
            slot_schemas = all_args
        return slot_schemas if slot_schemas else all_args

    def _get_slot_map(
        self, slot_schemas: list[ArgumentDefinition]
    ) -> dict[MQAChoice, ConstrainedKeyword]:
        """Return a mapping from QA options to slot-value pairs."""
        return build_categorical_choice_map(slot_schemas)


class MemorisedArgumentNameConstraintTemplate(HallucinatedArgNameConstraintTemplate):
    def __init__(
        self,
        schema: CommandCollection,
        train_schema: CommandCollection,
        template_factory: Callable[[], str] = memorised_arg_name_template_factory,
        optimise_prompt: bool = True,
    ):
        super().__init__(
            schema, template_factory=template_factory, optimise_prompt=optimise_prompt
        )
        environment = Environment()
        environment.filters["int2alpha"] = int2alpha
        environment.filters[
            "slot_definition_formatter"
        ] = SlotDefinitionFormatter().arg_definition_with_enums
        self._set_template_and_variables(environment)
        self._train_schema = train_schema

    def _get_memorised_argument_description_with_name(
        self, request: SlotConstraintRequest
    ) -> str:
        arg_name = request.predicted_slot
        try:
            arg_def = self._train_schema.get_arg_schema(
                request.memorisation_info.services[0], arg_name
            )
        # this assertion occurs if we want to use the paraphrase
        # prompt to map value object reference to schema slots
        # in certain scenarios
        except AssertionError:
            arg_def = self._schema.get_arg_schema(
                request.memorisation_info.services[0], arg_name
            )
        return f"'{arg_def.description.lower()}' ({arg_name})"

    def _get_memorised_argument_description(
        self, request: SlotConstraintRequest
    ) -> str:
        arg_name = request.predicted_slot
        arg_def = self._train_schema.get_arg_schema(
            request.memorisation_info.services[0], arg_name
        )
        return f"'{arg_def.description.lower()}'"

    @staticmethod
    def _get_memorised_argument_name(request: SlotConstraintRequest) -> str:
        return f"'{request.predicted_slot}'"

    @staticmethod
    def _get_predicted_value(request: SlotConstraintRequest) -> str:
        return request.predicted_slot_value


class HallucinatedBoolEnumArgValueTemplate(ConstraintTemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = bool_enum_arg_value_template_factory,
        optimise_prompt: bool = True,
    ):
        super().__init__(
            schema, template_factory=template_factory, optimise_prompt=optimise_prompt
        )
        environment = Environment()
        environment.filters["int2alpha"] = int2alpha
        self._set_template_and_variables(environment)

    def _get_arg_def(self, request: ValueConstraintRequest) -> ArgumentDefinition:
        return self._schema.get_arg_schema(request.service, request.argument)

    @staticmethod
    def _get_predicted_keyword(request: ValueConstraintRequest) -> str:
        assert request.argument is not None
        return f"'{request.argument} = {request.value}'"

    def get_prompt(
        self, request: ValueConstraintRequest
    ) -> HallucinatedSlotConstraintPrompt:
        template_vars = self._get_template_variables(request)
        prompt = self._template.render(
            **template_vars,
            undefined=StrictUndefined,
        )
        return HallucinatedSlotConstraintPrompt.model_validate(
            {
                "prompt": prompt,
                "slot_mapping": self._get_slot_map([self._get_arg_def(request)]),
            }
        )

    def _get_slot_map(
        self, slot_schemas: list[ArgumentDefinition] | dict[ArgumentDefinition, Any]
    ) -> dict[MQAChoice, ConstrainedKeyword]:
        return build_categorical_choice_map(slot_schemas, include_none_choice=False)


ConstraintTemplate = (
    HallucinatedArgNameConstraintTemplate
    | HallucinatedArgNameCatValueConstraintTemplate
    | HallucinatedBoolEnumArgValueTemplate
    | MemorisedArgumentNameConstraintTemplate
)
