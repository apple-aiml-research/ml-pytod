#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import random
from copy import copy
from typing import Callable

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel

from pytod.command import (
    ArgumentDefinition,
    CommandCollection,
    ServiceCommand,
    get_arg_description,
    get_arg_type,
)
from pytod.interpreter.metadata import NLG_CALL_TOOL
from pytod.prompting.pytod_text2text_header_formatter_utils import (
    CompletedTaskStack,
    ObjectReferenceInfo,
    RequestableInfo,
    StackItem,
    ValueType,
)
from pytod.prompting.template_factories import (
    complete_transaction_template_factory,
    complete_transaction_template_factory_resolved_carryover_arg_values,
    developer_turn_confirmed_argument_instructions,
    developer_turn_following_confirmation,
    developer_turn_following_iteration_list,
    developer_turn_following_selection,
    hidden_descriptions_display,
    selected_entity_template_factory,
    selected_entity_template_factory_resolved_carryover_arg_values,
    shown_descriptions_display,
    task_stack_template_factory,
)
from pytod.prompting.utils import TemplateMixin
from pytod.utils import random_subset


class DeveloperTurn(BaseModel):
    prompt: str


class StackedQuery(BaseModel):
    prompt: str


class StackedTransaction(BaseModel):
    prompt: str


class ObjectRefFormatter:
    @staticmethod
    def format(info: ObjectReferenceInfo) -> str:
        match info.value_type:
            case ValueType.WILDCARD if not info.system_arg:
                descr = info.argument_definition.description.lower()
                slot = info.reference.split(".")[1]
                return f"the command reference '{info.current_task_reference}.{slot}' to refer to {descr}"  # noqa
            case ValueType.CATEGORICAL | ValueType.RESOLUTION_ERROR | ValueType.PUBLIC_VARIABLE | ValueType.SYSTEM_ARGUMENT | ValueType.WILDCARD:  # noqa
                descr = info.argument_definition.description.lower()
                return f"'{info.reference}' to refer to {descr}"
            case ValueType.NON_CATEGORICAL:
                return f"'{info.reference}' instead of '{info.value.lower()}'"
            case _:
                raise TypeError(f"Unknown value type: {info.value_type}")


class StackDocsTemplate(TemplateMixin):
    def get_prompt(self, item: StackItem) -> str:
        return self._template.render(item=item, undefined=StrictUndefined)


class DisplayedDocsTemplate(StackDocsTemplate):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = shown_descriptions_display,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        environment.filters["get_arg_type"] = get_arg_type
        environment.filters["get_arg_description"] = get_arg_description
        self._set_template_and_variables(environment)


class HiddenDocsTemplate(StackDocsTemplate):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = hidden_descriptions_display,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        environment.filters["maybe_add_arg_type"] = self.maybe_add_arg_type

        self._set_template_and_variables(environment)

    def maybe_add_arg_type(
        self, attributes: list[str], task_schema: ServiceCommand
    ) -> list[str]:
        if len(attributes) > 1:
            return attributes
        arg_type = get_arg_type(attributes[0], task_schema)
        return [f"{attributes[0]}: {arg_type}"]


class CompleteQueryTemplate(TemplateMixin):
    """Template rendered when a non-transactional intent is completed."""

    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = selected_entity_template_factory,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        self._hidden_docs_template = HiddenDocsTemplate(schema)
        self._displayed_docs_template = DisplayedDocsTemplate(schema)
        self._default_order = ["_hidden_docs_template", "_displayed_docs_template"]
        environment.filters["docs_formatter"] = self.docstring_formatter
        self._set_template_and_variables(environment)

    def docstring_formatter(self, item: StackItem) -> list[str]:
        templates = [getattr(self, t) for t in copy(self._default_order)]
        if item.randomise:
            item.shuffle()
            random.shuffle(templates)
        prompts = [template.get_prompt(item) for template in templates]
        return prompts

    def get_prompt(self, item: StackItem) -> StackedQuery:
        template_vars = {"item": item, "triple_quote": self._get_triple_quote()}
        for var in self._template_variables:
            try:
                template_vars[var] = getattr(item, var)
            except AttributeError:
                if var not in template_vars:
                    raise AttributeError(f"Could not find variable: {var}")
        prompt = self._template.render(**template_vars, undefined=StrictUndefined)
        return StackedQuery.model_validate({"prompt": prompt})


class CompleteTransactionTemplate(CompleteQueryTemplate):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = complete_transaction_template_factory,
    ):
        super().__init__(schema, template_factory=template_factory)


class SelectedEntityDeveloperTurnTemplate(TemplateMixin):
    """Template for developer instruction that may be displayed as soon as
    the agent generates a `select` expression."""

    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = developer_turn_following_selection,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        environment.filters["object_ref_formatter"] = ObjectRefFormatter.format
        environment.filters["extract_variable"] = lambda x: x.split(".")[0]
        self._set_template_and_variables(environment)

    @staticmethod
    def _get_object_references(request: list[ObjectReferenceInfo]):
        return request

    def get_prompt(self, request: list[ObjectReferenceInfo]) -> DeveloperTurn:
        template_vars = self._get_template_variables(request)
        prompt = self._template.render(**template_vars, undefined=StrictUndefined)
        return DeveloperTurn.model_validate({"prompt": prompt})


class ConfirmedArgumentInstructionsDeveloperTurnTemplate(
    SelectedEntityDeveloperTurnTemplate
):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[
            [], str
        ] = developer_turn_confirmed_argument_instructions,
    ):
        super().__init__(schema, template_factory=template_factory)

    @staticmethod
    def _get_transaction(request: list[ObjectReferenceInfo]):
        return request[0].entity


class IterationDeveloperTurnTemplate(TemplateMixin):
    """Template for developer instructions which may be displayed as soon as
    the agent uses the `next` expression to iterate through entities returned
    by a call to the database."""

    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = developer_turn_following_iteration_list,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        environment.filters["inline_requestables"] = self._inline_requestables
        environment.filters["prop_formatter"] = self._prop_formatter
        self._set_template_and_variables(environment)

    @staticmethod
    def _inline_requestables(properties: list[ArgumentDefinition]) -> str:
        return ", ".join(
            [f"'{prop.name}' ({prop.description})" for prop in properties]
        ).lower()

    @staticmethod
    def _prop_formatter(prop: ArgumentDefinition) -> str:
        return f"{prop.name}: {prop.description}".lower()

    @staticmethod
    def _get_requestable_slots(request: RequestableInfo) -> list[ArgumentDefinition]:
        return request.properties

    @staticmethod
    def _get_nlg_call_tool(request: RequestableInfo) -> str:
        return NLG_CALL_TOOL

    @staticmethod
    def _get_entity_info(request: RequestableInfo) -> str:
        assert request.entity is not None, request.entity_var is not None
        return f" the '{request.entity}' object ({request.entity_var})"

    @staticmethod
    def _get_usage_example(request: RequestableInfo) -> str:
        if request.randomise:
            n_items = min(len(request.properties), 2)
            if n_items == 2:
                n_items = 1 if random.random() > 0.5 else n_items
                if len(request.properties) > 3:
                    n_items = 3 if random.random() > 0.8 else n_items
            example = random_subset(list(request.properties), n_items=n_items)
        else:
            example = request.properties[:2]
        pos_args = ", ".join([f"{request.entity_var}.{e.name}" for e in example])
        return f"{NLG_CALL_TOOL}({pos_args})"

    def get_prompt(self, request: RequestableInfo) -> str:
        template_vars = self._get_template_variables(request)
        if template_vars["requestable_slots"]:
            return self._template.render(**template_vars, undefined=StrictUndefined)
        return ""


class ConfirmationPropertyListingDeveloperTurnTemplate(IterationDeveloperTurnTemplate):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = developer_turn_following_confirmation,
    ):
        super().__init__(schema, template_factory=template_factory)

    @staticmethod
    def _get_transaction(request: RequestableInfo) -> str:
        return request.active_tool

    @staticmethod
    def _get_object_ref(request: RequestableInfo) -> str:
        return request.entity_var


class CompletedTaskFormatter:
    def __init__(
        self,
        query_template: CompleteQueryTemplate,
        transaction_template: CompleteTransactionTemplate,
    ):
        self._query_template = query_template
        self._transaction_template = transaction_template

    def __call__(self, *args, **kwargs):
        assert len(args) == 1 and isinstance(args[0], StackItem)
        return self.format(*args)

    def format(self, item: StackItem) -> str:
        if item.task_schema.is_transactional:
            return self._format_transaction(item)
        return self._format_query(item)

    def _format_transaction(self, item: StackItem) -> str:
        return self._transaction_template.get_prompt(item).prompt

    def _format_query(self, item: StackItem) -> str:
        return self._query_template.get_prompt(item).prompt


class CompletedTasksTemplate(TemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        value_object_references: bool = True,
        template_factory: Callable[[], str] = task_stack_template_factory,
    ):
        """

        Parameters
        ----------
        schema
        value_object_references
            If True, carried over values are displayed as attribute access on
            an existing entity or command. The language of the instructions
            displayed on task completion depends on this.
        template_factory
            A function which returns the template to be rendered and displayed
            at the start of the prompt.
        """
        super().__init__(schema, template_factory)
        environment = Environment()
        if value_object_references:
            stack_formatter = CompletedTaskFormatter(
                query_template=CompleteQueryTemplate(schema),
                transaction_template=CompleteTransactionTemplate(schema),
            )
        else:
            stack_formatter = CompletedTaskFormatter(
                query_template=CompleteQueryTemplate(
                    schema,
                    template_factory=selected_entity_template_factory_resolved_carryover_arg_values,
                ),
                transaction_template=CompleteTransactionTemplate(
                    schema,
                    template_factory=complete_transaction_template_factory_resolved_carryover_arg_values,  # noqa
                ),
            )
        environment.filters["completed_task_formatter"] = stack_formatter
        self._set_template_and_variables(environment)

    @staticmethod
    def _get_task_completion_stack(stack: CompletedTaskStack):
        return stack

    def get_prompt(self, stack: CompletedTaskStack) -> str:
        template_vars = self._get_template_variables(stack)
        prompt = self._template.render(**template_vars, undefined=StrictUndefined)
        return prompt
