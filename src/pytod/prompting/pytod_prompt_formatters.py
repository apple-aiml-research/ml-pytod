#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Formatters used to create text-to-text examples for training purposes."""
from functools import singledispatchmethod

from omegaconf import DictConfig

from pytod.command import ServiceCommand
from pytod.prompting.pytod_conversation_history_formatters import (
    ConversationHistoryFormatter,
    FormattedConversationHistory,
    RenderedEntitiesConversationHistoryFormatter,
)
from pytod.prompting.pytod_header_formatters import IntentFormatterType
from pytod.prompting.pytod_target_formatters import FormattedCompletion, TargetFormatter
from pytod.prompting.pytod_text2text_formatters_utils import VariableMapper
from pytod.prompting.pytod_text2text_header_formatter_utils import (
    CompletedTaskStack,
    EntityInfo,
)
from pytod.prompting.templates import CompletedTasksTemplate
from pytod.prompting.text2text_example_parsers import (
    ConversationExample,
    InferenceConversationExample,
)
from pytod.pytod_types.aliases import APIDescription
from pytod.pytod_types.pytod import Intent

FormattedPrompt = str
FormattedExample = tuple[
    FormattedPrompt, FormattedCompletion, FormattedConversationHistory
]
FormattedInferenceExample = tuple[
    FormattedPrompt, FormattedCompletion | None, FormattedConversationHistory
]


class PromptFormatterBase:
    def __init__(self, config: DictConfig | None = None):
        self._config = config


class PromptFormatter(PromptFormatterBase):
    def __call__(
        self, conversation_history: list[str], intents: list[APIDescription], **kwargs
    ) -> str:
        history_str = "\n".join(conversation_history)
        intents[-1] = intents[-1].strip("\n")
        api_str = "\n".join([f"- {x}" for x in intents])
        return f"functions with optional arguments:\n{api_str}\n\nconversation:\n{history_str}".lower()  # noqa


class TaskStackPromptFormatter(PromptFormatterBase):
    def __init__(self, config: DictConfig | None = None):
        super().__init__(config)
        self.template = CompletedTasksTemplate(
            self._config.schema, value_object_references=config.value_object_references
        )

    def __call__(
        self,
        conversation_history: list[str],
        intents: list[APIDescription],
        entities: list[EntityInfo],
        mapper: VariableMapper,
        **kwargs,
    ):
        stack = CompletedTaskStack.build(
            entities, mapper, randomise=self._config.randomise_prompt_elements
        )
        header = self.template.get_prompt(stack)
        history_str = "\n".join(conversation_history)
        api_str = "\n".join([f"- {x}" for x in intents])
        return f"{header}\n\n{api_str}\n\nconversation:\n{history_str}".lower().strip()


PromptFormatterType = PromptFormatter | TaskStackPromptFormatter


class ConversationFormatter:
    def __init__(
        self,
        conversation_fmt: ConversationHistoryFormatter,
        prompt_fmt: PromptFormatter,
        target_fmt: TargetFormatter,
        intent_fmt: IntentFormatterType,
        config: DictConfig | None = None,
    ):
        self.conversation_fmt = conversation_fmt
        self.prompt_fmt = prompt_fmt
        self.target_fmt = target_fmt
        self.intent_fmt = intent_fmt
        self._config = config
        self._mapper: VariableMapper | None = None

    def reset(self):
        """Reset the state of all formatters."""
        self._mapper = None
        self.conversation_fmt.reset()

    @singledispatchmethod
    def format(
        self,
        conversation: ConversationExample,
        intents: list[Intent] | list[ServiceCommand],
    ) -> FormattedExample:
        """Formats the various conversation parts using specific formatters.
        Returns a triple with the formatting output of the prompt, target and conversation
        formatters, respectively.
        """
        raise NotImplementedError(
            f"Cannot format {conversation.__class__.__qualname__}"
        )

    @format.register
    def _(
        self,
        conversation: ConversationExample,
        intents: list[Intent] | list[ServiceCommand],
    ) -> FormattedExample:
        """Formats examples for training and next action prediction"""
        formatted_conversation, source = self._get_source(conversation, intents)
        target = self.target_fmt(conversation.target_turns, self._mapper)
        self.reset()
        return source, target, formatted_conversation

    @format.register
    def _(
        self,
        conversation: InferenceConversationExample,
        intents: list[Intent] | list[ServiceCommand],
    ) -> FormattedInferenceExample:
        formatted_conversation, source = self._get_source(conversation, intents)
        target = None
        if conversation.target_turns is not None:
            target = self.target_fmt(conversation.oracle_target_turns, self._mapper)
        self.reset()
        return source, target, formatted_conversation

    def _get_source(
        self,
        conversation: ConversationExample | InferenceConversationExample,
        intents: list[Intent] | list[ServiceCommand],
    ) -> tuple[list[str], str]:
        formatted_conversation, var_mapper = self.conversation_fmt(
            conversation.source_turns
        )
        formatted_apis: list[APIDescription] = [
            self.intent_fmt(
                intent,
                n_source_turns=len(conversation.source_turns),
                transcript_start_ndx=var_mapper.start_ndx,
            )
            for intent in intents
        ]
        self._mapper = var_mapper
        source = self.prompt_fmt(
            conversation_history=formatted_conversation, intents=formatted_apis
        )
        return formatted_conversation, source


class RenderedEntitiesConversationFormatter(ConversationFormatter):
    def __init__(
        self,
        conversation_fmt: RenderedEntitiesConversationHistoryFormatter,
        prompt_fmt: TaskStackPromptFormatter,
        target_fmt: TargetFormatter,
        intent_fmt: IntentFormatterType,
        config: DictConfig | None = None,
    ):
        self.conversation_fmt = conversation_fmt
        self.prompt_fmt = prompt_fmt
        self.target_fmt = target_fmt
        self.intent_fmt = intent_fmt
        self._config = config
        self._mapper: VariableMapper | None = None

    def _get_source(
        self,
        conversation: ConversationExample | InferenceConversationExample,
        apis: list[ServiceCommand],
    ) -> tuple[list[str], str]:
        self.conversation_fmt.active_service = conversation.service
        formatted_conversation, var_mapper = self.conversation_fmt(
            conversation.source_turns
        )
        self._mapper = var_mapper
        formatted_apis: list[APIDescription] = [
            self.intent_fmt(
                intent,
                n_source_turns=len(conversation.source_turns),
                transcript_start_ndx=var_mapper.start_ndx,
            )
            for intent in apis
        ]
        source = self.prompt_fmt(
            conversation_history=formatted_conversation,
            intents=formatted_apis,
            entities=self.conversation_fmt.entities,
            mapper=var_mapper,
        )
        # print(source)
        return formatted_conversation, source


ConversationFormatterType = (
    ConversationFormatter | RenderedEntitiesConversationFormatter
)
