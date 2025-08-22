#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Type

from omegaconf import DictConfig

from pytod.command import ServiceCommand
from pytod.pytod_types.aliases import APIDescription, ToolName
from pytod.pytod_types.pytod import Intent
from pytod.sgd_invocations import (
    CacheItem,
    DeterministicSeenInvocationsSampler,
    InvocationsCache,
    SlotCoverageSampler,
)
from pytod.toolbox.toolbox import get_command_name, get_entity_name
from pytod.utils import save_json, typed_partial

logger = logging.getLogger(__name__)


class IntentFormatter:
    def __init__(self, config: DictConfig | None = None):
        self._config = config

    def __call__(self, intent: Intent, **kwargs) -> APIDescription:
        lines = [f"{intent.name}( # {intent.description}"]
        for slot in intent.slots:
            if slot.type == "enum":
                lines.append(
                    f"{slot.name}: {slot.type}, choices: {slot.choices} # {slot.description}"
                )
            else:
                lines.append(f"{slot.name}: {slot.type}, # {slot.description}")
        if intent.result:
            lines.append(f") -> {intent.result.type}(")
            for property in intent.result.properties:
                if property.type == "enum":
                    lines.append(
                        f"{property.name}: {property.type}, "
                        f"choices: {property.choices} # {property.description}"
                    )
                else:
                    lines.append(
                        f"{property.name}: {property.type}, # {property.description}"
                    )
            lines.append(")")
        else:
            lines.append(")")

        return "\n".join(lines)


class ServiceCommandFormatter:
    def __init__(self, config: DictConfig | None = None):
        self._config = config
        self.randomise_prompt_elements = True
        self.mark_optional_slots = False
        self._lines: list[str] = []
        if config is not None:
            toolbox_config = config.toolbox
            command_name_config_params = {
                "skip_service_variations": toolbox_config.skip_service_variations,
                "use_snake_case": toolbox_config.use_snake_case,
                "anonymize_service": toolbox_config.anonymize_service,
            }
            self.get_command_name = partial(
                get_command_name, **command_name_config_params
            )
            self.get_entity_name = partial(
                get_entity_name, **{"use_snake_case": toolbox_config.use_snake_case}
            )
            self.randomise_prompt_elements = config.randomise_prompt_elements
            self.mark_optional_slots = config.mark_optional_slots
        else:
            self.get_command_name = lambda intent_name, service_name: intent_name
            self.get_entity_name = lambda name: name

    def reset(self):
        self._lines = []

    def __call__(self, command: ServiceCommand, **kwargs) -> APIDescription:
        cmd_name = self.get_command_name(
            intent_name=command.name, service_name=command.service
        )
        self._lines = [f"{cmd_name}( # {command.description}"]
        required = list(command.required_slots)
        optional = list(command.optional_slots)
        if self.randomise_prompt_elements:
            random.shuffle(required)
            random.shuffle(optional)
        for slot in required + optional:
            type_annotation = slot.data_type
            if self.mark_optional_slots and slot in optional:
                type_annotation = f"optional[{type_annotation}]"
            if slot.data_type == "enum":
                possible_vals = list(slot.possible_values)
                if self.randomise_prompt_elements:
                    random.shuffle(possible_vals)
                choices = " | ".join(possible_vals)
                self._lines.append(
                    f"{slot.name}: {type_annotation}, choices: {choices} # {slot.description}"
                )
            else:
                self._lines.append(
                    f"{slot.name}: {type_annotation}, # {slot.description}"
                )
        self.add_return_types(command)
        api_str = "\n".join(self._lines)
        self.reset()
        return api_str

    def add_return_types(self, command: ServiceCommand):
        entity_name = self.get_entity_name(command.entity_name)
        if command.api_returns:
            api_returns = list(command.api_returns)
            if self.randomise_prompt_elements:
                random.shuffle(api_returns)
            self._lines.append(f") -> {entity_name}(")
            for property in api_returns:
                self._lines.append(
                    f"{property.name}: {property.data_type}, # {property.description}"
                )
            self._lines.append(")")
        else:
            self._lines.append(f") -> {entity_name}()")


class HiddenReturnTypeServiceCommandFormatter(ServiceCommandFormatter):
    def add_return_types(self, command: ServiceCommand):
        entity_name = self.get_entity_name(command.entity_name)
        self._lines.append(f") -> {entity_name}(...) // ... marks hidden properties")


class APIUsageCommandFormatter(HiddenReturnTypeServiceCommandFormatter):
    template = "Example #{i}:\nuser: {query}\n{start_ndx} {expression}"

    def __init__(self, config: DictConfig | None = None):
        super().__init__(config)
        schema = config.schema
        split = schema.split
        tool_name_formatter: Type[get_command_name] = typed_partial(
            get_command_name,
            skip_service_variations=config.toolbox.skip_service_variations,
            use_snake_case=config.toolbox.use_snake_case,
            anonymize_service=config.toolbox.anonymize_service,
        )
        assert split in ["train", "dev", "test"], f"Unknown split {split}"
        self._split = split
        # sampler that allows us to prompt with seen prompts from training
        # during dev/test inference for seen intents
        self._train_sampler = DeterministicSeenInvocationsSampler(
            Path(config.train_sampled_invocations_path)
        )
        cache = InvocationsCache(split, tool_name_formatter=tool_name_formatter)
        cache.load(Path(config.cache_path))
        self._sampler = SlotCoverageSampler(
            cache, schema, randomise=config.randomise_prompt_elements
        )
        self._store_samples = split == "train"
        self._first_turn_only = config.first_turn_only
        self._examples_buffer: dict[ToolName, list[list[CacheItem]]] = defaultdict(list)

    def _sample_examples(
        self, intent: ToolName, n_source_turns: int | None
    ) -> list[CacheItem] | None:
        if n_source_turns is None and self._first_turn_only:
            raise ValueError(
                "Number of source turns unknown but API invocations"
                " are required for first turn only!"
            )
        # for intents seen in training, we sample a prompt that was seen during training
        if self._split in ["dev", "test"]:
            if (maybe_train_sample := self._train_sampler.sample(intent)) is not None:
                if self._first_turn_only:
                    if n_source_turns == 1:
                        return maybe_train_sample
                    return []
                return maybe_train_sample
        if self._first_turn_only:
            if n_source_turns == 1:
                examples = self._sampler.sample(intent)
                if self._store_samples and examples:
                    self._examples_buffer[intent].append(examples)
                if not examples:
                    logger.warning(f"No invocation examples for {intent}")
                return examples
            return []
        return self._sampler.sample(intent)

    def format_api_usage_examples(
        self, items: list[CacheItem] | None, transcript_start_ndx: int | None
    ) -> str:
        usage_examples = "Usage examples:\n\n"
        for i, item in enumerate(items):
            assert len(item) == 2
            example = self.template.format(
                **{
                    "i": i + 1,
                    "query": item[0]["query"],
                    "start_ndx": transcript_start_ndx,
                    "expression": item[1]["expression"],
                }
            )
            usage_examples += f"{example}\n"
        return usage_examples

    def __call__(self, command: ServiceCommand, **kwargs) -> APIDescription:
        transcript_start_ndx = kwargs.get("transcript_start_ndx")
        assert (
            transcript_start_ndx is not None
        ), "Can't format examples without transcript start index"
        api_def = super().__call__(command)
        examples: list[CacheItem] = self._sample_examples(
            command.tool_name, kwargs.get("n_source_turns")
        )
        if not examples:
            return api_def
        api_usage = self.format_api_usage_examples(examples, transcript_start_ndx)
        return f"{api_def}\n\n{api_usage}"

    def save_sampled_invocations(self, odir: Path):
        if not self._examples_buffer:
            logger.warning("No examples have been buffered, there is nothing to save!")
        save_json(self._examples_buffer, odir / "seen_invocations.json")


IntentFormatterType = (
    ServiceCommandFormatter
    | IntentFormatter
    | HiddenReturnTypeServiceCommandFormatter
    | APIUsageCommandFormatter
)
