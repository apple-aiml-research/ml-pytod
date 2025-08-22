#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import ast
import logging
import random
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Type

from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.command import CommandCollection, ServiceCall, ServiceCommand
from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import IntentName, SlotName
from pytod.pytod_types.sgd_conversation import Author, Turn, UserDialogueAct
from pytod.pytod_types.transcript import ProgramStatement
from pytod.sgd_user_behaviour import (
    get_turns_matching_service_intent,
    user_states_intention,
)
from pytod.text_formatter import PythonFunctionServiceCallFormatter
from pytod.toolbox.toolbox import get_command_name
from pytod.utils import load_json, nested_defaultdict, save_json

USER_INTENT = "user_intention"

logger = logging.getLogger(__name__)


def get_call_turn_hash(turn: Turn) -> Optional[tuple[str]]:
    """Hash the turns according to the slots mentioned by the user."""
    assert turn.author == Author.USER
    if not user_states_intention(turn):
        return
    to_hash = []
    all_actions = chain(*turn.user_actions.values())
    slot_providing_acts = [UserDialogueAct.SELECT, UserDialogueAct.INFORM]
    for a in all_actions:
        match a.act:
            case UserDialogueAct.INFORM_INTENT:
                to_hash.append(USER_INTENT)
            case slot_providing_act if slot_providing_act in slot_providing_acts:
                if a.slot:
                    to_hash.append(a.slot)
            case _:
                continue

    return tuple(sorted(to_hash)) or None


CacheItem = tuple[
    Optional[dict[Literal["author", "query"], str]],
    dict[Literal["author", "index", "expression"], str],
]
ToolName = str
# a hash containing the slots that have been mentioned in the current turn
InvocationHash = tuple[SlotName]


class InvocationsCache:
    """Catch diverse invocation examples on the fly."""

    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        *,
        tool_name_formatter: Type[get_command_name],
    ):
        self.cache: defaultdict[
            ToolName, defaultdict[InvocationHash, list[CacheItem]]
        ] = nested_defaultdict(list, depth=2)
        self.tool_name_formatter = tool_name_formatter
        self._user_turn: Optional[Turn] = None
        self._service: Optional[str] = None
        self._missing_invocations: list[ToolName] = []
        self.tool_name_to_service: dict[str, str] = {}
        self.tool_name_to_intent: dict[str, str] = {}
        self._split = split
        use_snake_case = tool_name_formatter.keywords["use_snake_case"]
        self._command_formatter = PythonFunctionServiceCallFormatter(
            {"convert_camel_case": use_snake_case, "quote_values": True}
        )

    @property
    def user_turn(self):
        return self._user_turn

    @user_turn.setter
    def user_turn(self, turn: Turn):
        assert turn.author == Author.USER
        self._user_turn = turn

    @user_turn.deleter
    def user_turn(self):
        self._user_turn = None

    @property
    def service(self):
        return self._service

    @service.setter
    def service(self, service: str):
        self._service = service

    @service.deleter
    def service(self):
        self._service = None

    def maybe_cache_instruction(self, element: ProgramStatement, item: CacheItem):
        """Creates a hash for the current program statement if it
        represents an invocation."""

        call_turn_tags = [
            "call",
            "call_with_params",
        ]
        service = self.service
        intent = self.user_turn.dialogue_state[service].active_intent
        tool_name = self.tool_name_formatter(service_name=service, intent_name=intent)
        self.tool_name_to_service[tool_name] = service
        self.tool_name_to_intent[tool_name] = intent
        match element.tag:
            case tg if tg in call_turn_tags:
                assert self.service is not None
                assert self.user_turn is not None
                hash_ = get_call_turn_hash(self.user_turn)
                self.cache[tool_name][hash_].append(item)
            case _:
                # we only cache example invocations for now
                pass

    def harvest_from_index(self, iterator: SGDIterator, index_key: list[str]):
        def format_as_task_start(service: str, intent: str) -> str:
            return f"{service}({intent}"

        def get_item(turn: Turn, service: str, tool_name: str) -> CacheItem:
            if len(turn.user_actions) > 1:
                print("check id", id)
            assert turn.author == Author.USER
            item = [{"query": f"{turn.text}", "author": "User"}]
            param_dict = {}
            if UserDialogueAct.INFORM in turn.user_actions_dict[service]:
                inform_actions = turn.user_actions_dict[service][UserDialogueAct.INFORM]
                param_dict = {a.slot: a.values[0] for a in inform_actions}
            service_call_dict = {
                "method": tool_name,
                "parameters": param_dict,
                "service": "",
            }
            expression = self._command_formatter.call_to_text(
                ServiceCall.model_validate(service_call_dict)
            )
            item.append({"expression": expression, "author": "System", "index": 0})
            return item

        will_harvest = False
        self._missing_invocations = deepcopy(self._sampler.missing_invocations)
        if self._missing_invocations:
            logger.info(
                "Harvesting additional invocations from multi-domain conversations"
            )
            will_harvest = True

        while self._missing_invocations:
            next_harvested = self._missing_invocations.pop()
            service, intent = (
                self.tool_name_to_service[next_harvested],
                self.tool_name_to_intent[next_harvested],
            )
            for _, conversation in iterator.index_iterator(
                self._split,
                index_key=index_key,
                service=service,
                startswith_task=format_as_task_start(service, intent),
            ):
                logger.debug(
                    f"Harvesting invocations from leaf {index_key} for {service}.{intent}"
                )
                harvested_turns = get_turns_matching_service_intent(
                    conversation, service, intent, stop_at_first_turn=False
                )
                items = [
                    get_item(turn, service, next_harvested) for turn in harvested_turns
                ]
                turn_hashes = [get_call_turn_hash(turn) for turn in harvested_turns]
                for item, hash_ in zip(items, turn_hashes):
                    self.cache[next_harvested][hash_].append(item)
        if will_harvest:
            logger.info("Harvesting complete, adding remaining invocations to toolbox")

    def save(self, path: Path):
        o_path = path / "invocations_cache.json"
        logger.info(f"Saving invocations cache at {o_path}")
        serialised = {
            "tools": {},
            "tool_name_to_service": self.tool_name_to_service,
            "tool_name_to_intent": self.tool_name_to_intent,
        }
        for tool_name, examples in self.cache.items():
            serialised["tools"][tool_name] = {str(k): v for k, v in examples.items()}
        save_json(serialised, o_path)

    def load(self, path: Path):
        logger.info(f"Loading invocations cache from path {path}")
        data = load_json(path)
        deserialised = {}
        for tool_name, examples in data["tools"].items():
            deserialised[tool_name] = {}
            for k, v in examples.items():
                if k == "None":
                    logger.warning(f"Ignoring cache key None for tool {tool_name}")
                    continue
                deserialised[tool_name][ast.literal_eval(k)] = v
        self.cache = deserialised
        self.tool_name_to_intent = data["tool_name_to_intent"]
        self.tool_name_to_service = data["tool_name_to_service"]


class DiverseInvocationsSampler:
    def __init__(self, cache: InvocationsCache):
        self._cache: dict[ToolName, dict[InvocationHash, list[CacheItem]]] = cache.cache
        self.missing_invocations: list[ToolName] = []

    @staticmethod
    def random_pop(elements: list[Any]) -> Any:
        assert elements, "Nothing left to sample"
        random_index = random.randrange(len(elements))
        return elements.pop(random_index)

    def select_k_random_subset(self, elements: list[str], k: int) -> list[str]:
        return random.sample(elements, k)

    def sample(
        self, intent: IntentName, n_samples: int | None
    ) -> list[CacheItem] | None:
        sampled = []
        if intent not in self._cache:
            logger.warning(f"No examples invocations for intent {intent} were found")
            self.missing_invocations.append(intent)
            return
        this_intent_cache = self._cache[intent]
        assert this_intent_cache
        # more samples than unique keys
        if n_samples is None:
            n_samples = len(this_intent_cache)
        if n_samples > len(this_intent_cache):
            while True:
                for slot_hash, invocations in this_intent_cache.items():
                    sampled.append(list(self.random_pop(invocations)))
                    n_samples -= 1
                self.drop_empty_leaves(this_intent_cache)
                if n_samples == 0 or not this_intent_cache:
                    break
            return sampled
        slot_hashes = self.select_k_random_subset(
            list(this_intent_cache.keys()), n_samples
        )
        for s_hash in slot_hashes:
            invocations = this_intent_cache[s_hash]
            sampled.append(self.random_pop(invocations))
        return sampled

    def drop_empty_leaves(
        self, intent_invocations: dict[InvocationHash, list[CacheItem]]
    ):
        to_remove = [
            intent_hash
            for intent_hash, intent_invocations in intent_invocations.items()
            if not intent_invocations
        ]
        while to_remove:
            next_leaf_pruned = to_remove.pop()
            intent_invocations.pop(next_leaf_pruned)


class SlotCoverageSampler:
    """Sample a variable number of examples so that each slot in the schema
    appears in at least one example. The algorithm iterates through all
    the required + optional slots of an intent and chooses a turn where that
    slot is mentioned. The slots mentioned in each sampled turn are stored
    in a buffer. The algorithm stops when the buffer contains all the slots
    in the intent schema.

    Parameters
    ----------
    randomise
        Select random examples so that all slots are covered.
    hash_selection_function
        If randomise=False, this function decides which of possible
        slot combinations is selected for sampling an example. Defaults
        to median-like approach, where the median number of slots is
        mentioned in an example.
    """

    def __init__(
        self,
        cache: InvocationsCache,
        schema: CommandCollection,
        randomise: bool = False,
        hash_selection_function: Callable = lambda x: sorted(x)[len(x) // 2],
    ):
        self._schema = schema
        self._cache = cache
        self._randomise = randomise
        self._hash_selection_function = hash_selection_function

    def caches_tool(self, intent: ToolName) -> bool:
        return intent in self._cache.cache

    def get_intent_schema(self, tool_name: ToolName) -> ServiceCommand:
        service = self._cache.tool_name_to_service[tool_name]
        intent = self._cache.tool_name_to_intent[tool_name]
        command = self._schema.get(service, intent)
        assert command.tool_name == tool_name
        return command

    def get_intent_slots(self, command: ServiceCommand) -> list[SlotName]:
        all_slots = list(command.optional_slots + command.required_slots)
        if self._randomise:
            random.shuffle(all_slots)
        return [s.name for s in all_slots]

    def sample(self, intent: ToolName) -> list[CacheItem] | None:
        """Sample a variable-length sequence of examples which
        demonstrates the usage of all slots in the intent schema."""
        if not self.caches_tool(intent):
            logger.warning(f"There are no cache entries for tool {intent}")
            return []
        this_intent_samples = self._cache.cache[intent]
        intent_schema = self.get_intent_schema(intent)
        all_slots = self.get_intent_slots(intent_schema)
        covered_slots = set()
        examples = [self._select_example(this_intent_samples[(USER_INTENT,)])]
        for slot in all_slots:
            if slot in covered_slots:
                continue
            try:
                turn_hash = self._select_hash(slot, this_intent_samples)
            except (IndexError, ValueError):
                logger.warning(
                    f"No examples found for argument {slot}, intent {intent}"
                )
                continue
            this_slot_example = self._select_example(this_intent_samples[turn_hash])
            examples.append(this_slot_example)
            covered_slots.update([s for s in turn_hash if s != USER_INTENT])
            if covered_slots == set(all_slots):
                break
        if self._randomise:
            random.shuffle(examples)
        return examples

    def _select_hash(
        self, slot: SlotName, this_intent_samples: dict[InvocationHash, list[CacheItem]]
    ):
        """Select a turn in which `slot` is mentioned."""
        compatible = defaultdict(list)
        for hash_ in this_intent_samples:
            if slot in hash_:
                compatible[len(hash_)].append(hash_)
        hash_lengths = list(compatible.keys())
        if self._randomise:
            sampled_length = random.choice(hash_lengths)
            return random.choice(compatible[sampled_length])
        return compatible[self._hash_selection_function(hash_lengths)][0]

    def _select_example(self, examples: list[CacheItem]) -> CacheItem:
        if self._randomise:
            return random.choice(examples)
        return examples[0]


class DeterministicSeenInvocationsSampler:
    """Samples the kth seen set of API invocations in training for
    a given intent."""

    def __init__(self, sampled_examples_path: Path, index: int = 0):
        if not sampled_examples_path.exists():
            logger.warning(
                f"Could not find invocations seen in training at {sampled_examples_path}"
            )
            self._cache = {}
        else:
            self._cache: dict[ToolName, list[list[CacheItem]]] = load_json(
                sampled_examples_path
            )
        self._index = index

    def sample(self, intent: ToolName) -> list[CacheItem] | None:
        if intent not in self._cache:
            return
        logger.debug(f"Sampled {intent} from train cache!")
        return self._cache[intent][self._index]


def add_invocations(
    toolbox: list[dict[str, Any]],
    sampler: DiverseInvocationsSampler,
    field_name: str = "example_invocations",
    num_examples: int = 10,
):
    """Sample invocations from a cache and add them to the toolbox.

    Parameters
    ----------
    toolbox
        A list of dictionaries containing information about the SGD intents.
        These are parseable with the `pytod.pytod_types.pytod.Intent` class.
    sampler
        A sampler that enables sampling diverse invocations for intents from an invocations cache.
    field_name
        The name of the field under which the invocations are stored in the toolbox
        for every intent.
    num_examples
        How many invocations are to be stored for each example.
    """
    skipped_field = False
    for tool in toolbox:
        if field_name in tool and tool[field_name]:
            skipped_field = True
            continue
        tool[field_name] = []
        samples = sampler.sample(tool["name"], n_samples=num_examples)
        if samples is not None:
            if skipped_field:
                logger.info(f"Adding {len(samples)} invocations for {tool['name']}")
            tool[field_name] = samples


def collect_invocations(
    cfg: DictConfig,
    invocations_cache: InvocationsCache,
    toolbox: list[dict[str, Any]],
    sampler: DiverseInvocationsSampler,
):
    """Collect high-quality, diverse invocations for all the intents in the SGD conversations.


    Parameters
    ----------
    See `add_invocations`.
    """

    if not cfg.toolbox.collect_invocations:
        return
    logger.info("Adding invocations to toolbox")
    index_path = cfg.tree.index_path
    add_invocations(
        toolbox,
        sampler=sampler,
        field_name=cfg.toolbox.invocations_field_name,
        num_examples=cfg.toolbox.num_examples,
    )

    # invocations may not exist in single intent/single domain splits
    # but can be harvested from multi-domain conversations
    if cfg.toolbox.harvest_from_multi_domain:
        index_iterator = SGDIterator(
            cfg.tree.data_path,
            index_path,
            conversation_builder=cfg.tree.conversation_builder,
        )
        index_iterator.command_collections[cfg.split] = instantiate(
            cfg.command_collection
        )
        invocations_cache.harvest_from_index(index_iterator, ["multi_domain"])
        add_invocations(
            toolbox,
            sampler=sampler,
            field_name=cfg.toolbox.invocations_field_name,
            num_examples=cfg.toolbox.num_examples,
        )
