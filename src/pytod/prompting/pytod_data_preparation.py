#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import random
import re
from functools import singledispatch
from itertools import islice
from pathlib import Path
from typing import Generator, Iterator, Literal, TypeVar

from pytod.command import CommandCollection
from pytod.prompting.pytod_data_preparation_utils import (
    get_conversations,
    get_datapoint_idx,
    get_turn_apis,
)
from pytod.prompting.pytod_prompt_formatters import (
    ConversationFormatter,
    ConversationFormatterType,
)
from pytod.prompting.text2text_example_parsers import (
    ConversationExample,
    InferenceConversationExample,
)
from pytod.pytod_types.aliases import DialogueID
from pytod.pytod_types.pytod import AnyTurn, PyTODConversation, ServiceInfo, SystemTurn
from pytod.utils import get_sgd_shard_name

T = TypeVar("T")
logger = logging.getLogger(__name__)


def get_conversation_examples(
    in_dir: Path,
    split: Literal["train", "dev", "test"],
    command_collection: CommandCollection,
    say_sample_prob: float,
    limit: int | None,
    multiline_predictions: bool = False,
    multidomain_prompts: bool = False,
    return_only: set[DialogueID] | None = None,
    randomise_api_order: bool = True,
    render_entities: bool = False,
) -> Generator[ConversationExample, None, None]:
    """Yields a single conversation example."""
    conversations_it = islice(get_conversations(in_dir, return_only=return_only), limit)
    for conversation in conversations_it:
        if multiline_predictions:
            yield from multiline_conversation_to_conversation_examples(
                conversation=conversation,
                apis=command_collection,
                randomise_api_order=randomise_api_order,
                multidomain_prompts=multidomain_prompts,
                render_entities=render_entities,
            )
        else:
            yield from conversation_to_conversation_examples(
                conversation=conversation,
                apis=command_collection,
                split=split,
                say_sample_prob=say_sample_prob,
                randomise_api_order=randomise_api_order,
                multidomain_prompts=multidomain_prompts,
            )


def conversation_to_conversation_examples(
    conversation: PyTODConversation,
    apis: CommandCollection,
    split: str,
    say_sample_prob: float,
    randomise_api_order: bool = True,
    multidomain_prompts: bool = False,
) -> Iterator[ConversationExample]:
    """Create several training examples from a given PyTOD conversation."""

    def is_target(turn: AnyTurn, say_sample_prob: float) -> bool:
        if not isinstance(turn, SystemTurn):
            return False

        tool_name = turn.get_tool_name()
        if tool_name == "say":
            return random.random() <= say_sample_prob
        else:
            return tool_name not in {"perform", "len", "suggest", "show", "slice"}

    # List of IDs of turns in the conversation. For now, we simply use the datapoint IDs
    action_ids: list[str] = []
    service_info: list[ServiceInfo] = conversation.service_info

    for i, turn in enumerate(conversation.turns):
        action_ids.append(f"{conversation.id}_{i}")
        if not is_target(
            turn, say_sample_prob=say_sample_prob if split == "train" else 1.0
        ):
            continue
        assert isinstance(turn, SystemTurn)
        assert service_info is not None
        service = service_info[i].next_service
        intent = conversation.service_info[i].active_intent
        none_intent = conversation.service_info[i].none_intent
        this_turn_apis = get_turn_apis(
            apis, service_info[i], multidomain_prompts=multidomain_prompts
        )
        if randomise_api_order:
            random.shuffle(this_turn_apis)
        yield ConversationExample(
            id=conversation.id,
            action_ids=action_ids,
            source_turns=conversation.turns[:i],
            target_turns=[turn],
            service=service,
            intent=intent,
            none_intent=none_intent,
            apis=this_turn_apis,
        )


def multiline_conversation_to_conversation_examples(
    conversation: PyTODConversation,
    apis: CommandCollection,
    randomise_api_order: bool = True,
    multidomain_prompts: bool = True,
    render_entities: bool = False,
) -> Iterator[ConversationExample]:
    """Create several training examples from a given PyTOD conversation.
    Unlike `multiline_conversation_to_conversation_examples` this iterator
    creates two examples for each turn where a transition from a non-transactional
    (query) intent to another intent.

    Parameters
    ----------
    conversation
    apis
        The APIs to display in the prompt header.
    randomise_api_order
        If `True`, vary the order of the APIs listed in the prompt header across examples.
    multidomain_prompts
        If `True`, the APIs from both the previous and current service are displayed in
        the prompt header for turns annotated with more than one semantic frame.
    render_entities
        If `True`, an additional example is created for each multi-domain turn, to allow
        predicting the subsequent API calls conditioned on a rendered entity.
    """
    i = 0
    this_example_target = []
    this_example_source = []
    this_turn_apis = None
    while i < len(conversation.turns):
        current_turn = conversation.turns[i]
        service_info = conversation.service_info[i]
        service = service_info.next_service
        intent = service_info.active_intent
        current_command = apis.get(service, intent)
        none_intent = service_info.none_intent
        if current_turn.author == "User":
            this_turn_apis = get_turn_apis(
                apis, service_info, multidomain_prompts=multidomain_prompts
            )
        assert this_turn_apis is not None
        if randomise_api_order:
            random.shuffle(this_turn_apis)
        match current_turn.author:
            case "User" | "Response":
                this_example_source.append(current_turn)
                i += 1
                continue
            case "System":
                tool_name = current_turn.get_tool_name()
                match tool_name:
                    case "len" | "perform" | "suggest" | "show" | "slice":
                        if this_example_target:
                            yield ConversationExample(
                                id=conversation.id,
                                action_ids=[],
                                source_turns=this_example_source,
                                target_turns=this_example_target,
                                service=service,
                                intent=intent,
                                current_task=current_command,
                                apis=this_turn_apis,
                                none_intent=none_intent,
                            )
                            # if we render the entities, then we ensure that
                            # we learn to predict the instructions given the
                            # rendered entities, so we create an additional
                            # example where `select` is on the source side
                            if render_entities and len(this_example_target) > 1:
                                if this_example_target[0].get_tool_name() == "select":
                                    yield ConversationExample(
                                        id=conversation.id,
                                        action_ids=[],
                                        # ensure the select instruction is the last one in the
                                        # source, so that the model is trained to predict
                                        # the next instructions given the rendered entity
                                        source_turns=this_example_source
                                        + [this_example_target[0]],
                                        # all the other instructions apart from the entity
                                        target_turns=this_example_target[1:],
                                        service=service,
                                        intent=intent,
                                        current_task=current_command,
                                        # we only prompt with the APIs of the service that comes
                                        # next in this case - the other APIs are displayed in
                                        # different part of the prompt
                                        apis=get_turn_apis(apis, service_info),
                                    )
                            this_example_source.append(current_turn)
                            i += 1
                            this_example_source = conversation.turns[:i]
                            this_example_target = []
                            this_turn_apis = get_turn_apis(
                                apis,
                                conversation.service_info[i],
                                multidomain_prompts=multidomain_prompts,
                            )
                        else:
                            this_example_source.append(current_turn)
                            i += 1
                            continue
                    case "say":
                        this_example_target.append(current_turn)
                        yield ConversationExample(
                            id=conversation.id,
                            action_ids=[],
                            source_turns=this_example_source,
                            target_turns=this_example_target,
                            service=service,
                            intent=intent,
                            current_task=current_command,
                            apis=this_turn_apis,
                            none_intent=none_intent,
                        )
                        if render_entities and len(this_example_target) > 1:
                            if this_example_target[0].get_tool_name() == "select":
                                yield ConversationExample(
                                    id=conversation.id,
                                    action_ids=[],
                                    # ensure the select instruction is the last one in the
                                    # source, so that the model is trained to predict
                                    # the next instructions given the rendered entity
                                    source_turns=this_example_source
                                    + [this_example_target[0]],
                                    # all the other instructions apart from the entity
                                    target_turns=this_example_target[1:],
                                    service=service,
                                    intent=intent,
                                    current_task=current_command,
                                    # we only prompt with the APIs of the service that comes
                                    # next in this case - the other APIs are displayed in
                                    # different part of the prompt
                                    apis=get_turn_apis(apis, service_info),
                                )
                        assert not any(
                            t.get_tool_name == "select" for t in this_example_target
                        )
                        i += 1
                        this_example_source = conversation.turns[:i]
                        this_example_target = []
                    case _:
                        this_example_target.append(current_turn)
                        i += 1
                        continue
            case "Hint" | "Signal":
                if this_example_target:
                    yield ConversationExample(
                        id=conversation.id,
                        action_ids=[],
                        source_turns=this_example_source,
                        target_turns=this_example_target,
                        service=service,
                        intent=intent,
                        apis=this_turn_apis,
                        current_task=current_command,
                        none_intent=none_intent,
                    )
                    if render_entities and len(this_example_target) > 1:
                        # see above for rationale on this additional example
                        if this_example_target[0].get_tool_name() == "select":
                            yield ConversationExample(
                                id=conversation.id,
                                action_ids=[],
                                # ensure the select instruction is the last one in the
                                # source, so that the model is trained to predict
                                # the next instructions given the rendered entity
                                source_turns=this_example_source
                                + [this_example_target[0]],
                                # all the other instructions apart from the entity
                                target_turns=this_example_target[1:],
                                service=service,
                                intent=intent,
                                current_task=current_command,
                                # we only prompt with the APIs of the service that comes
                                # next in this case - the other APIs are displayed in
                                # different part of the prompt
                                apis=get_turn_apis(apis, service_info),
                            )
                    while current_turn.author in {"Signal", "Hint"}:
                        i += 1
                        current_turn = conversation.turns[i]
                    this_example_source = conversation.turns[:i]
                    this_example_target = []
                else:
                    this_example_source.append(current_turn)
                    i += 1
                    continue


@singledispatch
def get_text2text_example(
    conversation: ConversationExample | InferenceConversationExample,
    formatter: ConversationFormatter,
) -> dict:
    """Convert a structured conversation example to text2text format."""
    raise ValueError("Unknown conversation type")


@get_text2text_example.register
def _(
    conversation: ConversationExample, formatter: ConversationFormatterType
) -> dict[str, str]:
    """Utility for creating examples during training and NAP datasets."""
    assert conversation.target_turns
    source, target, _ = formatter.format(conversation, conversation.apis)
    assert target is not None
    dial_id = re.sub(r"(train_|dev_|test_)", "", conversation.id)
    datapoint_idx = get_datapoint_idx(
        conversation.source_turns, conversation.target_turns
    )
    return {
        "datapoint_id": f"{conversation.id}-{datapoint_idx}",
        "dialogue_id": dial_id,
        "shard_name": get_sgd_shard_name(dial_id),
        "prompt": source,
        "completion": target,
        "service": conversation.service,
        "intent": conversation.intent,
        "apis": [cmd.name for cmd in conversation.apis],
        "none_intent": conversation.none_intent,
    }


FeatureName = Literal[
    "datapoint_id",
    "dialogue_id",
    "shard_name",
    "prompt",
    "completion",
    "service",
    "apis",
    "intent",
    "none_intent",
]


@get_text2text_example.register
def _(
    conversation: InferenceConversationExample, formatter: ConversationFormatterType
) -> dict[FeatureName, str]:
    """Utility for creating examples during inference with predicted states."""
    sep = formatter.target_fmt.expression_list_sep
    target = (
        f" {sep} ".join([str(t.expression) for t in conversation.oracle_target_turns])
        .lower()
        .strip()
    )
    source, _, _ = formatter.format(conversation, conversation.apis)
    dial_id = re.sub(r"(train_|dev_|test_)", "", conversation.id)
    datapoint_id = conversation.datapoint_id
    if datapoint_id is None:
        datapoint_id = get_datapoint_idx(
            conversation.source_turns, conversation.oracle_target_turns
        )
        if any(
            t.get_tool_name() == "undefined_target"
            for t in conversation.oracle_target_turns
        ):
            datapoint_id = f"{conversation.id}-{datapoint_id}u"
        else:
            datapoint_id = f"{conversation.id}-{datapoint_id}"
    return {
        "datapoint_id": datapoint_id,
        "dialogue_id": dial_id,
        "shard_name": get_sgd_shard_name(dial_id),
        "prompt": source,
        "completion": target,
        "service": conversation.service,
        "apis": [cmd.name for cmd in conversation.apis],
        "intent": conversation.intent,
        "none_intent": conversation.none_intent,
    }
