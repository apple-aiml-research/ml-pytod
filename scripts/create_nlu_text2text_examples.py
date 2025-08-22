#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from copy import copy
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.command import ArgumentDefinition, CommandCollection
from pytod.inference.parser_supervisor import parse_answers
from pytod.iterators import IndexMetadata, SGDIterator
from pytod.prompting.pytod_nlu_formatters import UNK_VALUE, NLUExampleFormatter
from pytod.prompting.text2text_example_parsers import NLUConversationExample
from pytod.pytod_types.aliases import ServiceName, SlotName, SlotValue
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    SystemDialogueAct,
    Turn,
    UserDialogueAct,
)
from pytod.sgd_agent_behaviour import count_agent_questions
from pytod.transcript import APIInfo, get_current_api
from pytod.utils import get_sgd_shard_name, load_json, set_seed_no_gpu

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(
        resources.files("pytod.configs.pytod_finetuning.data_preparation")
        / "nlu_prompts"
    )


def split_into_service_spans(
    dialogue: Conversation, metadata: IndexMetadata, schema: CommandCollection
) -> dict[ServiceName, list[list[Turn]]]:
    """Split a conversation into multiple turn sequences, each corresponding to a service.
    If a service appears at multiple points in the conversation, multiple turn sequences
    are returned.
    """
    api_info = partial(get_current_api, label_novelty=False, index_metadata=metadata)
    i = 0
    prev_api: APIInfo | None = None
    service_mapping = defaultdict(list)
    current_service_stack = []
    while i < len(dialogue):
        this_turn = dialogue[i]
        this_turn.sgd_turn_idx = i
        if this_turn.author != Author.USER:
            i += 1
            continue
        prev_system_turn = None if i == 0 else dialogue[i - 1]
        next_system_turn = dialogue[i + 1]
        api: APIInfo = api_info(
            user_turn=this_turn,
            user_turn_index=i,
            conversation=dialogue,
            prev_api=prev_api,
            prev_system_turn=prev_system_turn,
            next_system_turn=next_system_turn,
            command_collection=schema,
        )
        if prev_api is None or (api.service == prev_api.service):
            current_service_stack.extend([this_turn, next_system_turn])
        else:
            assert api.service != prev_api.service
            service_mapping[prev_api.service].append(current_service_stack)
            current_service_stack = [this_turn, next_system_turn]
        prev_api = api
        i += 1
    if current_service_stack:
        service_mapping[prev_api.service].append(current_service_stack)
    return dict(service_mapping)


def get_nlu_target(
    service: ServiceName,
    user_turn: Turn,
    prev_sys_turn: Turn | None,
    schema: CommandCollection,
    dial_id: str,
) -> dict[ArgumentDefinition, SlotValue]:
    assert user_turn.author == Author.USER
    target = {}
    for a in user_turn.user_actions[service]:
        if a.act == UserDialogueAct.INFORM:
            arg_schema = schema.get_arg_schema(service, a.slot)
            target[arg_schema] = a.values[0]
        if a.act == UserDialogueAct.AFFIRM:
            assert prev_sys_turn is not None
            for sys_a in prev_sys_turn.system_actions[service]:
                if sys_a.act == SystemDialogueAct.REQUEST:
                    try:
                        assert sys_a.values
                    # assertion raised if the system requests a slot value and a slot
                    #  value confirmation at the same time
                    except AssertionError:
                        try:
                            assert sys_a.slot in {s.name for s in target}
                        except AssertionError:
                            logger.warning(
                                f"{service}({dial_id}): "
                                f"Could not find a value for slot {sys_a.slot} in target."
                            )
                        continue
                    arg_schema = schema.get_arg_schema(service, a.slot)
                    assert arg_schema
                    target[arg_schema] = sys_a.values[0]
    return target


def get_unknown_slots(
    turn: Turn, service: ServiceName, schema: CommandCollection
) -> list[ArgumentDefinition]:
    assert turn.author == Author.USER
    relevant_slots = set()
    for cmd in schema.get_service_commands(service):
        relevant_slots.update(
            {s.name for s in (*cmd.required_slots, *cmd.optional_slots)}
        )
    state = turn.dialogue_state[service].slot_values
    slot_schemas = [
        schema.get_arg_schema(service, s) for s in relevant_slots if s not in state
    ]
    assert not any(def_ is None for def_ in slot_schemas)
    return slot_schemas


def span_to_examples(
    service: ServiceName,
    schema: CommandCollection,
    span: list[Turn],
    span_idx: int,
    dial_id: str,
) -> list[NLUConversationExample]:
    examples = []
    for i, turn in enumerate(span):
        example = {
            "id": f"nlu_{dial_id}_{service.lower()}_span_{span_idx}_turn_{i}",
            "service": service,
        }
        if turn.author == Author.USER or (
            turn.author == Author.SYSTEM and count_agent_questions(turn) == 0
        ):
            continue
        example["source_turns"] = span[: i + 2]
        example["target"] = get_nlu_target(
            service, span[i + 1], span[i], schema, dial_id
        )
        example["unknown_val_slots"] = get_unknown_slots(span[i + 1], service, schema)
        examples.append(NLUConversationExample(**example))
    return examples


def get_first_turn_example(
    dialogue: Conversation,
    schema: CommandCollection,
) -> NLUConversationExample:
    first_turn = dialogue[0]
    service = list(first_turn.dialogue_state.keys())[0]
    return NLUConversationExample(
        id=f"nlu_{dialogue.id}_{service.lower()}_span_0_turn_0",
        service=service,
        source_turns=[first_turn],
        target=get_nlu_target(service, first_turn, None, schema, dialogue.id),
        unknown_val_slots=get_unknown_slots(first_turn, service, schema),
    )


def write_to_txt_format(df: pd.DataFrame, out_parquet_path: Path):
    out_txt_file = Path(out_parquet_path).with_suffix(".txt")
    with out_txt_file.open("w") as fh:
        for datapoint_id, row in df.iterrows():
            try:
                fh.write("turn_idx: " + str(row.turn_idx) + "\n")
                fh.write("parser map: " + str(row.parser_map) + "\n")
                fh.write("gold completion: " + str(row.gold_completion) + "\n")
                fh.write("unseen: " + str(row.unseen) + "\n")
            # not in the train dataset schema to ensure compat with pytod
            # training data
            except AttributeError:
                pass
            fh.write("datapoint: " + str(datapoint_id) + "\n")
            fh.write("shard: " + str(row.shard_name) + "\n")
            fh.write("dialogue: " + str(row.dialogue_id) + "\n")
            fh.write("service: " + str(row.service) + "\n")
            fh.write("intent: " + str(row.intent) + "\n")
            fh.write(row.prompt + "\n")
            fh.write(row.completion + "\n")
            fh.write("\n" + "-" * 100 + "\n")


def check_target_parsing(
    liniarised_target: str,
    example: NLUConversationExample,
    parser_map: dict[str, SlotName],
):
    target = {k.name: v for k, v in example.target.items()}
    lc_target = {k: v.lower() for k, v in target.items()}
    answers = parse_answers(liniarised_target)
    parsed_target = {parser_map[str(q_idx)]: answers[q_idx] for q_idx in answers}
    try:
        assert target == parsed_target or lc_target == parsed_target
    except AssertionError:
        assert len(example.source_turns) == 1
        first_turn_lc = copy(lc_target)
        for k, v in answers.items():
            if v == UNK_VALUE:
                first_turn_lc[parser_map[str(k)]] = UNK_VALUE
        assert first_turn_lc == parsed_target


@hydra.main(config_name="nlu_prompts", config_path=get_config_path())
def prepare_examples(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed_no_gpu(cfg.random)
    index_path = Path(cfg.index_path)
    formatter: NLUExampleFormatter = instantiate(cfg.formatter)
    first_turn_formatter: NLUExampleFormatter = instantiate(cfg.first_turn_formatter)
    index_info: dict[str, Any] = load_json(index_path / "metadata.json")
    index_metadata = IndexMetadata.model_validate(index_info)
    schema = CommandCollection(cfg.schema_path)
    train_schema = CommandCollection(cfg.train_schema_path)
    iterator = SGDIterator(cfg.data_path)
    processed_examples = []
    first_turn_examples = []
    split = cfg.split
    deny_list = cfg.deny_list.get(split, [])
    for _, dialogue in iterator.split_iterator(split):
        # logger.info(f"Processing dialogue: {dialogue.id}")
        if dialogue.id in deny_list:
            logger.info(f"Skipping dialogue {dialogue.id}")
            continue
        service_spans: dict[ServiceName, list[list[Turn]]] = split_into_service_spans(
            dialogue, index_metadata, schema
        )
        for service, spans in service_spans.items():
            for span_idx, span in enumerate(spans):
                span_examples: list[NLUConversationExample] = span_to_examples(
                    service, schema, span, span_idx, dialogue.id
                )
                for s in span_examples:
                    fmt_example = formatter.format(s)
                    last_user_turn = s.source_turns[-1]
                    turn_idx = last_user_turn.sgd_turn_idx
                    dialogue_state = last_user_turn.dialogue_state
                    assert turn_idx % 2 == 0
                    assert len(dialogue_state) == 1
                    intent = dialogue_state[service].active_intent
                    apis = [cmd.name for cmd in schema.get_service_commands(service)]
                    # nb: for train split we don't include parser_map
                    #  turn_idx attribs to ensure compatibility with
                    #  the pytod text2text dataset schema
                    example = {
                        "datapoint_id": s.id,
                        "dialogue_id": dialogue.id,
                        "shard_name": get_sgd_shard_name(dialogue.id),
                        "prompt": fmt_example.prompt,
                        "completion": fmt_example.target,
                        "service": service,
                        "intent": intent,
                        "apis": apis,
                        "none_intent": False,
                    }
                    if split in ["dev", "test"]:
                        example["parser_map"] = json.dumps(fmt_example.parser_map)
                        example["turn_idx"] = turn_idx
                        example["gold_completion"] = json.dumps(
                            {k.name: v.lower() for k, v in s.target.items()}
                        )
                        example["unseen"] = service not in train_schema.services
                        check_target_parsing(
                            example["completion"], s, fmt_example.parser_map
                        )
                    processed_examples.append(example)
        # create examples from first turns
        first_turn_example = get_first_turn_example(dialogue, schema)
        fmt_first_turn = first_turn_formatter.format(first_turn_example)
        dialogue_state = dialogue[0].dialogue_state
        service = list(dialogue_state.keys())[0]
        example = {
            "datapoint_id": first_turn_example.id,
            "dialogue_id": dialogue.id,
            "shard_name": get_sgd_shard_name(dialogue.id),
            "prompt": fmt_first_turn.prompt,
            "completion": fmt_first_turn.target,
            "service": service,
            "intent": dialogue_state[service].active_intent,
            "apis": [cmd.name for cmd in schema.get_service_commands(service)],
            "none_intent": False,
        }
        if split in ["dev", "test"]:
            example["parser_map"] = json.dumps(fmt_first_turn.parser_map)
            example["turn_idx"] = 0
            gold_completion = {
                k.name: v.lower() for k, v in first_turn_example.target.items()
            }
            for _, v in fmt_first_turn.parser_map.items():
                if v not in gold_completion:
                    gold_completion[v] = UNK_VALUE
            example["gold_completion"] = json.dumps(gold_completion)
            example["unseen"] = service not in train_schema.services
            check_target_parsing(
                example["completion"], first_turn_example, fmt_first_turn.parser_map
            )
        first_turn_examples.append(example)

    out_path = Path(cfg.output_dir)
    df = pd.DataFrame(processed_examples).set_index("datapoint_id")
    df_first_turn = pd.DataFrame(first_turn_examples).set_index("datapoint_id")
    df.to_parquet(out_path / f"{split}_nlu.parquet")
    df_first_turn.to_parquet(out_path / f"{split}_nlu_first.parquet")

    if cfg.debug:
        write_to_txt_format(df, out_path / f"{split}_nlu.parquet")
        write_to_txt_format(df_first_turn, out_path / f"{split}_nlu_first.parquet")


if __name__ == "__main__":
    prepare_examples()
