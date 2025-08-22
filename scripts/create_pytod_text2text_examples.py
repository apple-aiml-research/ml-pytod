#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import shutil
import time
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TypeVar

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.prompting.pytod_data_preparation import (
    get_conversation_examples,
    get_text2text_example,
)
from pytod.prompting.pytod_header_formatters import (
    APIUsageCommandFormatter,
    IntentFormatterType,
)
from pytod.prompting.pytod_prompt_formatters import (
    RenderedEntitiesConversationFormatter,
)
from pytod.prompting.text2text_example_parsers import ConversationExample
from pytod.pytod_types.aliases import DialogueID, ShardName
from pytod.sgd_conversation_builder import no_op_conversation_builder
from pytod.utils import (
    default_to_regular,
    load_dialog_histories,
    load_json,
    set_seed_no_gpu,
    write_shards,
)

logger = logging.getLogger(__name__)


T = TypeVar("T")
DIAL_SAMPLES_DIR = "sampled_dialogues"
_SGD_SPLITS = ["train", "dev", "test"]


def get_formatter_arg_overrides(
    split: Literal["train", "dev", "test"], config: DictConfig
) -> dict[str, Any]:
    """Get settings for formatter instantiation as a function of random settings.
    As a rule, variables start at x0 and prompt elements are not shuffled in dev/test,
    for reproducibility.
    """

    overrides = {}
    randomise_schema_element_order = split in config.randomise_schema_element_order
    randomise_start_variable = split in config.randomise_start_variable
    if not randomise_start_variable:
        overrides.update({"conversation_fmt": {"config": {"start_index": 0}}})
    if not randomise_schema_element_order:
        overrides.update(
            {"intent_fmt": {"config": {"randomise_prompt_elements": False}}}
        )
        overrides.update(
            {"prompt_fmt": {"config": {"randomise_prompt_elements": False}}}
        )
        if "conversation_fmt" not in overrides:
            overrides.update(
                {"conversation_fmt": {"config": {"randomise_prompt_elements": False}}}
            )
        else:
            overrides["conversation_fmt"]["config"].update(
                {"randomise_prompt_elements": False}
            )

    return overrides


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod_finetuning") / "data_preparation")


def process_dialogue_histories(
    evaluation_resource_pth: Path,
    testing_resources_path: Path,
    output_dir: Path,
    *,
    processor: Callable,
    render_entities: bool = False,
    return_only: set[DialogueID] | None = None,
):
    """Add DST targets to dialogue history files. These are used during inference for model
    output analysis. The following information is added:

        - dst_target_transcript - these are the targets where variable indices are according to
        the gold DST transcript (which includes some `say` turns in history)
        - dst_target_formatted - same targets as above, but with variable indices remapped
        to account for the absence of `say` turns in the history
        - transcript_pointers - for each target the (exclusive) index of the last turn from the gold
        DST transcript source turns
        - gold_dst_source - strings representing the gold input to the model, for which
        `dst_target_formatted` are the ground truth targets.
    """
    logger.info(
        "Merging oracle source and target information to dialogue history files..."
    )
    split, state_transcript_dir = (
        testing_resources_path.name,
        testing_resources_path.parent,
    )
    iterator = SGDIterator(
        state_transcript_dir, conversation_builder=no_op_conversation_builder
    )
    joined_histories: dict[ShardName, dict] = defaultdict(lambda: defaultdict(dict))
    # transcripts omitting some NLG calls (eg slot filling, confirmations, task outcome etc)
    histories = load_dialog_histories(str(evaluation_resource_pth))
    for fpath, state_transcript in iterator.split_iterator(
        split, return_only=return_only
    ):
        this_dial_id = state_transcript["id"].replace(f"{split}_", "")
        history = histories[this_dial_id]
        # in-place add transcript/formatted targets and dst gold source
        processor(
            dialogue_history=history,
            dst_transcript=state_transcript,
            rendered_entities=render_entities,
        )
        joined_histories[fpath.name][this_dial_id] = history
    history_dir = output_dir / "dialogue_histories"
    if not history_dir.exists():
        history_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing updated dialogue history shards to {history_dir}...")
    write_shards(default_to_regular(joined_histories), history_dir)


def maybe_save_seen_invocation_examples(
    split: str,
    intent_formatter: IntentFormatterType,
    odir: Path,
    indir: Path,
):
    """Save API usage examples shown in training prompts so that they
    can be re-used for seen intent prompts during inference on dev/test.
    """
    if split == "train" and isinstance(intent_formatter, APIUsageCommandFormatter):
        intent_formatter.save_sampled_invocations(odir)
        if (indir / "invocations_cache.json").exists():
            shutil.copy(indir / "invocations_cache.json", odir)


@hydra.main(config_name="offline_multiline", config_path=get_config_path())
def prepare_examples(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    start_time = time.time()
    set_seed_no_gpu(config.random)
    split = config.split
    assert split in _SGD_SPLITS, f"Split should be one of {_SGD_SPLITS}, got {split}"
    ids = config.ids
    if config.data_check:
        ids = load_json(
            Path(config.resources_path)
            / DIAL_SAMPLES_DIR
            / config.split
            / "dialogues.json"
        )
    formatter = instantiate(
        config.text2text_conversion, **get_formatter_arg_overrides(split, config.random)
    )
    render_entities = (
        True if isinstance(formatter, RenderedEntitiesConversationFormatter) else False
    )
    out_path = Path(config.output_dir)
    logger.info(f"Processing {split} ...")
    multiline_prediction = (
        config.text2text_conversion.target_fmt.config.program_newline_sep is not None
    )
    assistant_schema = instantiate(config.command_collection)
    in_data_path = Path(config.data_path)
    conv_examples: Iterator[ConversationExample] = get_conversation_examples(
        in_dir=in_data_path,
        split=split,
        command_collection=assistant_schema,
        # nb: this is not use in datasets where the agent predicts multiple instructions at once
        #  hence why missing from the config
        say_sample_prob=config.say_sample_prob
        if hasattr(config, "say_sample_prob")
        else 1.0,
        limit=config.limit,
        return_only=ids,
        multidomain_prompts=config.multidomain_prompts,
        multiline_predictions=multiline_prediction,
        randomise_api_order=split in config.random.randomise_schema_element_order,
        render_entities=render_entities,
    )
    processed_examples = []
    datapoint_ids = []
    for example in conv_examples:
        processed_examples.append(get_text2text_example(example, formatter))
        id_ = processed_examples[-1]["datapoint_id"]
        datapoint_ids.append(id_)
    assert len(datapoint_ids) == len(set(datapoint_ids)), "IDs should be unique"
    df = pd.DataFrame(processed_examples).set_index("datapoint_id")
    out_parquet_path = out_path / f"{split}.parquet"
    df.to_parquet(out_parquet_path)
    logging.info(f"Created {out_parquet_path} with {len(df)} rows")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time:.4f} seconds")
    if config.debug:
        out_txt_file = Path(out_parquet_path).with_suffix(".txt")
        with out_txt_file.open("w") as fh:
            for datapoint_id, row in df.iterrows():
                fh.write("datapoint: " + str(datapoint_id) + "\n")
                fh.write("shard: " + str(row.shard_name) + "\n")
                fh.write("dialogue: " + str(row.dialogue_id) + "\n")
                fh.write("service: " + str(row.service) + "\n")
                fh.write("intent: " + str(row.intent) + "\n")
                fh.write("no active intent: " + str(row.none_intent) + "\n")
                fh.write(row.prompt + "\n")
                fh.write(row.completion + "\n")
                fh.write("\n" + "-" * 100 + "\n")
        logging.info(f"Created {out_txt_file}")
    shutil.copy(in_data_path / "schema.json", out_path)
    # we do not add source/target info to dialogue histories because the processing
    # is non-deterministic, and we don't run inference on the train set anyway
    if split != "train":
        formatter_overrides = get_formatter_arg_overrides(split, config.random)
        history_processor = instantiate(
            config.history_processor,
            **{
                "conversation_formatter": instantiate(
                    config.text2text_conversion,
                    **formatter_overrides,
                )
            },
        )
        process_dialogue_histories(
            Path(config.evaluation_resource_pth),
            Path(config.testing_resource_pth),
            out_path,
            render_entities=render_entities,
            processor=history_processor,
            return_only=ids,
        )
    maybe_save_seen_invocation_examples(
        split, formatter.intent_fmt, out_path, in_data_path
    )


if __name__ == "__main__":
    prepare_examples()
