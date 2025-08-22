#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.preprocessing_pipelines import (
    add_followup_metadata,
    add_metadata_to_intent_schema,
    add_wildcard_info_to_slot_schema,
)
from pytod.pytod_types.aliases import ServiceName, SlotName
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import cast_vals_to_sorted_list, load_json, save_json, stringify_list

logger = logging.getLogger(__name__)

WILDCARD_VALUE = "dontcare"


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


def mine_wildcard_slots(config: DictConfig) -> dict[ServiceName, list[SlotName]]:
    """Build a map from service name to slots which can take wildcard values."""
    logger.info(
        f"Extracting wildcard values for splits: {stringify_list(list(config.splits))}"
    )
    wildcard_map = defaultdict(set)
    for split in config.splits:
        iterator = SGDIterator(
            config.data_path,
            conversation_builder=instantiate(config.conversation_builder),
        )
        for fpath, dial in iterator.split_iterator(
            split,
            return_only=set(config.ids) if config.ids is not None else None,
        ):
            for turn in dialogue_iterator(dial):
                if turn["speaker"] == "SYSTEM":
                    continue
                for frame in turn["frames"]:
                    service = frame["service"]
                    for action in frame["actions"]:
                        match action["act"]:
                            case "INFORM" if action["canonical_values"][
                                0
                            ] == WILDCARD_VALUE:
                                wildcard_map[service].add(action["slot"])

    return cast_vals_to_sorted_list(wildcard_map)


@hydra.main(config_name="schema_augmentation.yaml", config_path=get_config_path())
def augment_schema(config: DictConfig):
    outdir = Path(config.out_dir)
    wildcard_values = mine_wildcard_slots(config)
    for split in config.splits:
        logger.info(f"Augmenting schemata in {split} set ")
        schema_path = Path(config.data_path) / split / "schema.json"
        schema = load_json(schema_path)
        new_schema = add_metadata_to_intent_schema(
            schema, OmegaConf.to_container(config.metadata)
        )
        new_schema = add_followup_metadata(new_schema)
        new_schema = add_wildcard_info_to_slot_schema(new_schema, wildcard_values)
        save_json(new_schema, outdir / split / "schema.json")


if __name__ == "__main__":
    augment_schema()
