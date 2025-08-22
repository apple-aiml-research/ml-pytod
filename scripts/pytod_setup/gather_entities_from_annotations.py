#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from importlib import resources
from typing import Literal

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import Entity, IntentName, ServiceName
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import count_nested_dict_values, nested_defaultdict, save_json

logger = logging.getLogger(__name__)

SGD_SPLITS = ["train", "dev", "test"]
SlotName = str
CanonicalValue = str


def hash_query_result(
    result: dict[SlotName, CanonicalValue]
) -> tuple[tuple[SlotName, CanonicalValue]]:
    """Create a hashable object from a dict containing entity information."""
    return tuple(sorted(list(result.items()), key=lambda x: x[0]))


def serialise_result(result: dict[SlotName, CanonicalValue]) -> str:
    """Create a `str` object from an SGD database entry."""
    return json.dumps(hash_query_result(result))


def gather_counts(
    records: dict[ServiceName, dict[IntentName, list[Entity]]],
    records_with_duplicates: dict[ServiceName, dict[IntentName, list[Entity]]],
) -> dict[Literal["unique", "all"], dict[ServiceName, dict[IntentName, int]]]:
    """Count the total number of results for a given intent in the database."""
    return {
        "all": count_nested_dict_values(records_with_duplicates),
        "unique": count_nested_dict_values(records),
    }


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="db_build.yaml", config_path=get_config_path())
def gather_entities(config: DictConfig):
    """Extract a list of unique list of the database records from annotations."""
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    split = config.split
    logger.info(f"Extracting databases for split: {split}")
    assert split in SGD_SPLITS, f"Unknown split: {split}"
    seen_records = nested_defaultdict(set, depth=2)
    unique_records = nested_defaultdict(list, depth=2)
    unique_records_ids = nested_defaultdict(list, depth=3)
    records = nested_defaultdict(list, depth=2)
    cmd_collection = instantiate(config.command_collection)
    iterator = SGDIterator(
        config.data_path,
        conversation_builder=instantiate(config.conversation_builder),
    )
    for fpath, dial in iterator.split_iterator(
        config.split,
        return_only=set(config.ids) if config.ids is not None else None,
    ):
        call_id = defaultdict(lambda: defaultdict(int))
        for turn in dialogue_iterator(dial):
            for frame_ in turn["frames"]:
                if "service_call" in frame_:
                    service = frame_["service"]
                    intent = frame_["service_call"]["method"]
                    if cmd_collection.get(service, intent).is_transactional:
                        continue
                    results = frame_["service_results"]
                    matches_empty_query = (
                        False if frame_["service_call"]["parameters"] else True
                    )
                    call_id[service][intent] += 1
                    for idx, result in enumerate(results):
                        result_str = serialise_result(result)
                        records[service][intent].append(result)
                        order_info = {
                            "dialogue_id": dial["dialogue_id"],
                            "order": idx,
                            "call_id": call_id[service][intent],
                            "matches_empty_query": matches_empty_query,
                        }
                        if result_str not in seen_records[service][intent]:
                            seen_records[service][intent].add(result_str)
                            unique_records[service][intent].append(result)
                            if (
                                order_info
                                not in unique_records_ids[service][intent][result_str]
                            ):
                                unique_records_ids[service][intent][result_str].append(
                                    order_info
                                )
                        else:
                            if (
                                order_info
                                not in unique_records_ids[service][intent][result_str]
                            ):
                                unique_records_ids[service][intent][result_str].append(
                                    order_info
                                )
    logger.info(f"Finished extracting entities for split: {split}. Saving results.")
    save_json(unique_records, f"{split}.json")
    save_json(unique_records_ids, f"{split}_dial_ids.json")
    counts = gather_counts(unique_records, records)
    save_json(counts, f"{split}_counts.json")


if __name__ == "__main__":
    gather_entities()
