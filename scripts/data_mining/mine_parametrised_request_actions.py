#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf

from pytod.index_utils import invert_index
from pytod.iterators import SGDIterator
from pytod.sgd_conversation_builder import mine_request_parametrisation
from pytod.utils import default_to_regular, nested_defaultdict, store_data

logger = logging.getLogger(__name__)


ServiceName = str  # SGD service name
SlotName = str  # SGD slot name
DialogueID = str  # SGD dialogue ID
TaskSequence = (
    str  # a symbolic representation of the sequence of tasks the user completes
)

RequestedValueSlotsInfo = dict[
    Literal["user", "system"],
    dict[
        Literal["multivalue", "singlevalue"],
        dict[ServiceName, dict[SlotName, list[DialogueID]]],
    ],
]
"""Mapping containing information about slots for which the user/system request a specific value"""
RequestedValueTaskSequenceInfo = dict[
    Literal["user", "system"],
    dict[
        Literal["multivalue", "singlevalue"],
        dict[ServiceName, dict[TaskSequence, DialogueID]],
    ],
]


def add_task_sequence_info(
    index_path: Path,
    split: Literal["train", "dev", "test"],
    request_argument_slots: RequestedValueSlotsInfo,
) -> RequestedValueTaskSequenceInfo:
    """Transforms the list of dialogue IDs where REQUEST actions are parametrised
    to a dictionary where the key is the task sequence completed by the user and
    the value is a representative dialogue ID."""

    def get_task_sequence_mapping(
        dial_ids: list[str], inverted_index: dict[str, dict[str, str]]
    ) -> dict[str, str]:
        task_seq_map = {}

        def get_task_sequence(
            dial_id: str, inverted_index: dict[str, dict[str, str]]
        ) -> str:
            for _, index in inverted_index.items():
                if dial_id in index:
                    return index[dial_id]

        for dial_id in dial_ids:
            task_sequence = get_task_sequence(dial_id, inverted_index)
            if task_sequence in task_seq_map:
                continue
            task_seq_map[task_sequence] = dial_id
        return task_seq_map

    index_root_children = ["single_intent", "single_domain", "multi_domain"]
    inverted_index = {
        child: invert_index([child], index_path)[split] for child in index_root_children
    }
    mapped = nested_defaultdict(dict, depth=4)
    for author, multivalue_info_to_service_map in request_argument_slots.items():
        for (
            multivalue_key,
            service_to_slot_info_map,
        ) in multivalue_info_to_service_map.items():
            for (
                service,
                slot_name_to_dial_occurrence_map,
            ) in service_to_slot_info_map.items():
                for slot, dial_ids_list in slot_name_to_dial_occurrence_map.items():
                    task_seq_info = get_task_sequence_mapping(
                        dial_ids_list, inverted_index
                    )
                    store_data(
                        task_seq_info, mapped, [author, multivalue_key, slot, service]
                    )

    return default_to_regular(mapped)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod.data_mining") / ".")


@hydra.main(
    config_name="mine_request_parametrisation.yaml",
    config_path=get_config_path(),
)
def request_parametrisation(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    run_dir = Path(cfg.run_dir)
    for split in cfg.splits:
        split_path = run_dir / split
        if not split_path.exists():
            logger.info(f"Creating output directory {split_path}")
            split_path.mkdir(exist_ok=False, parents=True)
        collector = nested_defaultdict(list, depth=4)
        setattr(SGDIterator, "request_parametrisation_info", collector)
        iterator = SGDIterator(
            cfg.data_path,
            conversation_builder=partial(
                mine_request_parametrisation,
                request_parametrisation_collector=collector,
            ),
        )
        logger.info(f"Mining split: {split}")
        for _, dialogue in iterator.split_iterator(
            split,
            return_only=set(cfg.ids) if cfg.ids is not None else None,
        ):
            logger.info(f"Mining dialogue({dialogue.id}:{split})")

        this_split_mined_data = add_task_sequence_info(
            Path(cfg.index_path), split, iterator.request_parametrisation_info
        )
        with open(split_path / f"{split}_multivalue_offer_actions.json", "w") as f:
            json.dump(this_split_mined_data, f)


if __name__ == "__main__":
    request_parametrisation()
