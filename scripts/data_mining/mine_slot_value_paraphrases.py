#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, parse_obj_as

from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import (
    CanonicalValueCollector,
    ParaphraseValueCollector,
    ServiceName,
    SlotName,
    SlotValue,
    ValueParaphrase,
)
from pytod.sgd_conversation_builder import mine_slot_value_paraphrases
from pytod.utils import (
    append_to_values,
    default_to_regular,
    nested_defaultdict,
    store_data,
)

logger = logging.getLogger(__name__)


ValueParaphaseMap = dict[
    ServiceName, dict[SlotName, dict[SlotValue, list[ValueParaphrase]]]
]

ParaphaseList = list[list[ValueParaphrase]]


class SlotValueParaphrases(BaseModel):
    # see ValueParaphaseMap for actual type def
    map: dict[str, dict[str, dict[str, list[str]]]]
    # sub-lists describe semantically similar values
    cased_equivalence_lists: list[list[str]]
    uncased_equivalence_lists: list[list[str]]


def get_config_path() -> str:
    return str(resources.files("pytod.configs.ssa_mining.data_mining") / ".")


@hydra.main(
    config_name="slot_value_paraphrases.yaml",
    config_path=get_config_path(),
)
def mine_paraphrases(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    def lowercase(list_: set[str]) -> set[str]:
        return {el.lower() for el in list_}

    def get_paraphrases(
        collector: ParaphraseValueCollector, value_collector: CanonicalValueCollector
    ) -> SlotValueParaphrases:
        """
        1. Convert raw canonical map where the innermost keys are canonical values to
        a map where each innermost key is a value which appears in conversation and
        the values are lists of paraphrases of said value.

        2. Return cased and uncased equivalence lists of values that map to the same
        canonical value.
        """

        def get_value_and_paraphrases(
            val_paraphrases: list[ValueParaphrase],
        ) -> Optional[Iterable[tuple[SlotValue, list[ValueParaphrase]]]]:
            if len(val_paraphrases) == 1:
                return
            for i, value in enumerate(val_paraphrases):
                paraphrases = val_paraphrases[:i] + val_paraphrases[i + 1 :]
                yield value, paraphrases

        value_paraphrase_map = nested_defaultdict(list, depth=3)
        for service, slot_paraphrases_dict in collector.items():
            for slot, value_paraphrases_dict in slot_paraphrases_dict.items():
                for (
                    canonincal_value,
                    value_paraphrases,
                ) in value_paraphrases_dict.items():
                    this_canoical_value_dict = {
                        value: paraphrases
                        for value, paraphrases in get_value_and_paraphrases(
                            list(value_paraphrases)
                        )
                    }
                    for value, paraphrases in this_canoical_value_dict.items():
                        store_data(
                            paraphrases, value_paraphrase_map, [service, slot, value]
                        )
        equivalence_lists = [
            list(equivalent_values)
            for _, equivalent_values in value_collector.items()
            if len(equivalent_values) > 1
        ]
        uncased_equivalence_lists = []
        for _, equivalent_values in value_collector.items():
            vals = list(set(lowercase(equivalent_values)))
            if len(vals) > 1:
                uncased_equivalence_lists.append(vals)
        return parse_obj_as(
            SlotValueParaphrases,
            {
                "map": default_to_regular(value_paraphrase_map),
                "cased_equivalence_lists": equivalence_lists,
                "uncased_equivalence_lists": uncased_equivalence_lists,
            },
        )

    all_slot_paraphrases = {}
    run_dir = Path(cfg.run_dir)
    for split in cfg.splits:
        split_path = run_dir / split
        if not split_path.exists():
            logger.info(f"Creating output directory {split_path}")
            split_path.mkdir(exist_ok=False, parents=True)
        collector = nested_defaultdict(set, depth=3)
        canonical_value_collector = defaultdict(set)
        setattr(SGDIterator, "mine_slot_value_paraphrases", collector)
        setattr(SGDIterator, "canonical_value_collector", canonical_value_collector)
        iterator = SGDIterator(
            cfg.data_path,
            conversation_builder=partial(
                mine_slot_value_paraphrases,
                paraphrase_map_collector=collector,
                canonical_values_collector=canonical_value_collector,
            ),
        )
        logger.info(f"Mining split: {split}")
        for _, dialogue in iterator.split_iterator(
            split,
            return_only=set(cfg.ids) if cfg.ids is not None else None,
        ):
            logger.info(f"Mining dialogue({dialogue.id}:{split})")

        value_paraphrases = get_paraphrases(
            iterator.mine_slot_value_paraphrases, iterator.canonical_value_collector
        )
        append_to_values(all_slot_paraphrases, value_paraphrases.map)
        with open(split_path / f"{split}_slot_value_paraphrase_map.json", "w") as f:
            json.dump(value_paraphrases.map, f)
        with open(split_path / f"{split}_cased_equivalent_strings.json", "w") as f:
            json.dump(value_paraphrases.cased_equivalence_lists, f)
        with open(split_path / f"{split}_uncased_equivalent_strings.json", "w") as f:
            json.dump(value_paraphrases.uncased_equivalence_lists, f)
    with open(run_dir / "slot_value_paraphrase_map.json", "w") as f:
        json.dump(all_slot_paraphrases, f)


if __name__ == "__main__":
    mine_paraphrases()
