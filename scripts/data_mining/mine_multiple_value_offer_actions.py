#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from copy import deepcopy
from functools import partial
from importlib import resources
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.sgd_conversation_builder import mine_offered_slots
from pytod.utils import cast_vals_to_sorted_list, default_to_regular, nested_defaultdict

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod.data_mining") / ".")


@hydra.main(
    config_name="mine_values_offered.yaml",
    config_path=get_config_path(),
)
def mine_multiple_values_offer(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    all_multivalue_slots = {}
    run_dir = Path(cfg.run_dir)
    for split in cfg.splits:
        split_path = run_dir / split
        if not split_path.exists():
            logger.info(f"Creating output directory {split_path}")
            split_path.mkdir(exist_ok=False)
        collector = nested_defaultdict(set, depth=1)
        setattr(SGDIterator, "multivalue_offered_action_slots", collector)
        iterator = SGDIterator(
            cfg.data_path,
            conversation_builder=partial(mine_offered_slots, collector=collector),
        )
        logger.info(f"Mining split: {split}")
        for _, dialogue in iterator.split_iterator(
            split,
            return_only=set(cfg.ids) if cfg.ids is not None else None,
        ):
            logger.info(f"Mining dialogue({dialogue.id}:{split})")
        this_split_mined_data = cast_vals_to_sorted_list(
            default_to_regular(iterator.multivalue_offered_action_slots)
        )
        all_multivalue_slots.update(deepcopy(this_split_mined_data))
        with open(split_path / f"{split}_multivalue_offer_actions.json", "w") as f:
            json.dump(this_split_mined_data, f)
    with open(run_dir / "multivalue_offer_actions.json", "w") as f:
        json.dump(all_multivalue_slots, f)


if __name__ == "__main__":
    mine_multiple_values_offer()
